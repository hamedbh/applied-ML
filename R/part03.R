library(tidymodels)
library(AmesHousing)
ames <- make_ames() %>% 
    dplyr::select(-matches("Qu"))
set.seed(4595)
data_split <- initial_split(ames, strata = "Sale_Price")
ames_train <- training(data_split)
ames_test  <- testing(data_split)
lm_mod <- linear_reg() %>% 
    set_engine("lm")
perf_metrics <- metric_set(rmse, rsq, ccc)


# Feature Engineering -----------------------------------------------------

# Now dealing with preprocessing/feature engineering. Motivation: 

# May need to have predictors on the same scale
# May need to remove correlated variables with filters, or extract principal 
# components
# Transform scale of predictors
# Transform predictors to a more useful form, e.g. changing a date to the day 
# of the week
# Missing data imputation
# Adding new meaningful features (e.g. distance from an important landmark)

# recreate the formula used previously as a recipe, plus some extra stuff
mod_rec <- recipe(Sale_Price ~ Longitude + Latitude + Neighborhood, 
                  data = ames_train) %>% 
    step_log(Sale_Price, base = 10) %>% 
    
    # lump together small groups in Neigborhood 
    step_other(Neighborhood, 
               threshold = 0.05) %>% 
    
    # turn any factor to a dummy variable
    step_dummy(all_nominal())
mod_rec

# recipe() defines the pre proc
# prep() calculates stats from the training data
# bake() or juice() applies the pre proc to the data

mod_rec_trained <- mod_rec %>% 
    prep(training = ames_train, 
         verbose = TRUE)
mod_rec_trained
juice(mod_rec_trained)
bake(mod_rec_trained, new_data = ames_test)

# now build alternative recipe filtering out zero-variance predictors

mod_filter_rec <- recipe(Sale_Price ~ Longitude + Latitude + Neighborhood, 
                         data = ames_train) %>% 
    step_log(Sale_Price, base = 10) %>% 
    
    step_dummy(all_nominal()) %>% 
    
    step_zv(all_predictors())
mod_filter_rec

mod_filter_rec_trained <- mod_filter_rec %>% 
    prep(training = ames_train, 
         verbose = TRUE)
mod_filter_rec_trained
juice(mod_filter_rec_trained)
bake(mod_filter_rec_trained, new_data = ames_test)

# Which approach is better? 
# Depends on context. What would happen if there was a category not in the 
# training data that was in the test data?
d <- tibble(
    f = factor(letters[c(1, 2, 3, 3, 2, 3, 4)], 
               levels = letters[1:4]), 
    y = rnorm(7)
)
weird_rec <- recipe(y ~ f, 
                    data = d %>% 
                        slice(1:6)) %>% 
    step_dummy(all_nominal()) %>% 
    step_zv(all_predictors()) %>% 
    prep()
weird_rec
juice(weird_rec)

weird_rec %>% 
    bake(new_data = 
             d %>% 
             slice(7))
# # A tibble: 1 x 3
#           y   f_b   f_c
#       <dbl> <dbl> <dbl>
#     1  1.07     0     0
#
# the `d` category is missing from the baked dataset, so this seems shaky


# Interaction effects -----------------------------------------------------

price_breaks <- (1:6)*(10^5)

ames_train %>% 
    ggplot(aes(Year_Built, Sale_Price)) + 
    geom_point(alpha = 0.4) + 
    scale_y_log10() + 
    geom_smooth(method = "loess") + 
    theme_light()

# now separate by another variable
ames_train %>% 
    count(Central_Air) %>% 
    mutate(pct = n/sum(n))

library(MASS)
ames_train %>% 
    ggplot(aes(Year_Built, Sale_Price)) + 
    geom_point(alpha = 0.4) + 
    scale_y_log10() + 
    facet_wrap(~ Central_Air, nrow = 2) + 
    geom_smooth(method = "rlm") + 
    theme_light()

mod1 <- lm(log10(Sale_Price) ~ Year_Built + 
               Central_Air,                          data = ames_train)
mod2 <- lm(log10(Sale_Price) ~ Year_Built + 
               Central_Air + Year_Built:Central_Air, data = ames_train)
anova(mod1, mod2)

# can set up this interaction in the recipe
interact_rec <- recipe(Sale_Price ~ Year_Built + Central_Air, 
                       data = ames_train) %>% 
    step_log(Sale_Price) %>% 
    step_dummy(Central_Air) %>% 
    step_interact(~ starts_with("Central_Air"):Year_Built)

interact_rec

interact_rec %>% 
    prep(training = ames_train) %>% 
    juice() %>% 
    slice(153:157)


# Principal Component Analysis --------------------------------------------


bivariate_train %>% 
    ggplot(aes(A, B, colour = Class)) + 
    geom_point(alpha = 0.3) + 
    scale_colour_manual(values = c(One = "darkslateblue", 
                                   Two = "orangered")) + 
    theme_light() + 
    labs(
        x = "PredictorA", 
        y = "PredictorB"
    ) + 
    theme(legend.position = "top") 
# highly correlated, right-skew

# An inverse transformation resolves the skewness. 
bivariate_train %>% 
    mutate_at(
        vars(A, B), 
        ~ 1/.x
    ) %>% 
    ggplot(aes(A, B, colour = Class)) + 
    geom_point(alpha = 0.3) + 
    scale_colour_manual(values = c(One = "darkslateblue", 
                                   Two = "orangered")) + 
    theme_light() + 
    labs(
        x = "1/A", 
        y = "1/B"
    ) + 
    theme(legend.position = "top") 

# Can do this with a recipe
bivariate_rec <- recipe(
    Class ~ ., 
    data = bivariate_train
) %>% 
    step_BoxCox(all_predictors()) %>% 
    prep(training = bivariate_train)

inverse_train_data <- bivariate_rec %>% 
    juice()
inverse_test_data <- bivariate_rec %>% 
    bake(new_data = bivariate_test)

inverse_train_data %>% 
    ggplot(aes(A, B, colour = Class)) + 
    geom_point(alpha = 0.3) + 
    scale_colour_manual(values = c(One = "darkslateblue", 
                                   Two = "orangered")) + 
    theme_light() + 
    labs(
        x = "1/A", 
        y = "1/B"
    ) + 
    theme(legend.position = "top") 

# Now build on this by using PCA via recipe()
bivariate_pca_rec <- recipe(
    Class ~ A + B, 
    data = bivariate_train
) %>% 
    step_BoxCox(all_predictors()) %>% 
    step_normalize(all_predictors()) %>% 
    step_pca(all_predictors()) %>% 
    prep(training = bivariate_train)

pca_test <- bake(bivariate_pca_rec, 
                 new_data = bivariate_test)

# extend the range for plotting
pca_rng <- extendrange(c(pca_test$PC1, pca_test$PC2))

pca_test %>% 
    ggplot(aes(PC1, PC2, colour = Class)) + 
    geom_point(alpha = 0.2, cex = 1.5) + 
    scale_colour_manual(values = c(One = "darkslateblue", 
                                   Two = "orangered")) + 
    xlim(pca_rng) + 
    ylim(pca_rng) + 
    theme_light() + 
    theme(legend.position = "top")

# This shows an important aspect of PCA: the most important component with 
# respect to variation in the dataset may not be connected with the outcome. In 
# this case it is PC2 that is most predictive of Class. Can test this with 
# models

pc1_mod <- logistic_reg() %>% 
    set_engine("glm") %>% 
    fit(Class ~ PC1, 
        data = juice(bivariate_pca_rec))

pc2_mod <- logistic_reg() %>% 
    set_engine("glm") %>% 
    fit(Class ~ PC2, 
        data = juice(bivariate_pca_rec))

tibble(
    PC = paste0("PC", 1:2), 
    mod = list(pc1_mod, pc2_mod)
) %>% 
    mutate(
        preds = map(mod, 
                    ~ predict(.x, 
                              pca_test, 
                              type = "prob") %>% 
                        bind_cols(
                            pca_test %>% dplyr::select(Class)
                        )
        )
    ) %>% 
    transmute(PC, 
              auc = map_dbl(preds, 
                            ~ roc_auc(.x, 
                                      truth = Class, 
                                      .pred_One) %>% 
                                pull(.estimate)))


# Recipes and Models ------------------------------------------------------

(ames_rec <- recipe(
    Sale_Price ~ Bldg_Type + Neighborhood + Year_Built + 
        Gr_Liv_Area + Full_Bath + Year_Sold + Lot_Area +
        Central_Air + Longitude + Latitude,
    data = ames_train
) %>% 
    # log transform the outcome variable to deal with skew
    step_log(all_outcomes(), base = 10) %>% 
    # handle extreme skew in two of the predictors
    step_BoxCox(Lot_Area, Gr_Liv_Area) %>% 
    # clump smaller Neighborhood groups into other
    step_other(Neighborhood, threshold = 0.05) %>% 
    step_dummy(all_nominal()) %>% 
    step_interact(~ starts_with("Central_"):Year_Built) %>% 
    # create a B-spline expansion for geo variables to account for non-linearity
    step_ns(Longitude, Latitude, deg_free = 5) %>% 
    prep()
)

lm_fit <- lm_mod %>% 
    fit(Sale_Price ~ ., 
        data = juice(ames_rec))

glance(lm_fit$fit)

ames_test_processed <- bake(ames_rec, 
                            ames_test, 
                            all_predictors())

# Keeping track of separate objects and using juice() or bake() adds steps. 
# Use {workflows} objects to bundle the model and the pre-proc recipe together. 

(ames_wfl <- workflow() %>% 
        add_model(lm_mod) %>% 
        add_recipe(ames_rec))

(ames_wfl_fit <- fit(ames_wfl, ames_train))

predict(ames_wfl_fit, ames_test) %>% slice(1:5)

# Knowledge check
# Match function to package

# Fit a K-NN model - parsnip

# Extract holidays from dates - recipes

# Make a training/test split - rsample

# Bundle a recipe and model - workflows

# Is high in vitamin A - carrot

# Compute R2 - yardstick

# Bin a predictor (but seriously,...don't) - recipes

# none? - ggvis