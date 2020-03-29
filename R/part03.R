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
