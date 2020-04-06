library(tidymodels)

# Pick up from last section …

library(AmesHousing)
ames <- make_ames() %>% 
    dplyr::select(-matches("Qu"))
set.seed(4595)
data_split <- initial_split(ames, strata = "Sale_Price")
ames_train <- training(data_split)
ames_test  <- testing(data_split)
perf_metrics <- metric_set(rmse, rsq, ccc)

ames_rec <- recipe(
    Sale_Price ~ Bldg_Type + Neighborhood + Year_Built + 
        Gr_Liv_Area + Full_Bath + Year_Sold + Lot_Area +
        Central_Air + Longitude + Latitude,
    data = ames_train
) %>%
    step_log(Sale_Price, base = 10) %>%
    step_BoxCox(Lot_Area, Gr_Liv_Area) %>%
    step_other(Neighborhood, threshold = 0.05)  %>%
    step_dummy(all_nominal()) %>%
    step_interact(~ starts_with("Central_Air"):Year_Built) %>%
    step_bs(Longitude, Latitude, deg_free = 5)


# Resampling --------------------------------------------------------------

set.seed(2453)

cv_splits <- vfold_cv(ames_train)

cv_splits
cv_splits$splits[[1]]

cv_splits$splits[[1]] %>% 
    analysis() %>% 
    dim()

cv_splits$splits[[1]] %>% 
    assessment() %>% 
    dim()

knn_mod <- nearest_neighbor(neighbors = 5) %>% 
    set_engine("kknn") %>% 
    set_mode("regression")

knn_wfl <- workflow() %>% 
    add_model(knn_mod) %>% 
    add_formula(log10(Sale_Price) ~ Longitude + Latitude)

fit(knn_wfl, data = ames_train)

knn_res <- cv_splits %>% 
    mutate(
        workflows = map(
            splits, 
            ~ fit(knn_wfl, data = analysis(.x))
        )
    )
knn_res

knn_pred <- map2_dfr(
    knn_res$workflows, 
    knn_res$splits, 
    ~ predict(.x, assessment(.y)), 
    .id = "fold"
)
knn_pred

(prices <- map_dfr(
    knn_res$splits, 
    ~ assessment(.x) %>% select(Sale_Price)
) %>% 
        mutate(Sale_Price = log10(Sale_Price)))

(rmse_estimates <- knn_pred %>% 
        bind_cols(prices) %>% 
        group_by(fold) %>% 
        do(rmse = rmse(., Sale_Price, .pred)) %>% 
        unnest(cols = c(rmse))
)

mean(rmse_estimates$.estimate)

# Can actually do all of that automatically with {tune}

easy_eval <- fit_resamples(
    knn_wfl, 
    resamples = cv_splits, 
    control = control_resamples(
        save_pred = TRUE
    )
)
easy_eval

collect_predictions(easy_eval) %>% 
    arrange(.row) %>% 
    head()

collect_metrics(easy_eval)

collect_metrics(easy_eval, 
                summarize = FALSE)


# Model Tuning ------------------------------------------------------------


# Cannot estimate some parameters from the data

# Start using grid search. Elements required: 
# 1. Set of candidate hyperparameter values to evaluate (the grid)
# 2. A metric for model performance
# 3. A resampling scheme to reliably estimate out-of-sample performance

# Two main types of grid: regular and non-regular

# Regular grids

penalty()
mixture()

(glmn_param <- parameters(penalty(), mixture()))

glmn_grid <- glmn_param %>% 
    grid_regular(levels = c(10, 5)) #NB. different number for each

glmn_grid %>% head()
# The regular grid is on the transformed scale, then the values in the tibble 
# are on the original scale

# Non-regular grids are based on space-filling designs, keeping parameter 
# combinations away from each other.

set.seed(7454)

glmn_sfd <- glmn_param %>% 
    grid_max_entropy(size = 50)

glmn_sfd %>% head()

(glmn_set <- parameters(
    lambda = penalty(), 
    mixture()
))

(glmn_set <- glmn_set %>% 
        update(lambda = penalty(c(-5, -1))))

# in some cases the possible parameter values will depend on the data

(rf_set <- parameters(
    mtry(), 
    trees()
))

finalize(rf_set, 
         mtcars %>% 
             dplyr::select(-mpg))

?nearest_neighbor

# Params are neighbors, weight_func, and dist_power
knn_set <- parameters(neighbors(), 
           dist_power())

knn_grid <- knn_set %>% 
    grid_regular(levels = c(10, 4)) %>% 
    add_column(id = "Regular", 
               .before = 1)

knn_max_ent <- knn_set %>% 
    grid_max_entropy(size = 40) %>% 
    add_column(id = "Max Entropy", 
               .before = 1)

knn_latin <- knn_set %>% 
    grid_latin_hypercube(size = 40) %>% 
    add_column(id = "Latin Hypercube", 
               .before = 1)

bind_rows(knn_grid, knn_max_ent, knn_latin) %>% 
    ggplot(aes(neighbors, dist_power)) + 
    geom_point() + 
    facet_wrap(~ id) + 
    theme_light() + 
    labs(
        title = "Hyperparameter grids"
    )

# Tag parameters to be tuned
knn_mod <- nearest_neighbor(
    neighbors = tune(), 
    weight_func = tune()
) %>% 
    set_engine("kknn") %>% 
    set_mode("regression")

parameters(knn_mod)

set.seed(522)    

knn_grid <- knn_mod %>% 
    parameters() %>% 
    grid_regular(levels = c(15, 5))

ctrl <- control_grid(verbose = TRUE)

knn_tune <- tune_grid(
    knn_mod, 
    ames_rec, 
    resamples = cv_splits, 
    grid = knn_grid, 
    control = ctrl
)

knn_tune

knn_tune$.metrics[[1]]

show_best(knn_tune, "rmse")
