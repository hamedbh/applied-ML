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

