
# Regression Models -------------------------------------------------------

library(tidymodels)

# Using a new dataset, predicting number of people at L-train station in 
# Chicago
data("Chicago")

range(Chicago$date)
Chicago %>% 
    ggplot(aes(x = date, y = Clark_Lake)) + 
    stat_smooth(method = "gam") + 
    facet_wrap(~ Cubs_Home)

library(stringr)

us_hols <- timeDate::listHolidays() %>% 
    str_subset("(^US)|(Easter)")

chi_rec <- recipe(ridership ~ ., 
                  data = Chicago) %>% 
    step_holiday(date, holidays = us_hols) %>% 
    step_date(date) %>% 
    step_rm(date) %>% 
    step_dummy(all_nominal()) %>% 
    step_zv(all_predictors())

# For this sort of time series can use rolling periods for out of time testing
chi_folds <- rolling_origin(
    Chicago, 
    initial = 364 * 15, 
    assess = 7 * 4, 
    skip = 7 * 4, 
    cumulative = FALSE
)

chi_folds %>% nrow()


# Linear models -----------------------------------------------------------

# Use regularisation to mitigate the collinearity and reduce number of features

glmn_grid <- expand.grid(
    penalty = 10^(seq(-3, -1, length.out = 20)), 
    mixture = (0:5)/5
)

glmn_rec <- chi_rec %>% 
    step_normalize(all_predictors())

glmn_mod <- linear_reg(penalty = tune(), 
                       mixture = tune()) %>% 
    set_engine("glmnet")

ctrl <- control_grid(save_pred = TRUE)

# Can use parallel processing to get a big speed-up

library(doParallel)
cl <- makeCluster(2)
registerDoParallel(cl)

glmn_tune <- tune_grid(
    glmn_mod, 
    glmn_rec, 
    resamples = chi_folds, 
    grid = glmn_grid, 
    control = ctrl
)

stopCluster(cl)

rmse_vals <- collect_metrics(glmn_tune) %>% 
    filter(.metric == "rmse")

rmse_vals %>% 
    mutate(mixture = format(mixture)) %>% 
    ggplot(aes(penalty, mean, colour = mixture)) + 
    geom_line() + 
    geom_point() + 
    scale_x_log10()

# Compare with autoplot method
autoplot(glmn_tune)

show_best(glmn_tune, metric = "rmse")

(best_glmn <- select_best(glmn_tune, metric = "rmse"))

(glmn_pred <- collect_predictions(glmn_tune) %>% 
        # keep just the best model
        inner_join(
            best_glmn, 
            by = c("penalty", "mixture")
        )
)

# Can see the individual predictions, which were poor
ggplot(glmn_pred, aes(x = .pred, y = ridership)) + 
    geom_abline(col = "green") + 
    geom_point(alpha = .3) + 
    coord_equal() + 
    theme_light()

large_resid <- glmn_pred %>% 
    mutate(resid = ridership - .pred) %>% 
    arrange(desc(abs(resid))) %>% 
    slice(1:4)

library(lubridate)

Chicago %>% 
    slice(large_resid$.row) %>% 
    select(date) %>% 
    mutate(day = wday(date, label = TRUE)) %>% 
    bind_cols(large_resid)

# The large underprediction on 2016-03-12 was because of a Trump rally. Model 
# also overpredicted on two holidays. 4 July has an indicator in the list of 
# holidays, but Boxing Day (26 Dec) doesn't. 

# Now can prep a model with the best parameters

glmn_rec_final <- prep(glmn_rec)

glmn_mod_final <- finalize_model(glmn_mod, 
                                 best_glmn)

glmn_fit <- glmn_mod_final %>% 
    fit(ridership ~ ., 
        data = juice(glmn_rec_final))
glmn_fit

library(glmnet)

plot(glmn_fit$fit, xvar = "lambda")

# They advise against using predict(glmn_fit$fit), instead use the predict() 
# method on the object produced by fit().

# Can tidy up the coefficients and plot manually in ggplot
tidy_coefs <- glmn_fit %>% 
    broom::tidy() %>% 
    dplyr::filter(term != "(Intercept)") %>% 
    dplyr::select(-step, -dev.ratio)

# Get the closest value to the optimum lambda
delta <- abs(tidy_coefs$lambda - best_glmn$penalty)
lambda_opt <- tidy_coefs$lambda[which.min(delta)]

# keep the term labels for those with large coefficients and the right lambda
label_coefs <- tidy_coefs %>% 
    mutate(abs_estimate = abs(estimate)) %>% 
    dplyr::filter(abs_estimate >= 1.1) %>% 
    dplyr::distinct(term) %>% 
    inner_join(tidy_coefs, 
               by = "term") %>% 
    dplyr::filter(lambda == lambda_opt)

tidy_coefs %>% 
    ggplot(aes(lambda, estimate, group = term, colour = term, label = term)) + 
    geom_vline(xintercept = lambda_opt, 
               lty = 3) + 
    geom_line(alpha = 0.4) + 
    theme_light() + 
    theme(legend.position = "none") + 
    scale_x_log10() + 
    ggrepel::geom_text_repel(data = label_coefs, aes(x = 0.005))

library(vip)

vip(glmn_fit, 
    num_features = 20L, 
    lambda = best_glmn$penalty)


# Multivariate Adaptive Regression Splines --------------------------------

# Could let MARS decide on the number of terms, but tune for the dimensions
mars_mod <- mars(prod_degree = tune())

# But instead will tune both
mars_mod <- mars(num_terms = tune("mars terms"), 
                 prod_degree = tune(), 
                 prune_method = "none") %>% 
    set_engine("earth") %>% 
    set_mode("regression")

# Because MARS is based on linear regression need to deal with the 
# highly-correlated predictors. Use PCA this time.

mars_rec <- chi_rec %>% 
    step_normalize(one_of(!!stations)) %>% 
    step_pca(one_of(!!stations), num_comp = tune("pca comps"))


# Segue - Interative search methods ---------------------------------------

# Description of Bayesian optimisation in the slides is good, although the 
# LaTeX doesn't seem to have rendered correctly. 

# Optional step of adding model and recipe to a workflow

chi_wflow <- workflow() %>% 
    add_recipe(mars_rec) %>% 
    add_model(mars_mod)

chi_set <- chi_wflow %>% 
    parameters() %>% 
    update(
        `pca comps` = num_comp(c(0L, 20L)), # 0L would mean not using PCA
        `mars terms` = num_terms(c(2L, 100L))
    )

# Use parallel processing

library(doMC)
registerDoMC(cores = parallel::detectCores(logical = FALSE))

ctrl <- control_bayes(verbose = TRUE, 
                      save_pred = TRUE)
# Some defaults:
#   - Uses expected improvement with no trade-off. See ?exp_improve().
#   - RMSE is minimized
set.seed(7891)

# wrap the tuning function call in a conditional to read from disk if possible, 
# as the optimisation process takes ages!
tune_path <- here::here("data", "mars_tune.rds")

if (file.exists(tune_path)) {
    mars_tune <- readr::read_rds(tune_path)
} else {
    mars_tune <- tune_bayes(
        chi_wflow,
        resamples = chi_folds,
        iter = 25,
        param_info = chi_set,
        metrics = metric_set(rmse),
        initial = 4,
        control = ctrl
    )
    readr::write_rds(mars_tune, tune_path)
}

autoplot(mars_tune, type = "performance") + 
    theme_light()
autoplot(mars_tune, type = "marginals") + 
    theme_light()
autoplot(mars_tune, type = "parameters") + 
    theme_light()

show_best(mars_tune, "rmse")

best_mars <- select_best(mars_tune, "rmse")

mars_pred <- mars_tune %>% 
    collect_predictions() %>% 
    inner_join(
        best_mars, 
        by = c("mars terms", "prod_degree", "pca comps")
    )
ggplot(mars_pred, aes(x = .pred, y = ridership)) + 
    geom_abline(col = "green") + 
    geom_point(alpha = .3) + 
    coord_equal()


final_mars_wfl <- finalize_workflow(chi_wflow, 
                                    best_mars) %>% 
    fit(data = Chicago)

final_mars_wfl %>% 
    pull_workflow_fit() %>% 
    vip(num_features = 20L, type = "gcv") + 
    theme_light()
