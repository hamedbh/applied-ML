
# Classification Models ---------------------------------------------------

library(tidymodels)


# Measuring Performance in Classification ---------------------------------

two_class_example %>% head(4)

two_class_example %>% 
    conf_mat(truth = truth, estimate = predicted)

# can get certain metrics directly from the dataset
two_class_example %>% 
    accuracy(truth = truth, estimate = predicted)

two_class_example %>% 
    ppv(truth = truth, estimate = predicted)

# can also create a roc object
roc_obj <- two_class_example %>% 
    roc_curve(truth = truth, Class1)

# or just calculate the AUC
two_class_example %>% 
    roc_auc(truth = truth, Class1)

autoplot(roc_obj) + theme_light()


# Example Data ------------------------------------------------------------

# Using Amazon food reviews, some text processing involved too

library(modeldata)
data(small_fine_foods)

library(textrecipes)
library(textfeatures)

# quick helper function
count_to_binary <- function(x) {
    factor(ifelse(x != 0, "present", "absent"),
           levels = c("present", "absent"))
}

text_rec <- recipe(score ~ product + review,
                   data = training_data) %>% 
    update_role(product, new_role = "id") %>% 
    step_mutate(review_raw = review) %>% # creates a temporary copy
    # next step creates some standards numeric features from the text, e.g. 
    # words, punctuation
    step_textfeature(review_raw) %>% 
    # now tokenise and remove stop words (not always appropriate)
    step_tokenize(review) %>% 
    step_stopwords(review) %>% 
    step_stem(review) %>% 
    step_texthash(review, signed = FALSE, num_terms = 1024) %>% 
    step_mutate_at(starts_with("review_hash"), 
                   fn = count_to_binary) %>% 
    step_zv(all_predictors())

set.seed(8935)
text_folds <- vfold_cv(training_data, strata = "score")


# Classification Trees ----------------------------------------------------

tree_rec <- recipe(score ~ product + review,
                   data = training_data) %>% 
    update_role(product, new_role = "id") %>% 
    step_mutate(review_raw = review) %>% # creates a temporary copy
    step_textfeature(review_raw) %>% 
    step_tokenize(review) %>% 
    step_stopwords(review) %>% 
    step_stem(review) %>% 
    step_texthash(review, 
                  signed = FALSE, 
                  # now number of hash terms is a tuning parameter
                  num_terms = tune()) %>% 
    step_zv(all_predictors())

cart_mod <- decision_tree(cost_complexity = tune(), 
                          min_n = tune()) %>% 
    set_engine("rpart") %>% 
    set_mode("classification")

ctrl <- control_grid(save_pred = TRUE)

cart_wfl <- workflow() %>% 
    add_model(cart_mod) %>% 
    add_recipe(tree_rec)

set.seed(2553)

# wrap the tuning function call in a conditional to read from disk if possible
tune_path <- here::here("data", "cart_tune.rds")

if (file.exists(tune_path)) {
    cart_tune <- readr::read_rds(tune_path)
} else {
    cart_tune <- tune_grid(
        cart_wfl, 
        text_folds, 
        grid = 10, 
        metrics = metric_set(roc_auc), 
        control = ctrl
    )
    readr::write_rds(cart_tune, tune_path, compress = "xz", compression = 9L)
}
show_best(cart_tune, "roc_auc")
autoplot(cart_tune) + theme_light()

cart_pred <- collect_predictions(cart_tune)

cart_pred %>% 
    inner_join(
        select_best(cart_tune, "roc_auc"), 
        by = c("num_terms", "cost_complexity", "min_n")
    ) %>% 
    group_by(id) %>% 
    roc_curve(score, .pred_great) %>% 
    autoplot()

# Could pool the results across the folds and get something like an average
add_curve_data <- function(x) {
    collect_predictions(x) %>% 
        inner_join(select_best(x, "roc_auc")) %>% 
        roc_curve(score, .pred_great)
}

approx_roc_curves <- function(...) {
    curves <- map_dfr(
        list(...), 
        add_curve_data, 
        .id = "model"
    )
    
    default_cut <- curves %>% 
        group_by(model) %>% 
        arrange(abs(.threshold - .5)) %>% 
        slice(1)
    ggplot(curves) +
        aes(y = sensitivity, x = 1 - specificity, 
            col = model) +
        geom_abline(lty = 3) + 
        geom_step(direction = "vh") + 
        geom_point(data = default_cut) + 
        coord_equal() + 
        theme_light() + 
        theme(legend.position = "top")
}

approx_roc_curves(CART = cart_tune)


# Downsampling exercise ---------------------------------------------------

downsamp_rec <- recipe(score ~ product + review,
                       data = training_data) %>% 
    update_role(product, new_role = "id") %>% 
    step_downsample(score) %>% 
    step_mutate(review_raw = review) %>% # creates a temporary copy
    step_textfeature(review_raw) %>% 
    step_tokenize(review) %>% 
    step_stopwords(review) %>% 
    step_stem(review) %>% 
    step_texthash(review, 
                  signed = FALSE, 
                  # now number of hash terms is a tuning parameter
                  num_terms = tune()) %>% 
    step_zv(all_predictors())

downsamp_wfl <- workflow() %>% 
    add_model(cart_mod) %>% 
    add_recipe(downsamp_rec)

set.seed(2553)

# wrap the tuning function call in a conditional to read from disk if possible
downsamp_path <- here::here("data", "downsamp_tune.rds")

if (file.exists(downsamp_path)) {
    downsamp_tune <- readr::read_rds(downsamp_path)
} else {
    downsamp_tune <- tune_grid(
        downsamp_wfl, 
        text_folds, 
        grid = 10, 
        metrics = metric_set(roc_auc), 
        control = ctrl
    )
    readr::write_rds(downsamp_tune,
                     downsamp_path,
                     compress = "xz",
                     compression = 9L)
}
show_best(downsamp_tune, "roc_auc")
autoplot(downsamp_tune) + theme_light()

downsamp_pred <- collect_predictions(downsamp_tune)

downsamp_pred %>% 
    inner_join(
        select_best(downsamp_tune, "roc_auc"), 
        by = c("num_terms", "cost_complexity", "min_n")
    ) %>% 
    group_by(id) %>% 
    roc_curve(score, .pred_great) %>% 
    autoplot()

approx_roc_curves(CART = downsamp_tune)

# In this case the best performance for the downsampled model is slightly worse 
# than the one using the full dataset.

# Boosting ----------------------------------------------------------------


C5_mod <- boost_tree(trees = tune(), 
                     min_n = tune()) %>% 
    set_engine("C5.0") %>% 
    set_mode("classification")

C5_wfl <- update_model(cart_wfl, C5_mod)

# We will just modify our CART grid and add 
# a new parameter: 
set.seed(5793)

C5_grid <- collect_metrics(cart_tune) %>% 
    dplyr::select(min_n, num_terms) %>% 
    mutate(trees = sample(1:100, 10))

C5_path <- here::here("data", "C5_tune.rds")

if (file.exists(C5_path)) {
    C5_tune <- readr::read_rds(C5_path)
} else {
    C5_tune <- tune_grid(
        C5_wfl,
        text_folds,
        grid = C5_grid,
        metrics = metric_set(roc_auc),
        control = ctrl
    )
    readr::write_rds(C5_tune, C5_path, compress = "xz", compression = 9L)
}

approx_roc_curves(C5 = C5_tune, CART = cart_tune)

show_best(C5_tune, "roc_auc")

(best_C5 <- select_best(C5_tune, "roc_auc"))

C5_wfl_final <- C5_wfl %>% 
    finalize_workflow(best_C5) %>% 
    fit(data = training_data)

C5_wfl_final

test_probs <- C5_wfl_final %>% 
    predict(testing_data, type = "prob") %>% 
    bind_cols(testing_data %>% dplyr::select(score)) %>% 
    bind_cols(predict(C5_wfl_final, testing_data))

roc_auc(test_probs, score, .pred_great)

conf_mat(test_probs, score, .pred_class)

roc_values <- roc_curve(test_probs, score, .pred_great)

autoplot(roc_values)


# Naïve Bayes -------------------------------------------------------------

nb_rec <- tree_rec %>%
    step_mutate_at(starts_with("review_hash"), fn = count_to_binary)

library(discrim)
nb_mod <- naive_Bayes() %>% set_engine("klaR")

nb_path <- here::here("data", "nb_tune.rds")

if (file.exists(nb_path)) {
    nb_tune <- readr::read_rds(nb_path)
} else {
    nb_tune <- tune_grid(
        nb_mod,
        nb_rec,
        resamples = text_folds,
        grid = tibble(num_terms = floor(2^seq(8, 12, by = 0.5))),
        metrics = metric_set(roc_auc),
        control = ctrl
    )
    readr::write_rds(nb_tune, nb_path, compress = "xz", compression = 9L)
}

autoplot(nb_tune) + 
    scale_x_continuous(trans = log2_trans()) + 
    theme_light()

approx_roc_curves(CART = cart_tune, 
                  C5 = C5_tune, 
                  "Naive Bayes" = nb_tune)


# Some Cool Stuff ---------------------------------------------------------

# Can deploy models on SQL
library(tidypredict)
library(dbplyr)

lin_reg_fit <- lm(Sepal.Width ~ ., data = iris)

tidypredict_fit(lin_reg_fit)

tidypredict_sql(lin_reg_fit, con = simulate_dbi())
