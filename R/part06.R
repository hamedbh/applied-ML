
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
    ifelse(x != 0, "present", "absent")
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
    readr::write_rds(cart_tune, tune_path)
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
