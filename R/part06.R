
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

