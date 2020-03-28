library(tidymodels)
library(AmesHousing)

ames <- make_ames() %>% 
    select(-matches("Qu"))

nrow(ames)
# 2,930 obs

set.seed(4595)
data_split <- initial_split(
    ames, 
    prop = 0.75, 
    strata = "Sale_Price"
)

ames_train <- training(data_split)
ames_test <- testing(data_split)

nrow(ames_train)/nrow(ames_test)
nrow(ames_train)/nrow(ames)

# start by fitting a really simple linear model with a few predictors
simple_lm <- lm(
    log10(Sale_Price) ~ Longitude + Latitude, 
    data = ames_train
)

# create a tibble with the stats for each data point
simple_lm_values <- augment(simple_lm)
# get the coeffs
tidy(simple_lm)
# see that the model sucks!
glance(simple_lm)[1:3]

# could create linear models in lots of ways. use parsnip to get a unified 
# interface
spec_lin_reg <- linear_reg()
spec_lin_reg

lm_mod <- spec_lin_reg %>% 
    set_engine("lm")

lm_fit <- lm_mod %>% 
    fit(
        log10(Sale_Price) ~ Longitude + Latitude, 
        data = ames_train
    )
lm_fit
# coeffs same as above

# can use different interfaces
ames_train_log <- ames_train %>% 
    mutate(Sale_Price_Log = log10(Sale_Price))

fit_xy(
    lm_mod, 
    x = ames_train_log %>% 
        select(Longitude, Latitude), 
    y = ames_train_log %>% 
        pull(Sale_Price_Log)
)
# or engines
spec_stan <- spec_lin_reg %>% 
    set_engine("stan", 
               chains = 4, 
               iter = 1000)
fit_stan <- spec_stan %>% 
    fit(
        log10(Sale_Price) ~ Longitude + Latitude, 
        data = ames_train
    )

coef(fit_stan$fit)

# now switch to a different model type altogether, same interface
fit_knn <- nearest_neighbor(
    mode = "regression"
) %>% 
    set_engine("kknn") %>% 
    fit(
        log10(Sale_Price) ~ Longitude + Latitude, 
        data = ames_train
    )
fit_knn

# what follows is v bad practice to jump straight to test set, done here just 
# for teaching
test_pred <- lm_fit %>% 
    predict(ames_test) %>% 
    bind_cols(ames_test) %>% 
    mutate(log_price = log10(Sale_Price))

test_pred %>% 
    select(log_price, .pred)

perf_metrics <- metric_set(rmse, rsq, ccc)

test_pred %>% 
    perf_metrics(truth = log_price, 
                 estimate = .pred)
