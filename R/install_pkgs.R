cran_pkgs <- c(
    'AmesHousing',
    'C50',
    'devtools',
    'discrim',
    'earth',
    'ggthemes',
    'glmnet',   # See important note below
    'klaR',
    'lubridate',
    'modeldata',
    'party',
    'pROC',
    'rpart',
    'stringr',
    'textfeatures',
    'tidymodels'
)

install.packages(cran_pkgs, 
                 dependencies = TRUE, 
                 repos = "http://cran.rstudio.com")

devtools::install_github(c(
    "tidymodels/tidymodels",
    "tidymodels/tune",
    "tidymodels/textrecipes",
    "koalaverse/vip",
    "gadenbuie/countdown"
))
