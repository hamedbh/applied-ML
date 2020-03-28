
# Data Exploration --------------------------------------------------------

library(tidyverse)
library(AmesHousing)
ames <- make_ames()
ames
ames %>% 
    ggplot(aes(Sale_Price)) + 
    geom_density() + 
    # scale_x_log10()
    NULL
skimr::skim(ames)
# No missing data (seemingly)
# ── Data Summary ────────────────────────
# Values
# Name                       ames  
# Number of rows             2930  
# Number of columns          81    
# _______________________          
# Column type frequency:           
# factor                   46    
# numeric                  35   

# Factors
# Neighborhood has 28 categories

ames %>% 
    select_if(is.double) %>% 
    pivot_longer(everything()) %>% 
    ggplot(aes(value)) + 
    geom_density() + 
    facet_wrap(~ name, scales = "free")
# some right-skew variables
# some that seem to have little/no variance
# some are double that prob should be integer (e.g. Bsmt_Full_Bath)

ames %>% 
    select_if(is.integer) %>% 
    pivot_longer(everything()) %>% 
    ggplot(aes(value)) + 
    geom_bar() + 
    facet_wrap(~ name, scales = "free")
