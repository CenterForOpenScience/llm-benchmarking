library(tidyverse)
library(haven)
library(lme4)
library(lmerTest)
library(report)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


#"... [the study] finds substantially higher income for low-caste households residing
#in villages dominated by BACs [compared to villages dominated by upper caste]. (p. 240.)"





##### read data ----


# household.dta contains household-related data, and village.dta contains 
# village level data, the most important of which is whether the village is 
# high caste dominant or low caste dominant. This is coded in the domhigh variable.

household <- read_dta("./data/household.dta")
village <- read_dta("./data/village.dta")


# we have to change the variable types of village and domhigh, because we will
# use them to join our dataframes
village$village <- as.character(village$village)
village$domhigh <- as.character(village$domhigh)

# we create a separate dataframe to join with the household data
village_domhigh <- village[,c("village", "domhigh")]

# changing variable types in the household dataframe, too
household$village <- as.character(household$village)
household$caste <- as.character(household$caste)

# we create a joint dataframe and join by the village ID (village variable)
dat <- household %>% 
  select(hhcode, village, caste, totinc)

# we filter the records where the village dominance is not available
dat <- left_join(dat, village_domhigh) %>% 
  filter(domhigh == 1 | domhigh == 0)

##### analysis ----

#we define a random slope model
lmm_mod <- lmer(totinc ~ caste + domhigh + caste*domhigh + (1|village), 
                data = dat)
summary(lmm_mod)

sink("results.txt")
report(lmm_mod)
sink()

