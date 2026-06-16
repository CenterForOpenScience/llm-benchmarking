### Project: Multi100 Task #2
### Author ID: ZZY2R
### Claim to verify: "Our results confirm that priming the first lady's ethnicity increases support for President Yayi among her coethnics"

# clean workspace
rm(list=ls())

# set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# load packages
library(tidyverse)
library(skimr)
library(report)
library(sjPlot)
library(emmeans)

### and prepare data data
df <- read.csv('Benin2012survey.csv')

# filter data for analysis (keep only the three experimental conditions from the paper)
df <- df %>% 
  filter(passage %in% c('FonFemme', 'Femme', 'Control'))
table(df$passage)

# remove Yorouba (presient's) co-ethnics 
df <- df %>% 
  filter(ethnie != 'Yorouba')

# add the vote variable
df$vote <- ifelse(df$BoniVote == 1, 'yes',
                  ifelse(df$BoniVote == 0, 'no',
                         NA))
df$vote <- as.factor(df$vote)
table(df$vote)

# code ethnicity as Chantal coethnic vs. non-coethnic
df$coethnic <- ifelse(df$ethnie == 'Fon', 'co-ethnic(FON)', 'non-coethnic')
df$coethnic <- as.factor(df$coethnic)
table(df$coethnic)

# add the condition variable
df$condition <- ifelse(df$passage == 'Femme', 'wife',
                       ifelse(df$passage == 'FonFemme', 'wife(FON)',
                              'control'))
df$condition <- as.factor(df$condition)

### analyzes TASK 2
# claim to test: "Our results confirm that priming the first lady's ethnicity increases support for President Yayi among her coethnics"

m1 <- glm(vote ~ condition * coethnic, df, family = 'binomial')

# model results
round(summary(m1)$coef, 3)

summary(m1)
