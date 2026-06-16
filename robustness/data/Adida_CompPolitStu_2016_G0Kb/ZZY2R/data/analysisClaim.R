### Project: Multi100 task 1
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

### and prepare data data
df <- read.csv('Benin2012survey.csv')

# explore data
head(df)
glimpse(df)
str(df)
skim(df)
report(df)

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
table(df$condition)

# sample descriptives
report_participants(df)
table(df$coethnic)
table(df$condition)

### analyzes
#claim to test: "Our results confirm that priming the first lady's ethnicity increases support for President Yayi among her coethnics"

m0 <- glm(vote ~ condition + coethnic, df, family = 'binomial')
m1 <- glm(vote ~ condition * coethnic, df, family = 'binomial')
anova(m0, m1, test = 'Chisq') # the model with interactions is better

# model results
round(summary(m1)$coef, 3)

# plot model with 
sjPlot::plot_model(m1,
                   transform = NULL, 
                   show.intercept = T,
                   show.values = T, 
                   value.offset = .3)+
  theme_minimal()

# visualize interaction
sjPlot::plot_model(m1, type = 'int')+
  theme_minimal()  

emmeans::emmip(m1, coethnic ~ condition, CIs = T, plotit = T)+
  theme_minimal()

# post hocs
res <- emmeans::emmeans(m1, ~ condition * coethnic)
pairs(res)
emmeans::contrast(res,
                  'revpairwise',
                  by = 'coethnic',
                  adjust = 'tukey')

# results interpretation:

# A total of 1104 participants completed the questionnaire distributed in Benin in 2012. For the main analysis, I filtered the data in the following way. First, I excluded 637 participants who were assigned a different experimental condition than the three of interest: (a) "control," in which participants were not given a reference to President Yayi's wife (n = 140); (b) "wife," in which the text participants read mentioned that President Yayi had a wife (n = 132); and (c) "Fon wife," in which the text mentioned that President Yayi had a Fon wife (n = 141). I then excluded an additional 54 participants who identified as Yoruba, President Yayi's ethnic group (this subsample was also excluded from the analysis in the main article). The resulting sample included a total of 413 participants (mean age = 32.5, SD = 10.2, 47.7% female, 152 Fon [President's wife coethnic group], 261 non-coethnic).

# This analysis was conducted to verify the following claim in the abstract of the original paper: "Our results confirm that priming the First Lady's ethnicity increases support for President Yayi among her coethnics." To test this claim, I fitted a logistic regression model with the willingness to vote for President Yayi in elections (0 = no; 1 = yes) as the dependent variable, the experimental condition (control, wife, Fon wife) as the first predictor, and participants' coethnicity (coethnic [FON], non-coethnic ) as the second predictor. Considering that the claim refers to the interaction between the experimental condition (referred to as prime in the original paper) and participants' coethnicity (or lack thereof), I added the interaction terms between the two predictors.
# 
# The results of this model show no significant interaction between the experimental condition - disclosure of information about the President's wife (without reference to her ethnicity) - and the coethnicity of the participants, b = -0.20, SE = 0.64, z = -0.31 p = .758. However, there was a significant interaction between disclosure of information about the President's Fon wife and participant coethnicity, b = -1.58, SE = 0.55, z = -2.88, p = .004. Because of the presence of a significant interaction, I do not report the main effects. Visual inspection of the nature of the significant interaction shows trends in the same direction as in the original paper. Post-hoc analysis (b coefficients are given on a log odds ratio scale) with a Tukey p-value correction for multiple comparisons indeed revealed that the probability of voting for President Yayi was higher among Fon coethnics when participants were informed that his wife was Fon than when they were informed only that he had a wife, b = 1.42, SE = 0.53, z = 2.69, p = 0.020. Similarly, comparing the control condition, in which participants were not informed at all about the President's wife, to the experimental condition, in which they were informed that he had a Fon wife, increased the likelihood of voting among Fon coethnics, b = 1.04, SE = 0.43, z = 2.39, p =.045. There was no effect of the experimental condition among non-coethnics, with all ps > .196.

# Overall, the following claim "Our results confirm that priming the first lady's ethnicity increases support for President Yayi among her coethnics" was supported.