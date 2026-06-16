# Multi100 project analysis

# In my analysis I test the claim of the paper:
# Anderson (2011). Caste as an Impediment to Trade
# Claim: ... [the study] finds substantially higher income for low-caste households 
# residing in villages dominated by BACs [compared to villages dominated by upper caste]. (p. 240.) 

# TASK 2

# Produce a single, main result in terms of statistical families of z-, t-, F-, or χ² tests 
# (or their alternative or non-parametric versions). 
# Not using: District controls, Crop controls, Distance controls, Groundwater controls, Public goods controls,
#     Irrigation, credit, and tenancy variables, and their interaction term versions, instruments for water buyer status.


# The variable called domlow and totinc (total income) is in the focus of the analysis 
# domlow is a dummy variable that represents the village dominance where the households reside
# totinc is the total income of the household

# Load libraries
library(haven)
library(ggplot2)
library(tidyverse)
library(lmtest)
library(sandwich)
library(olsrr)
library(oaxaca)

# Import household data
df <- read_dta("household.dta")

# Remove observations where domlow variable is missing
df <- df[!is.na(df$domlow), ]

# Create factors from string variables
df$caste <- as.factor(df$caste)
df$domlow <- as.factor(df$domlow)

# Inspect the distribution of the dependent variable
summary(df$totinc)
hist(df$totinc, ylab = "Total income") # Total income is clearly skewed with a long tail toward larger values
boxplot(df$totinc, ylab = "Total income")
ggplot(df, aes(y=totinc, fill=domlow)) + 
  geom_boxplot()
# Total income seems higher in low caste dominated villages
# but there are many outliers

# Check observations with the highest income
df_ordered <- df[order(df$totinc), ]
tail(df_ordered[, c("totinc", "domlow")], 10) 
# The top 10 households with the highest income all live in low caste dominated villages

# Use log of total income
ggplot(df, aes(y=log(totinc), fill=domlow)) + 
  geom_boxplot()

#==================================
# Analysis with logarithmic income
#==================================

# Taking the log of the total income solves the issue of outliers

# Build regression model without the excluded predictors
# Village dominance is included as dummy variable
model1 <- lm(log(totinc) ~ bihar + literate + totland +
               cash + electric + 
               caste + domlow, data = df)
summary(model1)
# Robust standard errors
coeftest(model1, vcov = vcovHC(model1, type = "HC0"))
# domlow is not significant with p-value of 0.056

# All included variables are significant except for domlow
# No model selection is required
