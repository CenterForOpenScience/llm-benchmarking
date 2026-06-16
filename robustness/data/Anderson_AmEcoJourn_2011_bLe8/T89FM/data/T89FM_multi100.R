# Multi100 project analysis

# In my analysis I use multiple statistical methods to test the claim of the paper:
# Anderson (2011). Caste as an Impediment to Trade
# Claim: ... [the study] finds substantially higher income for low-caste households 
# residing in villages dominated by BACs [compared to villages dominated by upper caste]. (p. 240.) 

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
# Compare the log total income of households across village types with t test
domlow_log_totinc <- log(df[(df$domlow == 1), "totinc"])
domhigh_log_totinc <- log(df[(df$domlow == 0), "totinc"])
t.test(domlow_log_totinc, domhigh_log_totinc,
       alternative = "greater")
# The result of the t test suggest that log total income is significantly higher in 
# low-caste households residing in villages dominated by lower caste
# even at an 1% significance level


# Build regression model with all predictors
# Village dominance is included as dummy variable
model1 <- lm(log(totinc) ~ bihar + literate + borrow + totland + landirr + 
               dist1 + dist2 + dist3 + dist4 + dist5 + dist6 + dist8 + dist9 + dist10 + dist11 + 
               dist12 + dist13 + dist14 + dist16 + dist17 + dist18 + dist19 + dist20 + dist21 + 
               dist22 + dist23 + dist24 + dist25 + area + pmixdominant + tolaparea +
               gwdevelopment + rainfall + gwavailability + river + canal + noalkal + 
               nolog + nosoil + noflood + paddy + wheat + cereal + pulse + bulb + seed + cash +
               bus3 + tele3 + ps3 + pds3 + bank3 + pps3 + ms3 + ss3 + phc3 + hosp3 + electric + hhelec + 
               caste + domlow, data = df)
summary(model1)
# Robust standard errors
coeftest(model1, vcov = vcovHC(model1, type = "HC0"))
# domlow is not significant with p-value of 0.054

# Inspect residuals
ols_plot_resid_qq(model1)
ols_test_normality(model1) 
# Residuals are normally distributed according to the KS test at 5% signif level


# Omit predictors where there is not enough data
model2 <- lm(log(totinc) ~ bihar + literate + borrow + totland + landirr + 
               dist1 + dist2 + dist3 + dist4 + dist5 + dist6 + dist8 + dist9 + dist11 + 
               dist12 + dist13 + dist14 + dist16 + dist17 + dist18 + dist19 + dist20 + dist21 + 
               dist22 + dist23 + dist24 + area + pmixdominant + tolaparea +
               river + canal + noalkal + 
               nolog + nosoil + noflood + paddy + wheat + cereal + pulse + bulb + seed + cash +
               bus3 + tele3 + ps3 + pds3 + bank3 + pps3 + ms3 + ss3 + phc3 + hosp3 + electric + hhelec + 
               caste + domlow, data = df)
summary(model2)
# Robust standard errors
coeftest(model2, vcov = vcovHC(model2, type = "HC0"))
# domlow is not significant with p-value of 0.054

# Inspect residuals
ols_plot_resid_qq(model2) # Residuals are closer to normal distribution
ols_test_normality(model2) # Residuals are still not normally distributed
# Residuals are normally distributed according to the KS test at 5% signif level


# Apply backward elimination based on Akaike criteria
# but keep caste * domlow in the model anyway
model3 <- lm(log(totinc) ~ bihar + literate + totland + landirr + 
               dist1 + dist2 + dist3 + dist4 + dist5 + dist6 + dist8 + dist9 + dist11 + 
               dist12 + dist13 + dist14 + dist16 + dist17 + dist18 + dist19 + 
               dist20 + dist21 + dist22 + dist23 + dist24 + area + 
               river + canal + nolog + 
               nosoil + paddy + wheat + pulse + bulb + seed + 
               cash + ps3 + pds3 + pps3 + ms3 + ss3 + 
               phc3 + electric + hhelec + caste + domlow,
               data = df)
AIC(model3)
summary(model3)
# Robust standard errors
coeftest(model3, vcov = vcovHC(model3, type = "HC0"))
# domlow is not significant with p-value of 0.075

# Inspect residuals
ols_plot_resid_qq(model3)
ols_test_normality(model3) 
# Residuals are normally distributed according to the KS test at 5% signif level


#==================================
# Blinder-Oaxaca Decomposition
#==================================
# Use Blinder-Oaxaca Decomposition to estimate explained and unexplained differences
# for log total income with the most important variables

decomp_ln = oaxaca( formula = log(totinc) ~ bihar + literate + totland + landirr+ area + 
                   canal + nolog + 
                   nosoil + paddy + wheat + pulse + bulb + seed + 
                   cash + ps3 + pds3 + pps3 + ms3 + ss3 + 
                   phc3 + electric + hhelec + caste | domlow,
                 data = df, R = 100)
decomp_ln$y
# Log total income diff is 0.225
decomp_ln$twofold$overall
# Considering equal weights in the reference coefficients (where group.weight = 0.5)
# Explained diff is 0.21 while unexplained is 0.016
# However, standard error is high
plot(decomp_ln, decomposition = "twofold", group.weight = -1)


# To conclude, controlling for the right skewed distribution of the dependent variable 
# and for multiple other socioeconomic factors, I found that low-caste households 
# residing in villages dominated by BACs do not have significantly higher income 
# than the ones residing in villages dominated by upper caste.
