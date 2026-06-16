###Load necessary libraries
library(haven)
library(sm)
library(MASS)
library(pscl)


###Read data
setwd("C:/Users/INSERT YOUR FILE PATH HERE!")
dta <- read_dta("final_data.dta")
summary(dta)


###Distribution of complaints
sm.density(dta$complaints_2008, model="normal")
table(dta$first_A, dta$complaints_2008)


################################################################################################################################
###Dichotomized analysis of complaints (i.e., the distribution of complaints suggests that there are outliers in the total number of complaints.
###Hence, a dichotomized variable that only captures whether there was at least one complain vs. no complain is potentially more robust).

dta$complaints_binary <- ifelse(dta$complaints_2008==0, 0, 1)
addmargins(table(dta$first_A, dta$complaints_binary))

#Percentage of at least one complain, if name does not begin with "A"
177/1998

#Percentage of at least one complain, if name begins with "A"
72/295

#Test
chisq.test(dta$first_A, dta$complaints_binary)


################################################################################################################################
###GLM analyses: Poisson, Negative Binomial, and Zero-Inflated Models

###1) Poisson
mod.pois <- glm(complaints_2008 ~ first_A, family=poisson, data=dta)
summary(mod.pois)

#Goodness of fit test (p < .05 indicates a bad fit)
1 - pchisq(summary(mod.pois)$deviance, summary(mod.pois)$df.residual)

###2) Negative Bionomial
mod.nb <- glm.nb(complaints_2008 ~ first_A, data=dta)
summary(mod.nb)

#Goodness of fit test (p < .05 indicates a bad fit)
1 - pchisq(summary(mod.nb)$deviance, summary(mod.nb)$df.residual)

###Model Comparison: Poisson vs. NB --> NB is significantly better (i.e., p < .001)
pchisq(2 * (logLik(mod.nb) - logLik(mod.pois)), df = 1, lower.tail = FALSE)

###3) Zero-Inflated Models (Poisson & NB) --> zero-inflation is a problem for Poisson models but not for NB models
mod.zi.poi <- zeroinfl(complaints_2008 ~ first_A|1, data = dta)
mod.zi.poi2 <- zeroinfl(complaints_2008 ~ first_A|first_A, data = dta)
mod.zi.nb <- zeroinfl(complaints_2008 ~ first_A|1, data = dta, dist = "negbin")
mod.zi.nb2 <- zeroinfl(complaints_2008 ~ first_A|first_A, data = dta, dist = "negbin")
summary(mod.zi.poi)   #coefficient for zero-inflation is significant
summary(mod.zi.poi2)  #coefficients for zero-inflation are significant
summary(mod.zi.nb)    #coefficient for zero-inflation is not significant
summary(mod.zi.nb2)   #coefficients for zero-inflation are not significant


################################################################################################################################
###GLM analyses with control variables (based on NB because of its superior fit to the data)
mod.nb.c <- glm.nb(complaints_2008 ~ first_A + multiple_names + on_google + ad_spend_k + firm_age + chicago + emp_size, data=dta)
summary(mod.nb.c)


################################################################################################################################
###Replication of the t-Test reported in Table 2
t.test(complaints_2008 ~ first_A, data=dta)
