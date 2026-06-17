###Load necessary libraries
library(haven)
library(sm)
library(MASS)
library(pscl)


###Read data
setwd("C:/Users/INSERT YOUR FILE PATH HERE!")
dta <- read_dta("final_data.dta")
summary(dta)

###Your analysis should produce a single, main result in terms of statistical families of z-, t-, F-, or ?² tests (or their alternative or non-parametric versions).
###Use only the data of Illinois Plumbing Firms in your model. Do not restrict your sample to those firms that serve the metro Chicago area.
###Do not include the reviews received via yelp.com, Angie's List, and Consumer's Checkbook into your analysis. You should use the overall complaints number instead of complaints per employee.

################################################################################################################################
###GLM analyses with control variables (based on Negative Binomial because of its superior fit to the data)
mod.nb.c <- glm.nb(complaints_2008 ~ first_A + multiple_names + on_google + ad_spend_k + firm_age + chicago + emp_size, data=dta)
summary(mod.nb.c)
