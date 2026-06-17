rm(list=ls())
library(data.table) # need this package for loading dataset
library(sandwich) # need this package to compute robust standard errors
setwd('*****')

FinDT <- as.data.frame(read.csv('DAR data for 126zz.csv'))
FinDT <- na.omit(FinDT)

# Estimation 'glm.cluster'
myLogit <- glm.cluster(LeavingAgency ~ JobSat + Over40 + NonMinority
                       + SatPay + SatAdvan + PerfCul + Empowerment
                       + RelSup + Relcow + Over40xSatAdvan,
                       data=FinDT, cluster="Agency", 
                       family="binomial")
summary(myLogit)







