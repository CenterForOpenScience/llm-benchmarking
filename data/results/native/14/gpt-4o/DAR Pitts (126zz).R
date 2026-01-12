rm(list=ls())
library(data.table) # need this package for loading dataset
library(sandwich) # need this package to compute robust standard errors
setwd('/app/data')

FinDT <- as.data.frame(read.csv('Estimation Data - Pitts (126zz).csv'))
FinDT <- na.omit(FinDT)

# Estimation 'glm.cluster'
myLogit <- glm(LeavingAgency ~ JobSat + Over40 + NonMinority
               + SatPay + SatAdvan + PerfCul + Empowerment
               + RelSup + Relcow + Over40xSatAdvan,
               data=FinDT, family="binomial")

# Calculate clustered standard errors
library(sandwich)
library(lmtest)

# Function to calculate clustered standard errors
cluster_se <- function(model, cluster) {
  require(sandwich)
  require(lmtest)
  M <- length(unique(cluster))
  N <- length(cluster)
  K <- model$rank
  dfc <- (M/(M - 1)) * ((N - 1)/(N - K))
  uj <- apply(estfun(model), 2, function(x) tapply(x, cluster, sum))
  vcovCL <- dfc * sandwich(model, meat = crossprod(uj)/N)
  coeftest(model, vcovCL)
}

# Apply the function
cluster_se(myLogit, FinDT$Agency)







