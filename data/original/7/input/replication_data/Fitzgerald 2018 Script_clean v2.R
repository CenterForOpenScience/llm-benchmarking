## Daniel J. Mallinson
## Fitzgerald 2018 Replication Script

rm(list=ls()) #clear workspace

#install.packages(c("readstata13", "tidyverse", "reshape2", "prais", "panelAR")) #uncomment to install necessary packages

library(foreign)
library(readstata13)
library(tidyverse)
library(reshape2)
library(prais)
library(panelAR)

data <- read.dta13("compiled.dta")
hhsize <- read.dta13("hhsize.dta")
epa <- read.dta13("epa.dta")

## Reshape household size (hhsize) from wide to long
hhsize <- melt(hhsize, id.vars=c("State", "state_id_no", "state_fip"))
year <- c(rep(7,50), rep(8,50), rep(9,50), rep(10,50), rep(11,50),
	rep(12,50), rep(13,50), rep(14,50), rep(15,50), rep(16,50))
hhsize <- cbind(hhsize, year)

hhsize <- hhsize[c("State", "value", "year")]
names(hhsize)[2] <- "hhsize"

## Merge hhsize with rest of data
data <- merge(data, hhsize, by=c("State", "year"))
data <- merge(data, epa, by=c("State", "year"))

## Calculate Employed Population %
data$emppop_pct <- data$emppop/(data$pop*1000)*100

## Calculate Manufacturing % of GDP
data$manu_gdp <- data$manuf/data$gdp*100

## Log transform continuous variables

data[c("epa", "wrkhrs", "emppop_pct", "laborprod", "pop", "manu_gdp",
	"energy", "hhsize", "workpop")] <- log(data[c("epa", "wrkhrs", "emppop_pct", "laborprod", "pop", "manu_gdp",
	"energy", "hhsize", "workpop")])

#### Registration Analysis

## Draw sample for analysis set up
states <- unique(data$State)

group_var <- data %>% 
  group_by(State) %>%
  groups %>%
  unlist %>% 
  as.character

group_var

set.seed(42)
random_states <- data %>% 
  group_by(State) %>% 
  summarise() %>% 
  sample_n(5) %>% 
  mutate(unique_id=1:NROW(.))

random_states

sampledata <- data %>% 
  group_by(State)  %>% 
  right_join(random_states, by=group_var) %>%
  group_by_(group_var) 

sampledata <- sampledata[order(sampledata$State, sampledata$year),]

sampledata <- as.data.frame(sampledata)

## Replication models with 5% sample

model1 <- panelAR(epa ~ wrkhrs + emppop_pct + laborprod + pop + manu_gdp + 
	energy + hhsize + workpop + State + factor(year), data=sampledata, panelVar='State', timeVar='year', panelCorrMethod='pcse',singular.ok=TRUE, autoCorr="psar1", complete.case=TRUE)
summary(model1)

## Model with original years

model2 <- panelAR(epa ~ wrkhrs + emppop_pct + laborprod + pop + manu_gdp + 
          energy + hhsize + workpop + State + factor(year), data=sampledata[which(sampledata$year<14),], panelVar='State', timeVar='year', panelCorrMethod='pcse',singular.ok=TRUE, autoCorr="psar1", complete.case=TRUE)
summary(model2)

## Model with only new years
#Does not run, not enough data in sample

model3 <- panelAR(epa ~ wrkhrs + emppop_pct + laborprod + pop + manu_gdp + 
          energy + hhsize + workpop + State + factor(year), data=sampledata[which(sampledata$year>13),], panelVar='State', timeVar='year', panelCorrMethod='pcse',singular.ok=TRUE, autoCorr="psar1", complete.case=TRUE, rho.na.rm=TRUE)
summary(model3)

## Models with full data (Not yet run)

model4 <- panelAR(epa ~ wrkhrs + emppop_pct + laborprod + pop + manu_gdp + 
          energy + hhsize + workpop + State + factor(year), data=data, panelVar='State', timeVar='year', panelCorrMethod='pcse',singular.ok=TRUE, autoCorr="psar1", complete.case=TRUE)
summary(model4)

## Model with original years

model5 <- panelAR(epa ~ wrkhrs + emppop_pct + laborprod + pop + manu_gdp + 
          energy + hhsize + workpop + State + factor(year), data=data[which(data$year<14),], panelVar='State', timeVar='year', panelCorrMethod='pcse',singular.ok=TRUE, autoCorr="psar1", complete.case=TRUE)
summary(model5)

## Model with only new years

model6 <- panelAR(epa ~ wrkhrs + emppop_pct + laborprod + pop + manu_gdp + 
          energy + hhsize + workpop + State + factor(year), data=data[which(data$year>13),], panelVar='State', timeVar='year', panelCorrMethod='pcse',singular.ok=TRUE, autoCorr="psar1", complete.case=TRUE)
summary(model6)



