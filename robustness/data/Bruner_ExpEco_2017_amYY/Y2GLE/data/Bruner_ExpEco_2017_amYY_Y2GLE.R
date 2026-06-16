## set working directory ##
setwd("~/Documents/Nick-Grad/Multi100/DMB_ExEcon_2017_Data_Analysis/")

### load libraries ###
library(haven)
library(tidyverse)
library(plyr)

### read in the data ###
data <- read_dta("RiskData.dta")

### remove participants w/ age = 99 - indicates bad data ###
data <- data[!(data$age == 99), ]

### create unique ID ###
data$ID <- 1:nrow(data)

### make long format ###
data_l <- gather(data, Decision, Choice,
                 Risk11, Risk12, Risk13, Risk14, Risk15,
                 Risk16, Risk17, Risk18, Risk19, Risk110,
                 Risk21, Risk22, Risk23, Risk24, Risk25,
                 Risk26, Risk27, Risk28, Risk29, Risk210,
                 Risk31, Risk32, Risk33, Risk34, Risk35,
                 Risk36, Risk37, Risk38, Risk39, Risk310)

### recode the risk trials to have labels for which task version ###
data_l$Condition <- ifelse(grepl("Risk1", data_l$Decision),
                           "PV",
                           ifelse(grepl("Risk2", data_l$Decision),
                                  "RV",
                                  ifelse(grepl("Risk3", data_l$Decision),
                                         "LV", "")))

### sort by participant and then ###
### create trial numbers within each task version ###
data_l <- data_l[order(data_l$ID), ]

data_l$trial <- rep(seq(1:10), (106*3))

### recode the LV task so that the first four trials are flipped ###
data_l$Choice <- ifelse(data_l$Condition == "LV" & data_l$trial <= 4 & data_l$Choice == "1",
                       "0", ifelse(data_l$Condition == "LV" & data_l$trial <= 4 & data_l$Choice == "0",
                                   "1", data_l$Choice))

### convert to numeric ###
data_l$Choice <- as.numeric(data_l$Choice)

### summarize variables for each participant ###
summary_data <- (ddply(data_l, "ID", summarise, 
                       PV_Risk = sum(Choice[which(Condition == "PV")], na.rm = F),
                       RV_Risk = sum(Choice[which(Condition == "RV")], na.rm = F),
                       LV_Perform = sum(Choice[which(Condition == "LV")], na.rm = F)))

### calculate composite risk aversion score ###
summary_data$RiskAversion <- rowMeans(summary_data[, c("PV_Risk", "RV_Risk")])

### test model ###
summary(lm(summary_data$LV_Perform ~ summary_data$RiskAversion))

cor.test(summary_data$LV_Perform, summary_data$RiskAversion,
         method = "spearman") ### confirm robustness w/ non-parametric test
