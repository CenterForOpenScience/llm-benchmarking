# ------------------------------------------------------------------------------------------
# This file contains the code for the analyses to replicate a research claim in Yang et al (2013) for SCORE project.
#
# Authors: Miroslav Sirota, Marie Juanchich, Kelly Wolfe
# 
# last update: 16/June/2020
#
# This script file is licensed under a CC-BY 4.0 license. 
# see http://creativecommons.org/licenses/by/4.0/
# 
# written by Miroslav Sirota (email: msirota@essex.ac.uk)
# Please email me if you see any errors or have any questions.

# ------------------------------------------------------------------
# Load required libraries 
# ------------------------------------------------------------------
library(psych)
library(yarrr)

# ------------------------------------------------------------------
# clear workspace and load data 
# ------------------------------------------------------------------

#clear workspace
rm(list = ls())

#set working directory
setwd("/app/data")

#load data (Note: We have removed Qualtrics meta-data such as Location and coded willingness to pay variables)
data <- read.csv("/workspace/replication_data/Data_Cleaned_22102020.csv", header=T, check.names=FALSE)

#------------------------------------------------------------------------------
# Codebook 
# Form: Variable name / Variable label / Variable values 
#------------------------------------------------------------------------------
#
# Id/ Identification Number/ NA	
# AttCheck/ Reading Attention Check, part 1, multiple choice question/ 1-9 = sport disciplines, 10 = "other"	(Note: the correct response: "10")
# AttCheck_10_TEXT/ Reading Attention Check, part 2, textbox/ NA (Note: the correct response: "boxing")
# Lot_WTP/ Willingness to pay for a lottery ticket/ NA	
# Lot_WTPc/ Coded Willingness to pay for a lottery ticket/ NA	
# Lot_check/ Lotter ticket instructions interpretation check/ 1 = "0", 2 = "A 5$ gift card", 3 = "A 10$ gift card", 4 = "A 15$ gift card", 5 = "A 20$ gift card"	(Note: the correct response: 3)	
# Gift_WTP/ Willingness to pay for a lottery ticket/ NA	
# Gift_WTPc/ Coded Willingness to pay for a lottery ticket/ NA	
# Gift_check/ Gift instructions interpretation check/ 1 = "0", 2 = "A 5$ gift card", 3 = "A 10$ gift card", 4 = "A 15$ gift card", 5 = "A 20$ gift card"	(Note: the correct response: 3)	
# Gen/ "Gender"/ 1 = "Male", 2 = "Female"
# Dif/ "How difficult did you find the instructions?"/ 1 = "Not at all difficult", 7 = "Very difficult"
# Age/ "Age"/ NA
# Inc/ "Which of the following income categories best describes your annual household income?"/ 1 = "Less than $15,000", 2 = "$15,000 to $29,999", 3 = "$30,000 to $49,999", 4 = "$50,000 to $74,999", 5 = "$75,000 to $99,999", 6 = "$100,000 and over"

# ------------------------------------------------------------------------------
# Variable creation and recoding 
# ------------------------------------------------------------------------------


#Create the independent variable 

data$Cond <- 
data$Cond[data$Lot_check=="NA"&data$Gift_check=="NA"] <- NA
data$Cond[data$Lot_check>0] <- 0
data$Cond[data$Gift_check>0] <- 1

levels(data$Cond) <- list("lottery"=0,"gift"=1)

#Create the dependent variable 

data$WTP <- 
data$WTP <- ifelse(data$Cond==0, data$Lot_WTPc, ifelse(data$Cond==1, data$Gift_WTPc, NA))
data <- data[!is.na(data$WTP),]



#Create the checking variable

data$check <- 
data$check <- ifelse(data$Cond==0, data$Lot_check, ifelse(data$Cond==1, data$Gift_check, NA))

# ------------------------------------------------------------------------------
# Analysis 
# ------------------------------------------------------------------------------

#Critical test as in the original paper 

#Select those who answered correctly (correct == 3)
dataS <- data[data$check==3,]
write.csv(dataS, "/app/data/dataS.csv", fileEncoding="UTF-8")
#Parametric analysis for those who understood the instructions [CRITICAL TEST OF THE TARGETED HYPOTHESIS]
describeBy(dataS$WTP,dataS$Cond)
t.test(WTP~Cond, dataS)
cohen.d(dataS$WTP,dataS$Cond)


#Select those who answered 20 or less 
dataSC <- dataS[dataS$WTP<=20,]
#Parametric analysis for those who understood the instructions [CRITICAL TEST OF THE TARGETED HYPOTHESIS]
describeBy(dataSC$WTP,dataSC$Cond)
t.test(WTP~Cond, dataSC)
cohen.d(dataSC$WTP,dataSC$Cond)



