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
# (yarrr removed to avoid installation issues)

# ------------------------------------------------------------------
# clear workspace and load data 
# ------------------------------------------------------------------

#clear workspace#clear workspace# clear workspace
rm(list = ls())

# Determine script directory and construct data path
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
if(length(script_path) == 0) {
  script_dir <- "."
} else {
  script_dir <- dirname(script_path)
}
DATA_PATH <- file.path(script_dir, "Data_Cleaned_22102020.csv")

# Load data (Note: We have removed Qualtrics meta-data such as Location and coded willingness to pay variables)
if(!file.exists(DATA_PATH)) {
  stop(paste("Data file not found at", DATA_PATH))
}

data <- read.csv(DATA_PATH, header = TRUE, fileEncoding = "latin1", check.names = FALSE)

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


#Create the independent variable#Create variables ------------------------------------------------------

# Independent variable: framing condition (lottery vs gift)

data$Cond <- NA
# Participants assigned to lottery condition if they have a non-missing Lot_check value
idx_lottery <- !is.na(data$Lot_check) & data$Lot_check != ""
# Participants assigned to gift condition if they have a non-missing Gift_check value
idx_gift    <- !is.na(data$Gift_check) & data$Gift_check != ""

data$Cond[idx_lottery] <- "lottery"
data$Cond[idx_gift]    <- "gift"

data$Cond <- factor(data$Cond, levels = c("lottery", "gift"))

# Dependent variable: willingness to pay (numeric USD)

data$WTP <- ifelse(data$Cond == "lottery", data$Lot_WTPc,
                   ifelse(data$Cond == "gift",    data$Gift_WTPc, NA))

# Comprehension check variable: interpret the appropriate check depending on condition

data$check <- ifelse(data$Cond == "lottery", data$Lot_check,
                     ifelse(data$Cond == "gift",    data$Gift_check, NA))#Create the dependent variable
# Remove rows with missing WTP values
data <- data[!is.na(data$WTP), ]

# ----------------------------------------------------------------------------# ------------------------------------------------------------------------------
# Analysis 
# ------------------------------------------------------------------------------

#Critical test as in the original paper 

#Select those who answered correctly (correct == 3)
dataS <- data[data$check==3,]
write.csv(dataS, "dataS.csv")
#Parametric analysis for those who understood the instructions [CRITICAL TEST OF THE TARGETED HYPOTHESIS]
describeBy(dataS$WTP,dataS$Cond)
t.test(dataS$WTP ~ dataS$Cond)
cohen.d(dataS$WTP,dataS$Cond)


#Select those who answered 20 or less 
dataSC <- dataS[dataS$WTP<=20,]
#Parametric analysis for those who understood the instructions [CRITICAL TEST OF THE TARGETED HYPOTHESIS]
describeBy(dataSC$WTP,dataSC$Cond)
t.test(dataSC$WTP ~ dataSC$Cond)
cohen.d(dataSC$WTP,dataSC$Cond)



