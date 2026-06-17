options(scipen=999)
library(lmerTest)
library(dplyr)
library(tidyverse)
library(MuMIn)
library(reshape)
dat <- readRDS(file="replication_data/Final replication dataset.rds")

#Data transformation

#Creation of a new variable that classifies participants in a monolingual (0) and in a bilingual group (1)
dat$bilingual = ifelse(dat$I03_ST_A_S26A == (2 | 3), 0, 1)
dat <- dat[!(is.na(dat$bilingual)),]


#Data exclusion: Excluding students who speak English (the target language) at home
dat<-subset(dat, I03_ST_A_S27B==0)



#Creation of an average score for writting, reading and listeting
dat$ave_writing<- (dat$PV1_WRIT_C+dat$PV2_WRIT_C+dat$PV3_WRIT_C+dat$PV4_WRIT_C+dat$PV5_WRIT_C)/5
dat$ave_reading<-(dat$PV1_READ+dat$PV2_READ+dat$PV3_READ+dat$PV4_READ+dat$PV5_READ)/5
dat$ave_listening<-(dat$PV1_LIST+dat$PV2_LIST+dat$PV3_LIST+dat$PV4_LIST+dat$PV5_LIST)/5
dat$average_english <- rowMeans(dat[ , c('ave_writing', 'ave_reading', 'ave_listening')], na.rm=TRUE)


#Converting Cultural Capital into a continous variable
dat$Cultural_capital = ifelse(dat$SQt21i01 == "0-10 books", 0, 
                              ifelse(dat$SQt21i01 == "11-25 books", 1, 
                                     ifelse(dat$SQt21i01 == "26-100 books", 2, 
                                            ifelse(dat$SQt21i01== "101-200 books", 3,
                                                   ifelse(dat$SQt21i01== "201-500 books", 4,   
                                                          ifelse(dat$SQt21i01== "More than 500 books", 5,""))))))

dat$Cultural_capital<-as.numeric(dat$Cultural_capital)


#Exclusing observations with no oweights or weights == 0
# Three datasets are created for each dimenion (Writing, Reading, Listening)
dat<-dat %>% filter(FSW_WRIT_TR > 0 | FSW_READ_TR > 0 | FSW_LIST_TR > 0 )



#Centering continous variables (for each of the three datasets)
#Centering age
dat$c_age<-scale(dat$I08_ST_A_S02A, center = TRUE, scale = FALSE)

#Centering SES
dat$c_HISEI<-scale(dat$HISEI, center = TRUE, scale = FALSE)

#Converting to Z-scores the variable "parental education" and "cultural capital"
dat$Z_Parental<-scale(dat$PARED, center = TRUE, scale = TRUE)
dat$Z_Cultural<-scale(dat$Cultural_capital, center = TRUE, scale = TRUE)

#Function to calculate standardized estimates
stdCoef.merMod <- function(object) {
  sdy <- sd(getME(object,"y"))
  sdx <- apply(getME(object,"X"), 2, sd)
  sc <- fixef(object)*sdx/sdy
  se.fixef <- coef(summary(object))[,"Std. Error"]
  se <- se.fixef*sdx/sdy
  return(data.frame(stdcoef=sc, stdse=se))
}


#Three-level model

results_m2<-lmer(average_english ~ 1+ bilingual + factor(SQt01i01) + c_age + c_HISEI + Z_Parental + Z_Cultural + (1|country_id/school_id), data=dat)
summary(results_m2)
r.squaredGLMM(results_m2)
stdCoef.merMod(results_m2)

# SECOND EXPLORATORY ANALYSIS

#Three multilevel models are fitted separately on each English dimension, namely writing, reading and listening skills.

data <- readRDS(file="Final replication dataset.rds")

#Transformation
data$bilingual = ifelse(data$I03_ST_A_S26A == (2 | 3), 0, 1)


#Data exclusion: Exclusing students who speak English (the target language) at home
data<-subset(data, I03_ST_A_S27B==0)


#Creation of an average score for writting, reading and listeting
data$ave_writing<- (data$PV1_WRIT_C+data$PV2_WRIT_C+data$PV3_WRIT_C+data$PV4_WRIT_C+data$PV5_WRIT_C)/5
data$ave_reading<-(data$PV1_READ+data$PV2_READ+data$PV3_READ+data$PV4_READ+data$PV5_READ)/5
data$ave_listening<-(data$PV1_LIST+data$PV2_LIST+data$PV3_LIST+data$PV4_LIST+data$PV5_LIST)/5


#Converting Cultural Capital into a continous variable
data$Cultural_capital = ifelse(data$SQt21i01 == "0-10 books", 0, 
                               ifelse(data$SQt21i01 == "11-25 books", 1, 
                                      ifelse(data$SQt21i01 == "26-100 books", 2, 
                                             ifelse(data$SQt21i01== "101-200 books", 3,
                                                    ifelse(data$SQt21i01== "201-500 books", 4,   
                                                           ifelse(data$SQt21i01== "More than 500 books", 5,""))))))

data$Cultural_capital<-as.numeric(data$Cultural_capital)


dat_writing<-data %>% filter(FSW_WRIT_TR > 0)
dat_reading<-data %>% filter(FSW_READ_TR > 0)
dat_listening<- data %>% filter(FSW_LIST_TR > 0)

#Centering continous variables (for each of the three datasets)
#Centering age
dat_writing$c_age<-scale(dat_writing$I08_ST_A_S02A, center = TRUE, scale = FALSE)
dat_reading$c_age<-scale(dat_reading$I08_ST_A_S02A, center = TRUE, scale = FALSE)
dat_listening$c_age<-scale(dat_listening$I08_ST_A_S02A, center = TRUE, scale = FALSE)

#Centering SES
dat_writing$c_HISEI<-scale(dat_writing$HISEI, center = TRUE, scale = FALSE)
dat_reading$c_HISEI<-scale(dat_reading$HISEI, center = TRUE, scale = FALSE)
dat_listening$c_HISEI<-scale(dat_listening$HISEI, center = TRUE, scale = FALSE)


#Converting to Z-scores the variable "parental education" and "cultural capital"
dat_writing$Z_Parental<-scale(dat_writing$PARED, center = TRUE, scale = TRUE)
dat_reading$Z_Parental<-scale(dat_reading$PARED, center = TRUE, scale = TRUE)
dat_listening$Z_Parental<-scale(dat_listening$PARED, center = TRUE, scale = TRUE)

dat_writing$Z_Cultural<-scale(dat_writing$Cultural_capital, center = TRUE, scale = TRUE)
dat_reading$Z_Cultural<-scale(dat_reading$Cultural_capital, center = TRUE, scale = TRUE)
dat_listening$Z_Cultural<-scale(dat_listening$Cultural_capital, center = TRUE, scale = TRUE)


#Three-leve models with control variables
writing<-lmer(ave_writing ~ 1+ bilingual + factor(SQt01i01) + c_age + c_HISEI + Z_Parental + Z_Cultural + (1|country_id/school_id), data=dat_writing)
summary(writing)
r.squaredGLMM(writing)

reading<-lmer(ave_reading ~ 1+ bilingual + factor(SQt01i01) + c_age + c_HISEI + Z_Parental + Z_Cultural + (1|country_id/school_id), data=dat_reading)
summary(reading)
r.squaredGLMM(reading)

listening<-lmer(ave_listening ~ 1+ bilingual + factor(SQt01i01) + c_age + c_HISEI + Z_Parental + Z_Cultural +(1|country_id/school_id), data=dat_listening)
summary(listening)
r.squaredGLMM(listening)

#Obtaining standardized estimates for all models
stdCoef.merMod(writing)
stdCoef.merMod(reading)
stdCoef.merMod(listening)