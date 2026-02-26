# Project name: Seaton_AmEduResJourn_2010_Blxd_3053
# Script author: James Field (jamesfield6912@gmail.com)
# Last updated: November 19, 2020
# Replication project website: https://osf.io/mu4rs/

# Import data set from GitHub
 #install.packages("readr")
 library (readr)
 urlfile="https://raw.githubusercontent.com/jamiefield/jamiefield.github.io/master/files/PISA2012.replication.RDS"
 dat_seaton <- readRDS(url(urlfile))
 
#install.packages("dplyr")
 library(dplyr)
 dat_seaton$uniqueSchoolID <- paste0(dat_seaton$SCHOOLID, "|", dat_seaton$CNT) #this creates a *unique* school ID
 dat_seaton$uniqueStudentID <- paste0(dat_seaton$STIDSTD, "|", dat_seaton$uniqueSchoolID) #this creates a *unique* student ID
 
 length(unique(dat_seaton$uniqueStudentID)) #count number of unique students
 length(unique(dat_seaton$uniqueSchoolID)) #count number of unique schools
 length(unique(dat_seaton$CNT)) #count number of unique countries
 
# Remove missing data from math self-concept, the DV (see codes provided in PISA data dictionary file)
 dat_seaton <- filter(dat_seaton, SCMAT <= 997)
 dat_seaton <- filter(dat_seaton, !is.na(SCMAT))
 
 length(unique(dat_seaton$uniqueStudentID)) #count number of unique students
 length(unique(dat_seaton$uniqueSchoolID)) #count number of unique schools
 length(unique(dat_seaton$CNT)) #count number of unique countries
 
# Remove missing data from plausible values, the IV (see codes provided in PISA data dictionary file)
 dat_seaton <- filter(dat_seaton, PV1MATH <= 997)
 dat_seaton <- filter(dat_seaton, PV2MATH <= 997)
 dat_seaton <- filter(dat_seaton, PV3MATH <= 997)
 dat_seaton <- filter(dat_seaton, PV4MATH <= 997)
 dat_seaton <- filter(dat_seaton, PV5MATH <= 997)
 dat_seaton <- filter(dat_seaton, !is.na(PV1MATH))
 dat_seaton <- filter(dat_seaton, !is.na(PV2MATH))
 dat_seaton <- filter(dat_seaton, !is.na(PV3MATH))
 dat_seaton <- filter(dat_seaton, !is.na(PV4MATH))
 dat_seaton <- filter(dat_seaton, !is.na(PV5MATH))
 
 length(unique(dat_seaton$uniqueStudentID)) #count number of unique students
 length(unique(dat_seaton$uniqueSchoolID)) #count number of unique schools
 length(unique(dat_seaton$CNT)) #count number of unique countries
 
# Remove missing data from memorization, the moderator (see codes provided in PISA data dictionary file)
 dat_seaton <- filter(dat_seaton, MEMOR <= 997)
 dat_seaton <- filter(dat_seaton, !is.na(MEMOR))
 
 length(unique(dat_seaton$uniqueStudentID)) #count number of unique students
 length(unique(dat_seaton$uniqueSchoolID)) #count number of unique schools
 length(unique(dat_seaton$CNT)) #count number of unique countries

# Delete observations nested in schools with n <= 10
 dat_seaton <- dat_seaton %>% group_by(uniqueSchoolID) %>% filter(n() > 10) #remove the schools with n <= 10
 
 length(unique(dat_seaton$uniqueStudentID)) #count number of unique students
 length(unique(dat_seaton$uniqueSchoolID)) #count number of unique schools
 length(unique(dat_seaton$CNT)) #count number of unique countries
 
 # Remove irrelevant columns
 dat_seaton <- subset(dat_seaton, select = -c(HISEI:BELONG, CSTRAT:ELAB))

 # Select 5% random sample of imported data
 #set.seed(2020)
 #dat_seaton <- sample_n(dat_seaton, round(0.20*nrow(dat_seaton)))


# Standardize the five plausible values for mathematics ability (i.e., the predictor variables; see p. 404)
# See https://stackoverflow.com/questions/15215457/standardize-data-columns-in-r
 dat_seaton$PV1MATH_z <- ((dat_seaton$PV1MATH - mean(dat_seaton$PV1MATH)) / sd(dat_seaton$PV1MATH))
 dat_seaton$PV2MATH_z <- ((dat_seaton$PV2MATH - mean(dat_seaton$PV2MATH)) / sd(dat_seaton$PV2MATH))
 dat_seaton$PV3MATH_z <- ((dat_seaton$PV3MATH - mean(dat_seaton$PV3MATH)) / sd(dat_seaton$PV3MATH))
 dat_seaton$PV4MATH_z <- ((dat_seaton$PV4MATH - mean(dat_seaton$PV4MATH)) / sd(dat_seaton$PV4MATH))   
 dat_seaton$PV5MATH_z <- ((dat_seaton$PV5MATH - mean(dat_seaton$PV5MATH)) / sd(dat_seaton$PV5MATH))

# Standardize mathematics self-concept (i.e., the outcome variable; see p. 404)
 dat_seaton$SCMAT_z <- ((dat_seaton$SCMAT - mean(dat_seaton$SCMAT)) / sd(dat_seaton$SCMAT)) 

# Standardize memorization (i.e., the moderator of interest; see p. 404) 
 dat_seaton$MEMOR_z <- ((dat_seaton$MEMOR - mean(dat_seaton$MEMOR)) / sd(dat_seaton$MEMOR))

# Estimate school average ability for PV1MATH : PV5MATH (see p. 404)
 dat_seaton <- dat_seaton %>%
        group_by(uniqueSchoolID) %>%
        mutate(school_PV1MATH_z = mean(PV1MATH_z),
               school_PV2MATH_z = mean(PV2MATH_z),
               school_PV3MATH_z = mean(PV3MATH_z),
               school_PV4MATH_z = mean(PV4MATH_z),
               school_PV5MATH_z = mean(PV5MATH_z))

# Create cross products (see p. 404)
 dat_seaton$CROSS1 <- dat_seaton$MEMOR_z * dat_seaton$school_PV1MATH_z
 dat_seaton$CROSS2 <- dat_seaton$MEMOR_z * dat_seaton$school_PV2MATH_z
 dat_seaton$CROSS3 <- dat_seaton$MEMOR_z * dat_seaton$school_PV3MATH_z
 dat_seaton$CROSS4 <- dat_seaton$MEMOR_z * dat_seaton$school_PV4MATH_z
 dat_seaton$CROSS5 <- dat_seaton$MEMOR_z * dat_seaton$school_PV5MATH_z

# Create quadratic terms
 dat_seaton$PV1MATH_z_sq <- dat_seaton$PV1MATH_z^2
 dat_seaton$PV2MATH_z_sq <- dat_seaton$PV2MATH_z^2
 dat_seaton$PV3MATH_z_sq <- dat_seaton$PV3MATH_z^2
 dat_seaton$PV4MATH_z_sq <- dat_seaton$PV4MATH_z^2
 dat_seaton$PV5MATH_z_sq <- dat_seaton$PV5MATH_z^2

########################################
########################################

#install.packages("lme4")
library(lme4)
#install.packages("car")
library(car)
#install.packages("lmerTest")
library(lmerTest)
#install.packages("afex")
library(afex)

# Helpful online sources
# https://stackoverflow.com/questions/53034261/warning-lme4-model-failed-to-converge-with-maxgrad
# https://www.r-bloggers.com/three-ways-to-get-parameter-specific-p-values-from-lmer/
# https://stats.stackexchange.com/questions/22988/how-to-obtain-the-p-value-check-significance-of-an-effect-in-a-lme4-mixed-mode

########################################
## ESTIMATE A SET OF FIVE MULTILEVEL MODELING REGRESSION ANALYSES (ONE FOR EACH PLAUSIBLE VALUE)
########################################

# Plausible value #1
 model1 <- lmer(SCMAT_z ~ PV1MATH_z + PV1MATH_z_sq + school_PV1MATH_z + MEMOR_z + CROSS1 + (PV1MATH_z + PV1MATH_z_sq + MEMOR_z|uniqueSchoolID) + (PV1MATH_z + PV1MATH_z_sq + school_PV1MATH_z + MEMOR_z|CNT), data = dat_seaton, weights = W_FSTUWT, REML = TRUE)
 summary(model1)

 cross1Result <- coef(summary(model1))[6, "Estimate"] #Store moderator parameter
 cross1SE <- (coef(summary(model1))[6, "Std. Error"]) #Store moderator SE
 cross1T <- (coef(summary(model1))[6, "t value"]) #Store moderator t value
 cross1P <- 2 * (1-pnorm(abs(cross1T))) #Store moderator p value

# Plausible value #2
 model2 <- lmer(SCMAT_z ~ PV2MATH_z + PV2MATH_z_sq + school_PV2MATH_z + MEMOR_z + CROSS2 + (PV2MATH_z + PV2MATH_z_sq + MEMOR_z|uniqueSchoolID) + (PV2MATH_z + PV2MATH_z_sq + school_PV2MATH_z + MEMOR_z|CNT), data = dat_seaton, weights = W_FSTUWT, REML = TRUE)
 summary(model2)
 
 cross2Result <- coef(summary(model2))[6, "Estimate"] #Store moderator parameter
 cross2SE <- (coef(summary(model2))[6, "Std. Error"]) #Store moderator SE
 cross2T <- (coef(summary(model2))[6, "t value"]) #Store moderator t value
 cross2P <- 2 * (1-pnorm(abs(cross2T))) #Store moderator p value

# Plausible value #3
 model3 <- lmer(SCMAT_z ~ PV3MATH_z + PV3MATH_z_sq + school_PV3MATH_z + MEMOR_z + CROSS3 + (PV3MATH_z + PV3MATH_z_sq + MEMOR_z|uniqueSchoolID) + (PV3MATH_z + PV3MATH_z_sq + school_PV3MATH_z + MEMOR_z|CNT), data = dat_seaton, weights = W_FSTUWT, REML = TRUE)
 summary(model3)
 
 cross3Result <- coef(summary(model3))[6, "Estimate"] #Store moderator parameter
 cross3SE <- (coef(summary(model3))[6, "Std. Error"]) #Store moderator SE
 cross3T <- (coef(summary(model3))[6, "t value"]) #Store moderator t value
 cross3P <- 2 * (1-pnorm(abs(cross3T))) #Store moderator p value
 
# Plausible value #4
 model4 <- lmer(SCMAT_z ~ PV4MATH_z + PV4MATH_z_sq + school_PV4MATH_z + MEMOR_z + CROSS4 + (PV4MATH_z + PV4MATH_z_sq + MEMOR_z|uniqueSchoolID) + (PV4MATH_z + PV4MATH_z_sq + school_PV4MATH_z + MEMOR_z|CNT), data = dat_seaton, weights = W_FSTUWT, REML = TRUE)
 summary(model4)
 
 cross4Result <- coef(summary(model4))[6, "Estimate"] #Store moderator parameter
 cross4SE <- (coef(summary(model4))[6, "Std. Error"]) #Store moderator SE
 cross4T <- (coef(summary(model4))[6, "t value"]) #Store moderator t value
 cross4P <- 2 * (1-pnorm(abs(cross4T))) #Store moderator p value
 

# Plausible value #5
 model5 <- lmer(SCMAT_z ~ PV5MATH_z + PV5MATH_z_sq + school_PV5MATH_z + MEMOR_z + CROSS5 +  (PV5MATH_z + PV5MATH_z_sq + MEMOR_z|uniqueSchoolID) + (PV5MATH_z + PV5MATH_z_sq + school_PV5MATH_z + MEMOR_z|CNT), data = dat_seaton, weights = W_FSTUWT, REML = TRUE)
 summary(model5)
 
 cross5Result <- coef(summary(model5))[6, "Estimate"] #Store moderator parameter
 cross5SE <- (coef(summary(model5))[6, "Std. Error"]) #Store moderator SE
 cross5T <- (coef(summary(model5))[6, "t value"]) #Store moderator t value
 cross5P <- 2 * (1-pnorm(abs(cross5T))) #Store moderator p value


########################################
# SAVE MODEL RESULTS
########################################
 model1Output <- summary(model1)
 model2Output <- summary(model2)
 model3Output <- summary(model3)
 model4Output <- summary(model4)
 model5Output <- summary(model5)

 capture.output(model1Output, file = "model1Output.txt")
 capture.output(model2Output, file = "model2Output.txt")
 capture.output(model3Output, file = "model3Output.txt")
 capture.output(model4Output, file = "model4Output.txt")
 capture.output(model5Output, file = "model5Output.txt")

########################################
# ESTIMATE THE *FINAL* PARAMETER ESTIMATE FOR THE FOCAL CLAIM
########################################

# Take average of multiplicative results (see p. 405 od original article and Step 2 on p. 120 of PISA Data Analysis Manual)

 library(plyr)
 crossResults <- as.data.frame(rbind(cross1Result, cross2Result, cross3Result, cross4Result, cross5Result))
 names(crossResults)[names(crossResults) == "V1"] <- "Coefficient"
 final_crossB <- mean(crossResults$Coefficient)
 final_crossB #This is the final parameter for the moderating effect under investigation

########################################
# ESTIMATE THE *FINAL* PARAMETER STANDARD ERROR FOR THE FOCAL CLAIM
########################################

 M <- 5 #There are five plausible values

# Estimate final sampling variance (see #3 on page 120 of PISA Data Analysis Manual)
 sv1 <- cross1SE^2
 sv2 <- cross2SE^2
 sv3 <- cross3SE^2
 sv4 <- cross4SE^2
 sv5 <- cross5SE^2

 final_sampling_variance <- (sv1 + sv2 + sv3 + sv4 + sv5) / M
 final_sampling_variance

# Estimate imputation variance (see #4 of PISA Data Analysis Manual)
 sq_diff_1 <- (cross1Result - final_crossB)^2
 sq_diff_2 <- (cross2Result - final_crossB)^2
 sq_diff_3 <- (cross3Result - final_crossB)^2
 sq_diff_4 <- (cross4Result - final_crossB)^2
 sq_diff_5 <- (cross5Result - final_crossB)^2

 sum_sq_diff <- sq_diff_1 + sq_diff_2 + sq_diff_3 + sq_diff_4 + sq_diff_5

 final_imputation_variance <- sum_sq_diff / (M-1)  #See #4 on page 120 of PISA Data Analysis Manual
 final_imputation_variance

# Estimate final error variance (see #5 of PISA Data Analysis Manual)
 final_error_variance <- final_sampling_variance + (1.2*final_imputation_variance)
 final_error_variance

# Estimate final standard error (see #6 of PISA Data Analysis Manual)
 finalSE <- sqrt(final_error_variance)
 finalSE #This is the final SE for the moderating effect under investigation

# Estimate final p-value 
# See https://rdrr.io/cran/metaRNASeq/man/fishercomb.html
 cross_P_results <- as.data.frame(rbind(cross1P, cross2P, cross3P, cross4P, cross5P))
 names(cross_P_results)[names(cross_P_results) == "V1"] <- "pValues"

 #install.packages("metaRNASeq")
 library(metaRNASeq)
 #vignette("metaRNASeq") #see page 7

 finalP <- fishercomb(cross_P_results$pValues)
 finalP$adjpval #This is the final p-value for the moderating effect under investigation

## PRINT FINAL FOCAL ANALYSIS RESULTS
 final_crossB
 finalSE
 finalP$adjpval

 #install.packages("rlang")
 library("rlang")
 #if (!requireNamespace("BiocManager", quietly = TRUE))
 #        install.packages("BiocManager")
 #BiocManager::install("multtest")
 library(multtest)
 #install.packages("metap", repos = c("http://rstudio.org/_packages",
 #          "http://cran.rstudio.com"))
 library(metap)

 getwd()
 write.csv(cross_P_results, "cross_P_results.csv")

 allmetap(cross_P_results$pValues, method = "all") #https://cran.r-project.org/web/packages/metap/metap.pdf


