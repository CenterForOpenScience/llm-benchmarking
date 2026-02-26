##============================================================================
## Load required packages 

Packages <- c("readxl","readr","tidyverse","ggplot2","kableExtra","pander","dplyr","emmeans","sjstats","tibble","reshape2")
lapply(Packages, library, character.only = TRUE)

##=============================================================================

## Read data
setwd('~/Dropbox/Research/Brand/') # set the directory
dat <- read_csv("data.csv") # read the dataset
items <- read_xlsx("ItemsList.xlsx") # read the stimuli dataset

# Cleaning data
dat <- dplyr::select(dat, -c(StartDate:First)) #delete first several unncessary columns (e.g., start time, etc.)
dat <- dat[3:nrow(dat),] # delete the first two rows that are explanations of the columns from Qualtrics
dat <- data.frame(sapply(dat[,1:ncol(dat)], as.numeric)) # change the columns as numeric
dat <- tibble::rowid_to_column(dat, "ID") # Impose ID to each participant 
dat$ID <- as.factor(dat$ID) # Make the ID variable as a factor variable 
dat <- dplyr::select(dat, -starts_with("Familiarity")) # Delete familiarity ratings since they are not parts of the main DVs
dat <- dplyr::select(dat, -contains("Bipol")) # Delete the Utilitarian vs. Hedonic bipolar ratings since they are not the main DVs

dat <- dat %>% filter(Attention1 == 7 & Attention2 == 1) # Exclude subjects who failed to provide the correct answers to the two attention check questions 

##=============================================================================

## Create separate data frames for the three conditions (control vs. Brand A vs. Brand B)

dat.control <- dat %>% filter(Cond == 1 | Cond == 2) # Control condition 
dat.control <- dat.control %>% 
  select(-contains("Util_1") & -contains("Util_2")) %>% 
  select(-contains("Symbol_1") & -contains("Symbol_2")) %>% 
  select(-contains("Bipol_1") & -contains("Bipol_2"))

dat.brandA <- dat %>% filter(Cond == 3) # Brand A condition
dat.brandA <- dat.brandA %>% 
  select(-ends_with("Util") & -contains("Util_2")) %>% 
  select(-ends_with("Symbol") & -contains("Symbol_2")) %>% 
  select(-ends_with("Bipol") & -contains("Bipol_2"))

dat.brandB <- dat %>% filter(Cond == 4) # Brand B condition 
dat.brandB <- dat.brandB %>% 
  select(-ends_with("Util") & -contains("Util_1")) %>% 
  select(-ends_with("Symbol") & -contains("Symbol_1")) %>% 
  select(-ends_with("Bipol") & -contains("Bipol_1"))

##=============================================================================

## Reshape each condition's data frame from wide to long

control_long <- reshape(dat.control, 
                idvar = c("ID","Cond"), 
                timevar = "ItemCatID",
                varying = list(c(2,4,6,8,10,12,14,16), c(3,5,7,9,11,13,15,17)), 
                v.names = c("Benefits", "Symbols"), 
                direction = "long")

brandA_long <- reshape(dat.brandA, 
                        idvar = c("ID","Cond"), 
                        timevar = "ItemCatID",
                        varying = list(c(2,4,6,8,10,12,14,16), c(3,5,7,9,11,13,15,17)), 
                        v.names = c("Benefits", "Symbols"), 
                        direction = "long")

brandB_long <- reshape(dat.brandB, 
                       idvar = c("ID","Cond"), 
                       timevar = "ItemCatID",
                       varying = list(c(2,4,6,8,10,12,14,16), c(3,5,7,9,11,13,15,17)), 
                       v.names = c("Benefits", "Symbols"), 
                       direction = "long")

## create a new variable "Condition" that distinguishes between the Category and Brand conditions
control_long$Condition <- "Category"
brandA_long$Condition <- "Brand"
brandB_long$Condition <- "Brand"

##================================================================================

combined <- rbind(control_long, brandA_long, brandB_long) # Combined the three conditions' data frames into one data frame
combined <- combined %>% full_join(items, by = c("ItemCatID", "Cond")) # merge the combined data with the stimuli data frame
combined <- combined %>% mutate(ScoreDiff = Benefits - Symbols) # Create the main DV (difference between benefits ratings - hedonic rationgs)

# Collapse the data by each participant, condition, and product types
dat_collapsed <- combined %>% 
  group_by(ID, ProductType, Condition) %>% 
  summarise(ScoreDiff = mean(ScoreDiff))

##================================================================================
## Descriptive Statistics by Items

combined %>% 
  group_by(Condition, Category) %>%
  summarise(Mean=mean(ScoreDiff)) %>% 
  dcast(Category ~ Condition, value.var="Mean") %>% 
  pander()

## Descriptive Statistics by Conditions

dat_collapsed %>%
  group_by(Condition, ProductType) %>%
  summarise(Mean=mean(ScoreDiff)) %>% 
  spread(Condition, Mean) %>%
  pander()

##================================================================================
## 2 (Level: NoBrand vs. Brand; between-subject) X 2 (Product Type: Utilitarian vs. Hedonic; within-subject) ANOVA

# ANOVA
model1 <- aov(ScoreDiff ~ Condition*ProductType + Error(ID/ProductType), data=dat_collapsed)
summary(model1)

# Effect Size
eta_sq(model1, partial=TRUE)

# Post-hoc Test
contrast.producttype <- emmeans(model1, pairwise ~ ProductType|Condition, adjust="tukey")
contrast.producttype$contrasts

contrast.condition <- emmeans(model1, pairwise ~ Condition|ProductType, adjust="tukey")
contrast.condition$contrasts



