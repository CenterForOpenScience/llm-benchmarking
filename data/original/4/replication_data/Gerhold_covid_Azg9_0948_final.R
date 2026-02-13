# Project name: Gerhold_covid_Azg9_0948 
# Script author: James Field (jamesfield6912@gmail.com)
# Last updated: November 16, 2020
# replication project website: https://osf.io/a7h9n/

# Import data set from GitHub
#install.packages("readr")
library (readr)
urlfile="https://raw.githubusercontent.com/jamiefield/jamiefield.github.io/master/files/data_gerhold.csv"
dat_gerhold <- read_csv(url(urlfile))

# Remove "missing" data (i.e., gender = 3)
#install.packages("dplyr")
library(dplyr)
dat_gerhold <- filter(dat_gerhold, gender != "3")

# Select 5% random sample of imported data
#set.seed(200187)
#dat_gerhold <- sample_n(dat_gerhold, round(0.05*nrow(dat_gerhold)))

# Get summary statistics for females and males
#install.packages("psych")
library(psych)
describe.by(dat_gerhold,group="female")

# Create female and male subgroups
female_group <- dat_gerhold[ which(dat_gerhold$female=='1'),] #this is the female group
male_group <- dat_gerhold[ which(dat_gerhold$female=='0'),] #this is the male group

## T-TEST SECTION BELOW ##

# Select "mh_anxiety_1" from female_group and male_group to test for homoscedasticity (i.e., are the groups homogenous). We do this by conducting a Fishers F-test
x <- female_group$mh_anxiety_1
y <- male_group$mh_anxiety_1

# If the p-value from this test is p > .05 (greater than .05), then you can assume that the variances of both samples are homogenous. 
var.test(x,y)

# Given that the corresponding p-value is greater than .05, we conclude that the variances of both samples are homogenouswe and, this, we run a classic Student's two-sample t-test by setting the parameter var.equal = TRUE
focalClaim <- t.test(x, y, var.equal = TRUE)
focalClaim

#interpretation: female mean score is statistically different (higher) than male mean score

##############################################################
##############################################################
##############################################################

# Select "mh_anxiety_3" from female_group and male_group to test for homoscedasticity (i.e., are the groups homogeneous). We do this by conducting a Fishers F-test
x <- female_group$mh_anxiety_3
y <- male_group$mh_anxiety_3

# If the p-value from this test is p > .05 (greater than .05), then you can assume that the variances of both samples are homogeneous. 
var.test(x,y)

# Given that the corresponding p-value is greater than .05, we conclude that the variances of both samples are homogenous we and, this, we run a classic Student's two-sample t-test by setting the parameter var.equal = TRUE
exploratory <- t.test(x, y, var.equal = TRUE)
exploratory

#interpretation: female mean score is statistically different (higher) than male mean score

##############################################################
##############################################################
##############################################################

## Calculate Cohen's d for focal claim replication
cohenD <- cohen.d(dat_gerhold$mh_anxiety_1, dat_gerhold$female)
cohenD$cohen.d[,2] ## effect size
cohenD$p ## corresponding p-value




