# Multi100 Replication of Christensen et al (2018)
# 5/12/2022

#### THE CLAIM ###
# Claim: "High openness to experience group’s semantic network would have a lower average shortest path length (ASPL) than the low openness to experience group’s network."

#### SETUP ####
# install.packages("SemNetCleaner")
# install.packages("NetworkToolbox")
# install.packages("tidyverse")
# install.packages("shiny")
library(SemNetCleaner)
library(NetworkToolbox)
library(tidyverse)
library(shiny)

#### GET DATA ####

fluency <- read_csv("FINAL fluency.csv")
open <- read_csv("FINAL open.csv")


# Median split on fluency:
open <- open %>%
  mutate(group = LATENT > median(LATENT))

open %>% 
  group_by(group) %>% 
  summarize(latentm = mean(LATENT))

#### SEMANTIC NETWORKS ####

# The textcleaner() function asks for interactive human feedback on words that are spelled
# incorrectly. I just rejected all words that were not automatically matched by the dictionary (response=5), even if
# I thought I *may* have been able to guess the intended meaning:

fluency.cleaned <- textcleaner(fluency, partBY = "row", 
                               dictionary = "animals", 
                               spelling = "US", allowPunctuations = "-")

# Get binarized matrix:
fluency.binary <- fluency.cleaned$responses$binary

#### THREE APPROACHES ####

# (1) Trying to reproduce the approach in Christensen et al (2018), in which one big semantic
# network is created for each group, and then this network is sub-sampled.

# (2) I'm worried that (1) can't account for a few or even one highly influential individuals. 
# So, instead of sub-sampling from one big network for each group, 
# bootstrap sub-samples of individuals, 
# and then calculate ASPL and openness-to-experience for each bootstrapped group.

# (3) Similar to (2), but create bins based on openness-to-experience, 
# and calculate ASPL within each group. 

# Here, we just use APPROACH 1, to generate a single t-statistic for subnetworks with 90% of nodes, as requested by the Task 2 instructions. 

#### APPROACH 1 ####
# Split into high- and low-openness groups (group 1 is High Openness)
fluency.binary1 <- fluency.binary[open$group,]
fluency.binary2 <- fluency.binary[!open$group,]

# How many people mentioned each word in each group?
wordCounts1 <- colSums(fluency.binary1)
wordCounts2 <- colSums(fluency.binary2)

# Which words were only mentioned once or never? 
common1 <- wordCounts1[wordCounts1 >= 2]
common2 <- wordCounts2[wordCounts2 >= 2]
commonWords <- intersect(names(common1), names(common2))

# Semantic networks with only common words: 
fluency.binary.common1 <- fluency.binary1[,commonWords]
fluency.binary.common2 <- fluency.binary2[,commonWords]



# Pairwise cosine similarity matrix for each word vector:
# (I know there's a more efficient linear algebra way to get the cosine similarity 
# matrix in one step... but this brute-force method is easier for me to implement on an airplane without wifi)

vectorLength <- function(v){ # Utility function for getting vector length: 
  sqrt(sum((v * v)^2))
}

cos1 <- matrix(NA, 
               ncol=ncol(fluency.binary.common1), 
               nrow=ncol(fluency.binary.common1))
for(i in 1:ncol(fluency.binary.common1)){
  for(j in 1:ncol(fluency.binary.common1)){
    col1 <- fluency.binary.common1[,i]
    col2 <- fluency.binary.common1[,j]
    cos1[i,j] <- as.numeric(col1 %*% col2/(vectorLength(col1) * vectorLength(col2)))
  }
}


cos2 <- matrix(NA, 
               ncol=ncol(fluency.binary.common2), 
               nrow=ncol(fluency.binary.common2))
for(i in 1:ncol(fluency.binary.common2)){
  for(j in 1:ncol(fluency.binary.common2)){
    col1 <- fluency.binary.common2[,i]
    col2 <- fluency.binary.common2[,j]
    cos2[i,j] <- as.numeric(col1 %*% col2/(vectorLength(col1) * vectorLength(col2)))
  }
}


# Following the original analysis, we apply the TMFG procedure to get rid of weak edges:
tmfg1 <- TMFG(cos1)$A
tmfg2 <- TMFG(cos2)$A


# Binarize the edges (1/0) to get final full networks for each group: 
finalNet1 <- matrix(as.numeric(tmfg1 > 0), nrow = nrow(tmfg1))
finalNet2 <- matrix(as.numeric(tmfg2 > 0), nrow = nrow(tmfg2))

# Bootstrap subnetworks with only 90% of nodes: 

num_boots <- 1000   # number of bootstraps
subNetSize <- .9    # subnetwork size

bootDF <- tibble()  # tibble for bootstrapped results
set.seed(42)
for(i in 1:num_boots){
  # print(i)
  # sample random nodes
  bootNodes <- sample(1:nrow(finalNet1), size = subNetSize * nrow(finalNet1), replace = F)
  
  # create subnetworks for both groups:
  bootNet1 <- finalNet1[bootNodes,bootNodes]
  bootNet2 <- finalNet2[bootNodes,bootNodes]
  
  # get ASPL for each subnetwork:
  aspl1 <- pathlengths(bootNet1)$ASPL
  aspl2 <- pathlengths(bootNet2)$ASPL
  
  # save results:
  bootDF <- bind_rows(bootDF, 
                      tibble(i, aspl1, aspl2))
}


# The moment of truth: 
mean(bootDF$aspl1)
mean(bootDF$aspl2)
t.test(bootDF$aspl1, bootDF$aspl2, paired=T)
tstat <- t.test(bootDF$aspl1, bootDF$aspl2, paired=T)$stat
dstat <- DescTools::CohenD(bootDF$aspl1, bootDF$aspl2)[1]
tstat
t.test(bootDF$aspl1, bootDF$aspl2, paired=T)$p.value
dstat


# Conclusion (1):
# ASPL is shorter for high-openness group (3.1) than low-openness group (4.1).
# And t = -77 and d = -2.7, which is similar to that reported in the paper. 
# We thus reproduced the main results. 
