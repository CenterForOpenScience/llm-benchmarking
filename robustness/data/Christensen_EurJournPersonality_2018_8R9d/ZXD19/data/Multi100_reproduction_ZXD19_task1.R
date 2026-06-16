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
dstat


# Conclusion (1):
# ASPL is shorter for high-openness group (3.1) than low-openness group (4.1).
# And t = -77 and d = -2.7, which is similar to that reported in the paper. 
# We thus reproduced the main results. 

#### APPROACH 2 ####
# Bootstrap at the subject level:
# Create randoms subsets of participants, then calculate ASPL and mean openness for each subset

num_boots <- 1000
subSampleN <- 20
bootSubResults <- tibble()

set.seed(42)
for(booti in 1:num_boots){
  print(booti)
  bootIDs <- sample(1:nrow(fluency.binary), subSampleN, replace = F)
  
  subFluency <- fluency.binary[bootIDs,]
  subMeanOpenness <- mean(open[bootIDs, ]$LATENT)
  
  # How many people mentioned each word?
  wordCounts <- colSums(subFluency)
  
  # Which words were only mentioned once or never? 
  common <- wordCounts[wordCounts >= 2]
  
  # Semantic networks with only common words: 
  subFluency.common <- subFluency[,names(common)]
  
  # Pairwise cosine similarity matrix for each word vector:
  # (I know there's a more efficient linear algebra way to get the cosine similarity 
  # matrix in one step... but this brute-force method is easier for me to implement on an airplane without wifi)
  
  
  cosM <- matrix(NA, 
                 ncol=ncol(subFluency.common), 
                 nrow=ncol(subFluency.common))
  for(i in 1:ncol(subFluency.common)){
    for(j in 1:ncol(subFluency.common)){
      col1 <- subFluency.common[,i]
      col2 <- subFluency.common[,j]
      cosM[i,j] <- as.numeric(col1 %*% col2/(vectorLength(col1) * vectorLength(col2)))
    }
  }
  
  # Following the original analysis, we apply the TMFG procedure to get rid of weak edges:
  tmfgSub <- TMFG(cosM)$A
  
  # Binarize the edges (1/0) to get final full networks for each group: 
  finalNetSub <- matrix(as.numeric(tmfgSub > 0), nrow = nrow(tmfgSub))
  asplSub <- pathlengths(finalNetSub)$ASPL
  
  bootSubResults <- bind_rows(bootSubResults, 
                              tibble(booti, open = subMeanOpenness, aspl = asplSub, 
                                     num_words = nrow(finalNetSub)))
}


# Conclusion (2)
# Significant association between openness and ASPL, but in the wrong direction:
cor.test(bootSubResults$open, bootSubResults$aspl)
summary(lm(aspl ~ open, bootSubResults))

# When we control for the number of words, the effect goes away, but remains 
# in the numerically unpredicted direction:
summary(lm(aspl ~ open + words, bootSubResults))

# Plot: 
bootSubResults %>% 
  ggplot(aes(x=open, y=aspl)) + 
  geom_point() + 
  geom_smooth(method="lm") + 
  theme_classic()

# Statistically significant association between openness-to-experience and ASPL, 
# but in the *wrong* direction — more openness-to-experience is associated with *longer* paths.

#### APPROACH 3 ####
# Sliding window analysis:
# Create subsets of participants, based on a sliding window of openness-to-experience:

window_width <- .1
step_size <- window_width
# step_size <- .01

slidingResults <- tibble()

set.seed(42)
for(stepi in seq(min(open$LATENT), max(open$LATENT) - window_width, step_size)){
  print(stepi)
  
  stepIDs <- open %>% 
    mutate(insideWindow = (LATENT >= stepi & LATENT < (stepi + window_width)))
  
  stepFluency <- fluency.binary[stepIDs$insideWindow,]
  stepMeanOpenness <- mean(subset(stepIDs, insideWindow==T)$LATENT)
  
  # How many people mentioned each word?
  wordCounts <- colSums(stepFluency)
  
  # Which words were only mentioned once or never? 
  common <- wordCounts[wordCounts >= 2]
  
  # Semantic networks with only common words: 
  stepFluency.common <- stepFluency[,names(common)]
  
  # Pairwise cosine similarity matrix for each word vector:
  # (I know there's a more efficient linear algebra way to get the cosine similarity 
  # matrix in one step... but this brute-force method is easier for me to implement on an airplane without wifi)
  
  
  cosM <- matrix(NA, 
                 ncol=ncol(stepFluency.common), 
                 nrow=ncol(stepFluency.common))
  for(i in 1:ncol(stepFluency.common)){
    for(j in 1:ncol(stepFluency.common)){
      col1 <- stepFluency.common[,i]
      col2 <- stepFluency.common[,j]
      cosM[i,j] <- as.numeric(col1 %*% col2/(vectorLength(col1) * vectorLength(col2)))
    }
  }
  
  # Following the original analysis, we apply the TMFG procedure to get rid of weak edges:
  tmfgStep <- TMFG(cosM)$A
  
  # Binarize the edges (1/0) to get final full networks for each group: 
  finalNetStep <- matrix(as.numeric(tmfgStep > 0), nrow = nrow(tmfgStep))
  asplStep <- pathlengths(finalNetStep)$ASPL
  
  slidingResults <- bind_rows(slidingResults, 
                              tibble(stepi, n = nrow(subset(stepIDs, insideWindow==T)),
                                     open = stepMeanOpenness, aspl = asplStep))
}


# Conclusion (3)
cor.test(filter(slidingResults, n > 20)$open, filter(slidingResults, n > 20)$aspl)
cor.test(slidingResults$open, slidingResults$aspl)

slidingResults %>% 
  ggplot(aes(x=open, y=aspl)) + 
  geom_point(aes(size=n)) + 
  geom_smooth(method="lm") + 
  theme_classic()

# No significant association between openness-to-experience and ASPL in bins based on openness.
# The numerical association goes in the *wrong* direction.



#### FINAL CONCLUSION ####

# Final conclusion: At an individual level, openness-to-experience does not seem 
# to be reliably associated with average shorter path length in semantic networks. 

# Greater openness-to-experience is only associated with shorter ASPL when the semantic
# networks are created by pooling across many individuals and then subsampling from that 
# large pooled semantic network. 

# When networks are created from bootstrapped samples of subets of all subjects, there is no 
# reliable association or it goes in the other direction. 