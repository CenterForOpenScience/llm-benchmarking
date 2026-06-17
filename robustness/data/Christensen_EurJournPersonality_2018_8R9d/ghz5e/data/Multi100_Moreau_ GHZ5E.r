#############
### ABOUT ###
#############

## Analyst: David Moreau, University of Auckland, d.moreau@auckland.ac.nz - ID = GHZ5E

## Paper Title: Remotely Close Associations: Openness to Experience and Semantic Memory Structure

## Paper ID: Christensen_EurJournPersonality_2018_8R9d

## Claim to test
##... the high openness to experience group’s network was more interconnected ... 
## than the low openness to experience group (p. 480.)


################
### ANALYSIS ###
################

# Load packages
library(SemNeT)
library(SemNetCleaner)
library(NetworkToolbox)

# Read data
latent <- read.csv('Final Open.csv')

# Combine latent variable with responses and create groups
comb <- as.data.frame(cbind(latent$no_int,IDedResp))
#changes column names
colnames(comb)[1:2] <- c("latent","id")
low <- comb[order(comb$latent),][1:258,]
high <- comb[order(comb$latent),][259:516,]

# Remove latent variable and ids
deLow <- low[,-c(1,2)]
deHigh <- high[,-c(1,2)]

# Remove responses given by one or no participants and equate nodes
finLow <- finalize(deLow)
finHigh <- finalize(deHigh)
eq <- equate(finLow,finHigh)
low <- eq$rmatA
high <- eq$rmatB

# Create cosine similarity matrices
cosLow <- similarity(finLow, method = "cosine"); diag(cosHigh) <- 1
cosHigh <- similarity(finHigh, method = "cosine"); diag(cosLow) <- 1

# Create networks
netLow <- TMFG(cosLow)$A
netHigh <- TMFG(cosHigh)$A

# Semantic network measures
low.meas <- semnetmeas(netLow, meas = c("ASPL", "CC", "Q")); low.meas
high.meas <- semnetmeas(netHigh, meas = c("ASPL", "CC", "Q")); high.meas

# Partial network bootstrapped approach
nodedrop <- list()
iter <- 1000
nodedrop$fifty <- partboot(high,low,n=ncol(high)*.50,iter=iter,corr="cosine",cores=4,weighted=TRUE)
nodedrop$sixty <- partboot(high,low,n=ncol(high)*.60,iter=iter,corr="cosine",cores=4)
nodedrop$seventy <- partboot(high,low,n=ncol(high)*.70,iter=iter,corr="cosine",cores=4)
nodedrop$eighty <- partboot(high,low,n=ncol(high)*.80,iter=iter,corr="cosine",cores=4)
nodedrop$ninety <- partboot(high,low,n=ncol(high)*.90,iter=iter,corr="cosine",cores=4)

# Partial network bootstrapped tests
partboot.test(nodedrop$ninety)
partboot.test(nodedrop$eighty)
partboot.test(nodedrop$seventy)
partboot.test(nodedrop$sixty)
partboot.test(nodedrop$fifty)

# ASPL (short path lengths indicate increased interconnectivity): LOW > HIGH
# CC (clustering coefficient): HIGH > LOW
# Q (modularity): LOW > HIGH

# This is consistent with the claim: the high openness to experience group’s network was more interconnected ... 
# than the low openness to experience group (p. 480.)

# For 90% partition, t = -65.69
