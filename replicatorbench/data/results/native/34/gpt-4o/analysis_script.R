# Load necessary libraries
library(lme4)
library(lmerTest)

# Load the dataset
sandra_replicate <- read.csv('/app/data/sandra_replicate.csv')

# Replication analysis
summary(lmer(scale(logRT) ~ scale(NFC) * trial * rewardlevel + blocknumber + (1|SubjectID), data = sandra_replicate, subset = accuracy==1))

# Follow-up simple effects
summary(lmer(scale(logRT) ~ HiNFC * trial * rewardlevel + blocknumber + (1|SubjectID), data = sandra_replicate, subset = accuracy==1))
summary(lmer(scale(logRT) ~ LoNFC * trial * rewardlevel + blocknumber + (1|SubjectID), data = sandra_replicate, subset = accuracy==1))

# Additional replication analysis - accuracy
summary(glmer(accuracy ~ scale(NFC) * trial * rewardlevel + blocknumber + (1|SubjectID), data = sandra_replicate, family="binomial"))

# Hypothesis 1 - executive function
summary(lmer(scale(logRT) ~ scale(Stroopindex) * trial * rewardlevel + blocknumber + (1|SubjectID), data = sandra_replicate, subset = accuracy==1))

# Hypothesis 2 (Intrinsic Motivation) Analyses
summary(lmer(scale(logRT) ~ scale(IMI) * trial * rewardlevel + blocknumber + (1|SubjectID), data = sandra_replicate, subset = accuracy==1))
