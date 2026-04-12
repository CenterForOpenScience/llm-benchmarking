# replicate_analysis.R
# This script replicates the focal analysis described in replication_info.json
# It fits a linear mixed-effects model predicting log-transformed RTs in a task-switching paradigm
# Author: automated replication bot

library(tidyverse)
library(lme4)
library(lmerTest)

# -----------------------------------------------------------------------------
# Helper: safe scale that returns numeric vector (not matrix)
scale_vec <- function(x) as.numeric(scale(x))

# -----------------------------------------------------------------------------
# 1. Load data -----------------------------------------------------------------# 1. Load data -----------------------------------------------------------------
# Attempt to locate dataset in common locations
possible_paths <- c(
  "replication_data/sandra_replicate.csv",                                     # same folder hierarchy as script
  "/workspace/replication_data/sandra_replicate.csv",                          # absolute path if script copied elsewhere
  "/app/data/original/34/input/replication_data/sandra_replicate.csv",         # mounted original relative path
  "/app/data/34/input/replication_data/sandra_replicate.csv",                  # alternative without 'original'
  "/app/data/replication_data/sandra_replicate.csv"                            # mounted top-level data folder
)

data_path <- NULL
for (p in possible_paths) {
  if (file.exists(p)) {
    data_path <- p
    break
  }
}

if (is.null(data_path)) {
  cat("Available files in current working directory:\n")
  print(list.files(getwd(), recursive = TRUE))
  stop("Dataset sandra_replicate.csv could not be located in expected paths.")
}

cat(sprintf("Reading dataset from %s\n", data_path))

df <- read_csv(data_path, show_col_types = FALSE)

# -----------------------------------------------------------------------------
# 2. Data cleaning / filtering --------------------------------------------------

# Keep only correct trials (accuracy == 1)

df <- df %>%
  filter(accuracy == 1)

# Remove rows with missing critical predictors
critical_vars <- c("NFC", "trial", "rewardlevel", "blocknumber", "logRT")
df <- df %>% drop_na(any_of(critical_vars))

# Ensure predictor coding is numeric (already -1 / 1 according to codebook)
# Convert to numeric just in case

df <- df %>%
  mutate(
    trial = as.numeric(trial),
    rewardlevel = as.numeric(rewardlevel),
    blocknumber = as.numeric(blocknumber),
    SubjectID = factor(SubjectID)
  )

# -----------------------------------------------------------------------------
# 3. Variable transformations ---------------------------------------------------

# Z-score Need for Cognition (NFC) and logRT

df <- df %>%
  mutate(
    NFC_z = scale_vec(NFC),
    logRT_z = scale_vec(logRT)
  )

# -----------------------------------------------------------------------------
# 4. Fit linear mixed-effects model --------------------------------------------

model_formula <- logRT_z ~ NFC_z * trial * rewardlevel + blocknumber + (1 | SubjectID)

cat("Fitting model:\n")
print(model_formula)

lmer_model <- lmer(model_formula, data = df, REML = FALSE)

model_summary <- summary(lmer_model)
print(model_summary)

# -----------------------------------------------------------------------------
# 5. Extract focal coefficient --------------------------------------------------

coefs <- coef(summary(lmer_model))

if (!"NFC_z:trial:rewardlevel" %in% rownames(coefs)) {
  stop("Focal interaction term not found in model coefficients. Please check coding.")
}

focal <- coefs["NFC_z:trial:rewardlevel", ]

# Create results data frame
results_df <- tibble(
  term = "NFC_z:trial:rewardlevel",
  estimate = focal["Estimate"],
  std_error = focal["Std. Error"],
  t_value = focal["t value"],
  p_value = focal["Pr(>|t|)"]
)

print("Focal coefficient:")
print(results_df)

# -----------------------------------------------------------------------------
# 6. Save results ---------------------------------------------------------------

results_path <- "/app/artifacts/focal_coefficient.csv"

readr::write_csv(results_df, results_path)

cat(sprintf("Results written to %s\n", results_path))
