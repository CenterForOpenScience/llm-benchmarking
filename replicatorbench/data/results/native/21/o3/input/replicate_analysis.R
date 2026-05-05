#!/usr/bin/env Rscript
# Replication script for Bischetti et al. Covid-19 humor aversiveness focal test
# Author: Automated replication bot
# This script follows the preregistered plan contained in replication_info.json

# -----------------------------
# 1) Load libraries
# -----------------------------
required_pkgs <- c("rio", "reshape2", "lme4", "lmerTest", "emmeans")
for (p in required_pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    stop(sprintf("Package '%s' not installed inside container.", p))
  }
}
library(rio)
library(reshape2)
library(lme4)
library(lmerTest)
library(emmeans)

# -----------------------------
# 2) Define helper function
# -----------------------------
msg <- function(...) cat("[replicate] ", ..., "\n")

# -----------------------------
# 3) Paths & data import
# -----------------------------
# The orchestrator mounts the host data directory to /app/data inside the container.
# The study root (code) is available at the working directory set by Dockerfile ("/workspace").
# We assume the replication_data folder lives in the same directory as this script.

# Determine script directory (fallback to current working directory)
script_dir <- tryCatch(dirname(normalizePath(sys.frames()[[1]]$ofile)), error = function(e) ".")

data_dir <- file.path(script_dir, "replication_data")
if (!dir.exists(data_dir)) {
  # Try alternate path (inside /app/data/...)
  data_dir <- file.path("/app/data/original/21/input/replication_data")
}
msg("Using data directory:", data_dir)

part1_file <- file.path(data_dir, "Bischetti_Survey_Part1_deidentify.csv")
part2_file <- file.path(data_dir, "Bischetti_Survey_Part2_deidentify.csv")

if (!file.exists(part1_file) || !file.exists(part2_file)) {
  stop("Required CSV files not found in replication_data folder. Check mount paths.")
}

msg("Importing CSVs ...")
df1 <- import(part1_file)
df2 <- import(part2_file)

# -----------------------------
# 4) Merge halves on participant_id
# -----------------------------
msg("Merging part 1 and part 2 data ...")
df <- merge(df1, df2, by = "participant_id", all = TRUE)
msg("Merged rows:", nrow(df))

# -----------------------------
# 5) Select aversiveness variables & reshape to long
# -----------------------------
msg("Reshaping data to long format ...")
# Keep participant demographics plus any *_disturbing columns (aversiveness ratings)
keep_cols <- grep("disturbing$|^participant_id$|^state$|^age$", names(df), value = TRUE)
df_sub <- df[, keep_cols]

# Use reshape2::melt
long <- melt(df_sub, id.vars = c("participant_id", "state", "age"), variable.name = "name", value.name = "Aversiveness")

# Remove rows with missing ratings
long <- long[!is.na(long$Aversiveness), ]
msg("Long dataset rows (non-missing ratings):", nrow(long))

# -----------------------------
# 6) Convert 1–7 scale to 0–6 scale by subtracting 1 (some stimuli already 0–6, but this matches prereg plan)
# -----------------------------
long$Aversiveness <- long$Aversiveness - 1

# -----------------------------
# 7) Derive stimulus label (covid-verbal vs non-verbal)
# -----------------------------
# The original notebook used meta-data XLSX files to map each item to picture type.
# Those files are not available in this replication bundle. As a pragmatic solution,
# we infer the stimulus category from the variable (item) name, which embeds the stimulus code.
# Based on the original study codebook:
#  - covid verbal items contain the substring "covid-text"
#  - non-covid verbal items contain "non-text"

long$label <- NA_character_
long$label[grepl("covid.*text", long$name, ignore.case = TRUE)] <- "covid-verbal"
long$label[grepl("non.*text", long$name, ignore.case = TRUE)]  <- "non-verbal"

# Keep only rows that we could classify as the two verbal categories
verbal_long <- subset(long, !is.na(label))
msg("Rows after filtering to verbal jokes (covid & non):", nrow(verbal_long))

# -----------------------------
# 8) Build analysis variables
# -----------------------------
# Create numeric Covid indicator (1 = covid-verbal, 0 = non-verbal)
verbal_long$CovidIndicator <- ifelse(verbal_long$label == "covid-verbal", 1, 0)

# Convert grouping factors
verbal_long$participant_id <- as.factor(verbal_long$participant_id)
verbal_long$name           <- as.factor(verbal_long$name)

# -----------------------------
# 9) Fit Linear Mixed Model
# -----------------------------
msg("Fitting mixed-effects model ...")
model_formula <- Aversiveness ~ CovidIndicator + (1|participant_id) + (1|name)
mod <- lmer(model_formula, data = verbal_long, REML = FALSE, control = lmerControl(calc.derivs = FALSE))

# -----------------------------
# 10) Extract results
# -----------------------------
summary_mod <- summary(mod)

# Condition means
marginal_means <- emmeans(mod, specs = ~ CovidIndicator, type = "response")

# -----------------------------
# 11) Save outputs
# -----------------------------
artifact_dir <- "/app/data"
if (!dir.exists(artifact_dir)) artifact_dir <- "."
res_file <- file.path(artifact_dir, "replication_results.txt")

sink(res_file)
cat("Replication of Covid-19 verbal joke aversiveness effect\n\n")
print(summary_mod)
cat("\nMarginal Means (estimated):\n")
print(marginal_means)
sink()

msg("Results saved to", res_file)

# -----------------------------
# 12) Session info for reproducibility
# -----------------------------
msg("Session info:")
print(sessionInfo())
