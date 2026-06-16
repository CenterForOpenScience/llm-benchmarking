#!/usr/bin/env Rscript
# Task1: Kendall's tau between average safe choices (PV+RV) and LV decision errors
# Follows planned path: exclude age==99; compute risk_avg_safe and risk3_sum_errors; test with Kendall's tau.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(readstata13)
  library(stats)
})

# Resolve data path robustly
paths <- c("/app/data/RiskData.dta", "data/RiskData.dta", "./data/RiskData.dta", "RiskData.dta", "./RiskData.dta")
data_path <- NA
for (p in paths) {
  if (file.exists(p)) { data_path <- p; break }
}
if (is.na(data_path)) {
  stop("RiskData.dta not found in expected locations.")
}

# Load data
bru_dat <- read.dta13(data_path)
bru_dat <- data.table(bru_dat)

# Exclude experimenter stations
bru_dat <- bru_dat[age != 99]

# Compute PV and RV safe-choice sums
bru_dat[, risk1_sum := Risk11 + Risk12 + Risk13 + Risk14 + Risk15 + Risk16 + Risk17 + Risk18 + Risk19 + Risk110]
bru_dat[, risk2_sum := Risk21 + Risk22 + Risk23 + Risk24 + Risk25 + Risk26 + Risk27 + Risk28 + Risk29 + Risk210]

# Average safe choices across PV and RV
bru_dat[, risk_avg_safe := (risk1_sum + risk2_sum) / 2]

# LV decision errors (exclude item 5 which is always correct)
bru_dat[, risk3_sum_errors := abs(Risk31 - 1) + abs(Risk32 - 1) + abs(Risk33 - 1) + abs(Risk34 - 1) + 
                                 Risk36 + Risk37 + Risk38 + Risk39 + Risk310]

# Remove missing rows for analysis variables
bru_use <- bru_dat[!is.na(risk_avg_safe) & !is.na(risk3_sum_errors)]

# Kendall's tau
ct <- suppressWarnings(cor.test(bru_use$risk_avg_safe, bru_use$risk3_sum_errors, method = "kendall", exact = FALSE))

estimate <- unname(ct$estimate)
p_value <- ct$p.value
n <- nrow(bru_use)
direction <- ifelse(estimate > 0, "positive", ifelse(estimate < 0, "negative", "zero"))

# Prepare artifact directory
out_dir <- "artifacts"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Write result JSON
res <- list(
  task_id = "Task1",
  metric = "kendalls_tau",
  estimate = estimate,
  p_value = p_value,
  direction = direction,
  sample_size = n,
  method = "Kendall's tau (cor.test)",
  notes = "Computed risk_avg_safe from PV and RV; risk3_sum_errors excluded item 5; excluded age==99."
)

jsonlite::write_json(res, file.path(out_dir, "task1_result.json"), auto_unbox = TRUE, pretty = TRUE)

cat(sprintf("Task1 complete. Kendall's tau = %.4f, p = %.4g, N = %d\n", estimate, p_value, n))
