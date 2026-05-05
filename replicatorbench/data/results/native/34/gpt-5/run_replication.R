#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  if (!requireNamespace("lme4", quietly = TRUE)) install.packages("lme4", repos = "https://cloud.r-project.org"); library(lme4)
  if (!requireNamespace("lmerTest", quietly = TRUE)) install.packages("lmerTest", repos = "https://cloud.r-project.org"); library(lmerTest)
  if (!requireNamespace("readr", quietly = TRUE)) install.packages("readr", repos = "https://cloud.r-project.org"); library(readr)
  if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr", repos = "https://cloud.r-project.org"); library(dplyr)
  if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite", repos = "https://cloud.r-project.org"); library(jsonlite)
})

# Resolve input and output paths
in_paths <- c("/app/data/sandra_replicate.csv", "replication_data/sandra_replicate.csv", "/workspace/replication_data/sandra_replicate.csv")
in_path <- NULL
for (p in in_paths) { if (file.exists(p)) { in_path <- p; break } }
if (is.null(in_path)) { stop("Could not find sandra_replicate.csv in /app/data or replication_data.") }

out_dir <- "/app/data"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Read data (skip wrapper header if present)
message(sprintf("Reading data from: %s", in_path))
header_line <- tryCatch(readLines(in_path, n = 1), error = function(e) "")
skip_header <- if (length(header_line) > 0 && grepl("^#!/", header_line)) 3 else 0
df <- readr::read_csv(in_path, show_col_types = FALSE, skip = skip_header, comment = "#")

# Ensure required columns exist
required_cols <- c("logRT","trial","rewardlevel","blocknumber","NFC","SubjectID","accuracy")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse=", "))) 

# Coerce types sensibly
num_cols <- c("logRT","trial","rewardlevel","blocknumber","NFC","accuracy")
for (nm in intersect(num_cols, names(df))) df[[nm]] <- suppressWarnings(as.numeric(df[[nm]]))
df$SubjectID <- as.factor(df$SubjectID)

# Subset to correct trials and complete cases for RT model
rt_df <- df %>% dplyr::filter(accuracy == 1) %>% dplyr::filter(stats::complete.cases(logRT, trial, rewardlevel, blocknumber, NFC, SubjectID))

if (nrow(rt_df) < 100) stop("Too few rows after filtering for RT model; check data.")

# Fit mixed-effects model on log RTs
formula_rt <- stats::as.formula("scale(logRT) ~ scale(NFC) * trial * rewardlevel + blocknumber + (1|SubjectID)")
message("Fitting lmer model for RTs...")
mod_rt <- lmerTest::lmer(formula_rt, data = rt_df, REML = FALSE)

# Extract summaries
summ_rt <- summary(mod_rt)
fixef_tab <- as.data.frame(coef(summ_rt))
fixef_tab$term <- rownames(fixef_tab)
rownames(fixef_tab) <- NULL

# Find the three-way interaction term robustly (order-insensitive)
all_terms <- fixef_tab$term
match_term <- function(term) {
  # normalize by removing spaces
  term <- gsub("\\s+", "", term)
  parts <- c("trial", "rewardlevel", "scale(NFC)")
  parts <- gsub("\\s+", "", parts)
  # Check if term contains all parts separated by ':'
  ok <- all(vapply(parts, function(x) grepl(x, term, fixed = TRUE), logical(1))) && grepl(":", term)
  ok && length(strsplit(term, ":")[[1]]) >= 3
}
tri_idx <- which(vapply(all_terms, match_term, logical(1)))
tri_row <- if (length(tri_idx) >= 1) tri_idx[1] else NA_integer_

primary <- list()
if (!is.na(tri_row)) {
  primary <- list(
    term = fixef_tab$term[tri_row],
    estimate = unname(fixef_tab$Estimate[tri_row]),
    std_error = unname(fixef_tab$`Std. Error`[tri_row]),
    t_value = unname(fixef_tab$`t value`[tri_row]),
    p_value = if (!is.null(fixef_tab$`Pr(>|t|)`)) unname(fixef_tab$`Pr(>|t|)`[tri_row]) else NA_real_,
    direction = ifelse(unname(fixef_tab$Estimate[tri_row]) > 0, "positive", ifelse(unname(fixef_tab$Estimate[tri_row]) < 0, "negative", "null"))
  )
} else {
  warning("Could not automatically locate the three-way interaction term; saving full table only.")
}

# Save outputs
# Avoid printing summary to file to bypass printing bug
# sink(file.path(out_dir, "lmer_summary.txt"))
# print(summ_rt)
# sink()

readr::write_csv(fixef_tab, file.path(out_dir, "fixed_effects.csv"))

result_obj <- list(
  model = "lmer(scale(logRT) ~ scale(NFC) * trial * rewardlevel + blocknumber + (1|SubjectID))",
  n_obs = nrow(rt_df),
  n_subjects = length(unique(rt_df$SubjectID)),
  primary_interaction = primary
)
jsonlite::write_json(result_obj, file.path(out_dir, "replication_results.json"), pretty = TRUE, auto_unbox = TRUE)

message("Analysis complete. Outputs written to /app/data.")
