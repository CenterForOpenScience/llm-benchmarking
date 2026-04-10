# Replication analysis script
# This script implements the preregistered focal test:
# baseline Realistic COVID-19 threat (T1) predicts follow-up Negative Affect (T2)
# after controlling for Symbolic threat (T1).

library(dplyr)    # data manipulation

# -----------------------------------------------------------------------------
# Load data --------------------------------------------------------------------# Load data --------------------------------------------------------------------
# Attempt several likely paths for the CSV depending on how the host folder is
# mounted inside the container.

possible_paths <- c(
  "/app/data/original/20/input/replication_data/Kachanoff_Survey_deidentify.csv",
  "replication_data/Kachanoff_Survey_deidentify.csv",
  "./replication_data/Kachanoff_Survey_deidentify.csv",
  "../replication_data/Kachanoff_Survey_deidentify.csv",
  "/app/data/replication_data/Kachanoff_Survey_deidentify.csv"
)

csv_path <- NA
for (p in possible_paths) {
  if (file.exists(p)) { csv_path <- p; break }
}

if (is.na(csv_path)) {
  stop("Could not locate Kachanoff_Survey_deidentify.csv in expected paths.")
}

cat("Using data file:", csv_path, "\n")

df <- read.csv(csv_path, stringsAsFactors = FALSE)df <- read.csv(csv_path, stringsAsFactors = FALSE)

# -----------------------------------------------------------------------------
# Create composite variables ----------------------------------------------------
# (Replicates the scoring logic shown in Analysis_updated.Rmd)

# Realistic and Symbolic COVID-19 threat (mean of 5 items each, 1–4 scale)
df$Realistic <- apply(df[, grepl("^covid_real[0-9]+$", names(df))], 1, mean, na.rm = TRUE)
df$Symbolic  <- apply(df[, grepl("^covid_symbolic[0-9]+$", names(df))], 1, mean, na.rm = TRUE)

# Negative affect (PANAS; sum of 10 items, 1–5 scale)
df$Negative  <- apply(df[, grepl("^negative[0-9]+$", names(df))], 1, sum,  na.rm = TRUE)

# -----------------------------------------------------------------------------
# Reshape to wide format (identify Time-1 vs Time-2 entries) --------------------

# Order rows by creation time so the *earliest* record per participant is T1
if (!"created" %in% names(df)) {
  stop("'created' timestamp column not found in data. Cannot determine waves.")
}

df <- df[order(df$created), ]

# Determine unique participant identifier column present in the dataset.
# The de-identified column is `participant_id`; the fallback is `PROLIFIC_PID`.

id_var <- if ("participant_id" %in% names(df)) "participant_id" else "PROLIFIC_PID"

# Flag second submission (TRUE == Time-2)
df$time2 <- duplicated(df[[id_var]])

df_time1 <- subset(df,  time2 == FALSE)  # baseline
 df_time2 <- subset(df, time2 == TRUE)   # follow-up

# Keep only necessary columns for each wave
cols_t1 <- c(id_var, "Realistic", "Symbolic")
cols_t2 <- c(id_var, "Negative")

df_wide <- merge(df_time1[, cols_t1], df_time2[, cols_t2], by = id_var)

# -----------------------------------------------------------------------------
# Fit preregistered regression --------------------------------------------------

model <- lm(Negative ~ Realistic + Symbolic, data = df_wide)
summary_out <- summary(model)

cat("\n================ Regression Summary ================\n")
print(summary_out)

# -----------------------------------------------------------------------------
# Save coefficient table --------------------------------------------------------

coef_df <- as.data.frame(summary_out$coefficients)
coef_df$term <- rownames(coef_df)
rownames(coef_df) <- NULL

results_path <- "/app/data/replication_results.csv"
write.csv(coef_df, results_path, row.names = FALSE)

cat("\nCoefficient table written to:", results_path, "\n")
