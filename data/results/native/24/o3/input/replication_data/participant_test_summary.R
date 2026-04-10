#!/usr/bin/env Rscript
library(dplyr)
library(readr)
library(broom)

# Load dataset (assumes script is executed within replication_data directory)
cat('Loading analysis-data.csv ...\n')
data <- read_csv('analysis-data.csv', show_col_types = FALSE)

# Ensure required columns exist
required_cols <- c('subject','condition','phase','correct_1d_a','correct_1d_b')
missing <- setdiff(required_cols, names(data))
if(length(missing) > 0){
  stop(paste('Missing columns in dataset:', paste(missing, collapse=',')))
}

# Compute participant-level 1D accuracy during Test phase
summary_df <- data %>%
  filter(phase == 'Test') %>%
  group_by(subject, condition) %>%
  summarise(acc_1d_a = mean(correct_1d_a, na.rm = TRUE),
            acc_1d_b = mean(correct_1d_b, na.rm = TRUE), .groups='drop') %>%
  mutate(acc_1d = pmax(acc_1d_a, acc_1d_b))

# Save participant level summary
write_csv(summary_df, 'participant_test_summary.csv')

# Run independent samples t-test (equal variances)
t_res <- t.test(acc_1d ~ condition, data = summary_df, var.equal = TRUE)
print(t_res)

# Tidy output
 tidy_out <- tidy(t_res)
 write_csv(tidy_out, 't_test_results.csv')

cat('\nT-test results saved to t_test_results.csv and participant summaries to participant_test_summary.csv\n')