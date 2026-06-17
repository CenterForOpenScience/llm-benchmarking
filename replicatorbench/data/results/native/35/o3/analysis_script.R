#!/usr/bin/env Rscript

# Tremoliere Generalizability Replication Analysis Script
# This script replicates the focal Salience x Scenario interaction using
# mixed-effects logistic regression and a 2×2 mixed ANOVA.
# It saves the key statistics to /app/data/tremoliere_replication_results.csv

# --------------------------
# 1. Load libraries
# --------------------------
require(tidyverse)      # includes dplyr, tidyr, ggplot2, etc.
require(lme4)           # for glmer
require(lmerTest)       # p values for lme4
require(sjPlot)         # descriptive tables / odds ratios
require(rstatix)        # anova_test convenience

# --------------------------
# 2. Read data
# --------------------------
input_file <- "Tremoliere_generalizability_score.csv"
if (!file.exists(input_file)) {
  stop(paste("Cannot find", input_file, "in working directory", getwd()))
}

df <- read.csv(input_file, stringsAsFactors = FALSE)

# --------------------------
# 3. Data cleaning and reshaping
# --------------------------
# Filter to rows that provided an appropriate response to manipulation check
# (satisfactory_manipulation_response1 == 1)

df_long <- df %>%
  # Keep needed columns
  select(ResponseId,
         death_salience, pain_salience,
         satisfactory_manipulation_response1,
         age, gender, gender_3_TEXT,
         politic_1, politic_2, politic_3,
         race, race_7_TEXT, race_8_TEXT,
         income, education, open, cond, salience, participant_uid,
         moral_accept, moral_accept1) %>%
  # Reshape to long format, two rows per participant
  pivot_longer(cols = c(moral_accept, moral_accept1),
               names_to = "variable",
               values_to = "moral_acceptability") %>%
  # Keep participants who passed manipulation check
  filter(satisfactory_manipulation_response1 == 1)

# Recode moral acceptability into binary 0/1 (0 = not utilitarian, 1 = utilitarian)
df_long <- df_long %>%
  mutate(moral_acceptability_01 = case_when(
    moral_acceptability == "1" ~ 0,
    moral_acceptability == "2" ~ 1,
    TRUE ~ NA_real_))

# Create variable for scenario type

df_long <- df_long %>%
  mutate(moral_scenario = case_when(
    variable == "moral_accept" ~ "impartial_beneficience",
    variable == "moral_accept1" ~ "partial_beneficience",
    TRUE ~ NA_character_))

# Convert to factors for analysis

df_long <- df_long %>%
  mutate(salience_fact = factor(salience),
         moral_scenario_fact = factor(moral_scenario))

# --------------------------
# 4. Mixed-effects logistic regression
# --------------------------

fit_glmer <- glmer(moral_acceptability_01 ~ salience_fact * moral_scenario_fact + (1|participant_uid),
                   data = df_long,
                   family = binomial,
                   control = glmerControl(optimizer = "bobyqa"))

# Summary statistics
model_summary <- summary(fit_glmer)
print(model_summary)

# --------------------------
# 5. ANOVA replication (2×2 mixed design)
# --------------------------
fit_anova <- anova_test(data = df_long,
                       dv = moral_acceptability_01,
                       wid = participant_uid,
                       between = salience_fact,
                       within = moral_scenario_fact)

anova_table <- get_anova_table(fit_anova)
print(anova_table)

# --------------------------
# 6. Extract focal interaction statistics
# --------------------------

# Interaction term name (depends on R's contrasts). We'll extract the last row of fixed effects that contains ':'
coef_df <- as.data.frame(coef(summary(fit_glmer)))
coef_df$Term <- rownames(coef(summary(fit_glmer)))
interaction_row <- coef_df %>% filter(grepl(":", Term))

if (nrow(interaction_row) == 0) {
  warning("No interaction term found – check model specification")
}

# For ANOVA, interaction row is where effect == "salience_fact:moral_scenario_fact"
interaction_anova <- anova_table %>% filter(Effect == "salience_fact:moral_scenario_fact")

# --------------------------
# 7. Save results to CSV
# --------------------------

results_path <- "/app/data/tremoliere_replication_results.csv"

# Prepare a concise results data frame
results_df <- tibble(
  analysis = c("glmer_interaction", "anova_interaction"),
  estimate_or_F = c(interaction_row$Estimate[1], interaction_anova$F[1]),
  std_error_or_df = c(interaction_row$`Std. Error`[1], interaction_anova$DFn[1]),
  test_statistic = c(interaction_row$`z value`[1], interaction_anova$F[1]),
  p_value = c(interaction_row$`Pr(>|z|)`[1], interaction_anova$p[1])
)

write.csv(results_df, results_path, row.names = FALSE)
cat("\nResults saved to", results_path, "\n")

# Also produce odds ratio table for completeness (printed but not saved)
tryCatch({
  sjPlot::tab_model(fit_glmer)
}, error = function(e) {
  message("Unable to generate sjPlot table: ", e$message)
})

cat("Analysis complete.\n")
