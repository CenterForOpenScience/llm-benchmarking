import pandas as pd
import statsmodels.api as sm

# Load processed data
score_all = pd.read_csv('/app/data/k17_processed_data.csv')

# Define the models
X1 = score_all[['age', 'gender', 'children', 'work_hours', 'work_days']]
X1 = sm.add_constant(X1)
Model_1 = sm.OLS(score_all['T3_panas_joviality'], X1).fit()

X2 = score_all[['age', 'gender', 'children', 'work_hours', 'work_days', 'T1_panas_joviality']]
X2 = sm.add_constant(X2)
Model_2 = sm.OLS(score_all['T3_panas_joviality'], X2).fit()

X3 = score_all[['age', 'gender', 'children', 'work_hours', 'work_days', 'T1_panas_joviality', 'req_control', 'req_mastery', 'req_relax', 'req_detach', 'hassles']]
X3 = sm.add_constant(X3)
Model_3 = sm.OLS(score_all['T3_panas_joviality'], X3).fit()

# Print summaries
print(Model_1.summary())
print(Model_2.summary())
print(Model_3.summary())

# Calculate R-squared differences
r_squared_diff_1_2 = Model_2.rsquared - Model_1.rsquared
r_squared_diff_2_3 = Model_3.rsquared - Model_2.rsquared

print(f'R-squared difference between Model 1 and Model 2: {r_squared_diff_1_2}')
print(f'R-squared difference between Model 2 and Model 3: {r_squared_diff_2_3}')

# ANOVA comparisons
anova_results_1_2 = sm.stats.anova_lm(Model_1, Model_2)
anova_results_2_3 = sm.stats.anova_lm(Model_2, Model_3)

print(anova_results_1_2)
print(anova_results_2_3)
