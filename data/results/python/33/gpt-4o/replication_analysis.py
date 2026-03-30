import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load the dataset
file_path = '/app/data/Ma-Kellams Replication Study Data.sav'
data = pd.read_spss(file_path)

# Compute necessary variables
# Recode PAS2 and PAS5
pas2_positive = data['PAS2'].replace({-3: 3, -2: 2, -1: 1, 0: 0, 1: -1, 2: -2, 3: -3})
pas5_positive = data['PAS5'].replace({-3: 3, -2: 2, -1: 1, 0: 0, 1: -1, 2: -2, 3: -3})

# Compute PASAverage
# Ensure PAS1, PAS3, and PAS4 are numeric
pas1_numeric = pd.to_numeric(data['PAS1'], errors='coerce')
pas3_numeric = pd.to_numeric(data['PAS3'], errors='coerce')
pas4_numeric = pd.to_numeric(data['PAS4'], errors='coerce')

# Compute PASAverage
# Convert PAS2 and PAS5 to numeric as well
pas2_positive = pd.to_numeric(pas2_positive, errors='coerce')
pas5_positive = pd.to_numeric(pas5_positive, errors='coerce')

# Compute PASAverage
pas_average = pas1_numeric.add(pas3_numeric, fill_value=0).add(pas4_numeric, fill_value=0).add(pas2_positive, fill_value=0).add(pas5_positive, fill_value=0)

# Add computed columns to the dataframe
data['PASAverage'] = pas_average

# Define the model formula
formula = 'PASAverage ~ C(Culture) * C(WritingCondition)'

# Fit the model
model = ols(formula, data=data).fit()

# Print the ANOVA table
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# Save the results
output_path = '/app/data/replication_results.csv'
anova_table.to_csv(output_path)
