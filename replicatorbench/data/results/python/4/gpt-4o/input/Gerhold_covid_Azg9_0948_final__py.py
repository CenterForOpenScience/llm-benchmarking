# Python translation of Gerhold_covid_Azg9_0948_final.R

import pandas as pd
from scipy import stats
import numpy as np

# Load the dataset
url = '/app/data/data_gerhold.csv'
data = pd.read_csv(url)

# Remove 'missing' data (i.e., gender = 3)
data = data[data['gender'] != 3]

# Create female and male subgroups
female_group = data[data['female'] == 1]
male_group = data[data['female'] == 0]

# T-TEST SECTION
# Test for homoscedasticity and perform t-test on mh_anxiety_1
x = female_group['mh_anxiety_1']
y = male_group['mh_anxiety_1']
var_test_result = stats.levene(x, y)
if var_test_result.pvalue > 0.05:
    focal_claim = stats.ttest_ind(x, y, equal_var=True)
else:
    focal_claim = stats.ttest_ind(x, y, equal_var=False)

# Test for homoscedasticity and perform t-test on mh_anxiety_3
x = female_group['mh_anxiety_3']
y = male_group['mh_anxiety_3']
var_test_result = stats.levene(x, y)
if var_test_result.pvalue > 0.05:
    exploratory = stats.ttest_ind(x, y, equal_var=True)
else:
    exploratory = stats.ttest_ind(x, y, equal_var=False)

# Calculate Cohen's d for focal claim replication
def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

cohen_d_value = cohen_d(female_group['mh_anxiety_1'], male_group['mh_anxiety_1'])

# Output results
print('Focal Claim T-test:', focal_claim)
print('Exploratory T-test:', exploratory)
print('Cohen\'s d:', cohen_d_value)
