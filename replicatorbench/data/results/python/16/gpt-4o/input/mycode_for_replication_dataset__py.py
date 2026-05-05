# Python translation of the Stata do-file
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load the dataset
file_path = '/app/data/replicationDataset_Malik2020_with.year.csv'
data = pd.read_csv(file_path)

# Convert 'date' to datetime
data['date2'] = pd.to_datetime(data['date'], format='%m/%d/%Y')

# Take a 5% random sample
sample_data = data.sample(frac=0.05, random_state=42)

# Focal analysis: Multilevel mixed-effects linear regression model
model_transit = smf.mixedlm('CMRT_transit ~ date2 + lockdown', sample_data, groups=sample_data['city']).fit()
print(model_transit.summary())

# Additional analysis: Multilevel mixed-effects linear regression model
model_residential = smf.mixedlm('CMRT_residential ~ date2 + lockdown', sample_data, groups=sample_data['city']).fit()
print(model_residential.summary())
