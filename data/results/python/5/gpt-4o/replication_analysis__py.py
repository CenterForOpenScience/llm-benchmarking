# Python translation of the Stata do-file for replication analysis

import pandas as pd
import numpy as np

# Load the dataset
file_path = '/app/data/replication_data.dta'
data = pd.read_stata(file_path)

# Generate systolic and diastolic blood pressure averages
# and determine hypertension status
systolic = (data['us07b1'] + data['us07c1']) / 2
diastolic = (data['us07b2'] + data['us07c2']) / 2

data['hypertension'] = np.where((systolic > 140) | (diastolic > 90), 1, 0)
data.loc[data[['us07b1', 'us07c1', 'us07b2', 'us07c2']].isnull().any(axis=1), 'hypertension'] = np.nan

# Determine under-diagnosis
# cd05 == 3 indicates not previously diagnosed by a doctor
data['under_diag'] = np.where((data['hypertension'] == 1) & (data['cd05'] == 3), 1, 0)
data.loc[data['hypertension'].isnull() | data['cd05'].isin([np.nan, 8]), 'under_diag'] = np.nan

# Generate years of education
# Initialize years of education
data['yrs_school'] = np.nan

# Replace years of education for those who never attended school
data.loc[data['dl04'] == 3, 'yrs_school'] = 0

# Process education levels and grades completed
# Elementary education
data.loc[(data['dl06'] == 2) & (data['dl07'] == 7), 'dl07'] = 6
data.loc[data['dl06'] == 2, 'yrs_school'] = data['dl07']

# Junior High education
data.loc[(data['dl06'] == 3) & (data['dl07'] == 7), 'dl07'] = 3
data['dl07'] = pd.to_numeric(data['dl07'], errors='coerce')
data.loc[data['dl06'] == 3, 'yrs_school'] = data['dl07'] + 6

# Senior High education
data.loc[(data['dl06'] == 5) & (data['dl07'] == 7), 'dl07'] = 3
data.loc[data['dl06'] == 5, 'yrs_school'] = data['dl07'] + 9

# University education
data.loc[(data['dl06'] == 61) & (data['dl07'] == 7), 'dl07'] = 4
data.loc[data['dl06'] == 61, 'yrs_school'] = data['dl07'] + 12

# Handle missing values for education
data.loc[data['dl06'].isin([14, 98, 99]), 'yrs_school'] = np.nan

# Save the processed data
data.to_csv('/app/data/processed_replication_data.csv', index=False)
