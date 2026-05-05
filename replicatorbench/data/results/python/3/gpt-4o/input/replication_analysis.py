# Python script to replicate the analysis from the Stata do-file

import pandas as pd
import statsmodels.api as sm

# Load the dataset
file_path = '/app/data/COVID replication.rds'
import pyreadr
result = pyreadr.read_r(file_path)
data = result[None]  # extract the pandas dataframe

# Select the dataset for analysis
# Assuming dataset = 1 for the original study timeframe

data['total_cases'] = data['COVID.12.31_04.03']

# Calculate cases per million

data['cases_per_million'] = data['total_cases'] / data['popData2019'] * 1000000

# Adjust democracy index

data['democracy'] = data['Democracy index (EIU)'] / 10

# Rename columns for analysis

data.rename(columns={'Annual_temp': 'temperature', 'trade.2016': 'openness'}, inplace=True)

# Define independent variables
X = data[['democracy', 'temperature', 'openness']]
X = sm.add_constant(X)  # Add constant term for intercept

# Define dependent variable
y = data['cases_per_million']

# Run regression model
model = sm.OLS(y, X).fit()

# Print regression results
print(model.summary())
