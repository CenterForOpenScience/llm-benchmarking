import pandas as pd
import statsmodels.api as sm

# Load the dataset
# Assuming the dataset is converted to CSV format for Python compatibility
#df = pd.read_csv('/app/data/COVID_replication.csv')

# For demonstration, creating a mock DataFrame
# Replace this with actual data loading
df = pd.read_stata('/app/data/COVID replication.dta')

# Select the dataset
dataset = 1  # Change this value to select different datasets

if dataset == 1:
    df['total_cases'] = df['COVID_12_31_04_03']
elif dataset == 2:
    df['total_cases'] = df['COVID_04_04_08_11']
elif dataset == 3:
    df['total_cases'] = df['COVID_12_31_08_11']
elif dataset == 4:
    df['total_cases'] = df['COVID_04_04_08_11'].fillna(df['COVID_12_31_08_11'])
else:
    df['total_cases'] = df['COVID_12_31_04_03'].fillna(df['COVID_04_04_08_11']).fillna(df['COVID_12_31_08_11'])

# Create cases per million
df['cases_per_million'] = df['total_cases'] / df['popData2019'] * 1000000

# Prepare independent variables
df['democracy'] = df['Democracy'] / 10
df['temperature'] = df['Annual_temp']
df['openness'] = df['trade_2016']

# Define the regression model
X = df[['democracy', 'temperature', 'openness']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = df['cases_per_million']

model = sm.OLS(y, X).fit()
print(model.summary())

# Note: Replace the mock data loading and processing with actual data operations
# Ensure the data file is available at '/app/data/COVID_replication.csv' or similar path
