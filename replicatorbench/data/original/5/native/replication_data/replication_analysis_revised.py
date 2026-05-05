import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the dataset
# Assuming the dataset is in CSV format and located in /app/data
df = pd.read_csv('/app/data/replication_dataset.csv')

# Time preferences
# df['time_A'] = np.nan
df.loc[(df['si21a'] == 1) | (df['si21e'] == 3) & (df['si21b'] == 1) & (df['si21c'] == 1), 'time_A'] = 4
df.loc[(df['si21a'] == 1) | (df['si21e'] == 3) & (df['si21b'] == 1) & (df['si21c'] == 2), 'time_A'] = 3
df.loc[(df['si21a'] == 1) | (df['si21e'] == 3) & (df['si21b'] == 2) & (df['si21d'] == 1), 'time_A'] = 2
df.loc[(df['si21a'] == 1) | (df['si21e'] == 3) & (df['si21b'] == 2) & (df['si21d'] == 2), 'time_A'] = 1
df.loc[df[['si21a', 'si21e', 'si21b', 'si21c', 'si21d']].isin([9]).any(axis=1), 'time_A'] = np.nan

df['time_B'] = np.nan
df.loc[(df['si22a'] == 1) | (df['si22e'] == 3) & (df['si22b'] == 1) & (df['si22c'] == 1), 'time_B'] = 4
df.loc[(df['si22a'] == 1) | (df['si22e'] == 3) & (df['si22b'] == 1) & (df['si22c'] == 2), 'time_B'] = 3
df.loc[(df['si22a'] == 1) | (df['si22e'] == 3) & (df['si22b'] == 2) & (df['si22d'] == 1), 'time_B'] = 2
df.loc[(df['si22a'] == 1) | (df['si22e'] == 3) & (df['si22b'] == 2) & (df['si22d'] == 2), 'time_B'] = 1
df.loc[df[['si22a', 'si22e', 'si22b', 'si22c', 'si22d']].isin([9]).any(axis=1), 'time_B'] = np.nan

df['time_preference'] = (df['time_A'] + df['time_B']) / 2
df['time_preference'] = df['time_preference'].round()

# Probit regression model
covariates = ['log_PCE', 'risk_preference', 'time_preference', 'distance', 'female', 'age', 'agesqrt']
X = df[covariates]
X = sm.add_constant(X)
y = df['under_diag']
probit_model = sm.Probit(y, X, missing='drop')
probit_results = probit_model.fit()
print(probit_results.summary())
