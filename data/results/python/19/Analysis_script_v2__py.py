# Python translation of the Stata do-file
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('/app/data/gelfand_replication_data.csv')

# Drop specified countries
drop_countries = ['Belgium', 'France', 'New Zealand', 'Norway', 'Pakistan', 'Venezuela']
df = df[~df['country'].isin(drop_countries)]

# Fill missing dates and impose zero cases
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df = df.set_index(['country', 'date']).asfreq('D', method='ffill').reset_index()
df['total_covid_per_million'] = df['total_covid_per_million'].fillna(method='ffill')
df['gdp'] = df['gdp'].fillna(method='ffill')

# Keep observations with more than 1 case per million
df = df[df['total_covid_per_million'] > 1]

# Log transform total cases
df['ltotalcases'] = np.log(df['total_covid_per_million'])

# Create time variable for each country
df['time'] = df.groupby('country').cumcount() + 1

# Drop observations beyond 30 days
df = df[df['time'] <= 30]

# Replace missing gini values
df['gini'] = df['gini_val'].combine_first(df['alternative_gini'])

# Estimate country-specific exponential growth regression
coeffs = []
for country, group in df.groupby('country'):
    if len(group) > 1:
        X = group[['time']]
        y = group['ltotalcases']
        model = sm.OLS(y, sm.add_constant(X)).fit()
        coeffs.append((country, model.params['time']))

coeffs_df = pd.DataFrame(coeffs, columns=['country', 'coeffs1'])

# Merge estimated coefficients with the main dataset
df = df.merge(coeffs_df, on='country')

# Create interaction term
df['eff_tight'] = df['efficiency'] * df['tightness']

# Regression analysis
import statsmodels.api as sm
X = df[['eff_tight', 'gdp', 'gini', 'median_age', 'efficiency', 'tightness']]
y = df['coeffs1']
model = sm.OLS(y, sm.add_constant(X)).fit()
print(model.summary())
