import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import iqr

# Load datasets
county_variables = pd.read_csv('/app/data/county_variables.csv')
transportation = pd.read_csv('/app/data/transportation.csv')

# Sample 5% of county_variables
county_variables_sample = county_variables.sample(frac=0.05, random_state=2982)

# Process transportation data
transportation['date'] = pd.to_datetime(transportation['date'])
transportation['prop_home'] = transportation['pop_home'] / (transportation['pop_home'] + transportation['pop_not_home'])

# Define time periods
conditions = [
    (transportation['date'] >= '2020-02-16') & (transportation['date'] <= '2020-02-29'),
    (transportation['date'] >= '2020-03-19') & (transportation['date'] <= '2020-04-01'),
    (transportation['date'] >= '2020-08-16') & (transportation['date'] <= '2020-08-29')
]
choices = ['AAA Reference', 'March', 'August']
transportation['time_period'] = np.select(conditions, choices, default=np.nan)

# Filter and group data
flat_data = transportation.dropna(subset=['time_period', 'pop_home'])
flat_data = flat_data.groupby(['time_period', 'fips', 'state']).agg({'prop_home': 'mean'}).reset_index()
flat_data = flat_data.sort_values(by=['state', 'fips', 'time_period'])
flat_data['prop_home_change'] = flat_data.groupby(['fips', 'state'])['prop_home'].transform(lambda x: 100 * (x / x.iloc[0] - 1))
flat_data = flat_data[flat_data['time_period'] != 'AAA Reference']

# Pivot data
flat_data = flat_data.pivot(index=['fips', 'state'], columns='time_period', values=['prop_home', 'prop_home_change']).reset_index()

# Merge with county variables
flat_data = flat_data.merge(county_variables_sample, on='fips', how='right')

# Calculate IQR of Trump support
trump_iqr = iqr(county_variables_sample['trump_share'].dropna())

# Prepare data for regression
flat_data['state'] = flat_data['state'].astype('category')
flat_data = flat_data[['prop_home_change_March', 'prop_home_change_August', 'income_per_capita', 'trump_share',
                       'male_percent', 'percent_black', 'percent_hispanic', 'percent_college', 'percent_retail',
                       'percent_transportation', 'percent_hes', 'prop_rural', 'ten_nineteen', 'twenty_twentynine',
                       'thirty_thirtynine', 'forty_fortynine', 'fifty_fiftynine', 'sixty_sixtynine', 'seventy_seventynine',
                       'over_eighty', 'state', 'fips']]
flat_data = flat_data.dropna()

# Scale percentages
percent_cols = [col for col in flat_data.columns if 'percent_' in col]
flat_data[percent_cols] = flat_data[percent_cols] * 100
flat_data['male_percent'] *= 100
flat_data['percent_college'] /= 100
flat_data['income_per_capita'] /= 1000

# Define regression function
def run_regression(dep_var, data):
    X = data.drop(columns=['fips', 'state', dep_var])
    y = data[dep_var]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

# Run regressions
model_march = run_regression('prop_home_change_March', flat_data)
model_august = run_regression('prop_home_change_August', flat_data)

# Print summaries
print(model_march.summary())
print(model_august.summary())

# Save regression tables
with open('/app/data/regression_table_march.txt', 'w') as f:
    f.write(model_march.summary().as_text())

with open('/app/data/regression_table_august.txt', 'w') as f:
    f.write(model_august.summary().as_text())

# Note: Spatial autocorrelation analysis is not included in this translation.