# Python translation of kavanagh_analysis.R

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

# Load datasets
county_variables = pd.read_csv('/app/data/county_variables.csv')
transportation = pd.read_csv('/app/data/transportation.csv')

# Sample 5% of the data
county_variables_sample = county_variables.sample(frac=0.05, random_state=2982)

# Process transportation data
transportation['date'] = pd.to_datetime(transportation['date'])
transportation['prop_home'] = transportation['pop_home'] / (transportation['pop_home'] + transportation['pop_not_home'])

# Define time periods
transportation['time_period'] = np.where(transportation['date'].between('2020-02-16', '2020-02-29'), 'AAA Reference',
                                         np.where(transportation['date'].between('2020-03-19', '2020-04-01'), 'March',
                                                  np.where(transportation['date'].between('2020-08-16', '2020-08-29'), 'August', np.nan)))

# Filter and group data
flat_data = transportation.dropna(subset=['time_period', 'pop_home'])
flat_data = flat_data.groupby(['time_period', 'fips', 'state']).agg({'prop_home': 'mean'}).reset_index()
flat_data = flat_data.sort_values(by=['state', 'fips', 'time_period'])

# Calculate prop_home_change
flat_data['prop_home_change'] = flat_data.groupby(['fips', 'state'])['prop_home'].transform(lambda x: 100 * (x / x.iloc[0] - 1))
flat_data = flat_data[flat_data['time_period'] != 'AAA Reference']

# Reshape data
flat_data = flat_data.pivot(index=['fips', 'state'], columns='time_period', values=['prop_home', 'prop_home_change']).reset_index()

# Merge with county variables
flat_data = flat_data.merge(county_variables_sample, on='fips', how='right')

# Calculate IQR of Trump support
trump_share = county_variables['trump_share'].dropna()
trump_iqr = np.subtract(*np.percentile(trump_share, [75, 25]))

# Variable construction
flat_data['state'] = flat_data['state'].astype('category')
flat_data = flat_data[['prop_home_change_March', 'prop_home_change_August', 'income_per_capita', 'trump_share',
                       'percent_male', 'percent_black', 'percent_hispanic', 'percent_college', 'percent_retail',
                       'percent_transportation', 'percent_hes', 'percent_rural', 'percent_10_19', 'percent_20_29',
                       'percent_30_39', 'percent_40_49', 'percent_50_59', 'percent_60_69', 'percent_70_79',
                       'percent_80_over', 'state', 'fips']]
flat_data = flat_data.apply(lambda x: x * 100 if x.name.startswith('percent_') else x)
flat_data['male_percent'] *= 100
flat_data['percent_college'] /= 100
flat_data['income_per_capita'] /= 1000

# Save processed data
flat_data.to_csv('/app/data/processed_flat_data.csv', index=False)

# Regression analysis
import statsmodels.formula.api as smf
from libpysal.weights import KNN
from spreg import GM_Lag

# Define regression formula
formula = 'prop_home_change_March ~ income_per_capita + trump_share + percent_male + percent_black + percent_hispanic + percent_college + percent_retail + percent_transportation + percent_hes + percent_rural + percent_10_19 + percent_20_29 + percent_30_39 + percent_40_49 + percent_50_59 + percent_60_69 + percent_70_79 + percent_80_over + C(state)'

# Fixed effect regression for March
model_march = smf.ols(formula, data=flat_data).fit()
print(model_march.summary())

# Fixed effect regression for August
formula_august = formula.replace('prop_home_change_March', 'prop_home_change_August')
model_august = smf.ols(formula_august, data=flat_data).fit()
print(model_august.summary())

# Spatial autocorrelation analysis
# Load county coordinates
counties = pd.read_csv('/app/data/county_coordinates.csv')
flat_data = flat_data.merge(counties, on='fips')

# Create spatial weights
coords = flat_data[['lon', 'lat']].values
knn = KNN.from_array(coords, k=5)

# Spatial lag model for March
model_march_spatial = GM_Lag.from_formula(formula, data=flat_data, w=knn)
print(model_march_spatial.summary)

# Spatial lag model for August
model_august_spatial = GM_Lag.from_formula(formula_august, data=flat_data, w=knn)
print(model_august_spatial.summary)

# Save regression results
with open('/app/data/regression_results.txt', 'w') as f:
    f.write(model_march.summary().as_text())
    f.write('\n\n')
    f.write(model_august.summary().as_text())
    f.write('\n\n')
    f.write(str(model_march_spatial.summary))
    f.write('\n\n')
    f.write(str(model_august_spatial.summary))

