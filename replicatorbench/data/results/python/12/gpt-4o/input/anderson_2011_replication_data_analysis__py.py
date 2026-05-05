# Python translation of the Stata do-file
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the dataset
data = pd.read_stata('/app/data/analysis_data.dta')
data['vill_id'] = data['vill_id'].astype(int)
data = data.dropna(subset=['land_owned'])
data['vill_id'] = data['vill_id'].astype(int)

# Define the regression formula
formula = 'raw_inc_per_acre ~ literate_hh + land_owned + locaste_land_v + stcode * caste'

# Run the regression with clustering by vill_id
model = smf.ols(formula, data=data).fit(cov_type='cluster', cov_kwds={'groups': data['vill_id']})

# Print the summary of the regression
print(model.summary())

# Exploratory analysis using net income per acre
formula_net = 'net_inc_per_acre ~ literate_hh + land_owned + locaste_land_v + stcode * caste'
model_net = smf.ols(formula_net, data=data).fit(cov_type='cluster', cov_kwds={'groups': data['vill_id']})
print(model_net.summary())

# Alternative analysis using only subset of cases in UP/B
data_subset = data[data['stcode'].isin([2, 15])]
model_subset = smf.ols(formula, data=data_subset).fit(cov_type='cluster', cov_kwds={'groups': data_subset['vill_id']})
print(model_subset.summary())
