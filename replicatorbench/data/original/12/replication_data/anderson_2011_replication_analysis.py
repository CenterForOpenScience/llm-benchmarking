import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load the data
data = pd.read_stata('/app/data/analysis_data.dta')

# Preregistered analysis (5% sample)
data_sample = data.sample(frac=0.05, random_state=234)

# Regression (Table 3) with all (both types of locaste)
# state code and caste interacted (because numbering differs by state)
model = ols('raw_inc_per_acre ~ literate_hh + land_owned + locaste_land_v + C(stcode) * C(caste)', data=data_sample).fit(cov_type='cluster', cov_kwds={'groups': data_sample['vill_id']})
print(model.summary())

# Final analysis
model_final = ols('raw_inc_per_acre ~ literate_hh + land_owned + locaste_land_v + C(stcode) * C(caste)', data=data).fit(cov_type='cluster', cov_kwds={'groups': data['vill_id']})
print(model_final.summary())

# Exploratory analysis using net income per acre
model_exploratory = ols('net_inc_per_acre ~ literate_hh + land_owned + locaste_land_v + C(stcode) * C(caste)', data=data).fit(cov_type='cluster', cov_kwds={'groups': data['vill_id']})
print(model_exploratory.summary())

# Alternative analysis using only subset of cases in UP/B (following original study)
data_subset = data[data['stcode'].isin([2, 15])]
model_alternative = ols('raw_inc_per_acre ~ literate_hh + land_owned + locaste_land_v + C(stcode) * C(caste)', data=data_subset).fit(cov_type='cluster', cov_kwds={'groups': data_subset['vill_id']})
print(model_alternative.summary())
