# Python translation of Fitzgerald 2018 replication script
# Loads compiled.dta, hhsize.dta, epa.dta, reshapes and merges, computes variables,
# logs variables, draws a 5-state random sample, and runs panel regressions

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

# All IO uses /app/data relative path in execution environment
DATA_DIR = '/app/data/original/7/python/replication_data'

# Load datasets
compiled_path = os.path.join(DATA_DIR, 'compiled.dta')
hhsize_path = os.path.join(DATA_DIR, 'hhsize.dta')
epa_path = os.path.join(DATA_DIR, 'epa.dta')

data = pd.read_stata(compiled_path)
hhsize = pd.read_stata(hhsize_path)
epa = pd.read_stata(epa_path)

# Reshape hhsize from wide to long
hh_cols = [c for c in hhsize.columns if c.startswith('hhsize')]
hh_long = hhsize.melt(id_vars=['State', 'state_id_no', 'statefip'], value_vars=hh_cols,
                      var_name='hhvar', value_name='hhsize')
# Extract year suffix from column names like 'hhsize07' -> 7
hh_long['year'] = hh_long['hhvar'].str.replace('hhsize', '').astype(int)
hh_long = hh_long[['State', 'hhsize', 'year']]

# Merge hhsize and epa into compiled data
data = data.merge(hh_long, how='left', on=['State', 'year'])
data = data.merge(epa, how='left', on=['State', 'year'])

# Compute employed population percentage and manufacturing percent of GDP
# In compiled.dta, emppop appears to be counts; original R computed: emppop/(pop*1000)*100
# Follow same formula (preserve original logic)
if 'emppop' in data.columns and 'pop' in data.columns:
    data['emppop_pct'] = data['emppop'] / (data['pop'] * 1000.0) * 100.0

if 'manuf' in data.columns and 'gdp' in data.columns:
    data['manu_gdp'] = data['manuf'] / data['gdp'] * 100.0

# Log-transform continuous variables (match R code)
log_vars = ['epa', 'wrkhrs', 'emppop_pct', 'laborprod', 'pop', 'manu_gdp', 'energy', 'hhsize', 'workpop']
for v in log_vars:
    if v in data.columns:
        # add small constant safeguard to avoid log(0)
        data[v] = data[v].replace([np.inf, -np.inf], np.nan)
        data[v] = data[v].astype(float)
        data[v] = np.log(data[v])

# Create group list
states = data['State'].unique()

# Set seed and draw 5 random states like the R script
np.random.seed(42)
random_states = pd.DataFrame({'State': np.random.choice(states, size=5, replace=False)})
random_states['unique_id'] = np.arange(1, len(random_states) + 1)

# Create sampledata by filtering data for selected states
sampledata = data.merge(random_states[['State']], on='State', how='inner')

# Order sampledata
sampledata = sampledata.sort_values(['State', 'year']).reset_index(drop=True)

# Define formula matching R: epa ~ wrkhrs + emppop_pct + laborprod + pop + manu_gdp + energy + hhsize + workpop + State + factor(year)
# In Python/statsmodels, include C(State) and C(year) as fixed effects
formula = 'epa ~ wrkhrs + emppop_pct + laborprod + pop + manu_gdp + energy + hhsize + workpop + C(State) + C(year)'

results = {}

# Model 1: on sampledata (full years included)
try:
    mod1 = smf.ols(formula=formula, data=sampledata).fit(cov_type='cluster', cov_kwds={'groups': sampledata['State']})
    results['model1'] = mod1
except Exception as e:
    results['model1_error'] = str(e)

# Model 2: original years (year < 14) -> corresponds to years 7-13 in R (2007-2013)
try:
    mod2 = smf.ols(formula=formula, data=sampledata[sampledata['year'] < 14]).fit(cov_type='cluster', cov_kwds={'groups': sampledata[sampledata['year'] < 14]['State']})
    results['model2'] = mod2
except Exception as e:
    results['model2_error'] = str(e)

# Model 4 & 5: full data equivalents
try:
    mod4 = smf.ols(formula=formula, data=data).fit(cov_type='cluster', cov_kwds={'groups': data['State']})
    results['model4'] = mod4
except Exception as e:
    results['model4_error'] = str(e)

try:
    mod5 = smf.ols(formula=formula, data=data[data['year'] < 14]).fit(cov_type='cluster', cov_kwds={'groups': data[data['year'] < 14]['State']})
    results['model5'] = mod5
except Exception as e:
    results['model5_error'] = str(e)

# Save model summaries to /app/data
out_path = '/app/data/original/7/python/replication_data/replication_model_summaries.txt'
with open(out_path, 'w') as f:
    for k, v in results.items():
        f.write(f'=== {k} ===\n')
        if isinstance(v, sm.regression.linear_model.RegressionResultsWrapper):
            f.write(v.summary().as_text())
            f.write('\n\n')
        else:
            f.write(str(v) + '\n\n')

print('Replication script completed. Summaries saved to', out_path)
