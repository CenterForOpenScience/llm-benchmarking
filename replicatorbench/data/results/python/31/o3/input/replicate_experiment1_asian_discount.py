import pandas as pd
import statsmodels.api as sm

# Load data# Load data
df = pd.read_stata('replication_data/REPExperiment1DataR2.dta')

# Keep Asian# Keep Asian subjects only
df_asian = df[df['asian'] == 1].copy()

# Compute midpoint of log discount rate bounds
# If either bound is missing, midpoint will be NaN, and that row will be dropped automatically by statsmodels.
df_asian['mid_lograte'] = (df_asian['lndiscratelo'] + df_asian['lndiscratehi']) / 2

# Define predictors# Drop rows with any missing values in outcome or predictors
cols = ['mid_lograte', 'givenprimingquestionnaire', 'largestakes', 'longterm', 'largelong', 'id']
df_model = df_asian[cols].dropna()

# Define predictors
X = df_model[['givenprimingquestionnaire', 'largestakes', 'longterm', 'largelong']]
X = sm.add_constant(X)

y = df_model['mid_lograte']

# Groups for clustering
clusters = df_model['id']

# Fit OLS# Fit OLS with clustering by subject id
model = sm.OLS(y, X, missing='drop')
results = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})

print(results.summary())

# Save summary to text file
with open('/app/data/replication_asian_discount_results.txt', 'w') as f:
    f.write(results.summary().as_text())
