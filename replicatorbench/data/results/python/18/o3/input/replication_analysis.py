import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

# Load data
import os
# Attempt primary path inside mounted data directory
PRIMARY_PATH = '/app/data/original/18/0203_o3/replication_data/replication_data_mkk9.csv'
FALLBACK_PATH = os.path.join(os.path.dirname(__file__), 'replication_data_mkk9.csv')
DATA_PATH = PRIMARY_PATH if os.path.exists(PRIMARY_PATH) else FALLBACK_PATH

df = pd.read_csv(DATA_PATH)

# Clean and prepare
# Drop missing median_age
df = df.dropna(subset=['median_age'])

# Ensure numeric types
for col in ['median_age', 'entrepreneurship', 'cy_cell']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Weight variable
weights = df['cy_cell']

# Fit weighted OLS with year fixed effects, clustered standard errors by country
# Using Patsy formula interface
formula = 'entrepreneurship ~ median_age + C(year)'

model = smf.wls(formula=formula, data=df, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': df['country']})

print(model.summary())

# Save coefficient table to CSV
coef_table = pd.DataFrame({'coef': model.params, 'std_err': model.bse, 't': model.tvalues, 'p_value': model.pvalues})
output_path = '/app/data/replication_country_year_median_age_regression.csv'
coef_table.to_csv(output_path, index=True)

print(f'Coefficient table saved to {output_path}')
