import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Path to data inside the container (resolve relative to this script location)
import os, pathlib
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / 'Estimation Data - Pitts (126zz).csv'

# Load data
df = pd.read_csv(DATA_PATH)

# Keep only the variables of interest
model_vars = ['LeavingAgency', 'JobSat', 'Over40', 'NonMinority',
              'SatPay', 'SatAdvan', 'PerfCul', 'Empowerment',
              'RelSup', 'Relcow', 'Over40xSatAdvan', 'Agency']

df = df[model_vars].dropna()

# Define formula matching the original specification
formula = 'LeavingAgency ~ JobSat + Over40 + NonMinority + SatPay + SatAdvan + PerfCul + Empowerment + RelSup + Relcow + Over40xSatAdvan'

# Fit logistic regression with cluster robust standard errors (clustered by Agency)
model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
result = model.fit(cov_type='cluster', cov_kwds={'groups': df['Agency']})

print(result.summary())

# Save summary to file
with open('/app/data/replication_logistic_summary.txt', 'w') as f:
    f.write(result.summary().as_text())

# Print the focal coefficient for convenience
coef = result.params['JobSat']
se = result.bse['JobSat']
print(f"JobSat coefficient: {coef:.3f}, SE: {se:.3f}")
