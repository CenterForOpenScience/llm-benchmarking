import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

DATA_PATH = 'replication_data/SCORE_ALL_DATA.csv'
ARTIFACTS_DIR = 'artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print('Loading data...')
df = pd.read_csv(DATA_PATH)
print('Data shape:', df.shape)

# Prepare variables: outcome S1_likely; rating distribution in TYPE; self-concept clarity in 'SCL compliant code '
# Recode TYPE into two factors: 'Bimodal' vs 'Unimodal' and 'Cl' (clear) vs 'Co' (conflicted)

def map_TYPE(t):
    # Expect codes like 'ClB', 'ClU', 'CoB', 'CoU'
    if pd.isnull(t):
        return np.nan
    t = str(t).strip()
    # clarity: Cl = Clear, Co = Conflicted
    clarity = 'Clear' if t.startswith('Cl') else ('Conflicted' if t.startswith('Co') else t[:2])
    # distribution: B = Bimodal, U = Unimodal
    dist = 'Bimodal' if t.endswith('B') else ('Unimodal' if t.endswith('U') else t[-1])
    return clarity + '_' + dist

print('Recoding TYPE...')
df['type_recoded'] = df['TYPE'].apply(map_TYPE)
print(df['type_recoded'].value_counts(dropna=False))

# Use the provided 'SCL compliant code ' as clarity measure; if it's coded 1/2, we'll treat 1 as Low clarity (or vice versa)
# Inspect unique values
print('SCL unique values:', df['SCL compliant code '].unique())

# For replication of interaction, create binary indicators: is_bimodal and is_clear (from type_recoded)
df['is_bimodal'] = df['type_recoded'].str.contains('Bimodal')
df['is_clear'] = df['type_recoded'].str.contains('Clear')

# Outcome
outcome = 'S1_likely'

# Drop missing
model_df = df[[outcome, 'is_bimodal', 'is_clear', 'SCL compliant code ']].dropna()
print('Model df shape after dropping missings:', model_df.shape)

# Create interaction terms
model_df['bimodal_x_clarity'] = model_df['is_bimodal'].astype(int) * model_df['is_clear'].astype(int)

# Fit linear model: outcome ~ is_bimodal * is_clear
model = smf.ols(formula=f"{outcome} ~ is_bimodal * is_clear", data=model_df).fit()
print(model.summary())

# Save coefficients to artifacts
coef_df = model.summary2().tables[1]
coef_df.to_csv(os.path.join(ARTIFACTS_DIR, 'regression_coefficients.csv'))
print('Saved regression coefficients to artifacts/regression_coefficients.csv')

# Also save a simple results json
res = {
    'params': model.params.to_dict(),
    'pvalues': model.pvalues.to_dict(),
    'rsquared': float(model.rsquared)
}
import json
with open(os.path.join(ARTIFACTS_DIR, 'execution_result.json'), 'w') as f:
    json.dump(res, f)
print('Saved execution_result.json')
