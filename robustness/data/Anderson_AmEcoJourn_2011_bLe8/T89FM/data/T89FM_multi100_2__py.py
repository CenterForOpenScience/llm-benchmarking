# Python translation of T89FM_multi100_2.R (Task 2)
# Ensures all IO uses /app/data with fallback to subfolder path.

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def load_household_data():
    candidates = [
        "/app/data/household.dta",
        "/app/data/AEJApp-2009-0289-data/household.dta"
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_stata(path, convert_categoricals=False)
    raise FileNotFoundError("household.dta not found under /app/data or /app/data/AEJApp-2009-0289-data")


df = load_household_data()

# Keep rows with domlow present
if "domlow" not in df.columns:
    raise ValueError("Expected column 'domlow' not found in dataset.")

df = df.copy()
df = df[~pd.isna(df["domlow"])].reset_index(drop=True)

# Ensure categorical encodings
for cat_col in ["caste", "borrow", "electric"]:
    if cat_col in df.columns:
        df[cat_col] = df[cat_col].astype("category")

df["domlow_cat"] = df["domlow"].astype("category")

# Log total income, guard against non-positive
if (df["totinc"] <= 0).any():
    df.loc[df["totinc"] <= 0, "totinc"] = np.nan

# Model per Task 2 restrictions: include bihar + literate + totland + cash + electric + caste + domlow
model_formula = (
    "np.log(totinc) ~ bihar + literate + totland + cash + electric + C(caste) + C(domlow_cat)"
)

m = smf.ols(model_formula, data=df).fit(cov_type='HC0')
# Extract domlow effect
param_keys = [k for k in m.params.index if k.startswith('C(domlow_cat)')]
summary = {
    "nobs": int(m.nobs),
    "rsq_adj": float(m.rsquared_adj),
}
if param_keys:
    key = param_keys[0]
    summary.update({
        "coef_domlow": float(m.params[key]),
        "se_domlow": float(m.bse[key]),
        "t_domlow": float(m.tvalues[key]),
        "p_domlow": float(m.pvalues[key])
    })

print("Task2 OLS with HC0 SE summary (restricted model):")
print(summary)
