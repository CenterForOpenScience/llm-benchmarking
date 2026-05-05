"""Replicate the main finding from Cohen et al. (2015):
ACT subsidies induce take-up of ACT.

This script translates the provided Stata do-file ("Cohen et al 2015 - Replication Analysis.do")
into Python.  The script:
1. Loads the replication dataset
2. Constructs the same variables used in the Stata code
3. Runs an OLS model with household-level clustered SEs replicating Panel A, Table 2
   (impact of any ACT subsidy on ACT take-up)
4. Saves the regression table to /app/data as a CSV for downstream comparison.

All file IO is restricted to /app/data in compliance with the execution guidelines.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
from pathlib import Path

# Set paths relative to script location for portability within container
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "ReplicationData_Cohen_AmEcoRev_2015_2lb5.dta"

# Results will still be written to the shared /app/data volume so they are easy to retrieve
OUTPUT_TABLE_PATH = Path("/app/data/replication_regression_results.csv")

# ----------------------------------------------------------------------------# -----------------------------------------------------------------------------

# 1. Load data
# -----------------------------------------------------------------------------
print(f"Loading dataset from {DATA_PATH.resolve()}")
df = pd.read_stata(DATA_PATH)
print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")

# -----------------------------------------------------------------------------
# 2. Variable construction (mirrors Stata do-file)
# -----------------------------------------------------------------------------

# Outcome: took_ACT = drugs_taken_AL (already 0/1 or NaN)
df['took_ACT'] = df['drugs_taken_AL']

# Focal treatment: act_subsidy (1 if voucher given, 0 otherwise; set NaN for 98)
df['act_subsidy'] = np.where(df['maltest_chw_voucher_given'] == 98, np.nan,
                             (df['maltest_chw_voucher_given'] == 1).astype(float))

# Household id (unique row id in Stata code)
df = df.reset_index(drop=False).rename(columns={'index': 'hh_id'})

# Convenience function to test presence of item codes within a string

def contains_code(series: pd.Series, code: str):
    """Return boolean Series indicating whether the whitespace-separated code is in the string."""
    return series.fillna('').str.split().apply(lambda codes: code in codes).astype(float)

# Household assets (ses_hh_items is a whitespace-separated list of numeric codes)
assets_codes = {
    'refrigerator': '3',
    'mobile': '5',
}
for var, code in assets_codes.items():
    df[var] = contains_code(df['ses_hh_items'], code)

# Toilet type dummies
# 2 = VIP/Ventilated improved pit; 5 = Composting; 8 = Other

df['vip_toilet'] = (df['ses_toilet_type'] == 2).astype(float)
df['composting_toilet'] = (df['ses_toilet_type'] == 5).astype(float)
df['other_toilet'] = (df['ses_toilet_type'] == 8).astype(float)

# Wall material dummies: 1 Stone, 7 Cement

df['stone_wall'] = (df['ses_wall_material'] == 1).astype(float)
df['cement_wall'] = (df['ses_wall_material'] == 7).astype(float)

# Number of sheep

df['num_sheep'] = df['ses_no_sheep']

# Subsample: maltest_where==1 (tested at CHW) and wave != 0 (post-baseline)
mask_sub = (df['maltest_where'] == 1) & (df['wave'] != 0)
df_sub = df.loc[mask_sub].copy()
print(f"Subsample size: {df_sub.shape[0]:,}")

# Drop observations with missing key variables
key_vars = ['took_ACT', 'act_subsidy']
df_sub = df_sub.dropna(subset=key_vars)
print(f"After dropping missing outcome/treatment: {df_sub.shape[0]:,}")

# -----------------------------------------------------------------------------
# 3. Regression (replicating Stata svy: reg with household clustering)
# -----------------------------------------------------------------------------

authors_covariates = ['refrigerator', 'mobile', 'vip_toilet', 'composting_toilet',
                      'other_toilet', 'stone_wall', 'cement_wall', 'num_sheep']
formula = 'took_ACT ~ act_subsidy + ' + ' + '.join(authors_covariates) + ' + C(cu_code)'
print("Running OLS with clustered SE (hh_id)...")
model = smf.ols(formula=formula, data=df_sub)
res = model.fit(cov_type='cluster', cov_kwds={'groups': df_sub['hh_id']})
print(res.summary())

# -----------------------------------------------------------------------------
# 4. Save results table
# -----------------------------------------------------------------------------
summary_df = res.summary2().tables[1]
summary_df.to_csv(OUTPUT_TABLE_PATH)
print(f"Regression results saved to {OUTPUT_TABLE_PATH}")
