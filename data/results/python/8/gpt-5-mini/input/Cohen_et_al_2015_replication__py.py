import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Paths
candidates = [
    "/app/data/original/8/python/replication_data/ReplicationData_Cohen_AmEcoRev_2015_2lb5.dta",
    "/workspace/replication_data/ReplicationData_Cohen_AmEcoRev_2015_2lb5.dta",
    "./replication_data/ReplicationData_Cohen_AmEcoRev_2015_2lb5.dta",
    "./data/original/8/python/replication_data/ReplicationData_Cohen_AmEcoRev_2015_2lb5.dta"
]
DTA_PATH = None
for p in candidates:
    if os.path.exists(p):
        DTA_PATH = p
        break
if DTA_PATH is None:
    raise FileNotFoundError("Could not locate ReplicationData_Cohen_AmEcoRev_2015_2lb5.dta in expected locations: %s" % candidates)

DATA_DIR = os.path.dirname(DTA_PATH)
OUT_SUMMARY = os.path.join(DATA_DIR, "Cohen_et_al_2015_regression_summary.txt")
OUT_COEFFS = os.path.join(DATA_DIR, "Cohen_et_al_2015_regression_coeffs.csv")

print("Loading data from:", DTA_PATH)
df = pd.read_stata(DTA_PATH)
print("Initial rows:", len(df))

# Outcome
if 'drugs_taken_AL' in df.columns:
    df['took_ACT'] = pd.to_numeric(df['drugs_taken_AL'], errors='coerce')
else:
    raise KeyError('drugs_taken_AL not found in dataset')

# Treatment
df['act_subsidy'] = (df['maltest_chw_voucher_given'] == 1).astype(float)
# set to missing where code == 98
df.loc[df['maltest_chw_voucher_given'] == 98, 'act_subsidy'] = np.nan

# Household id: assume each row is a (mostly) unique household as in the .do
df = df.reset_index(drop=True)
df['hh_id'] = np.arange(1, len(df) + 1)

# Strata
df['strata'] = df['cu_code']

# Assets: ses_hh_items contains codes as string; use substring search similar to strpos
if 'ses_hh_items' in df.columns:
    df['ses_hh_items'] = df['ses_hh_items'].fillna("")
    df['refrigerator'] = df['ses_hh_items'].astype(str).str.contains('3').astype(float)
    df['mobile'] = df['ses_hh_items'].astype(str).str.contains('5').astype(float)
else:
    df['refrigerator'] = np.nan
    df['mobile'] = np.nan

# Toilet types
if 'ses_toilet_type' in df.columns:
    df['vip_toilet'] = (df['ses_toilet_type'] == 2).astype(float)
    df['composting_toilet'] = (df['ses_toilet_type'] == 5).astype(float)
    df['other_toilet'] = (df['ses_toilet_type'] == 8).astype(float)
else:
    df['vip_toilet'] = np.nan
    df['composting_toilet'] = np.nan
    df['other_toilet'] = np.nan

# Wall material
if 'ses_wall_material' in df.columns:
    df['stone_wall'] = (df['ses_wall_material'] == 1).astype(float)
    df['cement_wall'] = (df['ses_wall_material'] == 7).astype(float)
else:
    df['stone_wall'] = np.nan
    df['cement_wall'] = np.nan

# Number of sheep
if 'ses_no_sheep' in df.columns:
    df['num_sheep'] = pd.to_numeric(df['ses_no_sheep'], errors='coerce')
else:
    df['num_sheep'] = np.nan

# Sampling weight
if 'weight' in df.columns:
    df['sampling_weight'] = pd.to_numeric(df['weight'], errors='coerce')
else:
    df['sampling_weight'] = 1.0

# Subset: maltest_where == 1 & wave != 0
df_sub = df[(df['maltest_where'] == 1) & (df['wave'] != 0)].copy()
print("Rows after subsetting (maltest_where==1 & wave!=0):", len(df_sub))

# Prepare regression data: outcome, treatment, covariates
covariates = ['refrigerator', 'mobile', 'vip_toilet', 'composting_toilet', 'other_toilet', 'stone_wall', 'cement_wall', 'num_sheep']
reg_vars = ['took_ACT', 'act_subsidy', 'hh_id', 'strata', 'sampling_weight'] + covariates
# Keep only these columns
for c in reg_vars:
    if c not in df_sub.columns:
        df_sub[c] = np.nan

# Drop rows with missing outcome or treatment
df_reg = df_sub.dropna(subset=['took_ACT', 'act_subsidy'])
print("Rows after dropping missing outcome/treatment:", len(df_reg))

# Create design matrix
# Strata dummies
strata_dummies = pd.get_dummies(df_reg['strata'].astype('category'), prefix='strata', drop_first=True)
X = pd.concat([df_reg[['act_subsidy'] + covariates].reset_index(drop=True), strata_dummies.reset_index(drop=True)], axis=1)
X = sm.add_constant(X)
y = df_reg['took_ACT'].astype(float)
weights = df_reg['sampling_weight'].astype(float).fillna(1.0)

# Ensure finite columns# Ensure finite columns
X = X.replace([np.inf, -np.inf], np.nan)
# Coerce all X columns to numeric where possible
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
# Ensure constant is present and numeric
if 'const' not in X.columns:
    X = sm.add_constant(X, has_constant='add')
else:
    X['const'] = pd.to_numeric(X['const'], errors='coerce')

# Drop rows with missing covariates (listwise deletion)
df_model = pd.concat([y.reset_index(drop=True), X.reset_index(drop=True), df_reg[['hh_id']].reset_index(drop=True), weights.reset_index(drop=True)], axis=1)
# Coerce outcome to numeric
df_model['took_ACT'] = pd.to_numeric(df_model['took_ACT'], errors='coerce')
# Coerce hh_id to numeric
df_model['hh_id'] = pd.to_numeric(df_model['hh_id'], errors='coerce')
# Coerce sampling weight to numeric
if 'sampling_weight' in df_model.columns:
    df_model['sampling_weight'] = pd.to_numeric(df_model['sampling_weight'], errors='coerce')

# Now drop rows with any missing values (listwise deletion)
df_model = df_model.dropna()
print("Rows in model after listwise deletion:", len(df_model))

# DEBUG: inspect X and y dtypes and samples before fitting
temp_X = df_model.drop(columns=['took_ACT', 'hh_id'])
print('\nDEBUG: temp_X dtypes:')
print(temp_X.dtypes)
print('\nDEBUG: temp_X head:')
print(temp_X.head())
print('\nDEBUG: took_ACT head:')
print(df_model['took_ACT'].head())

# proceed to define y and X
y = df_model['took_ACT'].astype(float)
X = df_model.drop(columns=['took_ACT', 'hh_id'])
# Remove sampling weight from regressors if present
if 'sampling_weight' in X.columns:
    X = X.drop(columns=['sampling_weight'])
# Convert boolean columns to integers
bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
for col in bool_cols:
    X[col] = X[col].astype(int)
# Coerce all X to float
X = X.apply(pd.to_numeric, errors='coerce').astype(float)
# Ensure no object dtypes remain
if X.select_dtypes(include=['object']).shape[1] > 0:
    raise ValueError('Non-numeric columns remain in X after coercion: %s' % X.select_dtypes(include=[object]).columns.tolist())

clusters = pd.to_numeric(df_model['hh_id'], errors='coerce').astype(int)
weights = pd.to_numeric(df_model['sampling_weight'], errors='coerce') if 'sampling_weight' in df_model.columns else None

# Fit WLS (with weights) or OLS
results = None
try:
    if weights is not None and not weights.isnull().all():
        model = sm.WLS(y, X, weights=weights)
        results = model.fit()
    else:
        model = sm.OLS(y, X)
        results = model.fit()
except Exception as e:
    # Fallback: try converting X/y to numpy arrays and refit
    print('Initial model fit failed, attempting with numpy arrays. Error:', e)
    try:
        y_np = np.asarray(y, dtype=float)
        X_np = np.asarray(X, dtype=float)
        if weights is not None and not np.isnan(weights).all():
            w_np = np.asarray(weights, dtype=float)
            model = sm.WLS(y_np, X_np, weights=w_np)
            results = model.fit()
        else:
            model = sm.OLS(y_np, X_np)
            results = model.fit()
    except Exception as e2:
        print('Fallback model fit also failed. Error:', e2)
        raise
# Cluster-robust SEs by hh_id
from statsmodels.stats.sandwich_covariance import cov_cluster
try:
    # Align clusters with X index
    groups = pd.Series(clusters.values, index=X.index)
    # compute clustered covariance
    clustered_cov = cov_cluster(results, groups)
    # extract robust standard errors
    robust_se = np.sqrt(np.diag(clustered_cov))
    # compute t-stats and p-values manually
    params = results.params.values
    tvalues = params / robust_se
    from scipy import stats
    df_resid = int(results.df_resid)
    pvalues = 2 * stats.t.sf(np.abs(tvalues), df_resid)
    # confidence intervals
    ci_lower = params - 1.96 * robust_se
    ci_upper = params + 1.96 * robust_se
    # create a simple container for later use
    class RobustResult:
        pass
    robust = RobustResult()
    robust.params = pd.Series(params, index=results.params.index)
    robust.bse = pd.Series(robust_se, index=results.params.index)
    robust.tvalues = pd.Series(tvalues, index=results.params.index)
    robust.pvalues = pd.Series(pvalues, index=results.params.index)
    def conf_int_func():
        return pd.DataFrame({'lower': ci_lower, 'upper': ci_upper}, index=results.params.index)
    robust.conf_int = conf_int_func
except Exception as e:
    print("Cluster robust SE computation failed, using default SEs. Error:", e)
    robust = results

# Extract coefficient for act_subsidy
coef_name = 'act_subsidy'
if coef_name in robust.params.index:
    coef = robust.params[coef_name]
    se = robust.bse[coef_name]
    t = robust.tvalues[coef_name]
    p = robust.pvalues[coef_name]
    conf_int = robust.conf_int().loc[coef_name].tolist()
else:
    coef = np.nan
    se = np.nan
    t = np.nan
    p = np.nan
    conf_int = [np.nan, np.nan]

# Save summary and coefficients
with open(OUT_SUMMARY, 'w') as f:
    f.write(results.summary().as_text())
    f.write('\n\nCluster-robust (by hh_id) results:\n')
    try:
        f.write(robust.summary().as_text())
    except Exception:
        f.write('Could not write robust summary.\n')

coef_table = pd.DataFrame({
    'term': robust.params.index,
    'estimate': robust.params.values,
    'std_error': robust.bse.values,
    't_value': robust.tvalues,
    'p_value': robust.pvalues
})
coef_table.to_csv(OUT_COEFFS, index=False)

print("Coefficient for act_subsidy:", coef)
print("SE:", se)
print("p-value:", p)
print("CI:", conf_int)
print("Wrote outputs to:", OUT_SUMMARY, OUT_COEFFS)
