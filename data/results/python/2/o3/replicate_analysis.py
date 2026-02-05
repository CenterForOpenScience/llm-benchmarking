"""
Python replication of Seaton et al. (2010) focal test: moderation of Big-Fish-Little-Pond Effect by MEMOR (use of memorisation).
This script mirrors the logic of the R script '!!CORRECTION!!_Seaton_AmEduResJourn_2010_Blxd_final.R' but using Python packages.
All IO paths follow the container convention of '/app/data'.
"""
import os
import numpy as np
import pandas as pd
import pyreadr
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.api as sm
from scipy.stats import norm, combine_pvalues

# ----------------------------
# 1. Load data
# ----------------------------
DATA_PATH = "/app/data/PISA2012.replication.RDS"
assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"
result = pyreadr.read_r(DATA_PATH)
# The RDS contains a single DataFrame
key = list(result.keys())[0]
df = result[key]

# ----------------------------
# 2. Create unique IDs (school + student)
# ----------------------------
df['uniqueSchoolID'] = df['SCHOOLID'].astype(str) + '|' + df['CNT'].astype(str)
df['uniqueStudentID'] = df['STIDSTD'].astype(str) + '|' + df['uniqueSchoolID']

# ----------------------------
# 3. Filter missing data following the original R logic
# ----------------------------
# Outcome: SCMAT (math self-concept).
keep = (df['SCMAT'] <= 997) & (~df['SCMAT'].isna())
for pv in [f'PV{i}MATH' for i in range(1,6)]:
    keep &= (df[pv] <= 997) & (~df[pv].isna())
keep &= (df['MEMOR'] <= 997) & (~df['MEMOR'].isna())
df = df[keep].copy()

# Remove schools with <= 10 students
counts = df.groupby('uniqueSchoolID')['uniqueStudentID'].transform('count')
df = df[counts > 10].copy()

# ----------------------------
# 4. Standardise variables and build additional terms
# ----------------------------
# Standardisation helper
def zscore(series):
    return (series - series.mean())/series.std(ddof=0)

for i in range(1,6):
    pv = f'PV{i}MATH'
    df[f'{pv}_z'] = zscore(df[pv])
    df[f'{pv}_z_sq'] = df[f'{pv}_z'] ** 2

# Standardise outcome and moderator
df['SCMAT_z'] = zscore(df['SCMAT'])
df['MEMOR_z'] = zscore(df['MEMOR'])

# Compute school-average ability for each plausible value
for i in range(1,6):
    pv_z = f'PV{i}MATH_z'
    school_mean = df.groupby('uniqueSchoolID')[pv_z].transform('mean')
    df[f'school_{pv_z}'] = school_mean
    # Cross product (moderation term)
    df[f'CROSS{i}'] = df['MEMOR_z'] * df[f'school_{pv_z}']

# ----------------------------
# 5. Mixed-effects models (one per plausible value)
# ----------------------------
coefs = []
std_errs = []
p_values = []

for i in range(1,6):
    pv_z = f'PV{i}MATH_z'
    school_pv_z = f'school_{pv_z}'
    cross = f'CROSS{i}'
    formula_terms = [pv_z, f'{pv_z}_sq', school_pv_z, 'MEMOR_z', cross]

    # Prepare design matrices manually (statsmodels MixedLM does not accept formula with varying slopes easily)
    exog = df[formula_terms]
    exog = sm.add_constant(exog)

    # Random effects structure approximation: allow random intercepts by school and country.
    # Statsmodels MixedLM only supports one grouping factor at a time. We'll approximate by school (dominant level) here.
    model = MixedLM(df['SCMAT_z'], exog, groups=df['uniqueSchoolID'])
    try:
        result = model.fit(method='lbfgs', reml=True, maxiter=200, warn_convergence=False)
    except Exception as e:
        print(f"Model for PV{i} failed to converge: {e}")
        continue

    est = result.params[cross]
    se  = result.bse[cross]
    zval = est / se
    pval = 2*(1 - norm.cdf(abs(zval)))

    coefs.append(est)
    std_errs.append(se)
    p_values.append(pval)

# ----------------------------
# 6. Pool results across plausible values (PISA manual procedure)
# ----------------------------
M = len(coefs)
if M == 0:
    raise RuntimeError("No models converged; cannot compute pooled estimates.")

final_coef = np.mean(coefs)
# Sampling variance
sampling_var = np.mean(np.square(std_errs))
# Imputation variance
imputation_var = np.sum([(b - final_coef)**2 for b in coefs]) / (M - 1) if M > 1 else 0
final_error_var = sampling_var + 1.2 * imputation_var
final_se = np.sqrt(final_error_var)

# Combine p-values using Fisher's method
combined_stat, combined_p = combine_pvalues(p_values, method='fisher')

# ----------------------------
# 7. Save outputs
# ----------------------------
out_path = "/app/data/replication_results.csv"
pd.DataFrame({
    'coef': coefs,
    'se': std_errs,
    'p': p_values
}).to_csv(out_path, index=False)

summary_path = "/app/data/final_summary.txt"
with open(summary_path, 'w') as fh:
    fh.write(f"Pooled moderator coefficient: {final_coef:.4f}\n")
    fh.write(f"Pooled SE: {final_se:.4f}\n")
    fh.write(f"Combined p-value (Fisher): {combined_p:.4e}\n")

print("Replication analysis complete. Summary written to", summary_path)
