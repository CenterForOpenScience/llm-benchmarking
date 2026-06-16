import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import mixedlm

# IO root
DATA_ROOT = os.environ.get("APP_DATA", "/app/data")
DATA_SUBDIR = os.environ.get("APP_DATA_SUBDIR", "AEJApp-2009-0289-data")

# Resolve data paths
VILLAGE_PATHS = [
    os.path.join(DATA_ROOT, 'village.dta'),
    os.path.join(DATA_ROOT, DATA_SUBDIR, 'village.dta')
]
HOUSEHOLD_PATHS = [
    os.path.join(DATA_ROOT, 'household.dta'),
    os.path.join(DATA_ROOT, DATA_SUBDIR, 'household.dta')
]

# Utility: outlier mask using 1.5*IQR on log-transformed series

def iqr_outlier_mask_log(y: pd.Series):
    ylog = np.log(y)
    q1 = np.nanpercentile(ylog, 25)
    q3 = np.nanpercentile(ylog, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (ylog >= lower) & (ylog <= upper)

# Village-level analysis (balance checks)

def run_village_balance():
    vfile = None
    for p in VILLAGE_PATHS:
        if os.path.exists(p):
            vfile = p
            break
    if vfile is None:
        print("[WARN] village.dta not found in expected locations. Skipping village-level analysis.")
        return None
    vil = pd.read_stata(vfile)
    if 'domhigh' not in vil.columns:
        print("[WARN] 'domhigh' not in village data. Skipping village-level analysis.")
        return None
    results = []
    for col in vil.columns:
        if col in ['village', 'domhigh']:
            continue
        s = vil[[col, 'domhigh']].dropna()
        if s.empty:
            continue
        x = s.loc[s['domhigh'] == 1, col].astype(float)
        y = s.loc[s['domhigh'] == 0, col].astype(float)
        if len(x) < 2 or len(y) < 2:
            continue
        try:
            sw_p = stats.shapiro(s[col].astype(float))[1]
        except Exception:
            sw_p = 0.0
        if sw_p < 0.05:
            try:
                _, pval = stats.mannwhitneyu(x, y, alternative='two-sided')
            except ValueError:
                pval = np.nan
            diff_loc = np.nanmedian(x) - np.nanmedian(y)
            normal = 'no'
        else:
            try:
                _, pval = stats.ttest_ind(x, y, equal_var=False, nan_policy='omit')
            except Exception:
                pval = np.nan
            diff_loc = np.nanmean(x) - np.nanmean(y)
            normal = 'yes'
        results.append({
            'variable': col,
            'normal.distribution': normal,
            'difference.in.location': diff_loc,
            'pval.uncorrected': pval
        })
    if not results:
        return None
    df = pd.DataFrame(results)
    try:
        df['pval.fdr'] = multipletests(df['pval.uncorrected'].values, method='fdr_bh')[1]
    except Exception:
        df['pval.fdr'] = np.nan
    print("=== Village-level balance tests (by domhigh) ===")
    print(df.to_string(index=False))
    return df

# Helper: derive district as idxmax across dist dummies when exactly one is 1

def derive_district_from_dummies(df: pd.DataFrame, dist_cols):
    if not all(col in df.columns for col in dist_cols):
        return df, False
    row_sums = df[dist_cols].sum(axis=1)
    # Only keep rows with exactly one district dummy set to 1
    valid = row_sums == 1
    district_idx = df.loc[valid, dist_cols].idxmax(axis=1)
    df.loc[valid, 'district'] = district_idx.str.replace('dist', '', regex=False).astype(float)
    # Set others to NaN
    df.loc[~valid, 'district'] = np.nan
    # Drop original dummy columns
    df = df.drop(columns=dist_cols)
    return df, True

# Fit with multiple optimizers up to 3 attempts

def fit_mixedlm_with_retries(formula, data, groups, vc_formula, max_attempts=3):
    methods = ['lbfgs', 'bfgs', 'cg', 'powell', 'nm']
    attempts = 0
    last_err = None
    for m in methods:
        try:
            md = mixedlm(formula, data=data, groups=groups, vc_formula=vc_formula)
            res = md.fit(method=m, reml=True, maxiter=500, disp=False)
            print(f"[INFO] MixedLM converged using method={m} after attempt {attempts+1}")
            return res, attempts+1
        except Exception as e:
            print(f"[WARN] MixedLM fit failed with method={m}: {e}")
            last_err = e
            attempts += 1
            if attempts >= max_attempts:
                break
    raise RuntimeError(f"All MixedLM attempts failed after {attempts} tries. Last error: {last_err}")

# Household-level analysis (Task1)

def run_task1_models():
    hfile = None
    for p in HOUSEHOLD_PATHS:
        if os.path.exists(p):
            hfile = p
            break
    if hfile is None:
        print("[ERROR] household.dta not found in expected locations")
        sys.exit(1)
    hh = pd.read_stata(hfile)

    # 0) Sample restriction: low-caste households (BAC, OBC, SC). Map based on observed levels
    low_caste_levels = {"Back agr", "Back oth", "ST/SC   "}
    if 'caste' in hh.columns:
        hh = hh.loc[hh['caste'].isin(low_caste_levels)].copy()

    # 1) District: derive from dist1..dist25 via idxmax when exactly one is 1
    dist_cols = [f'dist{i}' for i in range(1, 26)]
    if all(col in hh.columns for col in dist_cols):
        hh, _ = derive_district_from_dummies(hh, dist_cols)
    else:
        print("[WARN] District dummy columns not found. 'district' will be unavailable for variance components.")

    # 2) Crop controls: mean of paddy,wheat,cereal,pulse,bulb,seed,cash
    crop_vars = ['paddy','wheat','cereal','pulse','bulb','seed','cash']
    if all(col in hh.columns for col in crop_vars):
        hh['cropcontrol'] = hh[crop_vars].mean(axis=1)
        # Keep original crop columns out of the model
        hh = hh.drop(columns=crop_vars)

    # 3) Complete cases AFTER constructing derived variables
    hh = hh.dropna().copy()

    # 4) Outlier detection on log(totinc)
    if 'totinc' not in hh.columns:
        print("[ERROR] 'totinc' not in household data.")
        sys.exit(1)
    mask = iqr_outlier_mask_log(hh['totinc'])
    hhfilt = hh.loc[mask].copy()
    hhfilt['log_totinc'] = np.log(hhfilt['totinc'])

    if 'domlow' not in hhfilt.columns:
        print("[ERROR] 'domlow' not in household data after filtering.")
        sys.exit(1)

    # 5) Center/scale continuous predictors to aid convergence (exclude outcomes and identifiers)
    exclude_scale = {'log_totinc', 'totinc', 'domlow', 'village', 'district', 'bihar'}
    num_cols = hhfilt.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col not in exclude_scale:
            std = hhfilt[col].std()
            if pd.notnull(std) and std > 0:
                hhfilt[col] = (hhfilt[col] - hhfilt[col].mean()) / std

    # 6) Random effects structures
    groups_col = 'village'
    vc = {}
    if 'district' in hhfilt.columns and hhfilt['district'].notnull().any():
        vc['district'] = '0 + C(district)'
    if 'bihar' in hhfilt.columns and hhfilt['bihar'].notnull().any():
        vc['bihar'] = '0 + C(bihar)'

    # Model 1: log_totinc ~ domlow
    print("\n=== Task1: Mixed-effects Model 1: log_totinc ~ domlow; RE: village + (district, bihar) ===")
    try:
        mdf1, attempts1 = fit_mixedlm_with_retries("log_totinc ~ domlow", data=hhfilt, groups=hhfilt[groups_col], vc_formula=vc, max_attempts=3)
        print(mdf1.summary())
        if 'domlow' in mdf1.params.index:
            print({
                'model': 'Task1_Model1',
                'coef_domlow': float(mdf1.params['domlow']),
                't_domlow': float(getattr(mdf1, 'tvalues', mdf1.zvalues)['domlow']),
                'p_domlow': float(mdf1.pvalues['domlow'])
            })
    except Exception as e:
        print(f"[ERROR] MixedLM Model 1 failed after retries: {e}")

    # Model 2: add controls
    controls = [
        'literate','totland','C(caste)','cropcontrol','bus3','tele3','ps3','pds3','bank3','pps3','ms3','ss3','phc3','hosp3',
        'gwdevelopment','rainfall','gwavailability','river','canal','noalkal','nolog','nosoil','noflood','area','pmixdominant','tolaparea'
    ]
    existing_controls = []
    for term in controls:
        base = term.replace('C(','').replace(')','') if term.startswith('C(') else term
        if term == 'C(caste)':
            if 'caste' in hhfilt.columns:
                existing_controls.append(term)
        else:
            if base in hhfilt.columns and hhfilt[base].std() > 0:
                existing_controls.append(term)
    formula = 'log_totinc ~ ' + ' + '.join(existing_controls + ['domlow']) if existing_controls else 'log_totinc ~ domlow'

    print("\n=== Task1: Mixed-effects Model 2: "+formula+"; RE: village + (district, bihar) ===")
    try:
        mdf2, attempts2 = fit_mixedlm_with_retries(formula, data=hhfilt, groups=hhfilt[groups_col], vc_formula=vc, max_attempts=3)
        print(mdf2.summary())
        if 'domlow' in mdf2.params.index:
            print({
                'model': 'Task1_Model2',
                'coef_domlow': float(mdf2.params['domlow']),
                't_domlow': float(getattr(mdf2, 'tvalues', mdf2.zvalues)['domlow']),
                'p_domlow': float(mdf2.pvalues['domlow'])
            })
    except Exception as e:
        print(f"[ERROR] MixedLM Model 2 failed after retries: {e}")

if __name__ == '__main__':
    print("[INFO] Running Task1 translation in Python")
    run_village_balance()
    run_task1_models()
