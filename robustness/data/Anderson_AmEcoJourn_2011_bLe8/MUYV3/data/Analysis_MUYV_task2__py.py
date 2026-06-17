import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm

# IO root
DATA_ROOT = os.environ.get("APP_DATA", "/app/data")
DATA_SUBDIR = os.environ.get("APP_DATA_SUBDIR", "AEJApp-2009-0289-data")

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


def run_task2_models():
    hfile = None
    for p in HOUSEHOLD_PATHS:
        if os.path.exists(p):
            hfile = p
            break
    if hfile is None:
        print("[ERROR] household.dta not found in expected locations")
        sys.exit(1)
    hh = pd.read_stata(hfile)

    # 0) Sample restriction: low-caste households (BAC, OBC, SC). Based on observed levels
    low_caste_levels = {"Back agr", "Back oth", "ST/SC   "}
    if 'caste' in hh.columns:
        hh = hh.loc[hh['caste'].isin(low_caste_levels)].copy()

    # Restrict to complete cases
    hh = hh.dropna().copy()

    # Outlier detection on log(totinc)
    if 'totinc' not in hh.columns:
        print("[ERROR] 'totinc' not in household data.")
        sys.exit(1)
    mask = iqr_outlier_mask_log(hh['totinc'])
    hhfilt = hh.loc[mask].copy()
    hhfilt['log_totinc'] = np.log(hhfilt['totinc'])

    # Model 1: base with domlow and RE for bihar/village (two-level nested)
    if 'domlow' not in hhfilt.columns:
        print("[ERROR] 'domlow' not in household data after filtering.")
        sys.exit(1)

    groups_col = 'village'
    vc = {}
    if 'bihar' in hhfilt.columns:
        vc['bihar'] = '0 + C(bihar)'

    print("\n=== Task2: Mixed-effects Model 1: log_totinc ~ domlow; RE: village + (bihar) ===")
    try:
        md1 = mixedlm("log_totinc ~ domlow", data=hhfilt, groups=hhfilt[groups_col], vc_formula=vc)
        mdf1 = md1.fit(method='lbfgs', maxiter=200, disp=False)
        print(mdf1.summary())
        if 'domlow' in mdf1.params.index:
            print({
                'model': 'Task2_Model1',
                'coef_domlow': float(mdf1.params['domlow']),
                't_domlow': float(getattr(mdf1, 'tvalues', mdf1.zvalues)['domlow']),
                'p_domlow': float(mdf1.pvalues['domlow'])
            })
    except Exception as e:
        print(f"[WARN] MixedLM Model 1 failed: {e}")

    # Model 2: include literacy, totland, caste and all interactions with domlow
    inter_formula = 'log_totinc ~ literate * totland'
    if 'caste' in hhfilt.columns:
        inter_formula += ' * C(caste)'
    inter_formula += ' * domlow'

    print("\n=== Task2: Mixed-effects Model 2: "+inter_formula+"; RE: village + (bihar) ===")
    try:
        md2 = mixedlm(inter_formula, data=hhfilt, groups=hhfilt[groups_col], vc_formula=vc)
        mdf2 = md2.fit(method='lbfgs', maxiter=400, disp=False)
        print(mdf2.summary())
        if 'domlow' in mdf2.params.index:
            print({
                'model': 'Task2_Model2',
                'coef_domlow': float(mdf2.params['domlow']),
                't_domlow': float(getattr(mdf2, 'tvalues', mdf2.zvalues)['domlow']),
                'p_domlow': float(mdf2.pvalues['domlow'])
            })
    except Exception as e:
        print(f"[WARN] MixedLM Model 2 failed: {e}")

if __name__ == '__main__':
    print("[INFO] Running Task2 translation in Python")
    run_task2_models()
