import os
import json
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm
from scipy import stats

# All IO uses /app/data
DATA_PATH = "/app/data/RiskData.dta"

def main():
    # Read data
    try:
        df, meta = pyreadstat.read_dta(DATA_PATH)
    except FileNotFoundError:
        print(json.dumps({"status": "error", "message": f"Data file not found at {DATA_PATH}"}))
        return

    # Drop participants with age == 99 (bad data indicator in original code)
    if 'age' in df.columns:
        df = df[df['age'] != 99].copy()

    # Create unique ID
    df = df.reset_index(drop=True)
    df['ID'] = np.arange(1, len(df) + 1)

    # Columns to gather (PV=Risk1x, RV=Risk2x, LV=Risk3x)
    risk_cols = [
        *[f"Risk1{i}" for i in range(1, 11)],
        *[f"Risk2{i}" for i in range(1, 11)],
        *[f"Risk3{i}" for i in range(1, 11)],
    ]

    missing_cols = [c for c in risk_cols if c not in df.columns]
    if missing_cols:
        print(json.dumps({"status": "error", "message": "Missing expected columns", "missing_columns": missing_cols}))
        return

    # Long format
    data_l = df.melt(id_vars=['ID'], value_vars=risk_cols, var_name='Decision', value_name='Choice')

    # Condition label
    data_l['Condition'] = np.where(data_l['Decision'].str.contains('Risk1'), 'PV',
                            np.where(data_l['Decision'].str.contains('Risk2'), 'RV',
                              np.where(data_l['Decision'].str.contains('Risk3'), 'LV', '')))

    # Sort by participant then decision
    data_l = data_l.sort_values(['ID', 'Condition', 'Decision']).reset_index(drop=True)

    # Create trial numbers within each ID x Condition
    data_l['trial'] = data_l.groupby(['ID', 'Condition']).cumcount() + 1

    # Recode LV first four trials flipped
    # Original code flipped when Choice == "1" -> "0" and vice versa, then converted to numeric
    # Handle as string first, then cast
    choice_str = data_l['Choice'].astype(str)
    mask_lv = (data_l['Condition'] == 'LV') & (data_l['trial'] <= 4)
    choice_str = np.where(mask_lv & (choice_str == '1'), '0', choice_str)
    choice_str = np.where(mask_lv & (choice_str == '0'), '1', choice_str)
    # Convert to numeric; coerce errors to NaN
    data_l['Choice'] = pd.to_numeric(choice_str, errors='coerce')

    # Summaries per participant (propagate NA if any NA present, like R's na.rm=FALSE)
    def sum_no_skipna(s):
        # If any NA present, return NA; else sum
        return s.sum(skipna=False)

    summary = (
        data_l.groupby('ID').apply(
            lambda g: pd.Series({
                'PV_Risk': sum_no_skipna(g.loc[g['Condition'] == 'PV', 'Choice']),
                'RV_Risk': sum_no_skipna(g.loc[g['Condition'] == 'RV', 'Choice']),
                'LV_Perform': sum_no_skipna(g.loc[g['Condition'] == 'LV', 'Choice']),
            })
        ).reset_index()
    )

    # Composite Risk Aversion (row mean of PV_Risk and RV_Risk, propagate NA if either NA)
    summary['RiskAversion'] = summary[['PV_Risk', 'RV_Risk']].mean(axis=1, skipna=False)

    # Drop rows with NA in variables used for regression
    reg_data = summary.dropna(subset=['LV_Perform', 'RiskAversion']).copy()

    # Linear regression: LV_Perform ~ RiskAversion
    X = sm.add_constant(reg_data['RiskAversion'])
    y = reg_data['LV_Perform']
    model = sm.OLS(y, X).fit()

    # Spearman correlation for robustness (as in Task1)
    rho, p_spear = stats.spearmanr(reg_data['LV_Perform'], reg_data['RiskAversion'], nan_policy='omit')

    # Prepare output
    out = {
        'status': 'ok',
        'n_obs_regression': int(model.nobs),
        'coefficients': {
            'const': {
                'estimate': model.params.get('const', np.nan),
                'se': model.bse.get('const', np.nan),
                't': model.tvalues.get('const', np.nan),
                'p': model.pvalues.get('const', np.nan)
            },
            'RiskAversion': {
                'estimate': model.params.get('RiskAversion', np.nan),
                'se': model.bse.get('RiskAversion', np.nan),
                't': model.tvalues.get('RiskAversion', np.nan),
                'p': model.pvalues.get('RiskAversion', np.nan)
            }
        },
        'f_stat': None if model.fvalue is None else float(model.fvalue),
        'f_pvalue': None if model.f_pvalue is None else float(model.f_pvalue),
        'r_squared_adj': float(model.rsquared_adj),
        'spearman_rho': None if rho is None else float(rho),
        'spearman_p': None if p_spear is None else float(p_spear)
    }

    print(json.dumps(out))

if __name__ == "__main__":
    main()
