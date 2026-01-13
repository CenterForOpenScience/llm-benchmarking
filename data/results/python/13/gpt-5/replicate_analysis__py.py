import os
import json
import traceback
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# Local helper
from variable_mapping__py import infer_mapping, missing_vars

CANDIDATE_DATA_DIRS = [
    "/app/data/original/13/0112_python_gpt5/replication_data",
    "/workspace/replication_data",
    "/workspace/data/original/13/0112_python_gpt5/replication_data",
    "/app/data/replication_data"
]
OUT_DIR = "/workspace"

PREFERRED_FILES = [
    "data_clean_5pct.rds",
    "data_clean.rds",
    "data_imp_5pct.rds"
]

KEY_VARS = [
    "trstprl_rev", "imm_concern", "cntry"
]

CONTROL_VARS = [
    "happy_rev", "stflife_rev", "sclmeet_rev", "distrust_soc",
    "stfeco_rev", "hincfel", "stfhlth_rev", "stfedu_rev",
    "vote_gov", "vote_frparty", "lrscale", "hhinc_std", "agea", "educ", "female",
    "vote_share_fr", "socexp", "lt_imm_cntry", "wgi", "gdppc", "unemp"
]

WEIGHT_VAR = "pspwght"


def read_rds(path):
    """Read an RDS file using pyreadr. Returns pandas DataFrame."""
    import pyreadr
    res = pyreadr.read_r(path)
    # res is a dict-like; take first value
    if hasattr(res, 'items'):
        for _, df in res.items():
            return df
    # Fallback if pyreadr returns directly
    return res


def select_data_file():
    searched = []
    for d in CANDIDATE_DATA_DIRS:
        for fname in PREFERRED_FILES:
            fpath = os.path.join(d, fname)
            searched.append(fpath)
            if os.path.exists(fpath):
                return fpath
    raise FileNotFoundError(f"None of the preferred data files found. Searched: {searched}")


def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# Ensure all columns in a DataFrame are numeric where possible
# - categorical -> codes
# - bool -> int
# - object -> to_numeric (coerce)
def make_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    X2 = X.copy()
    for col in X2.columns:
        s = X2[col]
        if pd.api.types.is_categorical_dtype(s):
            X2[col] = s.cat.codes.astype(float)
        elif pd.api.types.is_bool_dtype(s):
            X2[col] = s.astype(int)
        elif pd.api.types.is_numeric_dtype(s):
            # already numeric
            continue
        else:
            X2[col] = pd.to_numeric(s, errors='coerce')
    return X2


def fit_mixedlm(y, X, groups):
    # MixedLM cannot handle perfect collinearity; drop any all-NaN columns
    X = X.loc[:, X.notna().any(axis=0)]
    model = MixedLM(y, X, groups=groups)
    result = model.fit(reml=True, method='lbfgs', maxiter=200, disp=False)
    return result


def fit_ols_fe_cluster(y, X, groups):
    # Add constant if not present
    if 'const' not in X.columns:
        X = sm.add_constant(X, has_constant='add')
    # Country fixed effects via pandas.get_dummies
    dummies = pd.get_dummies(groups, prefix='cntry', drop_first=True)
    X_fe = pd.concat([X, dummies], axis=1)
    model = sm.OLS(y, X_fe, missing='drop')
    result = model.fit(cov_type='cluster', cov_kwds={'groups': groups}, use_t=True)
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log_lines = []
    try:
        data_path = select_data_file()
        log_lines.append(f"Selected data file: {data_path}")
        df = read_rds(data_path)
        log_lines.append(f"Loaded data shape: {df.shape}")

        # Infer mapping
        mapping = infer_mapping(df.columns)
        missing_key = missing_vars(mapping, KEY_VARS)
        if missing_key:
            raise RuntimeError(f"Missing required variables in dataset: {missing_key}")

        # Build variable lists present in data
        present_controls = [v for v in CONTROL_VARS if mapping.get(v) is not None]
        vars_needed = [mapping[v] for v in KEY_VARS if mapping[v] is not None] + [mapping[v] for v in present_controls]
        if mapping.get(WEIGHT_VAR) is not None:
            vars_needed.append(mapping[WEIGHT_VAR])
        vars_needed = list(dict.fromkeys(vars_needed))  # unique, preserve order

        df_sub = df[vars_needed].copy()
        # Coerce numeric where appropriate
        num_like = [mapping[v] for v in KEY_VARS if v != 'cntry' and mapping.get(v) is not None]
        num_like += [mapping[v] for v in present_controls if mapping.get(v) is not None]
        if mapping.get(WEIGHT_VAR) is not None:
            num_like.append(mapping[WEIGHT_VAR])
        df_sub = coerce_numeric(df_sub, num_like)

        # Rename to standardized names for modeling convenience
        rename_map = {mapping[k]: k for k in mapping if mapping[k] is not None}
        df_sub = df_sub.rename(columns=rename_map)

        # Drop rows missing essential variables
        required_for_model = ['trstprl_rev', 'imm_concern', 'cntry']
        before = len(df_sub)
        df_sub = df_sub.dropna(subset=required_for_model)
        after = len(df_sub)
        log_lines.append(f"Rows after dropping missing on key vars: {after} (dropped {before - after})")

        # Prepare design matrices
        y = df_sub['trstprl_rev']
        X_vars = ['imm_concern'] + [v for v in CONTROL_VARS if v in df_sub.columns]
        X = df_sub[X_vars].copy()
        groups = df_sub['cntry']
        # Enforce numeric types and drop rows with any NaNs/Infs in model matrices
        X = make_numeric_df(X)
        y = pd.to_numeric(y, errors='coerce')
        df_model = pd.concat([y.rename('y'), X, groups.rename('groups')], axis=1)
        df_model.replace([np.inf, -np.inf], np.nan, inplace=True)
        before_mod = len(df_model)
        df_model = df_model.dropna()
        after_mod = len(df_model)
        log_lines.append(f"Rows after cleaning model matrices (drop NaN/Inf): {after_mod} (dropped {before_mod - after_mod})")
        y = df_model['y']
        groups = df_model['groups']
        X = df_model.drop(columns=['y', 'groups'])


        # Try MixedLM first (unweighted); statsmodels MixedLM doesn't support weights straightforwardly
        model_type = None
        try:
            X_mixed = sm.add_constant(X, has_constant='add')
            mixed_res = fit_mixedlm(y, X_mixed, groups=groups)
            model_type = 'MixedLM (random intercept by cntry, unweighted)'
            res = mixed_res
            coef = res.params.get('imm_concern', np.nan)
            se = res.bse.get('imm_concern', np.nan)
            pval = res.pvalues.get('imm_concern', np.nan)
            ci_low, ci_high = (coef - 1.96 * se, coef + 1.96 * se) if np.isfinite(se) else (np.nan, np.nan)
            nobs = int(res.nobs)
            details = res.summary().as_text()
        except Exception as e:
            log_lines.append(f"MixedLM failed: {repr(e)}. Falling back to OLS with country FE and cluster-robust SEs.")
            # OLS with FE and cluster by cntry; optionally use weights if available via WLS but combine with cluster is complex.
            ols_res = fit_ols_fe_cluster(y, X, groups)
            model_type = 'OLS with country fixed effects; cluster-robust SE by cntry'
            res = ols_res
            coef = res.params.get('imm_concern', np.nan)
            se = res.bse.get('imm_concern', np.nan)
            pval = res.pvalues.get('imm_concern', np.nan)
            ci = res.conf_int().loc['imm_concern'] if 'imm_concern' in res.conf_int().index else [np.nan, np.nan]
            ci_low, ci_high = float(ci[0]), float(ci[1])
            nobs = int(res.nobs)
            details = res.summary().as_text()

        # Save outputs
        results = {
            'data_file': os.path.basename(data_path),
            'model_type': model_type,
            'nobs': nobs,
            'coefficient_imm_concern': float(coef) if pd.notnull(coef) else None,
            'se_imm_concern': float(se) if pd.notnull(se) else None,
            'p_value_imm_concern': float(pval) if pd.notnull(pval) else None,
            'ci_imm_concern_95': [float(ci_low) if pd.notnull(ci_low) else None, float(ci_high) if pd.notnull(ci_high) else None],
            'controls_included': [v for v in CONTROL_VARS if v in X.columns],
            'groups': 'cntry',
            'notes': log_lines,
        }

        with open(os.path.join(OUT_DIR, 'replication_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        with open(os.path.join(OUT_DIR, 'model_summary.txt'), 'w') as f:
            f.write(details)
            f.write("\n\nLog notes:\n" + "\n".join(log_lines))

        print(json.dumps(results, indent=2))

    except Exception as ex:
        err_path = os.path.join(OUT_DIR, 'execution_error.log')
        tb = traceback.format_exc()
        with open(err_path, 'w') as f:
            f.write(tb)
        print(f"ERROR: {ex}")
        print(tb)
        raise


if __name__ == '__main__':
    main()
