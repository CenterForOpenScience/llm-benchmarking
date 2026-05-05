import os
import json
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Core modeling
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.mixed_linear_model import MixedLM

# Optional: RDS reader
try:
    import pyreadr
    HAS_PYREADR = True
except Exception:
    HAS_PYREADR = False

DATA_DIR = "/app/data"


def locate_data_file() -> str:
    """Locate the best available dataset file inside /app/data.
    Priority order mirrors likely completeness: full clean > imputed 5% > clean 5% > CSV fallbacks.
    """
    candidates = [
        "data_clean.rds",
        "data_imp_5pct.rds",
        "data_clean_5pct.rds",
        "data_clean.csv",
        "data_imp_5pct.csv",
        "data_clean_5pct.csv",
    ]
    for fname in candidates:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            return fpath
    raise FileNotFoundError(
        f"No supported dataset found in {DATA_DIR}. Expected one of: {', '.join(candidates)}"
    )


def load_dataset(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        return df
    if ext == ".rds":
        if not HAS_PYREADR:
            raise ImportError(
                "pyreadr is required to load .rds files. Please install pyreadr."
            )
        res = pyreadr.read_r(path)
        # read_r returns a dict of name->object (usually one dataframe)
        if len(res) == 1:
            df = list(res.values())[0]
        else:
            # Choose the largest dataframe if multiple objects
            df = max(res.items(), key=lambda kv: (hasattr(kv[1], 'shape'), getattr(kv[1], 'shape', (0, 0))[0]))[1]
        # Ensure pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            # Attempt to convert
            df = pd.DataFrame(df)
        return df
    raise ValueError(f"Unsupported file extension: {ext}")


def find_first_matching_column(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for c_low, c_orig in lower_map.items():
        if all(pat in c_low for pat in patterns):
            return c_orig
    # fallback: any of patterns contained
    for c_low, c_orig in lower_map.items():
        if any(pat in c_low for pat in patterns):
            return c_orig
    return None


def get_preferred_column(df: pd.DataFrame, ordered_candidates: List[str]) -> Optional[str]:
    """Return the first column present from an ordered list of exact candidate names (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in ordered_candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def reverse_scale(series: pd.Series, max_scale: Optional[float] = None) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    if max_scale is None:
        # Try to infer max scale (commonly 10 or 10.0)
        s_max = s.max(skipna=True)
        s_min = s.min(skipna=True)
        if pd.notna(s_min) and pd.notna(s_max) and s_min >= 0 and s_max <= 10:
            max_scale = 10
        else:
            max_scale = s_max
    return max_scale - s


def build_variables(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str], Optional[str], Optional[str]]:
    """
    Construct analysis dataframe with outcomes, key predictor, and controls when available.
    Returns:
      - analysis_df: dataframe with columns y_<outcome>, imm_concern, controls, group ids
      - used_cols: mapping of logical names to actual column names
      - outcomes: list of outcome names actually constructed
      - group_col: primary grouping column (country)
      - vc_col: variance component column (country_round) if available
    """
    used = {}

    # Identify grouping columns
    country_col = get_preferred_column(df, ["country", "cntry", "ctry", "nation"])
    round_col = get_preferred_column(df, ["round", "essround", "ess_round", "wave"])  

    if country_col is None:
        # Try pattern search
        cc = find_first_matching_column(df, ["cntry"]) or find_first_matching_column(df, ["country"])
        country_col = cc
    if round_col is None:
        rr = find_first_matching_column(df, ["round"]) or find_first_matching_column(df, ["wave"]) or find_first_matching_column(df, ["ess"])
        round_col = rr

    used['country'] = country_col if country_col else ""
    used['round'] = round_col if round_col else ""

    # Create nested id if possible
    vc_col = None
    if country_col is not None and round_col is not None:
        vc_col = "country_round_id"
        df[vc_col] = df[country_col].astype(str) + "_" + df[round_col].astype(str)
        used['country_round_id'] = vc_col

    # Outcomes: distrust in parliament, politicians, legal system
    # Try direct 'distrust' columns first; else reverse from trust measures.
    def get_distrust(measure: str, distrust_candidates: List[str], trust_candidates: List[str]) -> Optional[pd.Series]:
        # exact candidates
        col = get_preferred_column(df, distrust_candidates)
        if col is not None:
            used[f"distrust_{measure}"] = col
            return pd.to_numeric(df[col], errors='coerce')
        # search by pattern
        col = find_first_matching_column(df, ["distrust", measure])
        if col is not None:
            used[f"distrust_{measure}"] = col
            return pd.to_numeric(df[col], errors='coerce')
        # trust-based reverse
        col = get_preferred_column(df, trust_candidates)
        if col is None:
            col = find_first_matching_column(df, ["trust", measure])
        if col is not None:
            used[f"trust_{measure}"] = col
            return reverse_scale(df[col])
        return None

    y_parl = get_distrust(
        "parliament",
        ["distrust_parliament", "distrust_parl"],
        ["trust_parliament", "trstprl", "trust_parl"],
    )
    y_polit = get_distrust(
        "politicians",
        ["distrust_politicians", "distrust_polit"],
        ["trust_politicians", "trstplt", "trust_polit"],
    )
    y_legal = get_distrust(
        "legal",
        ["distrust_legal_system", "distrust_legal", "distrust_justice"],
        ["trust_legal_system", "trstlgl", "trust_legal", "trust_justice"],
    )

    outcomes = []
    if y_parl is not None:
        outcomes.append("parliament")
    if y_polit is not None:
        outcomes.append("politicians")
    if y_legal is not None:
        outcomes.append("legal")

    # Key predictor: concern about immigration
    imm_col = None
    # direct index if present
    imm_col = get_preferred_column(df, [
        "immig_concern", "immigration_concern", "concern_immigration",
        "immconc", "imm_concern", "immigration_attitudes_index"
    ])
    if imm_col is None:
        # pattern search
        imm_col = find_first_matching_column(df, ["immig", "conc"])
    imm_concern = None
    if imm_col is not None:
        used['imm_concern'] = imm_col
        imm_concern = pd.to_numeric(df[imm_col], errors='coerce')
    else:
        # Try to build from ESS core items: imbgeco, imueclt, imwbcnt
        econ = get_preferred_column(df, ["imbgeco"]) or find_first_matching_column(df, ["imbgeco"])  # economy bad-good
        cult = get_preferred_column(df, ["imueclt"]) or find_first_matching_column(df, ["imueclt"])  # undermine-enrich culture
        life = get_preferred_column(df, ["imwbcnt"]) or find_first_matching_column(df, ["imwbcnt"])  # worse-better place
        if econ and cult and life:
            used['imbgeco'] = econ
            used['imueclt'] = cult
            used['imwbcnt'] = life
            econ_rev = reverse_scale(df[econ])
            cult_rev = reverse_scale(df[cult])
            life_rev = reverse_scale(df[life])
            imm_concern = (econ_rev + cult_rev + life_rev) / 3.0
            used['imm_concern'] = 'constructed_from_imbgeco_imueclt_imwbcnt'
        else:
            warnings.warn("Could not identify immigration concern variable; analysis will fail.")

    # Controls (assembled opportunistically if present)
    controls = {}

    # Economic, health, education dissatisfaction
    if get_preferred_column(df, ["stfeco"]) or find_first_matching_column(df, ["stfeco"]):
        col = get_preferred_column(df, ["stfeco"]) or find_first_matching_column(df, ["stfeco"])
        controls['diss_economy'] = reverse_scale(df[col])
        used['stfeco'] = col
    if get_preferred_column(df, ["stfhlth"]) or find_first_matching_column(df, ["stfhlth"]):
        col = get_preferred_column(df, ["stfhlth"]) or find_first_matching_column(df, ["stfhlth"])
        controls['diss_health'] = reverse_scale(df[col])
        used['stfhlth'] = col
    if get_preferred_column(df, ["stfedu"]) or find_first_matching_column(df, ["stfedu"]):
        col = get_preferred_column(df, ["stfedu"]) or find_first_matching_column(df, ["stfedu"])
        controls['diss_education'] = reverse_scale(df[col])
        used['stfedu'] = col

    # Interpersonal distrust
    if get_preferred_column(df, ["ppltrst"]) or find_first_matching_column(df, ["ppltrst"]):
        col = get_preferred_column(df, ["ppltrst"]) or find_first_matching_column(df, ["ppltrst"])
        controls['interpersonal_distrust'] = reverse_scale(df[col])
        used['ppltrst'] = col

    # Winner/loser, far-right vote indicator (if present)
    win_col = get_preferred_column(df, ["winner", "in_winner"]) or find_first_matching_column(df, ["winner"])  
    if win_col:
        used['winner'] = win_col
        controls['winner'] = pd.to_numeric(df[win_col], errors='coerce')

    fr_col = find_first_matching_column(df, ["far", "right", "vote"])  # very permissive
    if fr_col:
        used['far_right_vote'] = fr_col
        controls['far_right_vote'] = pd.to_numeric(df[fr_col], errors='coerce')

    # Left-right, income, age, education, gender
    for cand, key in [
        (["lrscale"], 'lrscale'),
        (["hinctnta", "hinctnt"], 'income'),
        (["agea", "age"], 'age'),
        (["eduyrs", "eduyr", "edulev", "edulvla"], 'education'),
        (["gndr", "gender"], 'gender'),
    ]:
        col = get_preferred_column(df, cand)
        if col is None:
            for pat in cand:
                col = find_first_matching_column(df, [pat])
                if col:
                    break
        if col:
            used[key] = col
            series = pd.to_numeric(df[col], errors='coerce')
            if key == 'gender':
                # try to binary-encode if string labels
                if series.isna().all() and df[col].dtype == object:
                    # map common labels
                    mapped = df[col].astype(str).str.lower().map({
                        'male': 0, 'm': 0, '1': 1, 'female': 1, 'f': 1
                    })
                    series = pd.to_numeric(mapped, errors='coerce')
            controls[key] = series

    # Country/Country-round level controls (if present)
    for cand, key in [
        (["socprot", "social_protection", "socxpr"], 'socprot'),
        (["gdp_pc", "gdppc", "gdppercap"], 'gdppc'),
        (["unempl", "unemployment"], 'unemployment'),
        (["governance", "govqual", "wgi_index"], 'govqual'),
        (["long_term_immigration", "immig_longterm", "imm_country"], 'longterm_imm_country'),
        (["far_right_popularity", "far_right_share", "fr_share"], 'fr_popularity'),
    ]:
        col = get_preferred_column(df, cand)
        if col is None:
            for pat in cand:
                col = find_first_matching_column(df, [pat])
                if col:
                    break
        if col:
            used[key] = col
            controls[key] = pd.to_numeric(df[col], errors='coerce')

    # Assemble analysis dataframe
    analysis = pd.DataFrame()
    if y_parl is not None:
        analysis['y_parliament'] = y_parl
    if y_polit is not None:
        analysis['y_politicians'] = y_polit
    if y_legal is not None:
        analysis['y_legal'] = y_legal
    if imm_concern is not None:
        analysis['imm_concern'] = imm_concern

    for k, s in controls.items():
        analysis[f"ctrl_{k}"] = pd.to_numeric(s, errors='coerce')

    if country_col is not None:
        analysis['group_country'] = df[country_col].astype(str)
    if vc_col is not None:
        analysis[vc_col] = df[vc_col].astype(str)

    # Drop rows with missing key vars
    req_cols = ['imm_concern'] + [c for c in analysis.columns if c.startswith('y_')]
    analysis = analysis.dropna(subset=req_cols, how='any')

    return analysis, used, outcomes, ('group_country' if country_col is not None else None), vc_col


def fit_model_for_outcome(df_model: pd.DataFrame, y_col: str, group_col: Optional[str], vc_col: Optional[str]):
    # Build design matrices
    X_cols = ['imm_concern'] + [c for c in df_model.columns if c.startswith('ctrl_')]
    X = sm.add_constant(df_model[X_cols], has_constant='add')
    y = df_model[y_col]

    # Try MixedLM with nested random intercepts if possible
    if group_col is not None:
        try:
            if vc_col is not None and vc_col in df_model.columns and group_col != vc_col:
                # vc formula requires patsy syntax
                data = df_model.copy()
                # MixedLM.from_formula expects variables directly
                formula_rhs = " + ".join(X_cols)
                formula = f"{y_col} ~ {formula_rhs}"
                vc_formula = {"cr": f"0 + C({vc_col})"}
                md = MixedLM.from_formula(formula, groups=data[group_col], vc_formula=vc_formula, re_formula="1", data=data)
            else:
                data = df_model.copy()
                formula_rhs = " + ".join(X_cols)
                formula = f"{y_col} ~ {formula_rhs}"
                md = MixedLM.from_formula(formula, groups=data[group_col], re_formula="1", data=data)
            mdf = md.fit(method='lbfgs', reml=True, maxiter=500, disp=False)
            return {
                'method': 'MixedLM',
                'params': mdf.params.to_dict(),
                'bse': mdf.bse.to_dict(),
                'pvalues': mdf.pvalues.to_dict(),
                'converged': bool(getattr(mdf, 'converged', True)),
                'nobs': int(mdf.nobs),
                'notes': ''
            }
        except Exception as e:
            warnings.warn(f"MixedLM failed for {y_col} with error: {e}. Falling back to OLS with clustered SE if possible.")

    # Fallback: OLS with cluster-robust SE (single cluster on vc_col if available, else country)
    try:
        X = sm.add_constant(df_model[X_cols], has_constant='add')
        ols = OLS(y, X, missing='drop').fit(
            cov_type='cluster',
            cov_kwds={'groups': df_model[vc_col] if (vc_col is not None and vc_col in df_model.columns) else (df_model[group_col] if group_col is not None else None)}
        )
        return {
            'method': 'OLS_clustered',
            'params': ols.params.to_dict(),
            'bse': ols.bse.to_dict(),
            'pvalues': ols.pvalues.to_dict(),
            'nobs': int(ols.nobs),
            'notes': 'Clustered SEs by country-round if available, else country.'
        }
    except Exception as e:
        # Plain OLS
        ols = OLS(y, X, missing='drop').fit()
        return {
            'method': 'OLS',
            'params': ols.params.to_dict(),
            'bse': ols.bse.to_dict(),
            'pvalues': ols.pvalues.to_dict(),
            'nobs': int(ols.nobs),
            'notes': f'Plain OLS due to error: {e}'
        }


def main():
    out = {
        'data_file': None,
        'n_rows_loaded': None,
        'used_columns': {},
        'outcomes_modeled': [],
        'results': {},
        'messages': []
    }

    try:
        data_path = locate_data_file()
        out['data_file'] = os.path.basename(data_path)
    except Exception as e:
        out['messages'].append(f"Data location error: {e}")
        print(json.dumps(out, indent=2))
        return

    try:
        df = load_dataset(data_path)
        out['n_rows_loaded'] = int(df.shape[0])
    except Exception as e:
        out['messages'].append(f"Data load error: {e}")
        print(json.dumps(out, indent=2))
        return

    analysis_df, used_cols, outcomes, group_col, vc_col = build_variables(df)
    out['used_columns'] = used_cols
    out['outcomes_modeled'] = outcomes

    # Fit models for each available outcome
    for outcome in outcomes:
        y_col = f"y_{outcome}"
        try:
            res = fit_model_for_outcome(analysis_df.dropna(subset=[y_col, 'imm_concern']), y_col, group_col, vc_col)
            out['results'][outcome] = res
        except Exception as e:
            out['results'][outcome] = {'error': str(e)}

    # Save results
    results_path = os.path.join(DATA_DIR, "replication_results.json")
    try:
        with open(results_path, 'w') as f:
            json.dump(out, f, indent=2)
    except Exception as e:
        out['messages'].append(f"Failed to write results to {results_path}: {e}")

    # Also write a brief CSV summary (columns used and counts)
    try:
        cols_summary = pd.DataFrame([
            {'logical_name': k, 'dataset_column': v} for k, v in used_cols.items()
        ])
        cols_summary.to_csv(os.path.join(DATA_DIR, 'replication_columns_used.csv'), index=False)
    except Exception as e:
        out['messages'].append(f"Failed to write columns summary CSV: {e}")

    # Print to stdout for convenience
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
