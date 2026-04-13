import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Modeling
try:
    from linearmodels.panel import PanelOLS
except Exception as e:
    PanelOLS = None
from statsmodels.tools.tools import add_constant


def safe_log(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    s = s.where(s > 0, np.nan)
    return np.log(s)


def load_data(data_dir: Path) -> dict:
    compiled_path = data_dir / "compiled.dta"
    epa_path = data_dir / "epa.dta"
    hhsize_path = data_dir / "hhsize.dta"

    if not compiled_path.exists() or not epa_path.exists():
        raise FileNotFoundError("Missing required input files in /app/data: compiled.dta and/or epa.dta")

    compiled = pd.read_stata(compiled_path, convert_categoricals=False)
    epa = pd.read_stata(epa_path, convert_categoricals=False)

    # Attempt to load and reshape hhsize long as in R script
    hh_long = None
    if hhsize_path.exists():
        hh = pd.read_stata(hhsize_path, convert_categoricals=False)
        # Identify wide columns like hhsize07..hhsize16
        hh_cols = [c for c in hh.columns if c.lower().startswith("hhsize")]
        id_vars = []
        for cand in ["State", "state_id_no", "state_fip", "statefip"]:
            if cand in hh.columns:
                id_vars.append(cand)
        if hh_cols:
            long = hh.melt(id_vars=id_vars, value_vars=hh_cols, var_name="var", value_name="hhsize")
            # Map var like 'hhsize07' -> year 7
            def map_year(v):
                try:
                    suffix = ''.join([ch for ch in str(v) if ch.isdigit()])
                    return int(suffix)
                except Exception:
                    return np.nan
            long['year'] = long['var'].apply(map_year).astype('float').astype('Int64')
            # Ensure we have State and year
            if 'State' not in long.columns and 'state' in long.columns:
                long.rename(columns={'state': 'State'}, inplace=True)
            hh_long = long[['State', 'year', 'hhsize']].dropna(subset=['State', 'year'])

    return {"compiled": compiled, "epa": epa, "hh_long": hh_long}


def prepare_dataset(compiled: pd.DataFrame, epa: pd.DataFrame, hh_long: Optional[pd.DataFrame]) -> pd.DataFrame:
    # Ensure year is integer-ish
    for df in (compiled, epa):
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')

    # Merge compiled with EPA by State, year (left join to keep compiled scope)
    data = compiled.merge(epa[['State', 'year', 'epa']], on=['State', 'year'], how='left')

    # Merge hhsize if available; otherwise proceed without it
    if hh_long is not None and not hh_long.empty:
        data = data.merge(hh_long, on=['State', 'year'], how='left')
    else:
        data['hhsize'] = np.nan

    # Derived variables as in R
    # Employed population %: emppop / (pop*1000) * 100
    if all(col in data.columns for col in ['emppop', 'pop']):
        data['emppop_pct'] = data['emppop'] / (data['pop'] * 1000.0) * 100.0
    else:
        data['emppop_pct'] = np.nan

    # Manufacturing % of GDP: manuf / gdp * 100
    if all(col in data.columns for col in ['manuf', 'gdp']):
        data['manu_gdp'] = data['manuf'] / data['gdp'] * 100.0
    else:
        data['manu_gdp'] = np.nan

    # Log transform continuous variables as per R script
    log_vars = ["epa", "wrkhrs", "emppop_pct", "laborprod", "pop", "manu_gdp", "energy", "hhsize", "workpop"]
    for v in log_vars:
        if v in data.columns:
            data[v] = safe_log(data[v])
        else:
            data[v] = np.nan

    # Sort for stability
    data = data.sort_values(['State', 'year']).reset_index(drop=True)
    return data


def sample_states(data: pd.DataFrame, n_states: int = 5, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    states = pd.Series(data['State'].dropna().unique())
    if len(states) <= n_states:
        return states.tolist()
    chosen = states.sample(n=n_states, random_state=seed)
    return chosen.tolist()


def fit_panel_ols(df: pd.DataFrame, dep: str, indep_vars: list, cluster_entity: str = 'State'):
    # Prepare panel structure
    df = df.copy()
    # Drop NA in model columns
    cols_needed = [dep] + indep_vars + ['State', 'year']
    df = df[cols_needed].dropna()
    if df.empty:
        return None, "No data after dropping NA for variables: {}".format(cols_needed)

    # Index for panel
    df = df.set_index(['State', 'year'])

    y = df[dep]
    X = df[indep_vars]

    # FEs via entity_effects and time_effects
    if PanelOLS is None:
        return None, "linearmodels is not available"

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    try:
        # Cluster by entity (State)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        return res, None
    except Exception as e:
        return None, str(e)


def run_models_and_save(data: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select independent variables (exclude hhsize if it's mostly missing)
    indep_all = ["wrkhrs", "emppop_pct", "laborprod", "pop", "manu_gdp", "energy", "workpop"]
    # Include hhsize only if at least 80% non-missing
    if 'hhsize' in data.columns and data['hhsize'].notna().mean() > 0.8:
        indep_all.append('hhsize')

    dep = 'epa'

    # Sample 5 states dataset
    five_states = sample_states(data, n_states=5, seed=42)
    sampledata = data[data['State'].isin(five_states)].copy()

    # Helper to save results
    def save_result(res, name_prefix):
        out_txt = out_dir / f"{name_prefix}_summary.txt"
        out_json = out_dir / f"{name_prefix}_coefs.json"
        if res is None:
            with open(out_txt, 'w') as f:
                f.write("Model failed or no data")
            with open(out_json, 'w') as f:
                json.dump({}, f)
            return
        # Save text summary
        try:
            with open(out_txt, 'w') as f:
                f.write(str(res.summary))
        except Exception:
            with open(out_txt, 'w') as f:
                f.write(str(res))
        # Save coefficients and stats
        coefs = {}
        try:
            params = res.params
            bse = res.std_errors if hasattr(res, 'std_errors') else res.std_errors
            pvals = res.pvalues
            for k in params.index:
                coefs[k] = {
                    'coef': float(params[k]),
                    'se': float(bse[k]) if k in bse.index else None,
                    'pval': float(pvals[k]) if k in pvals.index else None
                }
        except Exception:
            pass
        with open(out_json, 'w') as f:
            json.dump(coefs, f, indent=2)

    # Define subsets by year threshold 14 as in R (years coded 7..16)
    year_var = 'year'

    # Model 1: sample all years
    res1, err1 = fit_panel_ols(sampledata, dep, indep_all)
    if err1:
        with open(out_dir / 'model1_error.txt', 'w') as f:
            f.write(err1)
    save_result(res1, 'model1_sample_all')

    # Model 2: sample, years < 14
    sd2 = sampledata[sampledata[year_var] < 14]
    res2, err2 = fit_panel_ols(sd2, dep, indep_all)
    if err2:
        with open(out_dir / 'model2_error.txt', 'w') as f:
            f.write(err2)
    save_result(res2, 'model2_sample_year_lt14')

    # Model 3: sample, years > 13
    sd3 = sampledata[sampledata[year_var] > 13]
    res3, err3 = fit_panel_ols(sd3, dep, indep_all)
    if err3:
        with open(out_dir / 'model3_error.txt', 'w') as f:
            f.write(err3)
    save_result(res3, 'model3_sample_year_gt13')

    # Model 4: full data all years
    res4, err4 = fit_panel_ols(data, dep, indep_all)
    if err4:
        with open(out_dir / 'model4_error.txt', 'w') as f:
            f.write(err4)
    save_result(res4, 'model4_full_all')

    # Model 5: full data years < 14
    d5 = data[data[year_var] < 14]
    res5, err5 = fit_panel_ols(d5, dep, indep_all)
    if err5:
        with open(out_dir / 'model5_error.txt', 'w') as f:
            f.write(err5)
    save_result(res5, 'model5_full_year_lt14')

    # Model 6: full data years > 13
    d6 = data[data[year_var] > 13]
    res6, err6 = fit_panel_ols(d6, dep, indep_all)
    if err6:
        with open(out_dir / 'model6_error.txt', 'w') as f:
            f.write(err6)
    save_result(res6, 'model6_full_year_gt13')

    # Write a simple manifest of outputs
    manifest = {
        'models': [
            'model1_sample_all',
            'model2_sample_year_lt14',
            'model3_sample_year_gt13',
            'model4_full_all',
            'model5_full_year_lt14',
            'model6_full_year_gt13'
        ]
    }
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    data_dir = Path("/app/data")
    artifacts_dir = data_dir / "artifacts"

    loaded = load_data(data_dir)
    data = prepare_dataset(loaded['compiled'], loaded['epa'], loaded['hh_long'])

    run_models_and_save(data, artifacts_dir)


if __name__ == "__main__":
    main()
