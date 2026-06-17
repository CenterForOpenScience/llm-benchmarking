#!/usr/bin/env python3
"""
Replication script: Democracy and COVID-19 infections (cross-country)
- Reads dataset(s) from /app/data (supports .dta, .rds, .csv)
- Harmonizes variable names
- Runs OLS regressions of cases per million on temperature, precipitation, openness, democracy, and population density
- Saves model outputs to /app/data
"""
import os
import json
import sys
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Prefer statsmodels for OLS with robust SE
import statsmodels.api as sm


def try_read_stata(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_stata(path)
        return df
    except Exception as e:
        # try pyreadstat if available for newer Stata files
        try:
            import pyreadstat
            df, meta = pyreadstat.read_dta(path)
            return df
        except Exception as e2:
            sys.stderr.write(f"Failed to read Stata file {path}: {e}; fallback error: {e2}\n")
            return None


def try_read_rds(path: str) -> Optional[pd.DataFrame]:
    try:
        import pyreadr
        res = pyreadr.read_r(path)
        # take the first object in the RDS
        for k, v in res.items():
            if isinstance(v, pd.DataFrame):
                return v
        # if not found but an object exists, try to convert
        for k, v in res.items():
            try:
                return pd.DataFrame(v)
            except Exception:
                continue
        return None
    except Exception as e:
        sys.stderr.write(f"Failed to read RDS file {path}: {e}\n")
        return None


def try_read_csv(paths: List[str]) -> Optional[pd.DataFrame]:
    for p in paths:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception as e:
                sys.stderr.write(f"Failed to read CSV {p}: {e}\n")
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def clean(c: str) -> str:
        c = c.strip().lower()
        c = re.sub(r"[^0-9a-zA-Z]+", "_", c)
        c = re.sub(r"_+", "_", c)
        c = c.strip("_")
        return c
    df = df.copy()
    df.columns = [clean(c) for c in df.columns]
    return df


def find_first_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for name in candidates:
        if name in cols:
            return name
    # try contains logic
    for cand in candidates:
        for col in df.columns:
            if cand == col:
                return col
    return None


def infer_variables(df: pd.DataFrame) -> Tuple[str, Dict[str, str]]:
    """
    Returns (dv_col, mapping) where mapping maps logical names to actual columns.
    Logical names: temperature, precipitation, openness, democracy, pop_density
    """
    candidates = {
        "dv": [
            "cases_per_million","cases_pm","confirmed_cases_per_million","total_confirmed_cases_per_million",
            "covid_cases_per_million","cases_million","cases_per_1m","cases_per_1_million","casepermillion",
            "case_per_million","casesper1m","cases_per_million_people","cases_per_million_pop","casespm"
        ],
        "total_cases": [
            "total_cases","confirmed","cases","total_confirmed","confirmed_cases",
            # observed in provided dataset
            "covid_12_31_04_03"
        ],
        "population": ["population","pop","population_2018","pop2018","pop_2018","pop_2019","population2018","popdata2019"],
        "temperature": [
            "temperature","avg_temperature","yearly_avg_temperature","temp","mean_temperature",
            # observed in provided dataset
            "annual_temp"
        ],
        "precipitation": ["precipitation","precip","rainfall","yearly_avg_precipitation","avg_precipitation"],
        "openness": [
            "openness","trade_openness","openness_to_trade","trade_gdp","trade_percent_gdp","trade_of_gdp","openness_percent_gdp",
            # observed in provided dataset
            "trade_recent","imputed_trade","trade_2016"
        ],
        "democracy": [
            "democracy","democracy_index","dem_index","eiu_democracy_index","democracyind","democracy_idx",
            # observed in provided dataset (after cleaning)
            "democracy_index_eiu"
        ],
        "pop_density": ["population_density","pop_density","density","popdens","people_per_sq_km","pop_per_sq_km"],
        "country": ["country","name","country_name","country_name","country.code","country_code"]
    }

    mapping: Dict[str, str] = {}

    dv_col = find_first_column(df, candidates["dv"]) or ""

    # Independents
    for logical in ["temperature","precipitation","openness","democracy","pop_density","country"]:
        col = find_first_column(df, candidates[logical])
        if col:
            mapping[logical] = col

    # If dv missing, attempt construct from total_cases and population
    if not dv_col:
        total_cases_col = find_first_column(df, candidates["total_cases"]) or ""
        pop_col = find_first_column(df, candidates["population"]) or ""
        if total_cases_col and pop_col:
            new_dv = "cases_per_million"
            df[new_dv] = (pd.to_numeric(df[total_cases_col], errors="coerce") / pd.to_numeric(df[pop_col], errors="coerce")) * 1e6
            dv_col = new_dv
        else:
            raise ValueError("Could not identify dependent variable or construct it from total cases and population.")

    return dv_col, mapping


def run_ols(df: pd.DataFrame, dv: str, indep_cols: List[str]) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, Dict[str, float]]:
    sub = df[[dv] + indep_cols].copy()
    # coerce to numeric
    for c in [dv] + indep_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    y = sub[dv].astype(float)
    X = sub[indep_cols].astype(float)
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop')
    res = model.fit(cov_type='HC1')  # robust to heteroscedasticity
    metrics = {
        "n_obs": int(res.nobs),
        "r_squared": float(res.rsquared),
        "adj_r_squared": float(res.rsquared_adj),
        "cov_type": res.cov_type
    }
    return res, metrics


def main():
    # Inputs
    candidates = [
        "/app/data/COVID replication.dta",
        "/app/data/COVID_replication.dta",
        "/app/data/covid_replication.dta",
    ]
    dta_path = next((p for p in candidates if os.path.exists(p)), None)

    rds_candidates = [
        "/app/data/COVID replication.rds",
        "/app/data/COVID_replication.rds",
        "/app/data/covid_replication.rds",
    ]
    rds_path = next((p for p in rds_candidates if os.path.exists(p)), None)

    csv_candidates = [
        "/app/data/COVID replication.csv",
        "/app/data/COVID_replication.csv",
        "/app/data/covid_replication.csv",
    ]

    df = None
    if dta_path:
        df = try_read_stata(dta_path)
    if df is None and rds_path:
        df = try_read_rds(rds_path)
    if df is None:
        df = try_read_csv(csv_candidates)
    if df is None:
        raise FileNotFoundError("Could not read dataset. Place a Stata (.dta), RDS (.rds), or CSV version of the COVID replication dataset in /app/data.")

    df = standardize_columns(df)

    # Infer variables
    dv, mapping = infer_variables(df)

    # required independents for baseline model
    required_keys = ["temperature","precipitation","openness","democracy","pop_density"]
    missing = [k for k in required_keys if k not in mapping]
    if missing:
        sys.stderr.write(f"Warning: missing variables inferred for {missing}. The baseline model will omit those not found.\n")
    indep_cols = [mapping[k] for k in required_keys if k in mapping]

    # Run baseline model
    res1, metrics1 = run_ols(df, dv, indep_cols)

    # Prepare refined model by dropping non-significant covariates (p > 0.10), but keep democracy
    pvals = res1.pvalues.to_dict()
    refined_cols = []
    for c in indep_cols:
        if c == mapping.get("democracy", "democracy"):
            refined_cols.append(c)
        else:
            if pvals.get(c, 1.0) <= 0.10:
                refined_cols.append(c)
    # Ensure at least democracy present
    if mapping.get("democracy") not in refined_cols and mapping.get("democracy") in indep_cols:
        refined_cols.append(mapping.get("democracy"))

    res2, metrics2 = run_ols(df, dv, refined_cols)

    # Save outputs
    os.makedirs("/app/data", exist_ok=True)

    # Variable mapping
    mapping_out = {
        "dependent_variable": dv,
        "independent_variables": {k: mapping[k] for k in mapping if k in ["temperature","precipitation","openness","democracy","pop_density"]},
        "country_variable": mapping.get("country")
    }
    with open("/app/data/variable_mapping.json", "w") as f:
        json.dump(mapping_out, f, indent=2)

    # Model summaries
    def serialize_results(res, metrics, model_name, used_cols):
        params = res.params.to_dict()
        bse = res.bse.to_dict()
        tvals = res.tvalues.to_dict()
        pvals = res.pvalues.to_dict()
        return {
            "model": model_name,
            "n_obs": metrics["n_obs"],
            "r_squared": metrics["r_squared"],
            "adj_r_squared": metrics["adj_r_squared"],
            "cov_type": metrics["cov_type"],
            "dependent_variable": dv,
            "independent_variables": used_cols,
            "coefficients": params,
            "std_errors": bse,
            "t_values": tvals,
            "p_values": pvals,
            "democracy_coefficient": params.get(mapping.get("democracy","democracy")),
            "democracy_p_value": pvals.get(mapping.get("democracy","democracy"))
        }

    results_payload = {
        "baseline": serialize_results(res1, metrics1, "baseline_full", indep_cols),
        "refined": serialize_results(res2, metrics2, "refined_selected", refined_cols),
    }

    with open("/app/data/replication_results.json", "w") as f:
        json.dump(results_payload, f, indent=2)

    # Human-readable summaries
    with open("/app/data/replication_model_summary.txt", "w") as f:
        f.write("=== Baseline model (robust SE: HC1) ===\n")
        f.write(str(res1.summary()))
        f.write("\n\n=== Refined model (robust SE: HC1) ===\n")
        f.write(str(res2.summary()))
        f.write("\n")

    print("Replication completed. Outputs saved to /app/data:")
    print("- variable_mapping.json")
    print("- replication_results.json")
    print("- replication_model_summary.txt")


if __name__ == "__main__":
    main()
