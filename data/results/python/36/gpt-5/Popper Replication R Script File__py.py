#!/usr/bin/env python3
"""
Python translation of 'Popper Replication R Script File.R' for replicating Popper & Amit (2009) core SEM analyses.

IO policy: All reads/writes use /app/data.

Functions:
- Loads replication CSVs from /app/data (flexible filename matching for CFA/SEM and correlations files).
- Fits CFA (4-factor) and One-Factor measurement models.
- Fits SEM structural models (Model 1–6 as specified in the R script) using semopy.
- Extracts standardized coefficients and fit indices; saves to /app/data/sem_results.json.
- Computes simple Pearson correlation between Openness total and Leadership experiences total from the correlations dataset; saves to /app/data/cor_results.json and /app/data/cor_matrix.csv.

Note:
- Item-level alpha computations in the R script are omitted because item-level data are not included in provided CSVs; parcels and totals are available.
- Ensure that 'semopy' and 'scipy' are available in the environment.
"""
import os
import json
import glob
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# Optional dependencies
try:
    from semopy import Model, calc_stats
except Exception as e:
    Model = None
    calc_stats = None

try:
    from scipy import stats
except Exception as e:
    stats = None

DATA_DIR = "/app/data"


def find_file(patterns):
    """Find first existing file anywhere under /app/data matching any of the glob patterns (list), recursively."""
    for pat in patterns:
        # Search recursively under DATA_DIR
        search_pattern = os.path.join(DATA_DIR, "**", pat)
        matches = glob.glob(search_pattern, recursive=True)
        if matches:
            # Prefer deterministic ordering
            return sorted(matches)[0]
    return Nonereturn None    return None


def load_cfa_sem_data() -> pd.DataFrame:
    # Try various naming variants for the CFA/SEM file
    candidates = [
        "Popper_Data for CFA and SEM.csv",
        "Popper_Data*SEM.csv",
        "*Popper*SEM*.csv",
        "*CFA*SEM*.csv",
    ]
    fpath = find_file(candidates)
    if not fpath:
        raise FileNotFoundError(
            "Could not locate the CFA/SEM dataset in /app/data. Expected a file like 'Popper_Data for CFA and SEM.csv'."
        )
    df = pd.read_csv(fpath)
    return df


def load_correlations_data() -> Optional[pd.DataFrame]:
    candidates = [
        "Popper Data for Correlations.csv",
        "*Popper*Correlation*.csv",
        "*Correlations*.csv",
    ]
    fpath = find_file(candidates)
    if not fpath:
        return None
    return pd.read_csv(fpath)


def fit_sem(model_desc: str, data: pd.DataFrame) -> Dict[str, Any]:
    if Model is None:
        raise RuntimeError("semopy is not available. Please install semopy.")
    m = Model(model_desc)
    m.fit(data)
    # Extract parameter table with standardized estimates
    try:
        # semopy >= 2.3
        params = m.inspect(std_est=True)
    except Exception:
        # Fallback: try without std, we will compute standardized later if needed
        params = m.inspect()
        params["Std.Estimate"] = params.get("Estimate", np.nan)
    # Compute fit indices
    try:
        fit_stats = calc_stats(m)
        # Convert to plain dict
        fit_dict = {k: (v.item() if hasattr(v, "item") else v) for k, v in fit_stats.items()}
    except Exception:
        fit_dict = {}
    # Convert params to serializable list of dicts
    params_rec = params.to_dict(orient="records") if hasattr(params, "to_dict") else []
    return {"params": params_rec, "fit": fit_dict}


def extract_path(params_rec, lval, op, rval) -> Optional[Dict[str, Any]]:
    for row in params_rec:
        if row.get("lval") == lval and row.get("op") == op and row.get("rval") == rval:
            return row
    return None


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load data
    df_cfa = load_cfa_sem_data()

    # Ensure required columns for parcels exist
    required_cols = [
        "AvoidC_Par1", "AvoidD_Par2", "AttachX_Par3",
        "STAI_Par1", "STAI_Par2", "STAI_Par3",
        "Open_Par1", "Open_Par2", "Open_Par3",
        "Lead_Par1", "Lead_Par2", "Lead_Par3",
    ]
    missing = [c for c in required_cols if c not in df_cfa.columns]
    if missing:
        raise ValueError(f"Missing expected parcel columns in SEM dataset: {missing}")

    # Define measurement models
    popper_measurement = (
        "ATTACH=~ AvoidC_Par1 + AvoidD_Par2 + AttachX_Par3\n"
        "ANXIETY=~ STAI_Par1 + STAI_Par2 + STAI_Par3\n"
        "OPEN=~ Open_Par1 + Open_Par2 + Open_Par3\n"
        "LEAD=~ Lead_Par1 + Lead_Par2 + Lead_Par3\n"
    )

    one_factor_measurement = (
        "OneFactor=~ AvoidC_Par1 + AvoidD_Par2 + AttachX_Par3 +\n"
        "STAI_Par1 + STAI_Par2 + STAI_Par3 + Open_Par1 + Open_Par2 + Open_Par3 +\n"
        "Lead_Par1 + Lead_Par2 + Lead_Par3\n"
    )

    # Structural models (lavaan-like syntax)
    model1 = popper_measurement + (
        "ANXIETY~ a*ATTACH\n"
        "OPEN~ c*ATTACH\n"
        "LEAD~ b*ANXIETY + d*OPEN\n"
        "ab := a*b\n"
        "cd := c*d\n"
    )

    model2 = popper_measurement + (
        "ANXIETY~ a*ATTACH\n"
        "OPEN~ c*ATTACH\n"
        "LEAD~ b*ANXIETY + d*OPEN + e*ATTACH\n"
        "ab := a*b\n"
        "cd := c*d\n"
    )

    model3 = popper_measurement + (
        "LEAD~ ATTACH + OPEN + ANXIETY\n"
    )

    model4 = popper_measurement + (
        "ATTACH~ a*ANXIETY\n"
        "OPEN~ c*ANXIETY\n"
        "LEAD~ b*ATTACH + d*OPEN + e*ANXIETY\n"
        "ab := a*b\n"
        "cd := c*d\n"
    )

    model5 = popper_measurement + (
        "ANXIETY~ a*OPEN\n"
        "ATTACH~ c*OPEN\n"
        "LEAD~ b*ANXIETY + d*ATTACH + e*OPEN\n"
        "ab := a*b\n"
        "cd := c*d\n"
    )

    model6 = popper_measurement + (
        "ANXIETY~~ATTACH\n"
        "OPEN~ b*ATTACH\n"
        "LEAD~ a*ANXIETY + c*OPEN\n"
        "bc := b*c\n"
    )

    results = {"measurement": {}, "structural": {}}

    # Fit measurement models
    try:
        res_meas = fit_sem(popper_measurement, df_cfa)
        results["measurement"]["four_factor"] = res_meas
    except Exception as e:
        results["measurement"]["four_factor_error"] = str(e)

    try:
        res_one = fit_sem(one_factor_measurement, df_cfa)
        results["measurement"]["one_factor"] = res_one
    except Exception as e:
        results["measurement"]["one_factor_error"] = str(e)

    # Fit structural models 1–6
    models = [("model1", model1), ("model2", model2), ("model3", model3),
              ("model4", model4), ("model5", model5), ("model6", model6)]

    for name, desc in models:
        try:
            res = fit_sem(desc, df_cfa)
            # Extract focal path OPEN -> LEAD when present
            path = extract_path(res.get("params", []), lval="LEAD", op="~", rval="OPEN")
            if path:
                res["focal_path_OPEN_to_LEAD"] = {
                    "Estimate": path.get("Estimate"),
                    "Std.Estimate": path.get("Std.Estimate") or path.get("Std. Estimate"),
                    "SE": path.get("SE"),
                    "z": path.get("z"),
                    "p-value": path.get("p-value") or path.get("pval")
                }
            results["structural"][name] = res
        except Exception as e:
            results["structural"][f"{name}_error"] = str(e)

    # Save SEM results
    sem_out = os.path.join(DATA_DIR, "sem_results.json")
    with open(sem_out, "w") as f:
        json.dump(results, f, indent=2)

    # Correlations from totals dataset (optional)
    df_cor = load_correlations_data()
    if df_cor is not None and stats is not None:
        # Standardize column names for robustness
        cols_map = {c.lower().strip(): c for c in df_cor.columns}
        open_col = None
        lead_col = None
        for key in cols_map:
            if "open" in key and "total" in key:
                open_col = cols_map[key]
            if ("leader" in key or "lead" in key) and "total" in key:
                lead_col = cols_map[key]
        cor_results = {}
        if open_col and lead_col:
            x = pd.to_numeric(df_cor[open_col], errors="coerce")
            y = pd.to_numeric(df_cor[lead_col], errors="coerce")
            mask = x.notna() & y.notna()
            r, p = stats.pearsonr(x[mask], y[mask])
            cor_results = {"openness_total_col": open_col, "lead_total_col": lead_col,
                           "n": int(mask.sum()), "pearson_r": float(r), "p_value": float(p)}
        # Also save full correlation matrix of numeric columns
        num_df = df_cor.select_dtypes(include=[np.number])
        cor_mat_path = os.path.join(DATA_DIR, "cor_matrix.csv")
        if not num_df.empty:
            num_df.corr().to_csv(cor_mat_path, index=True)
        cor_out = os.path.join(DATA_DIR, "cor_results.json")
        with open(cor_out, "w") as f:
            json.dump(cor_results, f, indent=2)

    # Done
    print(f"Saved SEM results to {sem_out} and (if available) correlations to /app/data.")


if __name__ == "__main__":
    main()
