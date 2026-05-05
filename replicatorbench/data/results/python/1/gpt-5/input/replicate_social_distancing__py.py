#!/usr/bin/env python3
import os
import sys
import subprocess

# Ensure required Python packages are available under the current interpreter

def ensure_packages():
    try:
        import pandas  # noqa: F401
        import numpy   # noqa: F401
        import statsmodels  # noqa: F401
        import patsy   # noqa: F401
        return
    except Exception as e:
        print(f"Missing deps or import error ({e}). Installing required packages at runtime...")
        cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--timeout", "240",
               "pandas==1.5.3", "numpy==1.23.5", "statsmodels==0.13.5", "patsy==0.5.3"]
        subprocess.run(cmd, check=True)

ensure_packages()

import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.formula.api as smf

DATA_DIR = "/app/data"
RESULTS_SUMMARY_CSV = os.path.join(DATA_DIR, "regression_results_summary.csv")
IQR_EFFECTS_CSV = os.path.join(DATA_DIR, "iqr_effects_trump_share.csv")
MODEL_SUMMARY_TXT = os.path.join(DATA_DIR, "model_summary_March.txt")

MARCH_START = pd.to_datetime("2020-03-19")
MARCH_END = pd.to_datetime("2020-03-28")
AUG_START = pd.to_datetime("2020-08-16")
AUG_END = pd.to_datetime("2020-08-29")
REF_START = pd.to_datetime("2020-02-16")
REF_END = pd.to_datetime("2020-02-29")


def load_data():
    trans_path = os.path.join(DATA_DIR, "transportation.csv")
    cov_path = os.path.join(DATA_DIR, "county_variables.csv")
    os.makedirs(DATA_DIR, exist_ok=True)
    if not (os.path.exists(trans_path) and os.path.exists(cov_path)):
        ws_trans = "/workspace/replication_data/transportation.csv"
        ws_cov = "/workspace/replication_data/county_variables.csv"
        if os.path.exists(ws_trans) and os.path.exists(ws_cov):
            try:
                import shutil
                shutil.copyfile(ws_trans, trans_path)
                shutil.copyfile(ws_cov, cov_path)
                print("Copied input data from /workspace/replication_data to /app/data.")
            except Exception as e:
                print(f"Warning: could not copy data to /app/data ({e}). Will read directly from workspace.")
                trans_path, cov_path = ws_trans, ws_cov
        else:
            raise FileNotFoundError("Expected transportation.csv and county_variables.csv in /app/data or /workspace/replication_data")
    trans = pd.read_csv(trans_path)
    cov = pd.read_csv(cov_path)
    return trans, cov


def prep_transportation(trans: pd.DataFrame) -> pd.DataFrame:
    trans = trans.copy()
    trans["date"] = pd.to_datetime(trans["date"])  # ISO format expected

    denom = trans["pop_home"].fillna(0) + trans["pop_not_home"].fillna(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        trans["prop_home"] = np.where(denom > 0, trans["pop_home"] / denom, np.nan)

    def period_label(d):
        if REF_START <= d <= REF_END:
            return "Reference"
        if MARCH_START <= d <= MARCH_END:
            return "March"
        if AUG_START <= d <= AUG_END:
            return "August"
        return None

    trans["period"] = trans["date"].apply(period_label)
    trans = trans[trans["period"].notna()].copy()

    grp = trans.groupby(["fips", "state", "period"], as_index=False)["prop_home"].mean()
    wide = grp.pivot_table(index=["fips", "state"], columns="period", values="prop_home").reset_index()

    for per in ["March", "August"]:
        if per in wide.columns and "Reference" in wide.columns:
            wide[f"prop_home_change_{per}"] = 100.0 * (wide[per] / wide["Reference"] - 1.0)
        else:
            wide[f"prop_home_change_{per}"] = np.nan

    return wide


def prep_covariates(cov: pd.DataFrame) -> pd.DataFrame:
    cov = cov.copy()
    cov["income_per_capita_thousands"] = cov["income_per_capita"] / 1000.0
    cov["percent_college_prop"] = cov["percent_college"] / 100.0

    keep_cols = [
        "fips", "state_po", "trump_share", "income_per_capita_thousands",
        "percent_male", "percent_black", "percent_hispanic", "percent_college_prop",
        "percent_retail", "percent_transportation", "percent_hes", "percent_rural",
        "percent_under_5", "percent_5_9", "percent_10_14", "percent_15_19",
        "percent_20_24", "percent_25_34", "percent_35_44", "percent_45_54",
        "percent_55_59", "percent_60_64", "percent_65_74", "percent_75_84", "percent_85_over"
    ]
    if "state_po" not in cov.columns:
        cov["state_po"] = np.nan
    for col in keep_cols:
        if col not in cov.columns:
            cov[col] = np.nan
    cov = cov[keep_cols].rename(columns={"state_po": "state"})
    return cov


def fit_model(df: pd.DataFrame, outcome_col: str):
    model_df = df[[outcome_col, "trump_share", "income_per_capita_thousands", "percent_male", "percent_black",
                   "percent_hispanic", "percent_college_prop", "percent_retail", "percent_transportation",
                   "percent_hes", "percent_rural", "state",
                   "percent_under_5", "percent_5_9", "percent_10_14", "percent_15_19",
                   "percent_20_24", "percent_25_34", "percent_35_44", "percent_45_54",
                   "percent_55_59", "percent_60_64", "percent_65_74", "percent_75_84", "percent_85_over"
                   ]].dropna()

    num_cols = [c for c in model_df.columns if c != "state"]
    model_df[num_cols] = model_df[num_cols].apply(pd.to_numeric, errors='coerce')
    model_df = model_df.dropna()

    predictors = [
        "trump_share", "income_per_capita_thousands", "percent_male", "percent_black",
        "percent_hispanic", "percent_college_prop", "percent_retail", "percent_transportation",
        "percent_hes", "percent_rural",
        "percent_under_5", "percent_5_9", "percent_10_14", "percent_15_19",
        "percent_20_24", "percent_25_34", "percent_35_44", "percent_45_54",
        "percent_55_59", "percent_60_64", "percent_65_74", "percent_75_84", "percent_85_over",
        "C(state)"
    ]
    formula = f"{outcome_col} ~ " + " + ".join(predictors)

    model = smf.ols(formula=formula, data=model_df).fit(cov_type='HC1')
    return model, model_df


def compute_iqr_effect(model, var_name: str, base_df: pd.DataFrame):
    q1, q3 = base_df[var_name].quantile([0.25, 0.75])
    iqr = q3 - q1
    coef = model.params.get(var_name, np.nan)
    se = model.bse.get(var_name, np.nan)
    iqr_effect = coef * iqr
    iqr_se = se * iqr
    ci_low = iqr_effect - 1.96 * iqr_se
    ci_high = iqr_effect + 1.96 * iqr_se
    pval = model.pvalues.get(var_name, np.nan)
    return {
        "variable": var_name,
        "iqr": iqr,
        "coef": coef,
        "se": se,
        "iqr_effect": iqr_effect,
        "iqr_se": iqr_se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": pval
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    trans, cov = load_data()
    trans_wide = prep_transportation(trans)
    covars = prep_covariates(cov)

    merged = pd.merge(trans_wide, covars, on="fips", how="right")
    if "state_y" in merged.columns and "state_x" in merged.columns:
        merged["state"] = merged["state_y"].combine_first(merged["state_x"])
    elif "state_y" in merged.columns:
        merged["state"] = merged["state_y"]
    for col in ["state_x", "state_y"]:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)

    outcome = "prop_home_change_March"
    model, model_df = fit_model(merged, outcome)

    with open(MODEL_SUMMARY_TXT, "w") as f:
        f.write(model.summary().as_text())

    key_vars = ["trump_share", "income_per_capita_thousands", "percent_male", "percent_black",
                "percent_hispanic", "percent_college_prop", "percent_retail", "percent_transportation",
                "percent_hes", "percent_rural"]
    rows = []
    for v in key_vars:
        if v in model.params.index:
            rows.append({
                "variable": v,
                "coef": model.params[v],
                "se": model.bse[v],
                "p_value": model.pvalues[v]
            })
    pd.DataFrame(rows).to_csv(RESULTS_SUMMARY_CSV, index=False)

    iqr_res = compute_iqr_effect(model, "trump_share", model_df)
    pd.DataFrame([iqr_res]).to_csv(IQR_EFFECTS_CSV, index=False)

    print("Analysis complete.")
    print(f"Saved: {RESULTS_SUMMARY_CSV}, {IQR_EFFECTS_CSV}, {MODEL_SUMMARY_TXT}")


if __name__ == "__main__":
    main()
