import json
import math
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = "/app/data/1-s2.0-S1090513816301118-mmc1.csv"
OUTPUT_PATH = "/app/data/results_task1.json"


def partial_corr_xy_z(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, Any]:
    """
    Compute partial correlation r_{xy.z} and associated t, df, and p-value by
    residualizing x and y on z (with intercept) and correlating residuals.
    df = n - k - 2 where k = 1 (number of controls) => df = n - 3.
    """
    # Ensure 1-D and drop NaNs jointly
    df = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    n = len(df)
    if n < 4:
        return {"n": n, "r": np.nan, "t": np.nan, "df": max(n - 3, np.nan), "p": np.nan}

    X = np.column_stack([np.ones(n), df["z"].to_numpy(dtype=float)])

    # OLS residuals for x ~ 1 + z
    beta_x, *_ = np.linalg.lstsq(X, df["x"].to_numpy(dtype=float), rcond=None)
    x_hat = X @ beta_x
    res_x = df["x"].to_numpy(dtype=float) - x_hat

    # OLS residuals for y ~ 1 + z
    beta_y, *_ = np.linalg.lstsq(X, df["y"].to_numpy(dtype=float), rcond=None)
    y_hat = X @ beta_y
    res_y = df["y"].to_numpy(dtype=float) - y_hat

    # Pearson correlation of residuals
    if res_x.std(ddof=1) == 0 or res_y.std(ddof=1) == 0:
        r = np.nan
        t_stat = np.nan
        p_val = np.nan
        dfree = n - 3
    else:
        r, _ = stats.pearsonr(res_x, res_y)
        dfree = n - 3
        if np.isfinite(r):
            # t for partial correlation
            denom = max(1e-12, 1 - r ** 2)
            t_stat = r * math.sqrt(dfree / denom)
            p_val = 2 * stats.t.sf(abs(t_stat), df=dfree)
        else:
            t_stat = np.nan
            p_val = np.nan

    return {"n": n, "r": float(r) if np.isfinite(r) else np.nan, "t": float(t_stat) if np.isfinite(t_stat) else np.nan, "df": int(dfree), "p": float(p_val) if np.isfinite(p_val) else np.nan}


def main():
    # Read data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Variables
    cols_needed = ["DSM5_Total", "MiniK_Total", "HKSS_Total", "Age"]
    for c in cols_needed:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in dataset.")

    # Cast to numeric (coerce errors) to be safe
    for c in cols_needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Partial correlations controlling for Age
    res_miniK = partial_corr_xy_z(df["DSM5_Total"].values, df["MiniK_Total"].values, df["Age"].values)
    res_hkss = partial_corr_xy_z(df["DSM5_Total"].values, df["HKSS_Total"].values, df["Age"].values)

    # Descriptives
    desc = {}
    for c in ["DSM5_Total", "MiniK_Total", "HKSS_Total", "Age"]:
        series = df[c].dropna()
        desc[c] = {
            "n": int(series.shape[0]),
            "mean": float(series.mean()) if series.shape[0] else np.nan,
            "std": float(series.std(ddof=1)) if series.shape[0] > 1 else np.nan,
            "min": float(series.min()) if series.shape[0] else np.nan,
            "q1": float(series.quantile(0.25)) if series.shape[0] else np.nan,
            "median": float(series.median()) if series.shape[0] else np.nan,
            "q3": float(series.quantile(0.75)) if series.shape[0] else np.nan,
            "max": float(series.max()) if series.shape[0] else np.nan,
        }

    out = {
        "task": "Task1",
        "analysis": "Partial correlation of DSM5_Total with MiniK_Total and HKSS_Total controlling for Age",
        "dataset": os.path.basename(DATA_PATH),
        "results": {
            "DSM5_vs_MiniK__partial_Age": res_miniK,
            "DSM5_vs_HKSS__partial_Age": res_hkss
        },
        "descriptives": desc
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
