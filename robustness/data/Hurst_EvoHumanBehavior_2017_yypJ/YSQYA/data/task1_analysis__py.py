import json
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "/app/data/1-s2.0-S1090513816301118-mmc1.csv"
OUT_PATH = "/app/data/task1_results.json"

REQ_VARS = ["DSM5_Total", "MiniK_Total", "HKSS_Total", "Age"]


def load_data(path):
    df = pd.read_csv(path)
    return df


def drop_missing(df, cols):
    return df.dropna(subset=cols).copy()


def partial_corr_resid(x, y, controls):
    # Regress out controls from x and y, correlate residuals
    Xc = sm.add_constant(controls)
    model_x = sm.OLS(x, Xc, missing='drop').fit()
    model_y = sm.OLS(y, Xc, missing='drop').fit()
    rx = x - model_x.fittedvalues
    ry = y - model_y.fittedvalues
    # Align indices
    common = rx.dropna().index.intersection(ry.dropna().index)
    rx = rx.loc[common]
    ry = ry.loc[common]
    if len(common) < 3:
        return np.nan, np.nan, len(common)
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p), int(len(common))


def ols_with_age(df, y_var, x_var, age_var):
    sub = df.dropna(subset=[y_var, x_var, age_var]).copy()
    if sub.shape[0] < 3:
        return {
            "n": int(sub.shape[0]),
            "coef": np.nan,
            "se": np.nan,
            "t": np.nan,
            "p": np.nan,
            "beta_std": np.nan
        }
    Y = sub[y_var].astype(float)
    X = sm.add_constant(sub[[x_var, age_var]].astype(float))
    model = sm.OLS(Y, X).fit()
    coef = model.params.get(x_var, np.nan)
    se = model.bse.get(x_var, np.nan)
    tval = model.tvalues.get(x_var, np.nan)
    pval = model.pvalues.get(x_var, np.nan)
    # standardized beta: coef * sd(X)/sd(Y)
    sd_x = float(np.nanstd(sub[x_var].astype(float), ddof=1))
    sd_y = float(np.nanstd(sub[y_var].astype(float), ddof=1))
    beta_std = float(coef * sd_x / sd_y) if (sd_x > 0 and sd_y > 0 and pd.notnull(coef)) else np.nan
    return {
        "n": int(sub.shape[0]),
        "coef": float(coef) if pd.notnull(coef) else np.nan,
        "se": float(se) if pd.notnull(se) else np.nan,
        "t": float(tval) if pd.notnull(tval) else np.nan,
        "p": float(pval) if pd.notnull(pval) else np.nan,
        "beta_std": beta_std
    }


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df = load_data(DATA_PATH)

    # Ensure required columns exist
    missing = [c for c in REQ_VARS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Coerce to numeric where appropriate
    for c in REQ_VARS:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Partial correlations controlling for Age
    pc_minik = {
        "pair": ["MiniK_Total", "DSM5_Total"],
        "control": ["Age"]
    }
    r_minik, p_minik, n_minik = partial_corr_resid(df["MiniK_Total"], df["DSM5_Total"], df[["Age"]])
    pc_minik.update({"r": r_minik, "p": p_minik, "n": n_minik})

    pc_hkss = {
        "pair": ["HKSS_Total", "DSM5_Total"],
        "control": ["Age"]
    }
    r_hkss, p_hkss, n_hkss = partial_corr_resid(df["HKSS_Total"], df["DSM5_Total"], df[["Age"]])
    pc_hkss.update({"r": r_hkss, "p": p_hkss, "n": n_hkss})

    # OLS regressions with Age control
    ols_minik = ols_with_age(df, "DSM5_Total", "MiniK_Total", "Age")
    ols_minik.update({"y": "DSM5_Total", "x": "MiniK_Total", "controls": ["Age"]})

    ols_hkss = ols_with_age(df, "DSM5_Total", "HKSS_Total", "Age")
    ols_hkss.update({"y": "DSM5_Total", "x": "HKSS_Total", "controls": ["Age"]})

    results = {
        "task": "Task1",
        "dataset": os.path.basename(DATA_PATH),
        "partial_correlations": {
            "MiniK_Total__DSM5_Total|Age": pc_minik,
            "HKSS_Total__DSM5_Total|Age": pc_hkss
        },
        "ols": {
            "DSM5_on_MiniK_Age": ols_minik,
            "DSM5_on_HKSS_Age": ols_hkss
        }
    }

    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(json.dumps({
        "status": "ok",
        "message": "Task1 completed",
        "output": OUT_PATH
    }))


if __name__ == "__main__":
    main()
