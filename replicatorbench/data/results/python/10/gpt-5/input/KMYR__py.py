import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from sklearn.covariance import MinCovDet
from scipy.stats import chi2
from linearmodels.panel import PanelOLS


def find_data_path():
    candidates = [
        "/app/data/finaldata_noNA.csv",
        "/app/data/original/10/python/replication_data/finaldata_noNA.csv",
        "/app/data/replication_data/finaldata_noNA.csv",
        os.path.join(os.path.dirname(__file__), "replication_data", "finaldata_noNA.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find finaldata_noNA.csv. Checked: {candidates}")


def robust_outlier_mask(df, cols, quantile=0.999):
    sub = df[cols].dropna()
    if sub.empty:
        return pd.Series(False, index=df.index)
    try:
        mcd = MinCovDet(random_state=0, support_fraction=None).fit(sub.values)
        d2 = mcd.mahalanobis(sub.values)
        threshold = chi2.ppf(quantile, df=len(cols))
        flags = pd.Series(d2 > threshold, index=sub.index)
        mask = pd.Series(False, index=df.index)
        mask.loc[flags.index] = flags.values
        return mask
    except Exception as e:
        warnings.warn(f"MinCovDet failed ({e}); falling back to no outlier removal.")
        return pd.Series(False, index=df.index)


def make_period_dummies(df):
    # Initialize all to 0
    periods = {
        "DUM70to74": (1970, 1974),
        "DUM75to79": (1975, 1979),
        "DUM80to84": (1980, 1984),
        "DUM85to89": (1985, 1989),
        "DUM90to94": (1990, 1994),
        "DUM95to99": (1995, 1999),
        "DUM00to04": (2000, 2004),
        "DUM05to09": (2005, 2009),
        "DUM10to14": (2010, 2014),
        "DUM15to18": (2015, 2018),
    }
    for name, (lo, hi) in periods.items():
        df[name] = ((df["year"] >= lo) & (df["year"] <= hi)).astype(int)
    return df


def main():
    out_dir = "/app/data/original/10/python"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    data_path = find_data_path()
    log(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    log(f"Loaded shape: {df.shape}; columns: {list(df.columns)}")

    # Expected columns
    required = ["country", "year", "gdp", "pop", "totalimport", "totalexport", "unemp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Construct key variables as in do-file
    df["NAff"] = df["gdp"] / df["pop"]
    # Avoid divide-by-zero for gdp
    df["IMS"] = df.apply(lambda r: np.nan if r["gdp"] == 0 else r["totalimport"] / (r["gdp"] * 10000.0), axis=1)
    df["EXS"] = df.apply(lambda r: np.nan if r["gdp"] == 0 else r["totalexport"] / (r["gdp"] * 10000.0), axis=1)

    # Robust outlier detection on [NAff, IMS, EXS, unemp]
    out_mask = robust_outlier_mask(df, ["NAff", "IMS", "EXS", "unemp"], quantile=0.999)
    n_out = int(out_mask.sum())
    log(f"Flagged outliers: {n_out}")
    df = df.loc[~out_mask].copy()
    log(f"Shape after outlier removal: {df.shape}")

    # Period dummies
    df = make_period_dummies(df)

    # Sort and create within-country lags
    df = df.sort_values(["country", "year"]).copy()
    for var in ["IMS", "EXS", "unemp"]:
        df[f"L_{var}"] = df.groupby("country")[var].shift(1)

    # Panel index
    df = df.set_index(["country", "year"]).sort_index()

    # Define outcome and regressors
    y = df["NAff"]
    dummy_cols = [
        "DUM70to74", "DUM75to79", "DUM80to84", "DUM85to89", "DUM90to94",
        "DUM95to99", "DUM00to04", "DUM05to09", "DUM10to14", "DUM15to18"
    ]
    X = df[["L_IMS", "L_EXS", "L_unemp"] + dummy_cols]

    # Drop missing rows
    valid = y.notna() & X.notna().all(axis=1)
    y = y.loc[valid]
    X = X.loc[valid]
    log(f"Final regression sample: {len(y)} observations; entities: {y.index.get_level_values(0).nunique()}, years: {y.index.get_level_values(1).nunique()}")

    # Fit PanelOLS with entity fixed effects; cluster standard errors by country and year if possible
    res = None
    env_desc = "Python PanelOLS entity FE with clustered SEs"
    try:
        clusters = X.reset_index()[["country", "year"]]
        mod = PanelOLS(y, X, entity_effects=True)
        res = mod.fit(cov_type="clustered", clusters=clusters)
        log("Model fit with two-way clustering by country and year.")
    except Exception as e:
        warnings.warn(f"Two-way clustering failed ({e}); falling back to cluster by country.")
        try:
            clusters = X.reset_index()[["country"]]
            mod = PanelOLS(y, X, entity_effects=True)
            res = mod.fit(cov_type="clustered", clusters=clusters)
            log("Model fit with one-way clustering by country.")
        except Exception as e2:
            warnings.warn(f"Clustered SEs failed ({e2}); falling back to robust.")
            mod = PanelOLS(y, X, entity_effects=True)
            res = mod.fit(cov_type="robust")
            env_desc = "Python PanelOLS entity FE with robust SEs"
            log("Model fit with robust SEs.")

    # Summaries
    summary_txt = res.summary.as_text()
    results_path = os.path.join(out_dir, "replication_results.txt")
    with open(results_path, "w") as f:
        f.write("Replication of NAff ~ L_IMS + L_EXS + L_unemp + period dummies with country FE\n")
        f.write(f"Data: {data_path}\n")
        f.write(f"Outliers removed: {n_out}\n")
        f.write(f"Final N: {len(y)}; Countries: {y.index.get_level_values(0).nunique()}; Years: {y.index.get_level_values(1).nunique()}\n")
        f.write(f"Environment: {env_desc}\n\n")
        f.write(summary_txt)
    log(f"Saved regression summary to {results_path}")

    # Extract key estimates
    params = res.params
    b1 = float(params.get("L_IMS", np.nan))
    se = np.nan
    pval = np.nan
    ci_low = np.nan
    ci_high = np.nan
    try:
        se = float(res.std_errors.get("L_IMS", np.nan))
    except Exception:
        pass
    try:
        pval = float(res.pvalues.get("L_IMS", np.nan))
    except Exception:
        pass
    try:
        ci = res.conf_int().loc["L_IMS"]
        ci_low = float(ci[0])
        ci_high = float(ci[1])
    except Exception:
        pass

    estimates = {
        "outcome": "NAff",
        "predictor": "L_IMS",
        "estimate": b1,
        "std_error": se,
        "p_value": pval,
        "conf_int_95": [ci_low, ci_high],
        "n_obs": int(len(y)),
        "n_countries": int(y.index.get_level_values(0).nunique()),
        "n_years": int(y.index.get_level_values(1).nunique()),
        "outliers_removed": int(n_out),
        "environment": env_desc,
    }

    est_path = os.path.join(out_dir, "replication_estimates.json")
    with open(est_path, "w") as f:
        json.dump(estimates, f, indent=2)
    log(f"Saved estimates to {est_path}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
