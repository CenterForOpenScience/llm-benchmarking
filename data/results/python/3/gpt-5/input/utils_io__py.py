import os
import re
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    import pyreadr  # type: ignore
    HAS_PYREADR = True
except Exception:
    HAS_PYREADR = False


def normalize_col(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def build_norm_map(df: pd.DataFrame) -> dict:
    return {normalize_col(c): c for c in df.columns}


def find_col(df: pd.DataFrame, candidates: list) -> str:
    norm_map = build_norm_map(df)
    norm_cands = [normalize_col(c) for c in candidates]
    # Exact normalized match first
    for nc in norm_cands:
        if nc in norm_map:
            return norm_map[nc]
    # Fuzzy token containment as fallback
    for nc in norm_cands:
        tokens = [t for t in nc.split("_") if t]
        for norm_col, orig_col in norm_map.items():
            if all(t in norm_col for t in tokens):
                return orig_col
    return None


def try_read_rds(path: str) -> pd.DataFrame:
    if not HAS_PYREADR:
        return None
    res = pyreadr.read_r(path)
    for _, v in res.items():
        if isinstance(v, pd.DataFrame):
            return v
    return None


def load_dataset() -> tuple:
    candidates = [
        "/app/data/COVID replication.rds",
        "/app/data/replication_data/COVID replication.rds",
        "/app/data/COVID_replication.rds",
        "/app/data/replication_data/COVID_replication.rds",
        "/app/data/COVID replication.dta",
        "/app/data/replication_data/COVID replication.dta",
        "/app/data/COVID_replication.dta",
        "/app/data/replication_data/COVID_replication.dta",
    ]
    for p in candidates:
        if os.path.exists(p):
            if p.lower().endswith(".rds"):
                df = try_read_rds(p)
                if df is not None:
                    return df, p
            elif p.lower().endswith(".dta"):
                df = pd.read_stata(p)
                return df, p
    raise FileNotFoundError(
        'Could not find dataset in /app/data. Expected "COVID replication.rds" or ".dta".'
    )


def robust_ols(y: pd.Series, X: pd.DataFrame, cov_type: str = "HC1"):
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, Xc, missing="drop")
    fit = model.fit()
    rob = fit.get_robustcov_results(cov_type=cov_type)
    return rob


def export_regression_csv(results, out_csv_path: str) -> pd.DataFrame:
    rows = []
    params = results.params
    bse = results.bse
    tvals = results.tvalues
    pvals = results.pvalues
    conf = results.conf_int(alpha=0.05)
    for term in params.index:
        ci_low, ci_high = conf.loc[term, 0], conf.loc[term, 1]
        rows.append(
            {
                "term": term,
                "coef": float(params[term]),
                "std_err": float(bse[term]),
                "t": float(tvals[term]),
                "p_value": float(pvals[term]),
                "conf_low": float(ci_low),
                "conf_high": float(ci_high),
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv_path, index=False)
    return out_df


def append_model_summary(out_csv_path: str, r2: float, nobs: int) -> str:
    summ_path = out_csv_path.replace(".csv", "_model_summary.json")
    meta = {"r_squared": float(r2), "nobs": int(nobs)}
    with open(summ_path, "w") as f:
        json.dump(meta, f, indent=2)
    return summ_path


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# Override load_dataset to also search /workspace paths (bundled data)
def load_dataset() -> tuple:
    candidates = [
        "/app/data/COVID replication.rds",
        "/app/data/replication_data/COVID replication.rds",
        "/app/data/COVID_replication.rds",
        "/app/data/replication_data/COVID_replication.rds",
        "/app/data/COVID replication.dta",
        "/app/data/replication_data/COVID replication.dta",
        "/app/data/COVID_replication.dta",
        "/app/data/replication_data/COVID_replication.dta",
        # Also check repository-mounted workspace locations
        "/workspace/replication_data/COVID replication.rds",
        "/workspace/COVID replication.rds",
        "/workspace/replication_data/COVID_replication.rds",
        "/workspace/COVID_replication.rds",
        "/workspace/replication_data/COVID replication.dta",
        "/workspace/COVID replication.dta",
        "/workspace/replication_data/COVID_replication.dta",
        "/workspace/COVID_replication.dta",
    ]
    for p in candidates:
        if os.path.exists(p):
            if p.lower().endswith(".rds"):
                df = try_read_rds(p)
                if df is not None:
                    return df, p
            elif p.lower().endswith(".dta"):
                df = pd.read_stata(p)
                return df, p
    raise FileNotFoundError(
        'Could not find dataset. Searched /app/data and /workspace for "COVID replication.rds" or .dta variants.'
    )

# Override export_regression_csv to handle both numpy arrays and pandas Series from statsmodels
def export_regression_csv(results, out_csv_path: str) -> pd.DataFrame:
    import numpy as _np
    import pandas as _pd

    params = results.params
    # Determine parameter names and numeric arrays
    if hasattr(params, "index"):
        names = list(params.index)
        params_vals = _np.asarray(params)
    else:
        try:
            names = list(results.model.exog_names)
        except Exception:
            names = [f"param_{i}" for i in range(len(_np.asarray(params)))]
        params_vals = _np.asarray(params)

    bse = _np.asarray(results.bse)
    tvals = _np.asarray(results.tvalues)
    pvals = _np.asarray(results.pvalues)
    conf = results.conf_int(alpha=0.05)
    conf_is_df = hasattr(conf, "loc")

    rows = []
    for i, term in enumerate(names):
        coef = float(params_vals[i])
        se = float(bse[i] if bse.ndim > 0 else bse)
        tstat = float(tvals[i] if tvals.ndim > 0 else tvals)
        pval = float(pvals[i] if pvals.ndim > 0 else pvals)
        if conf_is_df:
            ci_low = float(conf.loc[term, 0])
            ci_high = float(conf.loc[term, 1])
        else:
            ci_low = float(conf[i, 0])
            ci_high = float(conf[i, 1])
        rows.append(
            {
                "term": term,
                "coef": coef,
                "std_err": se,
                "t": tstat,
                "p_value": pval,
                "conf_low": ci_low,
                "conf_high": ci_high,
            }
        )

    out_df = _pd.DataFrame(rows)
    out_df.to_csv(out_csv_path, index=False)
    return out_df
