import os
import json
import sys
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
# Replaced scikit-learn PCA and StandardScaler with NumPy-based implementation
import statsmodels.api as sm

warnings.filterwarnings("ignore")

DATA_DIR = "/app/data"
OUTPUT_DIR = os.path.join(DATA_DIR, "replication_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg: str):
    print(msg)
    sys.stdout.flush()


def safe_read_stata(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_stata(path, convert_categoricals=False)
        return df
    except Exception as e:
        log(f"ERROR reading Stata file {path}: {e}")
        return None


def infer_id_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    # Try to infer country, election-year, and election-date columns
    candidates_country = [
        "country", "ctry", "cname", "countryname", "state"
    ]
    candidates_year = [
        "year", "election_year", "elec_year", "yr"
    ]
    candidates_date = [
        "elect", "election_date", "date", "edate"
    ]
    country_col = next((c for c in candidates_country if c in df.columns), None)
    year_col = next((c for c in candidates_year if c in df.columns), None)
    date_col = next((c for c in candidates_date if c in df.columns), None)
    if country_col is None:
        raise ValueError("Could not find country identifier column.")
    if year_col is None and date_col is None:
        raise ValueError("Could not find election year or election date column.")
    return country_col, year_col, date_col


def find_position_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    # Try to find precomputed policy dimensions in common names
    econ_candidates = [
        "econ", "economic", "econ_dim", "economic_dimension", "econ_pos"
    ]
    soc_candidates = [
        "soc", "social", "soc_dim", "social_dimension", "soc_pos"
    ]
    econ_col = next((c for c in econ_candidates if c in df.columns), None)
    soc_col = next((c for c in soc_candidates if c in df.columns), None)
    return econ_col, soc_col


def select_cmp_policy_columns(df: pd.DataFrame) -> List[str]:
    # CMP policy variables often look like per101, per102, ..., or have names starting with 'per'/'cmp' and numeric suffixes
    cols = []
    for c in df.columns:
        lc = str(c).lower()
        if lc.startswith("per") or lc.startswith("cmp") or lc.startswith("cat"):
            # numeric columns only
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
        # Also include some other common CMP share columns
        if any(tok in lc for tok in ["share", "percent", "prop"]) and pd.api.types.is_numeric_dtype(df[c]):
            # exclude obviously non-policy like 'prop' from CPDS merge later by size check
            cols.append(c)
    # Deduplicate
    cols = list(dict.fromkeys(cols))
    # Heuristic: exclude columns with very few non-missing values
    cols = [c for c in cols if df[c].notna().sum() > 0.5 * len(df)]
    return cols


def build_positions_with_pca(df: pd.DataFrame, policy_cols: List[str]) -> Tuple[pd.Series, pd.Series]:
    # Standardize then PCA to two components using NumPy (no scikit-learn dependency)
    X = df[policy_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how='all')
    # Mean imputation per column
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
    # Standardize columns to mean 0, std 1
    means = X.mean(axis=0)
    stds = X.std(axis=0).replace(0, 1.0)
    Xs = (X - means) / stds
    Xs = Xs.values
    # Compute covariance matrix
    cov = np.cov(Xs, rowvar=False)
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort eigenvectors by descending eigenvalues
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    # Project data onto the top 2 principal components
    comps = Xs.dot(eigvecs[:, :2])
    econ_scores = pd.Series(comps[:, 0], index=df.index)
    soc_scores = pd.Series(comps[:, 1], index=df.index)
    return econ_scores, soc_scores

def compute_seat_shares(df: pd.DataFrame) -> Optional[pd.Series]:
    # Try to construct seat shares per country-election
    seat_candidates = ["seats", "seat", "seat_count", "seats_total", "parl_seats", "legseats"]
    vote_candidates = ["votes", "vote_share", "voteshare", "vote", "pct_vote"]
    seat_col = next((c for c in seat_candidates if c in df.columns), None)
    if seat_col is not None:
        # Need a group by country-election
        try:
            country_col, year_col, date_col = infer_id_columns(df)
        except Exception:
            return None
        if date_col and np.issubdtype(df[date_col].dtype, np.datetime64):
            grp_keys = [country_col, date_col]
        else:
            grp_keys = [country_col, year_col] if year_col else [country_col]
        shares = df.groupby(grp_keys)[seat_col].transform(lambda s: s / s.sum())
        return shares
    # fallback to vote share if seats missing
    vote_col = next((c for c in vote_candidates if c in df.columns), None)
    if vote_col is not None:
        try:
            country_col, year_col, date_col = infer_id_columns(df)
        except Exception:
            return None
        if date_col and np.issubdtype(df[date_col].dtype, np.datetime64):
            grp_keys = [country_col, date_col]
        else:
            grp_keys = [country_col, year_col] if year_col else [country_col]
        shares = df.groupby(grp_keys)[vote_col].transform(lambda s: s / s.sum())
        return shares
    return None


def compute_party_counts(df: pd.DataFrame, seat_shares: Optional[pd.Series]) -> Tuple[pd.Series, pd.Series]:
    # Effective number of parties and relevance count (>=1% share in current election as approximation)
    try:
        country_col, year_col, date_col = infer_id_columns(df)
    except Exception as e:
        raise e
    if date_col and np.issubdtype(df[date_col].dtype, np.datetime64):
        grp_keys = [country_col, date_col]
    else:
        grp_keys = [country_col, year_col] if year_col else [country_col]

    if seat_shares is None:
        # default to simple count of parties per election
        log("Seat shares not found; defaulting to simple count per election.")
        count = df.groupby(grp_keys).size()
        enp = count.copy().astype(float)
        # broadcast back to rows
        enp_series = df.set_index(grp_keys).index.map(dict(enp)).astype(float)
        count_series = df.set_index(grp_keys).index.map(dict(count)).astype(float)
        return enp_series, count_series

    # ENP = 1 / sum(s_i^2)
    enp = df.assign(_share=seat_shares).groupby(grp_keys)["_share"].agg(lambda s: 1.0 / np.sum(np.square(s)))
    enp_series = df.set_index(grp_keys).index.map(dict(enp)).astype(float)
    # relevance count >= 1%
    rel = df.assign(_share=seat_shares, _rel=lambda d: (d["_share"] >= 0.01).astype(int))
    rel_count = rel.groupby(grp_keys)["_rel"].sum()
    count_series = df.set_index(grp_keys).index.map(dict(rel_count)).astype(float)
    return enp_series, count_series


def compute_dispersion(df: pd.DataFrame, econ_col: str, soc_col: str) -> pd.DataFrame:
    country_col, year_col, date_col = infer_id_columns(df)
    if date_col and np.issubdtype(df[date_col].dtype, np.datetime64):
        grp_keys = [country_col, date_col]
    else:
        grp_keys = [country_col, year_col] if year_col else [country_col]
    grp = df.groupby(grp_keys)
    disp_econ = grp[econ_col].agg(lambda s: s.max() - s.min())
    disp_soc = grp[soc_col].agg(lambda s: s.max() - s.min())
    out = pd.DataFrame({
        "disp_econ": disp_econ,
        "disp_soc": disp_soc,
    }).reset_index()
    return out


def merge_controls(disp_df: pd.DataFrame, cpds: Optional[pd.DataFrame]) -> pd.DataFrame:
    if cpds is None:
        return disp_df
    # Harmonize keys: merge on country + year if possible, else nearest election date
    country_col_d, year_col_d, date_col_d = infer_id_columns(disp_df)
    country_col_c, year_col_c, date_col_c = infer_id_columns(cpds)

    df = disp_df.copy()
    c = cpds.copy()
    # Ensure year as int where available
    if year_col_d and df[year_col_d].dtype != int:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df[year_col_d] = pd.to_numeric(df[year_col_d], errors='coerce').astype("Int64")
    if year_col_c and c[year_col_c].dtype != int:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c[year_col_c] = pd.to_numeric(c[year_col_c], errors='coerce').astype("Int64")

    # Prefer merge on country + year
    if year_col_d and year_col_c and df[year_col_d].notna().any() and c[year_col_c].notna().any():
        m = pd.merge(df, c, left_on=[country_col_d, year_col_d], right_on=[country_col_c, year_col_c], how='left', suffixes=("", "_cpds"))
        return m

    # Fallback: if both have election date, merge on exact; else return df
    if date_col_d and date_col_c and np.issubdtype(df[date_col_d].dtype, np.datetime64) and np.issubdtype(c[date_col_c].dtype, np.datetime64):
        m = pd.merge(df, c, left_on=[country_col_d, date_col_d], right_on=[country_col_c, date_col_c], how='left', suffixes=("", "_cpds"))
        return m
    return df


def fit_ols_cluster(df: pd.DataFrame, y: str, xvars: List[str], cluster: str):
    d = df[[y] + xvars + [cluster]].dropna()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return None
    X = sm.add_constant(d[xvars])
    model = sm.OLS(d[y], X)
    try:
        res = model.fit(cov_type='cluster', cov_kwds={'groups': d[cluster]})
    except Exception:
        res = model.fit()
    return res


def main():
    log("Starting replication analysis...")
    cmp_path = os.path.join(DATA_DIR, "CMP_final.dta")
    cpds_path = os.path.join(DATA_DIR, "CPDS_final.dta")

    cmp = safe_read_stata(cmp_path)
    if cmp is None:
        log("FATAL: Could not read CMP_final.dta from /app/data. Exiting.")
        return

    cpds = safe_read_stata(cpds_path)

    # Infer identifiers
    try:
        country_col, year_col, date_col = infer_id_columns(cmp)
    except Exception as e:
        log(f"FATAL: {e}")
        return

    # Find or build positions
    econ_col, soc_col = find_position_columns(cmp)
    if econ_col is None or soc_col is None:
        log("Precomputed policy dimensions not found; attempting PCA-based construction of two policy dimensions.")
        policy_cols = select_cmp_policy_columns(cmp)
        if len(policy_cols) < 5:
            log("FATAL: Not enough policy variables to construct PCA-based dimensions.")
            return
        econ_scores, soc_scores = build_positions_with_pca(cmp, policy_cols)
        cmp["econ_dim_auto"] = econ_scores
        cmp["soc_dim_auto"] = soc_scores
        econ_col = "econ_dim_auto"
        soc_col = "soc_dim_auto"

    # Compute seat shares
    seat_shares = compute_seat_shares(cmp)

    # Compute party counts
    enp_series, count_series = compute_party_counts(cmp, seat_shares)
    cmp["enp_election"] = enp_series
    cmp["count_relevance"] = count_series

    # Compute dispersion per election
    disp = compute_dispersion(cmp, econ_col, soc_col)

    # Merge counts into dispersion frame
    # We need one row per election, so aggregate counts to unique keys
    country_col_d, year_col_d, date_col_d = infer_id_columns(disp)
    # attach counts by taking a representative row per election from cmp
    keys = [country_col, date_col] if (date_col and np.issubdtype(cmp[date_col].dtype, np.datetime64)) else ([country_col, year_col] if year_col else [country_col])
    counts = cmp[keys + ["enp_election", "count_relevance"]].drop_duplicates()
    disp = pd.merge(disp, counts, left_on=[country_col_d] + ([date_col_d] if date_col_d in disp.columns else [year_col_d] if year_col_d in disp.columns else []), right_on=keys, how='left')

    # Merge CPDS controls if available
    merged = merge_controls(disp, cpds)

    # Prepare variables
    merged["log_disp_econ"] = np.log(merged["disp_econ"].replace(0, np.nan))
    merged["log_disp_soc"] = np.log(merged["disp_soc"].replace(0, np.nan))
    merged["log_count_parties"] = np.log(merged["count_relevance"].replace(0, np.nan))
    merged["log_enp"] = np.log(merged["enp_election"].replace(0, np.nan))

    # Try to find electoral system indicator in CPDS (e.g., 'prop')
    control_vars = []
    for c in ["prop", "pr", "pr_system", "smd", "electoral_rule"]:
        if c in merged.columns:
            control_vars.append(c)
            break

    # Fit models
    cluster_col = country_col_d
    results = {}

    for y in ["log_disp_econ", "log_disp_soc"]:
        for x in ["log_count_parties", "log_enp"]:
            xvars = [x] + control_vars
            res = fit_ols_cluster(merged, y, xvars, cluster_col)
            key = f"{y}~{'+'.join(xvars)}"
            if res is None:
                results[key] = {"status": "failed", "nobs": 0}
            else:
                summ = {
                    "params": res.params.to_dict(),
                    "bse": res.bse.to_dict(),
                    "tvalues": res.tvalues.to_dict(),
                    "pvalues": res.pvalues.to_dict(),
                    "nobs": int(res.nobs),
                    "rsquared": float(getattr(res, "rsquared", np.nan)),
                    "cov_type": getattr(res, "cov_type", "non-robust"),
                }
                results[key] = summ
                log(f"Fitted model {key}; N={summ['nobs']} R2={summ['rsquared']:.3f}")

    # Save outputs
    out_json = os.path.join(OUTPUT_DIR, "results_replication.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    merged_out = os.path.join(OUTPUT_DIR, "analysis_dataset.csv")
    merged.to_csv(merged_out, index=False)

    log(f"Saved results to {out_json} and analysis dataset to {merged_out}")


if __name__ == "__main__":
    main()
