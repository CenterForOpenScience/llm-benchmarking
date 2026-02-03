import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices

# All IO should use /app/data per instructions
DATA_DIR = "/app/data"

TRANSPORT_FILE = os.path.join(DATA_DIR, "transportation.csv")
COUNTY_FILE = os.path.join(DATA_DIR, "county_variables.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "replication_results.json")
OUTPUT_CSV = os.path.join(DATA_DIR, "regression_data.csv")

# Parameters matching original design as closely as possible
MARCH_START = "2020-03-19"
MARCH_END = "2020-03-28"
# For baseline, use February 2020 (pre-COVID) matched weekdays, as a practical proxy for the paper's reference week
BASELINE_START = "2020-02-01"
BASELINE_END = "2020-02-29"

# Distance bin midpoints in miles (approximate)
BIN_MIDPOINTS = {
    "trips_under_1": 0.5,
    "trips_1_3": 2.0,
    "trips_3_5": 4.0,
    "trips_5_10": 7.5,
    "trips_10_25": 17.5,
    "trips_25_50": 37.5,
    "trips_50_100": 75.0,
    "trips_100_250": 175.0,
    "trips_250_500": 375.0,
    "trips_over_500": 600.0,
}

TRIP_BINS = list(BIN_MIDPOINTS.keys())


def load_datasets():
    if not os.path.exists(TRANSPORT_FILE):
        raise FileNotFoundError(f"Missing {TRANSPORT_FILE}. Place transportation.csv in /app/data")
    if not os.path.exists(COUNTY_FILE):
        raise FileNotFoundError(f"Missing {COUNTY_FILE}. Place county_variables.csv in /app/data")
    trans = pd.read_csv(TRANSPORT_FILE)
    counties = pd.read_csv(COUNTY_FILE)
    return trans, counties


def compute_daily_mobility(trans: pd.DataFrame) -> pd.DataFrame:
    df = trans.copy()
    # Ensure date type
    df["date"] = pd.to_datetime(df["date"]) 

    # Fill NaNs in trip bins with zeros
    for c in TRIP_BINS:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
        else:
            # If any expected bin is missing, create it with zeros
            df[c] = 0.0

    # Compute total distance traveled per county-day using bin midpoints
    total_distance = None
    for c in TRIP_BINS:
        dist = df[c] * BIN_MIDPOINTS[c]
        total_distance = dist if total_distance is None else total_distance + dist
    df["total_distance"] = total_distance

    # Average distance per person (proxy for Unacast "average distance traveled per person")
    # population engaged in mobility = pop_home + pop_not_home
    denom = (df["pop_home"].fillna(0.0) + df["pop_not_home"].fillna(0.0))
    denom = denom.replace({0: np.nan})
    df["avg_dist_per_person"] = df["total_distance"] / denom
    # If denominator is zero/missing, set to 0 (no mobility)
    df["avg_dist_per_person"] = df["avg_dist_per_person"].fillna(0.0)

    # Weekday indicator for matched-baseline calculation
    df["weekday"] = df["date"].dt.day_name()
    return df


def compute_percent_change_from_baseline(df: pd.DataFrame) -> pd.DataFrame:
    # Filter baseline period (February 2020)
    base_mask = (df["date"] >= pd.to_datetime(BASELINE_START)) & (df["date"] <= pd.to_datetime(BASELINE_END))
    baseline = df.loc[base_mask].copy()
    if baseline.empty:
        raise ValueError("Baseline period has no data in February 2020; cannot compute matched-weekday baseline.")

    # Compute county-weekday baseline means in Feb 2020
    base_means = (
        baseline.groupby(["fips", "weekday"], as_index=False)["avg_dist_per_person"].mean()
        .rename(columns={"avg_dist_per_person": "baseline_weekday_mean"})
    )

    # Join baseline back to full df on fips and weekday
    df = df.merge(base_means, on=["fips", "weekday"], how="left")

    # Percent change relative to baseline, in percentage points
    # (current - baseline)/baseline * 100
    df["pct_change_vs_baseline"] = (
        (df["avg_dist_per_person"] - df["baseline_weekday_mean"]) / df["baseline_weekday_mean"]
    ) * 100.0

    return df


def aggregate_march_period(df: pd.DataFrame) -> pd.DataFrame:
    # Filter to March 19-28, 2020
    m_mask = (df["date"] >= pd.to_datetime(MARCH_START)) & (df["date"] <= pd.to_datetime(MARCH_END))
    m = df.loc[m_mask].copy()
    if m.empty:
        raise ValueError("March 19-28 period has no data; cannot proceed.")
    # Average across those dates per county
    agg = m.groupby("fips", as_index=False)["pct_change_vs_baseline"].mean()
    agg = agg.rename(columns={"pct_change_vs_baseline": "outcome_pct_change"})
    return agg


def prepare_regression_data(outcome: pd.DataFrame, counties: pd.DataFrame) -> pd.DataFrame:
    # Select relevant columns from county file
    needed_cols = [
        "fips", "state_po", "trump_share", "percent_college", "percent_male", "percent_black", "percent_hispanic",
        # Age distribution
        "percent_under_5", "percent_5_9", "percent_10_14", "percent_15_19", "percent_20_24", "percent_25_34",
        "percent_35_44", "percent_45_54", "percent_55_59", "percent_60_64", "percent_65_74", "percent_75_84",
        "percent_85_over",
        # Industries
        "percent_retail", "percent_transportation", "percent_hes",
        # Rurality
        "percent_rural",
        # Income
        "income_per_capita"
    ]
    missing = [c for c in needed_cols if c not in counties.columns]
    if missing:
        raise KeyError(f"Missing required columns in county_variables.csv: {missing}")

    reg = outcome.merge(counties[needed_cols], on="fips", how="inner")

    # Drop rows missing state or trump_share
    reg = reg.dropna(subset=["state_po", "trump_share"])

    # Ensure numeric types
    numeric_cols = [c for c in reg.columns if c not in ["state_po"]]
    reg[numeric_cols] = reg[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Drop any rows with missing outcome or covariates
    reg = reg.dropna()

    return reg


def run_regression(reg: pd.DataFrame) -> dict:
    # Build formula with state fixed effects
    controls = [
        "percent_college", "percent_male", "percent_black", "percent_hispanic",
        "percent_under_5", "percent_5_9", "percent_10_14", "percent_15_19", "percent_20_24", "percent_25_34",
        "percent_35_44", "percent_45_54", "percent_55_59", "percent_60_64", "percent_65_74", "percent_75_84",
        "percent_85_over",
        "percent_retail", "percent_transportation", "percent_hes",
        "percent_rural",
        "income_per_capita"
    ]

    formula = "outcome_pct_change ~ trump_share + " + " + ".join(controls) + " + C(state_po)"

    y, X = dmatrices(formula, data=reg, return_type="dataframe")

    # Clustered SEs by state
    groups = reg["state_po"].values
    ols = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})

    coef = ols.params.get("trump_share", np.nan)
    se = ols.bse.get("trump_share", np.nan)
    pval = ols.pvalues.get("trump_share", np.nan)
    ci_low, ci_high = ols.conf_int().loc["trump_share"].tolist()

    # Interquartile range effect size: original reported IQR = 20.3 percentage points
    IQR_ORIG = 0.203
    effect_per_iqr = coef * IQR_ORIG
    effect_ci_low = ci_low * IQR_ORIG
    effect_ci_high = ci_high * IQR_ORIG

    results = {
        "n_obs": int(ols.nobs),
        "states": int(reg["state_po"].nunique()),
        "coef_trump_share": float(coef),
        "se_trump_share": float(se),
        "p_trump_share": float(pval),
        "ci_trump_share": [float(ci_low), float(ci_high)],
        "iqr_used": IQR_ORIG,
        "effect_per_iqr_pp": float(effect_per_iqr),
        "effect_per_iqr_ci": [float(effect_ci_low), float(effect_ci_high)],
        "outcome_definition": "Percent change in average distance traveled per person (county-day) relative to Feb-2020 matched weekdays, averaged over 2020-03-19 to 2020-03-28 (negative = more social distancing)",
        "model_spec": {
            "formula": formula,
            "cov_type": "cluster",
            "cluster_by": "state_po"
        }
    }

    return results, ols, X, y


def main():
    trans, counties = load_datasets()
    df = compute_daily_mobility(trans)
    df = compute_percent_change_from_baseline(df)
    outcome = aggregate_march_period(df)

    reg = prepare_regression_data(outcome, counties)

    # Save regression data for inspection
    reg.to_csv(OUTPUT_CSV, index=False)

    results, ols, X, y = run_regression(reg)

    # Save results JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # Also print summary to stdout for logs
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
