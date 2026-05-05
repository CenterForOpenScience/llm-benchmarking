import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import timedelta
from typing import List

from utils_py import compute_ols_slope_time, summarize_country_level

DATA_PATH = "/app/data/original/19/0112_python_gpt5/replication_data/gelfand_replication_data.csv"
OUT_DIR = "/app/data"

# Candidate paths for the CSV inside the container
DATA_CANDIDATES = [
    "/app/data/original/19/0112_python_gpt5/replication_data/gelfand_replication_data.csv",
    "/workspace/replication_data/gelfand_replication_data.csv",
    "/app/data/data/original/19/0112_python_gpt5/replication_data/gelfand_replication_data.csv",
]

def resolve_data_path() -> str:
    for p in DATA_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not locate dataset. Tried: {DATA_CANDIDATES}")

DROP_COUNTRIES = ["Belgium", "France", "New Zealand", "Norway", "Pakistan", "Venezuela"]


def ensure_daily_continuity(df: pd.DataFrame) -> pd.DataFrame:
    """For each country, reindex to daily frequency, filling gaps between min and max date.
    Forward-fill total_covid_per_million and gdp, as in Stata tsfill + replace with lag."""
    def _fill_country(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date")
        # full daily range from min to max existing dates
        full_idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(full_idx)
        g.index.name = "date"
        # forward-fill totals and gdp to mimic replace with lag
        for col in ["total_covid_per_million", "gdp"]:
            if col in g.columns:
                g[col] = g[col].ffill()
        # country, tightness, efficiency, gini_val, alternative_gini, median_age can be ffilled/backfilled lightly
        # but we'll keep them as-is; aggregation step will pick first non-null
        # restore country name (constant within group)
        g["country"] = g["country"].ffill().bfill()
        # carry over other static vars where possible
        for col in [
            "tightness", "efficiency", "gini_val", "alternative_gini", "median_age",
            "pop_per_million", "obs_count_full", "obs_after_one_per_million", "obs_count_original",
            "running_total_by_country", "cases", "deaths", "gdp", "popData2019"
        ]:
            if col in g.columns:
                g[col] = g[col].ffill().bfill()
        return g.reset_index()

    out = (
        df.groupby("country", group_keys=False)
          .apply(_fill_country)
          .reset_index(drop=True)
    )
    return out


def run_analysis():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    data_path = resolve_data_path()
    print(f"Using data at: {data_path}")
    df = pd.read_csv(data_path)
    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop specified countries
    df = df[~df["country"].isin(DROP_COUNTRIES)].copy()

    # Ensure continuity per country akin to Stata tsfill
    df = ensure_daily_continuity(df)

    # Keep only observations where total_covid_per_million > 1
    df = df[df["total_covid_per_million"] > 1].copy()

    # Log transform
    df["ltotalcases"] = np.log(df["total_covid_per_million"])  # safe since we filter > 1

    # Within each country, create time index ordered by date
    df = df.sort_values(["country", "date"])  # ensure order
    df["time"] = df.groupby("country").cumcount() + 1

    # Drop observations beyond 30 days for each country
    df = df[df["time"] <= 30].copy()

    # Construct gini from gini_val, falling back to alternative_gini
    df["gini"] = df["gini_val"]
    if "alternative_gini" in df.columns:
        df.loc[df["gini"].isna(), "gini"] = df.loc[df["gini"].isna(), "alternative_gini"]

    # Compute per-country growth coefficients (slope of ltotalcases on time)
    slopes = []
    for country, g in df.groupby("country"):
        slope = compute_ols_slope_time(g, y_col="ltotalcases", t_col="time")
        if slope is not None:
            slopes.append({"country": country, "coeffs1": slope})
        else:
            # Will be dropped later if slope missing
            slopes.append({"country": country, "coeffs1": np.nan})
    coeffs_df = pd.DataFrame(slopes)

    # Save estimated coefficients
    coeffs_path = os.path.join(OUT_DIR, "estimatedcoefficients.csv")
    coeffs_df.to_csv(coeffs_path, index=False)

    # Create a single row per country with predictors/controls
    country_summary = summarize_country_level(
        df,
        cols={
            "tightness": "tightness",
            "efficiency": "efficiency",
            "gdp": "gdp",
            "gini": "gini",
            "median_age": "median_age",
        },
    )

    # Merge coefficients to predictors by country
    merged = pd.merge(country_summary, coeffs_df, on="country", how="inner")

    # Interaction term
    merged["eff_tight"] = merged["efficiency"] * merged["tightness"]

    # Listwise delete rows with missing needed variables
    model_vars = ["coeffs1", "eff_tight", "gdp", "gini", "median_age", "efficiency", "tightness"]
    model_df = merged.dropna(subset=model_vars).copy()

    # Fit OLS: coeffs1 ~ eff_tight + gdp + gini + median_age + efficiency + tightness
    X = model_df[["eff_tight", "gdp", "gini", "median_age", "efficiency", "tightness"]]
    X = sm.add_constant(X)
    y = model_df["coeffs1"].astype(float)

    ols_model = sm.OLS(y, X)
    ols_res = ols_model.fit()

    # Save model summary and coefficients
    summary_path = os.path.join(OUT_DIR, "interaction_model_summary.txt")
    with open(summary_path, "w") as f:
        f.write(str(ols_res.summary()))

    coef_df = ols_res.params.rename("coef").to_frame()
    coef_df["std_err"] = ols_res.bse
    coef_df["t"] = ols_res.tvalues
    coef_df["p_value"] = ols_res.pvalues
    conf_int = ols_res.conf_int(alpha=0.05)
    coef_df["ci_lower"] = conf_int[0]
    coef_df["ci_upper"] = conf_int[1]
    coef_path = os.path.join(OUT_DIR, "interaction_model_coefficients.csv")
    coef_df.to_csv(coef_path)

    # Key results for eff_tight
    key = {
        "n_countries_in_coeffs": int(coeffs_df["country"].nunique()),
        "n_countries_in_model": int(model_df.shape[0]),
        "eff_tight": {
            "coef": float(ols_res.params.get("eff_tight", np.nan)),
            "std_err": float(ols_res.bse.get("eff_tight", np.nan)),
            "t": float(ols_res.tvalues.get("eff_tight", np.nan)),
            "p_value": float(ols_res.pvalues.get("eff_tight", np.nan)),
            "ci_lower": float(coef_df.loc["eff_tight", "ci_lower"]) if "eff_tight" in coef_df.index else np.nan,
            "ci_upper": float(coef_df.loc["eff_tight", "ci_upper"]) if "eff_tight" in coef_df.index else np.nan,
        },
        "r_squared": float(ols_res.rsquared),
        "adj_r_squared": float(ols_res.rsquared_adj),
    }
    key_path = os.path.join(OUT_DIR, "interaction_model_key_results.json")
    with open(key_path, "w") as f:
        json.dump(key, f)

    # Print concise log to stdout
    print("Saved:")
    print(f"- Coefficients per country: {coeffs_path}")
    print(f"- Model summary: {summary_path}")
    print(f"- Model coefficients: {coef_path}")
    print(f"- Key results: {key_path}")
    print("Key eff_tight results:")
    print(json.dumps(key["eff_tight"], indent=2))


if __name__ == "__main__":
    run_analysis()
