#!/usr/bin/env python3
"""
Python translation of Analysis_script_v2.do for replication of Gelfand et al. COVID-19 tightness × efficiency claim.
All file IO assumed relative to /app/data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm

# Define directory relative to this script to avoid absolute-path issues
DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "gelfand_replication_data.csv"

# Countries flagged for exclusion by the original authors
EXCLUDE_COUNTRIES = [
    "Belgium",
    "France",
    "New Zealand",
    "Norway",
    "Pakistan",
    "Venezuela",
]


def load_and_preprocess():
    df = pd.read_csv(CSV_PATH)

    # Drop excluded countries
    df = df.loc[~df["country"].isin(EXCLUDE_COUNTRIES)].copy()

    # Convert date string YYYY-MM-DD to pandas datetime
    df["date1"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # Fill in missing dates per country (equivalent of tsset + tsfill in Stata)
    full_frames = []
    for country, g in df.groupby("country"):
        g = g.sort_values("date1").set_index("date1")
        # build full daily date range between min and max
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g_full = g.reindex(full_idx)
        g_full["country"] = country
        full_frames.append(g_full.reset_index().rename(columns={"index": "date1"}))
    df_full = pd.concat(full_frames, ignore_index=True)

    # Forward fill numerical columns that should remain constant/accumulate when missing
    ffill_cols = [
        "total_covid_per_million",
        "gdp",
    ]
    for col in ffill_cols:
        df_full[col] = df_full.groupby("country")[col].fillna(method="ffill")

    # After fill, drop rows where still missing total_covid_per_million (pre first observation)
    df_full = df_full.dropna(subset=["total_covid_per_million"])

    # Keep observations with total_covid_per_million > 1
    df_full = df_full[df_full["total_covid_per_million"] > 1].copy()

    # Log of total cases per million
    df_full["ltotalcases"] = np.log(df_full["total_covid_per_million"])

    # Create time variable within each country (day index starting at 1)
    df_full = df_full.sort_values(["country", "date1"])
    df_full["time"] = df_full.groupby("country").cumcount() + 1

    # Restrict to first 30 days
    df_full = df_full[df_full["time"] <= 30]

    # Create cleaned Gini variable
    df_full["gini"] = df_full["gini_val"].where(~df_full["gini_val"].isna(), df_full["alternative_gini"])

    return df_full


def estimate_growth_coefficients(df):
    coeffs = {}
    for country, g in df.groupby("country"):
        if g["time"].nunique() < 2:
            continue  # cannot estimate slope
        X = sm.add_constant(g["time"])
        model = sm.OLS(g["ltotalcases"], X, missing="drop").fit()
        coeffs[country] = model.params["time"]
    coeff_df = pd.DataFrame({"country": list(coeffs.keys()), "growth_coeff": list(coeffs.values())})
    return coeff_df


def prepare_country_level_data(df):
    # Take first row for each country to capture country-level attributes
    first_rows = df.sort_values("date1").groupby("country").first().reset_index()

    # Select variables of interest
    vars_keep = [
        "country",
        "efficiency",
        "tightness",
        "gdp",
        "gini",
        "median_age",
    ]
    country_df = first_rows[vars_keep]
    return country_df


def main():
    df = load_and_preprocess()
    coeff_df = estimate_growth_coefficients(df)
    country_df = prepare_country_level_data(df)

    # Merge
    merged = pd.merge(country_df, coeff_df, on="country", how="inner")

    # Interaction term
    merged["eff_tight"] = merged["efficiency"] * merged["tightness"]
    # Drop rows with missing values in any variables used for regression
    regression_vars = [
        "growth_coeff",
        "eff_tight",
        "gdp",
        "gini",
        "median_age",
        "efficiency",
        "tightness",
    ]
    merged = merged.dropna(subset=regression_vars)


    # Regression: growth_coeff ~ interaction + controls
    X_vars = [
        "eff_tight",
        "gdp",
        "gini",
        "median_age",
        "efficiency",
        "tightness",
    ]
    X = sm.add_constant(merged[X_vars])
    y = merged["growth_coeff"]
    model = sm.OLS(y, X).fit(cov_type="HC3")

    # Save summary to text file inside /app/data
    output_path = DATA_DIR / "replication_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())
    print("Regression results saved to", output_path)
    print(model.summary())


if __name__ == "__main__":
    main()
