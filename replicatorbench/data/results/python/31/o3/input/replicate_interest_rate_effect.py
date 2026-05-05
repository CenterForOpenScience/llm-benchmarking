#!/usr/bin/env python3
"""
Replicate the effect of ethnic identity salience on reservation interest rates among Asian-American subjects.
Per replication plan, we estimate an interval regression (AFT model) of the lower and upper bounds of the log
reservation interest rate on the treatment dummy and controls, using cluster-robust SEs at the participant level.

Outputs:
    - Prints coefficient table to stdout
    - Saves CSV 'replication_results.csv' to /app/data
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import LogNormalAFTFitter

# Attempt to locate replication_data directory. First, standard mounted path, else fallback.
DATA_DIR = Path("/app/data/original/31/input/replication_data")
if not DATA_DIR.exists():
    # When study path is mounted at /workspace, datasets may be directly under it.
    alt = Path(__file__).resolve().parent / "replication_data"
    if alt.exists():
        DATA_DIR = alt
    else:
        # Another fallback: /app/data/replication_data
        alt2 = Path("/app/data/replication_data")
        if alt2.exists():
            DATA_DIR = alt2
# Final check
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Could not locate replication_data directory. Tried: {DATA_DIR}, {alt}, {alt2}")
OUTPUT_CSV = Path("/app/data/replication_results.csv")

ROUND1_FILE = DATA_DIR / "REPExperiment1Data.dta"
ROUND2_FILE = DATA_DIR / "REPExperiment1DataR2.dta"

REQUIRED_COLS = [
    "id",
    "asian",
    "givenprimingquestionnaire",
    "largestakes",
    "longterm",
    "largelong",
    "lndiscratelo",
    "lndiscratehi",
]

def load_and_concat():
    """Load both round datasets and concatenate them."""
    try:
        import pyreadstat  # noqa: F401 – ensure dependency present
    except ImportError as e:
        print("Missing pyreadstat. Please install before running.", file=sys.stderr)
        raise e

    dfs = []
    for f in [ROUND1_FILE, ROUND2_FILE]:
        if not f.exists():
            raise FileNotFoundError(f"Dataset not found: {f}")
        df, meta = pyreadstat.read_dta(str(f))
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Filter Asian subjects and prepare censoring columns."""
    # Ensure required columns exist
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    # Keep Asian subjects only
    df = df[df["asian"] == 1].copy()

    # Replace missing upper/lower bounds with +-inf as per lifelines requirements
    df["lower"] = df["lndiscratelo"].astype(float)
    df["upper"] = df["lndiscratehi"].astype(float)

    # If upper bound is NaN => right-censored: set to +inf
    df.loc[df["upper"].isna(), "upper"] = np.inf
    # If lower bound is NaN (shouldn't happen) => left-censored: set to -inf
    df.loc[df["lower"].isna(), "lower"] = -np.inf

    # Select variables for model
    model_cols = [
        "givenprimingquestionnaire",
        "largestakes",
        "longterm",
        "largelong",
    ]
    df_model = df[["id", "lower", "upper", *model_cols]].copy()
    return df_model


def fit_interval_regression(df: pd.DataFrame):
    aft = LogNormalAFTFitter()
    formula = "givenprimingquestionnaire + largestakes + longterm + largelong"
    aft.fit(
        df,
        lower_bound_col="lower",
        upper_bound_col="upper",
        cluster_col="id",
        formula=formula,
        show_progress=False,
    )
    return aft


def main():
    warnings.filterwarnings("ignore")
    print("Loading datasets...")
    df_all = load_and_concat()
    print(f"Combined dataset shape: {df_all.shape}")
    df_prep = preprocess(df_all)
    print(f"Filtered Asian subjects dataset shape: {df_prep.shape}")

    # Fit model
    print("Fitting Log-Normal AFT model (interval regression)...")
    aft_model = fit_interval_regression(df_prep)

    # Extract coefficient for treatment dummy
    summary = aft_model.summary.reset_index()

    # Variable names in lifelines add 'coefficient_' prefix (e.g., 'coef_') – summary index indicates parameter type.
    # Let's tidy summary
    summary = summary.rename(columns={
        "coef": "coefficient",
        "se(coef)": "std_err",
        "p": "p_value",
    })

    # Save CSV
    summary.to_csv(OUTPUT_CSV, index=False)
    print("Model summary (key variables):")
    print(summary[["index", "coefficient", "std_err", "p_value"]])
    print(f"\nFull summary saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
