"""
Replication script translated from Stata .do to Python.
- Reads replicationDataset_Malik2020_with.year.csv from /app/data
- Processes date variable to numeric days since baseline (date2)
- Runs two multilevel mixed-effects linear regression models (transit and residential mobility)
  with random intercepts by city, on full data and on a 5% random sample (to mirror the .do)
- Saves results (coefficients, CIs) to JSON in /app/data
"""

import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Configure
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine base directory where script resides or where /app/data is mounted
BASE_DIR = os.path.dirname(__file__)
# Prefer mounted /app/data path if it exists, else use script-relative path
possible_app_data = '/app/data/original/16/0205_gpt5-mini/replication_data/replicationDataset_Malik2020_with.year.csv'
if os.path.exists(possible_app_data):
    DATA_PATH = possible_app_data
    OUTPUT_JSON = '/app/data/original/16/0205_gpt5-mini/replication_data/replication_results.json'
    PROCESSED_CSV = '/app/data/original/16/0205_gpt5-mini/replication_data/processed_data.csv'
else:
    DATA_PATH = os.path.join(BASE_DIR, 'replicationDataset_Malik2020_with.year.csv')
    OUTPUT_JSON = os.path.join(BASE_DIR, 'replication_results.json')
    PROCESSED_CSV = os.path.join(BASE_DIR, 'processed_data.csv')

SEED = 2025

def load_and_prepare(path):
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)

    # Parse date (MDY)
    df["date_parsed"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    if df["date_parsed"].isna().any():
        logger.warning("Some dates failed to parse and will be set as NaT")

    # Create numeric date2 as days since earliest date
    min_date = df["date_parsed"].min()
    df["date2"] = (df["date_parsed"] - min_date).dt.days

    # Ensure lockdown is numeric
    df["lockdown"] = pd.to_numeric(df["lockdown"], errors="coerce").fillna(0).astype(int)

    # Drop rows with missing outcome or city
    df = df.dropna(subset=["city", "date_parsed"])

    # Save processed csv for inspection
    df.to_csv(PROCESSED_CSV, index=False)
    logger.info(f"Processed data saved to {PROCESSED_CSV}")

    return df


def fit_mixedlm(df, formula, group_var):
    # statsmodels MixedLM via formula
    try:
        md = smf.mixedlm(formula, df, groups=df[group_var])
        mdf = md.fit(reml=False)
        return mdf
    except Exception as e:
        logger.exception(f"Model failed: {e}")
        return None


def summarize_model(mdf):
    if mdf is None:
        return None
    params = mdf.params.to_dict()
    try:
        conf = mdf.conf_int()
        conf_dict = {var: [float(conf.loc[var, 0]), float(conf.loc[var, 1])] for var in conf.index}
    except Exception:
        conf_dict = None

    return {
        "params": {k: float(v) for k, v in params.items()},
        "conf_int": conf_dict,
        "aic": float(getattr(mdf, 'aic', np.nan)),
        "bic": float(getattr(mdf, 'bic', np.nan)),
        "converged": bool(getattr(mdf, 'converged', True))
    }


def main():
    df = load_and_prepare(DATA_PATH)

    results = {
        "n_obs": int(df.shape[0]),
        "n_cities": int(df['city'].nunique()),
        "models": {}
    }

    # Focal model: CMRT_transit ~ date2 + lockdown with random intercept by city
    formula_transit = "CMRT_transit ~ date2 + lockdown"
    mdf_transit = fit_mixedlm(df, formula_transit, "city")
    results['models']['transit_full'] = summarize_model(mdf_transit)

    # Secondary model: CMRT_residential ~ date2 + lockdown
    formula_res = "CMRT_residential ~ date2 + lockdown"
    mdf_res = fit_mixedlm(df, formula_res, "city")
    results['models']['residential_full'] = summarize_model(mdf_res)

    # Also run on 5% random sample to mirror the Stata .do sample step
    sample_df = df.sample(frac=0.05, random_state=SEED)
    results['sample_n_obs'] = int(sample_df.shape[0])

    mdf_transit_samp = fit_mixedlm(sample_df, formula_transit, "city")
    results['models']['transit_sample'] = summarize_model(mdf_transit_samp)

    mdf_res_samp = fit_mixedlm(sample_df, formula_res, "city")
    results['models']['residential_sample'] = summarize_model(mdf_res_samp)

    # Save results to JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written to {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
