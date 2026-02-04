"""Replicate Malik et al. (2020) focal claim: effect of social distancing measures on mobility.

This script reads the replication dataset, transforms variables, and fits a multilevel
mixed-effects linear regression with a random intercept for each city. The model follows the
specification in the original study:

    Mobility_it = beta0 + beta1 * time_t + beta2 * lockdown_it + u_i + e_it

where u_i ~ N(0, sigma_u^2) is a city-level random intercept.

Outputs:
  - Prints model summary to stdout
  - Saves coefficients and confidence intervals to /app/data/mobility_replication_results.csv

All file IO is restricted to /app/data as required by SCORE OPS.
"""
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Paths (ensure to use /app/data)# Determine base directory (location of this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "replicationDataset_Malik2020_with.year.csv")
# Save results to same directory to stay within allowed path
OUTPUT_PATH = os.path.join(BASE_DIR, "mobility_replication_results.csv")

def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Convert date string to datetime and create numeric time variable (days since first date)
    df['date_dt'] = pd.to_datetime(df['date'])
    first_date = df['date_dt'].min()
    df['days_since_start'] = (df['date_dt'] - first_date).dt.days

    # Ensure proper data types
    df['lockdown'] = df['lockdown'].astype(int)

    # Mixed-effects linear regression: outcome = CMI, predictors = days_since_start + lockdown
    # Random intercept for city
    model = smf.mixedlm("CMI ~ days_since_start + lockdown", df, groups=df["city"])
    result = model.fit(reml=False)

    # Print summary
    print(result.summary())

    # Save key results
    coef = result.params
    conf_int = result.conf_int(alpha=0.05)
    out = pd.DataFrame({
        'term': coef.index,
        'estimate': coef.values,
        'ci_lower': conf_int[0].values,
        'ci_upper': conf_int[1].values
    })
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
