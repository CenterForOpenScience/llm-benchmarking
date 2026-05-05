"""
Replicate key analysis adapted to 2014 Afghanistan presidential election data.
Runs logistic regression of fraud on violence (linear and quadratic) with controls,
clusters standard errors by regional command, prints summary, and saves
coefficients to /app/data/replication_coefficients.csv.
"""

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "Afghanistan_Election_Violence_2014.csv"
OUTPUT_PATH = Path('/app/data/replication_coefficients.csv')


def main():
    """Load data, fit logistic model with clustered SEs, save coefficients."""
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Recode fraud
    df['fraud_bin'] = df['fraud'].map({'Fraud': 1, 'No Fraud': 0})

    # Create squared violence term
    df['sigact_5r_sq'] = df['sigact_5r'] ** 2

    # Drop rows with missing values in variables used in model
    model_vars = ['fraud_bin', 'sigact_5r', 'sigact_5r_sq', 'pcx', 'electric', 'pcexpend', 'dist', 'elevation', 'regcom']
    df = df.dropna(subset=model_vars).copy()

    # Model formula
    formula = (
        'fraud_bin ~ sigact_5r + sigact_5r_sq + pcx + electric + pcexpend + dist + elevation'
    )

    # Fit logistic model with cluster-robust SEs
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    clusters = df['regcom']
    result = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})

    print("=== Logistic regression with clustered standard errors ===")
    print(result.summary())

    # Save coefficients
    coef_df = result.params.to_frame(name='coef')
    coef_df['std_err'] = result.bse
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(OUTPUT_PATH)


if __name__ == "__main__":
    main()
