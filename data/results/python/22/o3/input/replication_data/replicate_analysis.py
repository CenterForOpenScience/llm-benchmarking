"""
Script: replicate_analysis.py
Purpose: Replicate the focal claim from Al-Tammemi et al. (2020) that severe psychological distress is inversely related to motivation for distance learning among university students.

1. Loads an anonymised survey dataset that contains responses to the 10-item Kessler Psychological Distress Scale (K10) and a single item measuring motivation for online distance learning.
2. Computes the K10 total score and classifies psychological distress levels according to published cut-offs (no/mild/moderate/severe).
3. Collapses the motivation item into three ordered categories (High, Medium, Low) or keeps the original 4-point Likert scale.
4. Runs a multinomial logistic regression (severe vs other) or binary logistic (severe vs non-severe) mirroring the original study. The key predictor is motivation. Controls (age, gender) are optional if present.
5. Prints regression table and saves it to "/app/data/replication_results.csv".

All file IO is done inside /app/data.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ---------- Parameters ----------
DATA_PATH = "/app/data/AlTammemi_Survey_deidentify.csv"  # Update to match mounted volume
OUTPUT_PATH = "/app/data/replication_results.csv"

# ---------- Helper functions ----------

def compute_k10_total(df):
    k10_items = [f"Kessler_{i}" for i in range(1, 11)]
    df["k10_total"] = df[k10_items].sum(axis=1, skipna=False)
    return df

def classify_distress(df):
    """Classify into None (<20), Mild (20-24), Moderate (25-29), Severe (>=30)."""
    bins = [-np.inf, 19, 24, 29, np.inf]
    labels = ["None", "Mild", "Moderate", "Severe"]
    df["distress_level"] = pd.cut(df["k10_total"], bins=bins, labels=labels)
    df["severe"] = (df["distress_level"] == "Severe").astype(int)
    return df

def prepare_motivation(df):
    # Assume 1 = strongly disagree (low motivation) ... 4 = strongly agree (high motivation)
    df = df.rename(columns={"online_learning_1": "motivation"})
    # Reverse-code so that higher = lower motivation if original coding opposite
    # Here we keep as is: higher number = higher motivation.
    return df

# ---------- Main analysis ----------

def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Drop rows with missing K10 or motivation
    df = compute_k10_total(df)
    df = classify_distress(df)
    df = prepare_motivation(df)

    analytic = df.dropna(subset=["severe", "motivation"])

    # Treat motivation as categorical reference = highest motivation (4)
    analytic["motivation_cat"] = pd.Categorical(analytic["motivation"], ordered=True)

    # Logistic regression: outcome severe (1) vs not severe (0)
    model = smf.logit("severe ~ C(motivation_cat)", data=analytic).fit(disp=False)
    print(model.summary())

    # Save coefficients
    results_df = model.summary2().tables[1]
    results_df.to_csv(OUTPUT_PATH)
    print(f"Results written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
