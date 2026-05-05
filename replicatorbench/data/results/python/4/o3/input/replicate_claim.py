"""
Replication script for Gerhold COVID-19 gender worry difference.
Loads the public data_gerhold.csv and tests whether women report higher worry (mh_anxiety_1) than men.
Outputs summary statistics and independent samples t-test results as well as Cohen's d.
All outputs printed to stdout.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# CONSTANTS# CONSTANTS
POSSIBLE_PATHS = [
    Path("/app/data/data_gerhold.csv"),
    Path(__file__).resolve().parent / "data_gerhold.csv",
    Path("/app/data/original/4/python/replication_data/data_gerhold.csv"),
]

for p in POSSIBLE_PATHS:
    if p.exists():
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("Could not locate data_gerhold.csv in expected locations.")

def cohens_d(x, y):
    """Compute Cohen's d for two independent samples."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / dof)
    return (x.mean() - y.mean()) / pooled_std


def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Optional: restrict to Germany to closely match original study
    df = df[df["CountryofLiving"].str.lower() == "germany"]

    # Remove entries with gender coded as 3 ("diverse" / missing)
    df = df[df["gender"].isin([1, 2])]

    # Create female indicator if not already present
    if "female" not in df.columns:
        df["female"] = np.where(df["gender"] == 2, 1, 0)  # Assuming 1=male, 2=female in raw

    # Outcome variable corresponding to general worry about COVID-19
    outcome = "mh_anxiety_1"
    if outcome not in df.columns:
        raise ValueError(f"Outcome variable '{outcome}' not found in dataset.")

    female_group = df[df["female"] == 1][outcome].dropna()
    male_group = df[df["female"] == 0][outcome].dropna()

    # Two-sample t-test (equal variances)
    t_res = stats.ttest_ind(female_group, male_group, equal_var=True)

    # Variance equality test (Levene)
    var_test = stats.levene(female_group, male_group)

    # Cohen's d
    d = cohens_d(female_group, male_group)

    print("Replication of gender difference in COVID-19 worry (mh_anxiety_1)")
    print("Sample sizes: female =", len(female_group), ", male =", len(male_group))
    print("Mean(female)=", round(female_group.mean(), 3), ", SD=", round(female_group.std(ddof=1), 3))
    print("Mean(male)=", round(male_group.mean(), 3), ", SD=", round(male_group.std(ddof=1), 3))
    print("\nLevene equality of variances test: W=%.3f, p=%.4f" % (var_test.statistic, var_test.pvalue))
    print("T-test (equal variances): t=%.3f, df=%d, p=%.4g" % (t_res.statistic, len(female_group)+len(male_group)-2, t_res.pvalue))
    print("Cohen's d:", round(d, 3))


if __name__ == "__main__":
    main()
