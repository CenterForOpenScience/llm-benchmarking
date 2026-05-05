import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf
import numpy as np
from statsmodels.stats.anova import anova_lm

# Path constants (all IO should be in /app/data)
DATA_PATH = "/app/data/Ma-Kellams Replication Study Data.sav"

# Load the SPSS .sav file
print("Loading dataset from", DATA_PATH)
df, meta = pyreadstat.read_sav(DATA_PATH, apply_value_formats=False)

############################################################
# --- Variable construction to match original SPSS syntax ---
############################################################

# 1. Binary wordcount flag (>=30 words)
if "WordCount" in df.columns:
    df["Wrote30WordsOrMore"] = (df["WordCount"] > 29).astype(int)

# 2. Ethnicity coding
if "CulturalBackground" in df.columns:
    df["EuropeanAmerican"] = (df["CulturalBackground"] == 1).astype(int)
    df["EastAsian"] = (df["CulturalBackground"] == 4).astype(int)
    df["Culture"] = pd.NA
    df.loc[df["EuropeanAmerican"] == 1, "Culture"] = 0  # 0 = European American
    df.loc[df["EastAsian"] == 1, "Culture"] = 1          # 1 = East Asian

# 3. PAS scale construction (reverse-coding items 2 and 5)
PAS_ITEMS = ["PAS1", "PAS2", "PAS3", "PAS4", "PAS5"]
# Check all required items exist
if all(item in df.columns for item in PAS_ITEMS):
    df["PAS2Positive"] = df["PAS2"] * -1
    df["PAS5Positive"] = df["PAS5"] * -1
    df["PASAverage"] = df[["PAS1", "PAS2Positive", "PAS3", "PAS4", "PAS5Positive"]].mean(axis=1)

# 4. Optional log‐transform of bail (skewed)
if "BailAmount" in df.columns:
    df["BailAmountLog"] = np.log(df["BailAmount"].clip(lower=0) + 1)

############################################################
# --- Main replication analysis: 2 (Culture) x 2 (Condition) ANOVA on PASAverage ---
############################################################

required_cols = {"Culture", "WritingCondition", "PASAverage"}
if required_cols.issubset(df.columns):
    print("Running 2x2 ANOVA on PASAverage…")

    # Convert categorical predictors to category dtype
    df["Culture"] = df["Culture"].astype("category")
    df["WritingCondition"] = df["WritingCondition"].astype("category")

    model = smf.ols("PASAverage ~ C(Culture) * C(WritingCondition)", data=df).fit()
    anova_results = anova_lm(model, typ=3)
    print("\nANOVA table (Type III):")
    print(anova_results)
else:
    print("Required columns for main analysis are missing. Available columns:\n", df.columns.tolist())

############################################################
# --- Save cleaned data (optional) ---
############################################################

df.to_csv("/app/data/ma_kellams_replication_cleaned.csv", index=False)
print("Cleaned dataset written to /app/data/ma_kellams_replication_cleaned.csv")
