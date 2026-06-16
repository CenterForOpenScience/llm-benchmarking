import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
from statsmodels.stats.anova import anova_lm

# All IO uses /app/data
BASE_DIR = "/app/data"
HOUSEHOLD_PATH = os.path.join(BASE_DIR, "AEJApp-2009-0289-data", "household.dta")
VILLAGE_PATH = os.path.join(BASE_DIR, "AEJApp-2009-0289-data", "village.dta")
OUT_PATH = os.path.join(BASE_DIR, "results_task2.txt")

warnings.filterwarnings("ignore")

def main():
    # Read data
    household = pd.read_stata(HOUSEHOLD_PATH)
    village = pd.read_stata(VILLAGE_PATH)

    # Prepare join keys and relevant columns
    village_dom = village.loc[:, ["village", "domhigh"]].copy()

    # Construct analysis dataframe
    dat = household.loc[:, ["hhcode", "village", "caste", "totinc"]].copy()
    dat = dat.merge(village_dom, on="village", how="left")

    # Filter to domhigh in {0,1}
    dat = dat[dat["domhigh"].isin([0, 1])].copy()

    # Drop rows with missing outcome or predictors
    dat = dat.dropna(subset=["totinc", "caste", "domhigh", "village"])  

    # Mixed model: totinc ~ caste + domhigh + caste:domhigh + (1|village)
    formula = "totinc ~ C(caste) + C(domhigh) + C(caste):C(domhigh)"

    try:
        model = smf.mixedlm(formula, data=dat, groups=dat["village"], re_formula="1")
        result = model.fit(reml=True, method="lbfgs", maxiter=200)
    except Exception as e:
        try:
            result = model.fit(reml=True, method="nm", maxiter=400)
        except Exception as e2:
            with open(OUT_PATH, "w") as f:
                f.write("Model fitting failed.\n")
                f.write(str(e))
                f.write("\n--- Retry error ---\n")
                f.write(str(e2))
            return

    # Omnibus F-test for dominance effect is not directly available in MixedLM.
    # We approximate by fitting a fixed-effects OLS with village dummies and running anova on C(domhigh).
    ols_formula = "totinc ~ C(caste) + C(domhigh) + C(caste):C(domhigh) + C(village)"
    try:
        ols_model = smf.ols(ols_formula, data=dat).fit()
        # Type II ANOVA to test C(domhigh)
        from statsmodels.stats.anova import anova_lm
        anova_tbl = anova_lm(ols_model, typ=2)
    except Exception as e:
        anova_tbl = None

    with open(OUT_PATH, "w") as f:
        f.write("Linear mixed model: totinc ~ caste + domhigh + caste:domhigh + (1|village)\n")
        f.write("Software: Python statsmodels MixedLM (REML)\n\n")
        f.write(str(result.summary()))
        if anova_tbl is not None:
            f.write("\n\nOLS fixed-effects (village dummies) ANOVA (Type II) for C(domhigh):\n")
            f.write(str(anova_tbl.loc[[c for c in anova_tbl.index if c.startswith('C(domhigh)')]]))
        else:
            f.write("\n\nANOVA for dominance effect could not be computed.")

if __name__ == "__main__":
    main()
