import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings

# All IO uses /app/data
BASE_DIR = "/app/data"
HOUSEHOLD_PATH = os.path.join(BASE_DIR, "AEJApp-2009-0289-data", "household.dta")
VILLAGE_PATH = os.path.join(BASE_DIR, "AEJApp-2009-0289-data", "village.dta")
OUT_PATH = os.path.join(BASE_DIR, "results_task1.txt")

warnings.filterwarnings("ignore")

def main():
    # Read data
    household = pd.read_stata(HOUSEHOLD_PATH)
    village = pd.read_stata(VILLAGE_PATH)

    # Prepare join keys and relevant columns
    village = village.copy()
    # Keep only needed columns for merge
    village_dom = village.loc[:, ["village", "domhigh"]].copy()

    # Ensure types similar to R code logic (treated as strings for join/factor behavior)
    # For pandas merge, types need not be strings as long as they match, but we mimic behavior
    # by leaving numeric IDs as-is and handling categorical coding in the model formula using C().

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
        # Retry with a different optimizer if needed
        try:
            result = model.fit(reml=True, method="nm", maxiter=400)
        except Exception as e2:
            with open(OUT_PATH, "w") as f:
                f.write("Model fitting failed.\n")
                f.write(str(e))
                f.write("\n--- Retry error ---\n")
                f.write(str(e2))
            return

    # Write summary to output
    with open(OUT_PATH, "w") as f:
        f.write("Linear mixed model: totinc ~ caste + domhigh + caste:domhigh + (1|village)\n")
        f.write("Software: Python statsmodels MixedLM (REML)\n\n")
        f.write(str(result.summary()))
        f.write("\n\nFixed effects coefficients:\n")
        fe = pd.DataFrame({
            "coef": result.fe_params,
            "se": result.bse_fe,
            "z": result.fe_params / result.bse_fe,
        })
        fe["pvalue"] = 2 * (1 - pd.Series(np.abs(fe["z"]))\
                               .apply(lambda z: 0.5 * (1 + np.math.erf(z / np.sqrt(2)))))
        # Above p-value via normal CDF approximation; use statsmodels if available
        try:
            from statsmodels.stats.weightstats import ztest
        except Exception:
            pass
        f.write(fe.to_string())

if __name__ == "__main__":
    main()
