import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# All IO must use /app/data
DATA_PATH = "/app/data/final_data.dta"

def main():
    df = pd.read_stata(DATA_PATH)

    # Negative Binomial with controls per Task2 instruction
    formula_controls = 'complaints_2008 ~ first_A + multiple_names + on_google + ad_spend_k + firm_age + chicago + emp_size'
    nb_c = smf.glm(formula=formula_controls, data=df, family=sm.families.NegativeBinomial()).fit()

    # Print main result in a standardized form (z-/t-/F-/chi2-family): here z from GLM
    coef = nb_c.params.get('first_A', float('nan'))
    zval = nb_c.tvalues.get('first_A', float('nan'))
    pval = nb_c.pvalues.get('first_A', float('nan'))
    print("RESULT|task2_negbin_controls|coef_first_A|{:.6f}".format(coef))
    print("RESULT|task2_negbin_controls|z_first_A|{:.6f}".format(zval))
    print("RESULT|task2_negbin_controls|p_first_A|{:.6g}".format(pval))

if __name__ == "__main__":
    main()
