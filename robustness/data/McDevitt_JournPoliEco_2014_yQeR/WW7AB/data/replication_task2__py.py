import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial as NBDiscrete

DATA_PATH = "/app/data/final_data.dta"
OUT_PATH_JSON = "/app/data/task2_results.json"


def load_data(path=DATA_PATH):
    df = pd.read_stata(path, convert_categoricals=False)
    return df.copy()


def main():
    df = load_data()
    # t-test as in do-file
    a1 = df.loc[df["first_A"] == 1, "complaints_2008"].astype(float)
    a0 = df.loc[df["first_A"] == 0, "complaints_2008"].astype(float)
    t_eq, p_eq = stats.ttest_ind(a1, a0, equal_var=True, nan_policy="omit")

    # Negative binomial: complaints_2008 ~ first_A
    y = df["complaints_2008"].values
    X = sm.add_constant(df[["first_A"]])
    nb = NBDiscrete(y, X).fit(disp=False, maxiter=400)
    b = float(nb.params["first_A"])
    se = float(nb.bse["first_A"]) 
    z = float(b / se)
    p = float(nb.pvalues["first_A"]) 

    results = {
        "ttest_equal_var_t": float(t_eq),
        "ttest_equal_var_p": float(p_eq),
        "nb_coef_first_A": b,
        "nb_se_first_A": se,
        "nb_z_first_A": z,
        "nb_p_first_A": p
    }

    with open(OUT_PATH_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
