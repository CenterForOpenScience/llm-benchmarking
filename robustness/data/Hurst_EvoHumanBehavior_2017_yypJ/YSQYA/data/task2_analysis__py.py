import json
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "/app/data/1-s2.0-S1090513816301118-mmc1.csv"
OUT_PATH = "/app/data/task2_results.json"

REQ_VARS = ["DSM5_Total", "MiniK_Total", "Age"]


def load_data(path):
    return pd.read_csv(path)


def partial_corr_resid(x, y, controls):
    Xc = sm.add_constant(controls)
    model_x = sm.OLS(x, Xc, missing='drop').fit()
    model_y = sm.OLS(y, Xc, missing='drop').fit()
    rx = x - model_x.fittedvalues
    ry = y - model_y.fittedvalues
    common = rx.dropna().index.intersection(ry.dropna().index)
    rx = rx.loc[common]
    ry = ry.loc[common]
    if len(common) < 3:
        return np.nan, np.nan, len(common)
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p), int(len(common))


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df = load_data(DATA_PATH)

    missing = [c for c in REQ_VARS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    for c in REQ_VARS:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    r, p, n = partial_corr_resid(df["MiniK_Total"], df["DSM5_Total"], df[["Age"]])

    results = {
        "task": "Task2",
        "dataset": os.path.basename(DATA_PATH),
        "partial_correlation": {
            "pair": ["MiniK_Total", "DSM5_Total"],
            "control": ["Age"],
            "r": r,
            "p": p,
            "n": n
        }
    }

    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(json.dumps({
        "status": "ok",
        "message": "Task2 completed",
        "output": OUT_PATH
    }))


if __name__ == "__main__":
    main()
