import json
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, t
from sklearn.linear_model import LinearRegression


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return residuals of y after linear regression on X (with intercept)."""
    # Ensure 2D for X and 1D for y
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    y_hat = model.predict(X)
    return (y - y_hat).ravel()


def main():
    data_path = "/app/data/Dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Required dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    vars_needed = ["MiniK_Total", "DSM5_Total", "Age"]
    missing = [v for v in vars_needed if v not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    dsub = df[vars_needed].dropna()
    n = len(dsub)
    if n < 3:
        raise ValueError("Not enough observations after listwise deletion for partial correlation.")

    # Residualize MiniK_Total and DSM5_Total on Age
    X = dsub[["Age"]].to_numpy()
    miniK_res = residualize(dsub["MiniK_Total"].to_numpy().astype(float), X)
    dsm5_res = residualize(dsub["DSM5_Total"].to_numpy().astype(float), X)

    r, p = pearsonr(miniK_res, dsm5_res)

    results = {
        "task_id": "Task2",
        "analysis": "partial_correlation_miniK_psychopathology",
        "variables": {
            "x": "MiniK_Total",
            "y": "DSM5_Total",
            "controls": ["Age"]
        },
        "statistics": {
            "partial_r": float(r),
            "p_value": float(p),
            "n": int(n)
        }
    }

    out_path = "/app/data/Multi100-CCBE4-Task2_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also print a concise line for logs
    print(json.dumps({"result": results["statistics"], "output_file": out_path}))


if __name__ == "__main__":
    main()
