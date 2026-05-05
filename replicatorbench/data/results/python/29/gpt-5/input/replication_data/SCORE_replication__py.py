import json
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

# IO paths (all IO must be under /app/data)
INPUT_CSV = "/app/data/SCORE_ALL_DATA.csv"
RESULTS_JSON = "/app/data/replication_results_study29.json"
RESULTS_TXT = "/app/data/replication_results_study29.txt"


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Expected data file at {INPUT_CSV}. Please place SCORE_ALL_DATA.csv into /app/data.")

    df = pd.read_csv(INPUT_CSV)

    # Ensure Exclusion is numeric, then keep only usable rows as per preregistered coding (0 = used in analysis)
    if "Exclusion" not in df.columns:
        raise ValueError("Exclusion column not found in dataset. Cannot apply preregistered exclusion criteria.")
    df["Exclusion"] = pd.to_numeric(df["Exclusion"], errors="coerce")
    d = df[df["Exclusion"] == 0].copy()

    # Outcome: average of S1_likely, S1_quality, S1_purchase (coerce to numeric first)
    required_dv = ["S1_likely", "S1_quality", "S1_purchase"]
    for col in required_dv:
        if col not in d.columns:
            raise ValueError(f"Required DV component '{col}' not found in dataset.")
    d[required_dv] = d[required_dv].apply(pd.to_numeric, errors="coerce")

    d["average"] = d[required_dv].mean(axis=1)

    # Contrast coding based on TYPE: Cl/Co (clarity), B/U (bimodal/unimodal)
    if "TYPE" not in d.columns:
        raise ValueError("TYPE column not found to reconstruct contrast codes.")

    # Normalize TYPE strings (strip whitespace)
    d["TYPE"] = d["TYPE"].astype(str).str.strip()

    # Map clarity: Cl -> 1 (clear), Co -> -1 (conflicted)
    def code_clarity(val):
        if isinstance(val, str):
            if val.startswith("Cl"):
                return 1
            if val.startswith("Co"):
                return -1
        return np.nan

    # Map experimental: B -> 1 (bimodal), U -> -1 (unimodal)
    def code_experimental(val):
        if isinstance(val, str):
            v = val.strip()
            if v.endswith("B"):
                return 1
            if v.endswith("U"):
                return -1
        return np.nan

    d["clarity"] = d["TYPE"].apply(code_clarity)
    d["experimental"] = d["TYPE"].apply(code_experimental)
    d["interaction"] = d["clarity"] * d["experimental"]

    # Ensure numeric types for modeling
    for col in ["clarity", "experimental", "interaction", "average"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # Drop rows with missing codes or DVs
    d = d.dropna(subset=["average", "clarity", "experimental", "interaction"]).copy()

    if d.shape[0] == 0:
        raise ValueError("No analyzable rows remain after exclusions and type conversions. Check the dataset and coding.")

    # Fit OLS: average ~ clarity + experimental + interaction (with intercept)
    X = d[["clarity", "experimental", "interaction"]].astype(float)
    X = sm.add_constant(X)
    y = d["average"].astype(float)

    try:
        model = sm.OLS(y, X, missing='drop').fit()
    except Exception as e:
        # Provide diagnostics to help debugging if dtype issues persist
        diag = {
            "X_dtypes": X.dtypes.apply(lambda t: str(t)).to_dict(),
            "y_dtype": str(y.dtype),
            "head_X": X.head().to_dict(orient="list"),
            "head_y": y.head().tolist(),
            "n": int(len(y))
        }
        raise RuntimeError(f"Failed to fit OLS due to: {e}. Diagnostics: {json.dumps(diag)[:500]}...")

    # Compute cell means for descriptive check
    cell_means = d.groupby("TYPE")["average"].mean().to_dict()
    cell_counts = d.groupby("TYPE")["average"].size().to_dict()

    # Package results
    coef_table = {}
    for name in ["const", "clarity", "experimental", "interaction"]:
        if name in model.params.index:
            coef_table[name] = {
                "b": float(model.params[name]),
                "se": float(model.bse[name]),
                "t": float(model.tvalues[name]),
                "p": float(model.pvalues[name])
            }

    results = {
        "n_after_exclusion_and_complete": int(len(d)),
        "cell_means_average": {k: float(v) for k, v in cell_means.items()},
        "cell_counts": {k: int(v) for k, v in cell_counts.items()},
        "coefficients": coef_table,
        "model_summary_short": {
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj)
        },
        "focal_test": {
            "term": "interaction",
            "hypothesized_direction": "negative",
            "estimate": coef_table.get("interaction", {}).get("b"),
            "se": coef_table.get("interaction", {}).get("se"),
            "t": coef_table.get("interaction", {}).get("t"),
            "p": coef_table.get("interaction", {}).get("p")
        }
    }

    # Save JSON
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # Save text summary
    with open(RESULTS_TXT, "w") as f:
        f.write("Replication of clarity x rating distribution on desirability (average)\n")
        f.write(f"N (analyzed): {results['n_after_exclusion_and_complete']}\n\n")
        f.write("Cell means (average):\n")
        for k, v in sorted(results["cell_means_average"].items()):
            f.write(f"  {k}: {v:.3f} (n={results['cell_counts'].get(k, 0)})\n")
        f.write("\nCoefficients:\n")
        for name, vals in coef_table.items():
            f.write(f"  {name}: b={vals['b']:.3f}, SE={vals['se']:.3f}, t={vals['t']:.2f}, p={vals['p']:.3f}\n")

    # Also print focal result to stdout for convenience
    inter = results["focal_test"]
    if inter["estimate"] is not None:
        print("Focal interaction (clarity x experimental): ", end="")
        print(f"b={inter['estimate']:.3f}, SE={inter['se']:.3f}, t={inter['t']:.2f}, p={inter['p']:.3f}")
    else:
        print("Focal interaction (clarity x experimental): estimate not available (check data)")


if __name__ == "__main__":
    main()
