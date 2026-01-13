import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=UserWarning)

CSV_PATH = "/app/data/Estimation Data - Pitts (126zz).csv"
SUMMARY_TXT = "/app/data/pitts_126zz_logit_summary.txt"
RESULTS_JSON = "/app/data/pitts_126zz_results.json"
ALT_CSV_PATH = "/workspace/replication_data/Estimation Data - Pitts (126zz).csv"
SOURCE_CSV = CSV_PATH

HYPOTHESIS = (
    "Among federal employees, higher overall job satisfaction (JobSat) is associated with a lower "
    "likelihood of intending to leave their agency within one year (LeavingAgency=1), controlling for covariates."
)

MODEL_VARS = [
    "LeavingAgency", "JobSat", "Over40", "NonMinority", "SatPay", "SatAdvan",
    "PerfCul", "Empowerment", "RelSup", "Relcow", "Over40xSatAdvan", "Agency"
]
PREDICTORS = [
    "JobSat", "Over40", "NonMinority", "SatPay", "SatAdvan",
    "PerfCul", "Empowerment", "RelSup", "Relcow", "Over40xSatAdvan"
]


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_data(path):
    if not os.path.exists(path):
        # Debug listing
        try:
            listing = os.listdir(os.path.dirname(path))
        except Exception:
            listing = []
        raise FileNotFoundError(f"CSV not found at {path}. Dir listing: {listing}")
    df = pd.read_csv(path)
    return df


def prepare_data(df):
    missing_cols = [c for c in MODEL_VARS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[MODEL_VARS].copy()

    # Coerce numeric predictors and outcome; leave Agency as-is (for clustering)
    numeric_cols = [c for c in MODEL_VARS if c != "Agency"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Listwise deletion
    df = df.dropna(subset=MODEL_VARS)

    # Ensure outcome is binary 0/1
    y_vals = sorted(df["LeavingAgency"].unique())
    if not set(y_vals).issubset({0, 1}):
        # Try to map if values are floats close to 0/1
        df["LeavingAgency"] = (df["LeavingAgency"] > 0.5).astype(int)

    return df


def fit_logit_cluster(df):
    y = df["LeavingAgency"].astype(int)
    X = df[PREDICTORS].copy()
    X = sm.add_constant(X, has_constant='add')

    model = sm.Logit(y, X)
    # Cluster-robust SEs by Agency directly in fit
    groups = df["Agency"].astype("category").cat.codes
    result = model.fit(disp=0, cov_type='cluster', cov_kwds={'groups': groups})
    robust_res = result  # already has robust covariance
    return result, robust_res, X


def compute_first_difference(result, X, col_name="JobSat"):
    base_pred = result.predict(X)
    sd = X[col_name].std(ddof=0)
    X_new = X.copy()
    X_new[col_name] = X_new[col_name] + sd
    new_pred = result.predict(X_new)
    fd = float(new_pred.mean() - base_pred.mean())
    return fd, float(sd)


def save_outputs(robust_res, result, df, first_diff, jobsat_sd):
    # Save summary text
    ensure_dir(SUMMARY_TXT)
    with open(SUMMARY_TXT, 'w') as f:
        f.write(robust_res.summary().as_text())
        f.write("\n\nNote: Covariance is cluster-robust by Agency.\n")

    # Extract key metrics for JobSat
    coef = float(robust_res.params.get("JobSat", np.nan))
    se = float(robust_res.bse.get("JobSat", np.nan))
    zval = float(robust_res.tvalues.get("JobSat", np.nan))
    pval = float(robust_res.pvalues.get("JobSat", np.nan))
    odds_ratio = float(np.exp(coef)) if np.isfinite(coef) else np.nan

    direction = "negative" if coef < 0 else ("positive" if coef > 0 else "null")
    sig = (
        "p < 0.001" if pval < 0.001 else
        "p < 0.01" if pval < 0.01 else
        "p < 0.05" if pval < 0.05 else
        "n.s."
    )

    results = {
        "hypothesis_tested": HYPOTHESIS,
        "N": int(df.shape[0]),
        "num_agencies": int(df["Agency"].nunique()),
        "model": "Logit with cluster-robust SEs (clusters=Agency)",
        "pseudo_r2": float(getattr(result, 'prsquared', np.nan)),
        "job_sat": {
            "coefficient": coef,
            "std_error": se,
            "z_value": zval,
            "p_value": pval,
            "odds_ratio": odds_ratio,
            "direction": direction,
            "significance": sig
        },
        "first_difference_job_sat_plus_1sd": {
            "delta_probability": first_diff,
            "job_sat_sd": jobsat_sd,
            "interpretation": "Average change in predicted probability of LeavingAgency when increasing JobSat by +1 SD (holding other covariates at observed values)."
        },
        "files": {
            "summary_txt": SUMMARY_TXT,
            "results_json": RESULTS_JSON,
            "source_csv": SOURCE_CSV
        }
    }

    ensure_dir(RESULTS_JSON)
    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    try:
        global SOURCE_CSV
        csv_path = CSV_PATH
        if (not os.path.exists(csv_path)) and os.path.exists(ALT_CSV_PATH):
            csv_path = ALT_CSV_PATH
        if not os.path.exists(csv_path):
            listings = {}
            for p in [os.path.dirname(CSV_PATH), os.path.dirname(ALT_CSV_PATH)]:
                try:
                    listings[p] = os.listdir(p)
                except Exception as e:
                    listings[p] = f"unavailable: {e}"
            raise FileNotFoundError(f"CSV not found at either '{CSV_PATH}' or '{ALT_CSV_PATH}'. Listings: {listings}")
        SOURCE_CSV = csv_path
        df = load_data(SOURCE_CSV)
        df = prepare_data(df)
        result, robust_res, X = fit_logit_cluster(df)
        fd, sd = compute_first_difference(result, X, col_name="JobSat")
        results = save_outputs(robust_res, result, df, fd, sd)
        print(json.dumps({"status": "success", "summary": {
            "N": results["N"],
            "num_agencies": results["num_agencies"],
            "job_sat_coef": results["job_sat"]["coefficient"],
            "job_sat_p": results["job_sat"]["p_value"],
            "first_diff": results["first_difference_job_sat_plus_1sd"]["delta_probability"]
        }}, indent=2))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
