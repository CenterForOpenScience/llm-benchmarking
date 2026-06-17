import json
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable

# Configuration
DATA_PATH = os.environ.get("SANDRA_DATA_PATH", "/app/data/sandra_replicate.csv")
RESULTS_JSON_PATH = "/app/data/replication_results.json"
SUMMARIES_TXT_PATH = "/app/data/replication_model_summaries.txt"


def zscore(series):
    return (series - series.mean()) / series.std(ddof=0)


def fit_mixedlm_rt(df, iv_name="NFC_z"):
    # Mixed-effects linear model on standardized log RT for correct trials
    formula = f"logRT_z ~ {iv_name} * trial * rewardlevel + blocknumber"
    # Prepare modeling dataframe: keep only needed cols, drop missing, reset index to avoid statsmodels index issues
    cols = ["logRT_z", iv_name, "trial", "rewardlevel", "blocknumber", "SubjectID"]
    df_m = df[cols].dropna().copy().reset_index(drop=True)
    model = smf.mixedlm(formula, data=df_m, groups=df_m["SubjectID"], re_formula="1")
    res = model.fit(method="lbfgs", maxiter=1000, disp=False)
    return res


def fit_gee_accuracy(df, iv_name="NFC_z"):
    # Approximate GLMM (binomial) with GEE and exchangeable correlation by SubjectID
    formula = f"accuracy ~ {iv_name} * trial * rewardlevel + blocknumber"
    fam = sm.families.Binomial()
    cov = Exchangeable()
    # Prepare modeling dataframe
    cols = ["accuracy", iv_name, "trial", "rewardlevel", "blocknumber", "SubjectID"]
    df_m = df[cols].dropna().copy().reset_index(drop=True)
    model = smf.gee(formula, groups="SubjectID", data=df_m, family=fam, cov_struct=cov)
    res = model.fit()
    return res


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Basic sanity
    required_cols = [
        "SubjectID", "accuracy", "logRT", "trial", "rewardlevel", "blocknumber", "NFC", "Stroopindex"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Ensure correct dtypes
    for col in ["trial", "rewardlevel", "blocknumber", "accuracy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive z-scored predictors and outcomes
    # RT model uses only correct trials
    df_rt = df.loc[df["accuracy"] == 1].copy()
    # Only keep rows with required variables non-missing
    req_rt = ["logRT", "trial", "rewardlevel", "blocknumber", "SubjectID", "NFC", "Stroopindex"]
    df_rt = df_rt[req_rt].dropna().copy().reset_index(drop=True)
    df_rt["logRT_z"] = zscore(df_rt["logRT"])  # standardize within analysis sample
    df_rt["NFC_z"] = zscore(df_rt["NFC"])
    df_rt["Stroopindex_z"] = zscore(df_rt["Stroopindex"])  # note: sign may differ from original paper's definition

    # Full sample for accuracy models
    df_acc = df.copy()
    req_acc = ["accuracy", "trial", "rewardlevel", "blocknumber", "SubjectID", "NFC", "Stroopindex"]
    df_acc = df_acc[req_acc].dropna().copy().reset_index(drop=True)
    df_acc["NFC_z"] = zscore(df_acc["NFC"])
    df_acc["Stroopindex_z"] = zscore(df_acc["Stroopindex"])  # see note above

    # Fit models
    summaries_text = []
    results = {"dataset_path": DATA_PATH}

    # NFC RT model (focal replication)
    res_nfc_rt = fit_mixedlm_rt(df_rt, iv_name="NFC_z")
    term = "NFC_z:trial:rewardlevel"
    nfc_rt_est = res_nfc_rt.params.get(term, np.nan)
    nfc_rt_p = res_nfc_rt.pvalues.get(term, np.nan)
    nfc_rt_se = res_nfc_rt.bse.get(term, np.nan) if hasattr(res_nfc_rt, "bse") else np.nan
    ci_nfc = res_nfc_rt.conf_int().loc[term].tolist() if term in res_nfc_rt.conf_int().index else [np.nan, np.nan]
    results["nfc_rt_three_way"] = {
        "term": term,
        "estimate": float(nfc_rt_est) if pd.notnull(nfc_rt_est) else None,
        "std_err": float(nfc_rt_se) if pd.notnull(nfc_rt_se) else None,
        "conf_int": ci_nfc,
        "p_value": float(nfc_rt_p) if pd.notnull(nfc_rt_p) else None
    }
    summaries_text.append("=== MixedLM (RT) with NFC three-way ===\n" + str(res_nfc_rt.summary()) + "\n\n")

    # Stroop RT model (secondary replication of EF effect)
    res_stroop_rt = fit_mixedlm_rt(df_rt, iv_name="Stroopindex_z")
    term_s = "Stroopindex_z:trial:rewardlevel"
    s_rt_est = res_stroop_rt.params.get(term_s, np.nan)
    s_rt_p = res_stroop_rt.pvalues.get(term_s, np.nan)
    s_rt_se = res_stroop_rt.bse.get(term_s, np.nan) if hasattr(res_stroop_rt, "bse") else np.nan
    ci_s = res_stroop_rt.conf_int().loc[term_s].tolist() if term_s in res_stroop_rt.conf_int().index else [np.nan, npnan]
    results["stroop_rt_three_way"] = {
        "term": term_s,
        "estimate": float(s_rt_est) if pd.notnull(s_rt_est) else None,
        "std_err": float(s_rt_se) if pd.notnull(s_rt_se) else None,
        "conf_int": ci_s,
        "p_value": float(s_rt_p) if pd.notnull(s_rt_p) else None
    }
    summaries_text.append("=== MixedLM (RT) with Stroop three-way ===\n" + str(res_stroop_rt.summary()) + "\n\n")

    # Accuracy models (approximate with GEE)
    try:
        res_nfc_acc = fit_gee_accuracy(df_acc, iv_name="NFC_z")
        term_acc = "NFC_z:trial:rewardlevel"
        nfc_acc_est = res_nfc_acc.params.get(term_acc, np.nan)
        nfc_acc_p = res_nfc_acc.pvalues.get(term_acc, np.nan)
        nfc_acc_se = res_nfc_acc.bse.get(term_acc, np.nan) if hasattr(res_nfc_acc, "bse") else np.nan
        ci_acc = res_nfc_acc.conf_int().loc[term_acc].tolist() if term_acc in res_nfc_acc.conf_int().index else [np.nan, np.nan]
        results["nfc_accuracy_three_way_gee"] = {
            "term": term_acc,
            "estimate": float(nfc_acc_est) if pd.notnull(nfc_acc_est) else None,
            "std_err": float(nfc_acc_se) if pd.notnull(nfc_acc_se) else None,
            "conf_int": ci_acc,
            "p_value": float(nfc_acc_p) if pd.notnull(nfc_acc_p) else None
        }
        summaries_text.append("=== GEE (Accuracy) with NFC three-way ===\n" + str(res_nfc_acc.summary()) + "\n\n")
    except Exception as e:
        results["nfc_accuracy_three_way_gee_error"] = str(e)

    try:
        res_s_acc = fit_gee_accuracy(df_acc, iv_name="Stroopindex_z")
        term_s_acc = "Stroopindex_z:trial:rewardlevel"
        s_acc_est = res_s_acc.params.get(term_s_acc, np.nan)
        s_acc_p = res_s_acc.pvalues.get(term_s_acc, np.nan)
        s_acc_se = res_s_acc.bse.get(term_s_acc, np.nan) if hasattr(res_s_acc, "bse") else np.nan
        ci_s_acc = res_s_acc.conf_int().loc[term_s_acc].tolist() if term_s_acc in res_s_acc.conf_int().index else [np.nan, np.nan]
        results["stroop_accuracy_three_way_gee"] = {
            "term": term_s_acc,
            "estimate": float(s_acc_est) if pd.notnull(s_acc_est) else None,
            "std_err": float(s_acc_se) if pd.notnull(s_acc_se) else None,
            "conf_int": ci_s_acc,
            "p_value": float(s_acc_p) if pd.notnull(s_acc_p) else None
        }
        summaries_text.append("=== GEE (Accuracy) with Stroop three-way ===\n" + str(res_s_acc.summary()) + "\n\n")
    except Exception as e:
        results["stroop_accuracy_three_way_gee_error"] = str(e)

    # Sample sizes
    n_subjects = df["SubjectID"].nunique()
    n_trials = len(df)
    results["sample_size"] = {"subjects": int(n_subjects), "trials": int(n_trials)}

    # Save outputs
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)

    with open(SUMMARIES_TXT_PATH, "w") as f:
        f.write("\n".join(summaries_text))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
