# Python translation of the R replication script
# Executes mixed effects models approximating lmer with nested random intercepts
# Saves outputs to /app/data

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import pyreadr
import statsmodels.formula.api as smf
import shutil

warnings.filterwarnings("ignore", category=UserWarning)
RUN_SUMMARY = []

OUT_DIR = "/app/data"
DATA_PATH = os.path.join(OUT_DIR, "Final replication dataset.rds")
os.makedirs(OUT_DIR, exist_ok=True)
# Resolve dataset path from common locations
ALT_PATHS = [
    os.path.join(OUT_DIR, "Final replication dataset.rds"),
    "/workspace/replication_data/Final replication dataset.rds",
    "/workspace/Final replication dataset.rds",
]
for _p in ALT_PATHS:
    if os.path.exists(_p):
        DATA_PATH = _p
        break
ALT_OUT_DIR = "/workspace/artifacts"
os.makedirs(ALT_OUT_DIR, exist_ok=True)


def log(msg: str):
    print(msg, flush=True)


def standardize_fixed_effects(result):
    try:
        fe_params = result.fe_params
        bse_fe = result.bse_fe
        exog = result.model.exog
        names = result.model.exog_names
        y = result.model.endog
        sdy = float(np.std(y, ddof=0)) if np.std(y, ddof=0) != 0 else np.nan
        sdx = np.std(exog, axis=0, ddof=0)
        rows = []
        for j, name in enumerate(names):
            # avoid division by zero
            if sdy is None or np.isnan(sdy) or sdy == 0 or sdx[j] == 0:
                stdcoef = np.nan
                stdse = np.nan
            else:
                stdcoef = fe_params[j] * (sdx[j] / sdy)
                stdse = bse_fe[j] * (sdx[j] / sdy)
            rows.append({
                "term": name,
                "stdcoef": stdcoef,
                "stdse": stdse
            })
        return pd.DataFrame(rows)
    except Exception as e:
        log(f"Failed to compute standardized coefficients: {e}")
        return None


def fit_mixedlm(formula: str, data: pd.DataFrame, group_col: str, country_col: str, model_id: str):
    pass

def fit_mixedlm_req(formula: str, data: pd.DataFrame, group_col: str, country_col: str, model_id: str, required_cols: list):
    # Drop NAs only on explicitly required columns and grouping columns, keep formula intact for Patsy
    needed_cols = set(required_cols or [])
    needed_cols.update([group_col, country_col])
    try:
        outcome = formula.split("~")[0].strip()
        if outcome:
            needed_cols.add(outcome)
    except Exception:
        pass
    dfm = data.dropna(subset=list(needed_cols)).copy()
    vc_formula = {"country": "0 + C(" + country_col + ")"}
    log(f"Fitting MixedLM for {model_id} with {len(dfm)} rows...")
    model = smf.mixedlm(formula=formula, data=dfm, groups=dfm[group_col], vc_formula=vc_formula)
    try:
        res = model.fit(reml=False, method="lbfgs", maxiter=200)
    except Exception as e:
        log(f"Primary optimizer failed ({e}); retrying with Nelder-Mead (may be slow)...")
        res = model.fit(reml=False, method="nm", maxiter=400)
    return res, dfm
    # Prepare data: drop NAs on variables used in the formula and grouping columns
    needed_cols = set([group_col, country_col])
    # Extract variable names from Patsy formula terms
    # Conservative: drop rows with any NA in involved columns
    for token in ["+", "~", "*", ":", "(", ")"]:
        formula = formula.replace(token, " ")
    for var in formula.split():
        if var and var not in ["1", "C"] and not var.startswith("C("):
            needed_cols.add(var)
    # Also include inside C()
    if "C(" in formula:
        start = 0
        while True:
            i = formula.find("C(", start)
            if i == -1:
                break
            j = formula.find(")", i)
            if j == -1:
                break
            v = formula[i+2:j].strip("() ")
            needed_cols.add(v)
            start = j + 1
    dfm = data.dropna(subset=list(needed_cols)).copy()

    # Build vc_formula for country random intercepts, approximating (1|country) + (1|country:school)
    # groups = school_id; variance component per country
    vc_formula = {"country": "0 + C(" + country_col + ")"}

    log(f"Fitting MixedLM for {model_id} with {len(dfm)} rows...")
    model = smf.mixedlm(formula=formula, data=dfm, groups=dfm[group_col], vc_formula=vc_formula)
    try:
        res = model.fit(reml=False, method="lbfgs", maxiter=200)
    except Exception as e:
        log(f"Primary optimizer failed ({e}); retrying with Nelder-Mead (may be slow)...")
        res = model.fit(reml=False, method="nm", maxiter=400)
    return res, dfm


def save_results(res, dfm, model_id: str):
    # Save textual summary
    summary_path = os.path.join(OUT_DIR, f"{model_id}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(res.summary().as_text())

    # Save fixed effects table
    fe = pd.DataFrame({
        "term": res.fe_params.index,
        "coef": res.fe_params.values,
        "std_err": res.bse_fe.values,
        "z": res.fe_params.values / res.bse_fe.values,
        "p_value": res.pvalues[res.fe_params.index].values,
    })
    ci = res.conf_int().loc[res.fe_params.index]
    fe["ci_lower"] = ci[0].values
    fe["ci_upper"] = ci[1].values
    fe.to_csv(os.path.join(OUT_DIR, f"{model_id}_fixed_effects.csv"), index=False)

    # Standardized coefficients
    std_df = standardize_fixed_effects(res)
    if std_df is not None:
        std_df.to_csv(os.path.join(OUT_DIR, f"{model_id}_std_coefs.csv"), index=False)

    # Naive variance explained metric (not GLMM R2): 1 - Var(resid)/Var(y)
    try:
        resid = res.resid
        y = res.model.endog
        r2_naive = float(1.0 - (np.var(resid, ddof=0) / np.var(y, ddof=0))) if np.var(y, ddof=0) > 0 else np.nan
    except Exception:
        r2_naive = np.nan

    meta = {
        "model_id": model_id,
        "n_obs": int(len(dfm)),
        "converged": bool(getattr(res, "converged", False)),
        "aic": float(getattr(res, "aic", np.nan)) if hasattr(res, "aic") else np.nan,
        "bic": float(getattr(res, "bic", np.nan)) if hasattr(res, "bic") else np.nan,
        "llf": float(getattr(res, "llf", np.nan)) if hasattr(res, "llf") else np.nan,
        "r2_naive": r2_naive,
    }
    with open(os.path.join(OUT_DIR, f"{model_id}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Mirror artifacts to ALT_OUT_DIR for host retrieval
    try:
        for fname in [f"{model_id}_summary.txt", f"{model_id}_fixed_effects.csv", f"{model_id}_std_coefs.csv", f"{model_id}_meta.json"]:
            src = os.path.join(OUT_DIR, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(ALT_OUT_DIR, fname))
    except Exception as e:
        log(f"Warning: failed to mirror artifacts for {model_id}: {e}")

    # Append bilingual term summary to RUN_SUMMARY
    try:
        term = "bilingual"
        if hasattr(res, "fe_params") and term in res.fe_params.index:
            coef = float(res.fe_params[term])
            se = float(res.bse_fe[term]) if term in res.bse_fe.index else float("nan")
            pval = float(res.pvalues[term]) if term in res.pvalues.index else float("nan")
            ci_df = res.conf_int()
            if term in ci_df.index:
                ci_lower = float(ci_df.loc[term, 0])
                ci_upper = float(ci_df.loc[term, 1])
            else:
                ci_lower = float("nan")
                ci_upper = float("nan")
            RUN_SUMMARY.append({
                "model_id": model_id,
                "term": term,
                "coef": coef,
                "std_err": se,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": pval,
                "direction": "positive" if coef > 0 else ("negative" if coef < 0 else "null")
            })
    except Exception as e:
        log(f"Warning: failed to append RUN_SUMMARY for {model_id}: {e}")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Ensure it is mounted to /app/data.")

    log("Loading RDS dataset via pyreadr...")
    rdr = pyreadr.read_r(DATA_PATH)
    # If single object, key is None
    if None in rdr.keys():
        df = rdr[None]
    else:
        # Take the first item
        df = list(rdr.items())[0][1]

    log(f"Loaded dataframe with shape {df.shape}")

    # Variable construction per R script
    # bilingual: 0 if I03_ST_A_S26A == 1, 1 otherwise (replicating the R logic where (2|3) coerces to 1)
    if "I03_ST_A_S26A" not in df.columns:
        raise KeyError("I03_ST_A_S26A not found in dataset")
    df["bilingual"] = (~(df["I03_ST_A_S26A"] == 1)).astype(float)
    df = df[~df["bilingual"].isna()].copy()

    # Exclude students who speak English at home
    if "I03_ST_A_S27B" not in df.columns:
        raise KeyError("I03_ST_A_S27B not found in dataset")
    df = df[df["I03_ST_A_S27B"] == 0].copy()

    # Create average scores
    for base, cols in {
        "ave_writing": ["PV1_WRIT_C","PV2_WRIT_C","PV3_WRIT_C","PV4_WRIT_C","PV5_WRIT_C"],
        "ave_reading": ["PV1_READ","PV2_READ","PV3_READ","PV4_READ","PV5_READ"],
        "ave_listening": ["PV1_LIST","PV2_LIST","PV3_LIST","PV4_LIST","PV5_LIST"],
    }.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns for {base}: {missing}")
        df[base] = df[cols].mean(axis=1, skipna=True)

    df["average_english"] = df[["ave_writing","ave_reading","ave_listening"]].mean(axis=1, skipna=True)

    # Cultural capital recode
    if "SQt21i01" not in df.columns:
        raise KeyError("SQt21i01 not found in dataset")
    cap_map = {
        "0-10 books": 0,
        "11-25 books": 1,
        "26-100 books": 2,
        "101-200 books": 3,
        "201-500 books": 4,
        "More than 500 books": 5,
    }
    df["Cultural_capital"] = df["SQt21i01"].map(cap_map)
    df["Cultural_capital"] = pd.to_numeric(df["Cultural_capital"], errors="coerce")

    # Filter: any of the weights > 0
    for w in ["FSW_WRIT_TR","FSW_READ_TR","FSW_LIST_TR"]:
        if w not in df.columns:
            raise KeyError(f"{w} not found in dataset")
    df = df[(df["FSW_WRIT_TR"] > 0) | (df["FSW_READ_TR"] > 0) | (df["FSW_LIST_TR"] > 0)].copy()

    # Centering and Z-scores
    if "I08_ST_A_S02A" not in df.columns:
        raise KeyError("I08_ST_A_S02A (age) not found in dataset")
    if "HISEI" not in df.columns:
        raise KeyError("HISEI not found in dataset")
    if "PARED" not in df.columns:
        raise KeyError("PARED not found in dataset")

    df["c_age"] = df["I08_ST_A_S02A"] - df["I08_ST_A_S02A"].mean()
    df["c_HISEI"] = df["HISEI"] - df["HISEI"].mean()

    def zscore(s: pd.Series):
        return (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else np.nan)

    df["Z_Parental"] = zscore(df["PARED"]) 
    df["Z_Cultural"] = zscore(df["Cultural_capital"]) 

    # IDs required for random effects
    for idcol in ["country_id", "school_id"]:
        if idcol not in df.columns:
            raise KeyError(f"{idcol} not found in dataset")

    # Factor variable present in formula
    if "SQt01i01" not in df.columns:
        raise KeyError("SQt01i01 not found in dataset")

    # Model 1: average_english
    formula_main = "average_english ~ 1 + bilingual + C(SQt01i01) + c_age + c_HISEI + Z_Parental + Z_Cultural"
    res_main, dfm_main = fit_mixedlm_req(formula_main, df, group_col="school_id", country_col="country_id", model_id="model_average_english", required_cols=["bilingual","SQt01i01","c_age","c_HISEI","Z_Parental","Z_Cultural"])
    save_results(res_main, dfm_main, model_id="model_average_english")

    # Separate datasets for writing, reading, listening
    datasets = {
        "writing": df[df["FSW_WRIT_TR"] > 0].copy(),
        "reading": df[df["FSW_READ_TR"] > 0].copy(),
        "listening": df[df["FSW_LIST_TR"] > 0].copy(),
    }

    for name, dsub in datasets.items():
        # Recompute centered and z-scored variables per subset as in R script
        dsub["c_age"] = dsub["I08_ST_A_S02A"] - dsub["I08_ST_A_S02A"].mean()
        dsub["c_HISEI"] = dsub["HISEI"] - dsub["HISEI"].mean()
        dsub["Z_Parental"] = zscore(dsub["PARED"]) 
        dsub["Z_Cultural"] = zscore(dsub["Cultural_capital"]) 
        outcome = f"ave_{name}"
        if outcome not in dsub.columns:
            raise KeyError(f"{outcome} not found in dataset")
        formula = f"{outcome} ~ 1 + bilingual + C(SQt01i01) + c_age + c_HISEI + Z_Parental + Z_Cultural"
        res, dfm = fit_mixedlm_req(formula, dsub, group_col="school_id", country_col="country_id", model_id=f"model_{name}", required_cols=["bilingual","SQt01i01","c_age","c_HISEI","Z_Parental","Z_Cultural", outcome])
        save_results(res, dfm, model_id=f"model_{name}")

    log("All models estimated. Results saved to /app/data.")
    # Persist and print summary
    try:
        summary_json = json.dumps(RUN_SUMMARY, indent=2)
        with open(os.path.join(OUT_DIR, "run_summary.json"), "w") as f:
            f.write(summary_json)
        with open(os.path.join(ALT_OUT_DIR, "run_summary.json"), "w") as f:
            f.write(summary_json)
        print("RUN_SUMMARY_JSON:" + summary_json, flush=True)
    except Exception as e:
        log(f"Warning: failed to write RUN_SUMMARY: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
