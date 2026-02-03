import os
import sys
import json
import pandas as pd
import numpy as np
# Ensure pyreadstat is available at runtime
try:
    import pyreadstat
except Exception as _e_import:
    # Log the specific import error for debugging
    try:
        print(f"pyreadstat import error (pre-install): {_e_import}", file=sys.stderr)
    except Exception:
        pass
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyreadstat==1.2.7"])  # pinned version
        try:
            import pyreadstat  # retry
        except Exception as _e_reimport:
            try:
                print(f"pyreadstat import error (post-install): {_e_reimport}", file=sys.stderr)
            except Exception:
                pass
            pyreadstat = None
    except Exception as _e_pip:
        try:
            print(f"pyreadstat pip install failed: {_e_pip}", file=sys.stderr)
        except Exception:
            pass
        pyreadstat = None
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "/app/data"
INPUT_CANDIDATES = [
    os.path.join(DATA_PATH, "GSSreplication.dta"),
    os.path.join(DATA_PATH, "GSSreplication.csv"),
    os.path.join(DATA_PATH, "gssreplication.dta"),
    os.path.join(DATA_PATH, "gssreplication.csv"),
]

LOG_PATH = os.path.join(DATA_PATH, "replication_log.txt")
SUMMARY_CSV = os.path.join(DATA_PATH, "replication_evolution_summary.csv")
RESULTS_JSON = os.path.join(DATA_PATH, "replication_evolution_results.json")


def log(msg: str):
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")
    print(msg)


def load_data():
    last_err = None
    for fp in INPUT_CANDIDATES:
        if os.path.exists(fp):
            try:
                if fp.lower().endswith(".dta"):
                    try:
                        df = pd.read_stata(fp, convert_categoricals=True)
                    except Exception as e_pd:
                        # Fallback to pyreadstat for broader Stata version support
                        try:
                            if pyreadstat is None:
                                raise RuntimeError("pyreadstat unavailable to read Stata file.")
                            df, meta = pyreadstat.read_dta(fp, apply_value_formats=True)
                        except Exception as e_pr:
                            raise e_pr
                else:
                    df = pd.read_csv(fp)
                log(f"Loaded dataset: {fp} with shape {df.shape}")
                return df, fp
            except Exception as e:
                last_err = e
                log(f"Failed to load {fp}: {e}")
    raise RuntimeError(f"Could not load dataset from candidates. Last error: {last_err}")


def find_evolution_column(df: pd.DataFrame):
    # Priority candidates by name
    preferred = [
        "evolved", "evol", "evolution", "humans_evolved", "evo", "evo_item",
        "evoltrue", "evoltf", "evoltf_corr", "evol_correct"
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for key in preferred:
        if key in cols_lower:
            return cols_lower[key]
    # Fallback: any column containing 'evol'
    for c in df.columns:
        cl = c.lower()
        if "evol" in cl or ("human" in cl and ("animal" in cl or "species" in cl)):
            return c
    return None


def recode_evolution(series: pd.Series) -> pd.Series:
    # Target: 1 if respondent endorses that humans evolved from other animals (correct), 0 otherwise.
    s = series.copy()
    if pd.api.types.is_categorical_dtype(s):
        s = s.astype(str)
    if s.dtype == object:
        s_str = s.astype(str).str.strip().str.lower()
        true_like = ["true", "agree", "correct", "developed", "develop", "yes", "1", "t"]
        false_like = ["false", "disagree", "incorrect", "not", "no", "0", "f"]
        # Common GSS phrasing: "Human beings developed from earlier species of animals" (True/False)
        # Many datasets store responses as 'true'/'false' or '1'/'2'. We'll handle both broadly.
        out = pd.Series(index=s.index, dtype="float")
        out.loc[s_str.isin(true_like) | s_str.str.contains("true", na=False) | s_str.str.contains("develop", na=False)] = 1
        out.loc[s_str.isin(false_like) | s_str.str.contains("false", na=False) | s_str.str.contains("not", na=False)] = 0
        # Any explicitly labeled like 'correct' for key
        out.loc[s_str.str.contains("correct", na=False)] = 1
        # Don't know, refused, NA remain NaN
        return out.astype("float")
    else:
        # numeric codes: try common encodings
        # If there are only two unique non-missing values, map max to 1 if value labels exist; otherwise assume 1=True, 2=False
        s_num = pd.to_numeric(s, errors="coerce")
        uniq = sorted([u for u in pd.unique(s_num) if pd.notnull(u)])
        if len(uniq) == 2:
            # Heuristic: if values are {0,1} use as-is; if {1,2} assume 1=True,2=False; if {2,1} similar; if {1,0} use as-is
            if set(uniq) == {0.0, 1.0}:
                return s_num.astype(float)
            elif set(uniq) == {1.0, 2.0}:
                return (s_num == 1.0).astype(float)
        # Otherwise, can't confidently map
        return pd.Series([np.nan]*len(s), index=s.index, dtype="float")


def find_class_column(df: pd.DataFrame):
    # Try to detect the latent class membership column
    candidates = [
        "class", "lca_class", "class3", "latent_class", "perspective", "persp", "class_label",
        "relig_science_class", "sci_rel_class"
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for key in candidates:
        if key in cols_lower:
            return cols_lower[key]
    # Fallback: any column containing 'class'
    for c in df.columns:
        if "class" in c.lower():
            return c
    # Fallback: any column containing 'perspective'
    for c in df.columns:
        if "perspect" in c.lower():
            return c
    return None


def normalize_class_values(series: pd.Series) -> pd.Series:
    s = series.copy()
    if pd.api.types.is_categorical_dtype(s):
        s = s.astype(str)
    if s.dtype != object:
        s = s.astype(str)
    s_lower = s.str.strip().str.lower()
    out = pd.Series(index=s.index, dtype=object)
    # Map by substring matching
    out.loc[s_lower.str.contains("trad", na=False)] = "Traditional"
    out.loc[s_lower.str.contains("modern", na=False)] = "Modern"
    out.loc[s_lower.str.contains("post", na=False)] = "Post-secular"
    # If nothing mapped, return original
    if out.notna().sum() == 0:
        # Try numeric categories 1,2,3 with labelled order assumption: 1=Traditional, 2=Modern, 3=Post-secular
        try:
            s_num = pd.to_numeric(series, errors="coerce")
            mapped = s_num.map({1: "Traditional", 2: "Modern", 3: "Post-secular"})
            if mapped.notna().sum() > 0:
                return mapped
        except Exception:
            pass
        return series.astype(str)
    return out


def main():
    os.makedirs(DATA_PATH, exist_ok=True)
    # Clear log
    try:
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)
    except Exception:
        pass

    df, used_path = load_data()

    evol_col = find_evolution_column(df)
    if evol_col is None:
        log("ERROR: Could not find an evolution-related column in the dataset.")
        sys.exit(2)
    log(f"Detected evolution column: {evol_col}")

    evol_binary = recode_evolution(df[evol_col])
    df["evol_binary"] = evol_binary
    log(f"After recoding, non-missing evolution responses: {df['evol_binary'].notna().sum()}")

    class_col = find_class_column(df)
    if class_col is None:
        log("ERROR: Could not find a latent-class column in the dataset.")
        sys.exit(2)
    log(f"Detected class column: {class_col}")

    class_norm = normalize_class_values(df[class_col])
    df["class_norm"] = class_norm

    # Keep only rows with valid outcome and class
    work = df.loc[df["evol_binary"].isin([0.0, 1.0]) & df["class_norm"].isin(["Traditional", "Modern", "Post-secular"])].copy()
    log(f"Analysis sample size: {len(work)}")

    if len(work) == 0:
        log("ERROR: No valid observations after filtering. Check variable mappings.")
        sys.exit(2)

    # Compute proportions by class
    summ = work.groupby("class_norm")["evol_binary"].agg(["count", "mean"]).reset_index()
    summ.columns = ["class", "n", "prop_evolved"]
    # Ensure all three classes present (if not, still proceed but note)
    present = set(summ["class"].tolist())
    for cls in ["Traditional", "Modern", "Post-secular"]:
        if cls not in present:
            log(f"WARNING: Class not present in analysis: {cls}")

    summ.to_csv(SUMMARY_CSV, index=False)
    log(f"Saved summary to {SUMMARY_CSV}")

    # Logistic regression: evol_binary ~ C(class_norm), reference = Traditional
    work["class_norm"] = pd.Categorical(work["class_norm"], categories=["Traditional", "Modern", "Post-secular"], ordered=False)
    try:
        model = smf.logit("evol_binary ~ C(class_norm)", data=work).fit(disp=False)
        params = model.params.to_dict()
        pvalues = model.pvalues.to_dict()
        # Extract Post-secular vs Traditional coefficient
        key_ps = "C(class_norm)[T.Post-secular]"
        key_mod = "C(class_norm)[T.Modern]"
        coef_ps = params.get(key_ps, float("nan"))
        pval_ps = pvalues.get(key_ps, float("nan"))
        coef_mod = params.get(key_mod, float("nan"))
        pval_mod = pvalues.get(key_mod, float("nan"))
        results = {
            "data_file": os.path.basename(used_path),
            "n_analysis": int(len(work)),
            "proportions": summ.to_dict(orient="records"),
            "logit": {
                "formula": "evol_binary ~ C(class_norm)",
                "coef_post_secular_vs_traditional": coef_ps,
                "p_post_secular_vs_traditional": pval_ps,
                "coef_modern_vs_traditional": coef_mod,
                "p_modern_vs_traditional": pval_mod,
                "llf": float(model.llf),
                "aic": float(model.aic),
                "bic": float(model.bic) if hasattr(model, "bic") else None
            }
        }
        with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        log(f"Saved results to {RESULTS_JSON}")
    except Exception as e:
        log(f"ERROR fitting logistic regression: {e}")
        # Save at least the proportions
        results = {
            "data_file": os.path.basename(used_path),
            "n_analysis": int(len(work)),
            "proportions": summ.to_dict(orient="records"),
            "logit_error": str(e)
        }
        with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()

# Weighted analysis support appended below
# We try to detect common GSS weights and run weighted summaries and GLM if available.

def detect_weight_column(df: pd.DataFrame):
    weight_candidates = [
        "wtssall", "wtssnr", "wtss", "weight", "wgt", "weight_var", "weight_final"
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for key in weight_candidates:
        if key in cols_lower:
            return cols_lower[key]
    # Fallback: any column containing 'wt'
    for c in df.columns:
        if "wt" in c.lower():
            return c
    return None


def weighted_group_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan
    v = values[mask].astype(float)
    w = weights[mask].astype(float)
    if w.sum() == 0:
        return np.nan
    return float((v * w).sum() / w.sum())


def run_weighted_analysis():
    try:
        df, used_path = load_data()
        evol_col = find_evolution_column(df)
        class_col = find_class_column(df)
        if evol_col is None or class_col is None:
            log("Weighted analysis skipped: missing key columns.")
            return
        weight_col = detect_weight_column(df)
        if weight_col is None:
            log("No survey weight detected; weighted analysis skipped.")
            return
        log(f"Detected weight column: {weight_col}")
        df["evol_binary"] = recode_evolution(df[evol_col])
        df["class_norm"] = normalize_class_values(df[class_col])
        work = df.loc[df["evol_binary"].isin([0.0, 1.0]) & df["class_norm"].isin(["Traditional", "Modern", "Post-secular"])].copy()
        work = work.loc[work[weight_col].notna() & (work[weight_col] > 0)].copy()
        if len(work) == 0:
            log("Weighted analysis has empty sample.")
            return
        # Weighted proportions by class
        wprops = []
        for cls, sub in work.groupby("class_norm"):
            prop = weighted_group_mean(sub["evol_binary"], sub[weight_col])
            wprops.append({"class": cls, "weighted_prop_evolved": prop, "n": int(len(sub))})
        # Weighted GLM (Binomial) with frequency weights
        work["class_norm"] = pd.Categorical(work["class_norm"], categories=["Traditional", "Modern", "Post-secular"], ordered=False)
        try:
            glm_model = smf.glm("evol_binary ~ C(class_norm)", data=work, family=sm.families.Binomial(), freq_weights=work[weight_col]).fit()
            params = glm_model.params.to_dict()
            pvalues = glm_model.pvalues.to_dict()
            key_ps = "C(class_norm)[T.Post-secular]"
            key_mod = "C(class_norm)[T.Modern]"
            coef_ps = params.get(key_ps, float("nan"))
            pval_ps = pvalues.get(key_ps, float("nan"))
            coef_mod = params.get(key_mod, float("nan"))
            pval_mod = pvalues.get(key_mod, float("nan"))
            weighted = {
                "weight_column": weight_col,
                "n_analysis_weighted": int(len(work)),
                "weighted_proportions": wprops,
                "glm_binomial": {
                    "formula": "evol_binary ~ C(class_norm)",
                    "coef_post_secular_vs_traditional": coef_ps,
                    "p_post_secular_vs_traditional": pval_ps,
                    "coef_modern_vs_traditional": coef_mod,
                    "p_modern_vs_traditional": pval_mod,
                    "aic": float(glm_model.aic) if hasattr(glm_model, "aic") else None
                }
            }
        except Exception as e:
            log(f"Weighted GLM failed: {e}")
            weighted = {
                "weight_column": weight_col,
                "n_analysis_weighted": int(len(work)),
                "weighted_proportions": wprops,
                "glm_error": str(e)
            }
        # Merge into existing results JSON if exists
        try:
            if os.path.exists(RESULTS_JSON):
                with open(RESULTS_JSON, "r", encoding="utf-8") as fh:
                    base = json.load(fh)
            else:
                base = {}
        except Exception:
            base = {}
        base["weighted_analysis"] = weighted
        with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
            json.dump(base, fh, indent=2)
        log("Appended weighted analysis to results JSON.")
    except Exception as e:
        log(f"run_weighted_analysis error: {e}")

# Auto-run weighted analysis when executed as script
if __name__ == "__main__":
    try:
        run_weighted_analysis()
    except Exception as _:
        pass

