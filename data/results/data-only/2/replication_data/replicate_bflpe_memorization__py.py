import os
import re
import json
import math
import numpy as np
import pandas as pd

# Ensure all IO uses /app/data
DATA_PATH = "/app/data"
INPUT_FILE = os.path.join(DATA_PATH, "PISA2012.replication.RDS")
RESULTS_JSON = os.path.join(DATA_PATH, "replication_results.json")
LOG_FILE = os.path.join(DATA_PATH, "replication_run_log.txt")
COLUMNS_CSV = os.path.join(DATA_PATH, "replication_columns.csv")


def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def find_column(df, patterns, description):
    """
    Find first column matching any of the regex patterns (case-insensitive).
    Returns column name or None.
    """
    cols = list(df.columns)
    for pat in patterns:
        regex = re.compile(pat, flags=re.IGNORECASE)
        matches = [c for c in cols if regex.search(c)]
        if matches:
            log(f"Found {description} candidates for pattern '{pat}': {matches[:5]}{'...' if len(matches)>5 else ''}")
            return matches[0]
    log(f"No column found for {description} with patterns: {patterns}")
    return None


def find_columns(df, pattern, description):
    regex = re.compile(pattern, flags=re.IGNORECASE)
    matches = [c for c in df.columns if regex.search(c)]
    log(f"Search {description} with pattern '{pattern}': found {len(matches)} matches: {matches}")
    return matches


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def combine_rubins(estimates, ses):
    # Rubin's rules for m imputations
    m = len(estimates)
    Q = np.array(estimates, dtype=float)
    U = np.array(ses, dtype=float) ** 2
    Qbar = Q.mean()
    Ubar = U.mean()
    B = Q.var(ddof=1)
    T = Ubar + (1 + 1/m) * B
    se = math.sqrt(T)
    # Normal approximation
    z = Qbar / se if se > 0 else np.nan
    from math import erf, sqrt
    # two-sided p-value using normal CDF via erf
    def norm_cdf(x):
        return 0.5 * (1 + erf(x / sqrt(2)))
    p = 2 * (1 - norm_cdf(abs(z))) if not np.isnan(z) else np.nan
    return {
        "m": m,
        "estimates": Q.tolist(),
        "ses": np.sqrt(U).tolist(),
        "Qbar": Qbar,
        "Ubar": Ubar,
        "B": B,
        "T": T,
        "combined_se": se,
        "z": z,
        "p_value": p,
    }


def fit_models(df, outcome_col, mem_col, school_col, country_col, pv_cols):
    import statsmodels.formula.api as smf

    results = []
    for pv in pv_cols:
        work = df[[outcome_col, mem_col, school_col, country_col, pv]].copy()
        work = work.replace([-np.inf, np.inf], np.nan)
        # Drop missing
        work = work.dropna()

        # Standardize variables
        work["z_outcome"] = zscore(work[outcome_col])
        work["z_memor"] = zscore(work[mem_col])
        work["ability"] = pd.to_numeric(work[pv], errors="coerce")

        # Compute school-average ability using current PV
        sch_mean = work.groupby(school_col)["ability"].transform("mean")
        work["sch_mean_ability"] = sch_mean

        # Standardize individual and school mean ability
        work["z_ind_ability"] = zscore(work["ability"])  # individual level
        work["z_sch_mean_ability"] = zscore(work["sch_mean_ability"])  # school average
        # Quadratic term for individual ability as in original
        work["z_ind_ability_sq"] = work["z_ind_ability"] ** 2

        # Interaction term
        work["z_interact"] = work["z_sch_mean_ability"] * work["z_memor"]

        # Remove any remaining missing after transforms
        work = work.dropna(subset=["z_outcome", "z_memor", "z_ind_ability", "z_ind_ability_sq", "z_sch_mean_ability", "z_interact", school_col, country_col])

        # Prepare MixedLM with random intercepts for school and country via variance components
        # Using formula interface for clarity
        # Fixed effects: z_sch_mean_ability, z_memor, interaction, z_ind_ability, z_ind_ability_sq
        # Random: intercept per school (groups=school), plus variance component per country
        formula = "z_outcome ~ z_sch_mean_ability + z_memor + z_interact + z_ind_ability + z_ind_ability_sq"

        try:
            model = smf.mixedlm(formula,
                                data=work,
                                groups=work[school_col],
                                re_formula="1",
                                vc_formula={"country": "0 + C(%s)" % country_col})
            fit = model.fit(reml=False, method='lbfgs', maxiter=200)
            params = fit.params.to_dict()
            b = params.get("z_interact", np.nan)
            se = fit.bse.get("z_interact", np.nan)
            n = work.shape[0]
            results.append({
                "pv": pv,
                "n": n,
                "coef": float(b) if pd.notnull(b) else None,
                "se": float(se) if pd.notnull(se) else None,
                "success": True,
                "converged": bool(getattr(fit, 'converged', True)),
                "summary": str(fit.summary())[:2000]  # truncate to avoid very large outputs
            })
            log(f"Fitted model for {pv}: coef={b}, se={se}, n={n}, converged={getattr(fit, 'converged', True)}")
        except Exception as e:
            log(f"Model fit failed for {pv}: {e}")
            results.append({
                "pv": pv,
                "n": work.shape[0],
                "coef": None,
                "se": None,
                "success": False,
                "error": str(e)
            })
    return results


def main():
    # Reset log
    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
        except Exception:
            pass

    # Try to read RDS via pyreadr
    try:
        import pyreadr
    except Exception as e:
        log("pyreadr not installed: %s" % e)
        raise

    if not os.path.exists(INPUT_FILE):
        log(f"Input file not found at {INPUT_FILE}. Attempting to locate and copy from workspace...")
        candidates = [
            os.path.join(os.path.dirname(__file__), "PISA2012.replication.RDS"),
            "/workspace/data/original/2/data-only/replication_data/PISA2012.replication.RDS",
            os.path.join(os.getcwd(), "PISA2012.replication.RDS"),
            os.path.join("/workspace", "data", "original", "2", "data-only", "replication_data", "PISA2012.replication.RDS"),
        ]
        src = None
        for c in candidates:
            if os.path.exists(c):
                src = c
                break
        if src:
            try:
                import shutil
                shutil.copyfile(src, INPUT_FILE)
                log(f"Copied dataset from {src} to {INPUT_FILE}")
            except Exception as e:
                log(f"Failed to copy dataset from {src} to {INPUT_FILE}: {e}")
        if not os.path.exists(INPUT_FILE):
            raise FileNotFoundError(f"PISA2012.replication.RDS not found. Checked: { [INPUT_FILE] + candidates }")

    log(f"Loading RDS from {INPUT_FILE}")
    rds = pyreadr.read_r(INPUT_FILE)
    # pyreadr returns a dict-like; take first object
    if len(rds.keys()) == 0:
        raise RuntimeError("RDS file contained no objects")
    df = None
    for key in rds.keys():
        df = rds[key]
        break

    if df is None:
        raise RuntimeError("Failed to load dataframe from RDS")

    # Persist column list
    pd.Series(df.columns).to_csv(COLUMNS_CSV, index=False)
    log(f"Loaded dataframe with shape {df.shape}")

    # Identify columns heuristically
    # Country
    country_col = find_column(df, [r"^CNT$", r"COUNTRY", r"country"], "country")
    # School ID (within country in PISA); may be SCHOOLID or SCHOOL_ID
    school_col_raw = find_column(df, [r"^SCHOOLID$", r"SCHOOL_ID", r"^SCHID$", r"^SCHOOL$"], "school id")
    # If both country and school present, create a combined school key to avoid collisions across countries
    if country_col and school_col_raw:
        df["_school_key"] = df[country_col].astype(str) + "_" + df[school_col_raw].astype(str)
        school_col = "_school_key"
    else:
        school_col = school_col_raw or "SCHOOLID"

    # Outcome: Mathematics self-concept
    outcome_col = find_column(df, [r"SCMAT", r"SCMNA", r"SELF.*CON.*MATH", r"MATH.*SELF"], "math self-concept")
    # Memorization index
    mem_col = find_column(df, [r"^MEMOR", r"MEMORIZ", r"MEMORY", r"ST..MEM"], "memorization strategy")
    # Plausible values for math ability
    pv_cols = find_columns(df, r"^PV[1-5].*MATH$", "plausible values for math ability")
    if not pv_cols:
        # Try a broader pattern
        pv_cols = find_columns(df, r"PV[1-5].*MATH", "plausible values for math ability (broad)")
    pv_cols = sorted(pv_cols)[:5]

    required = {
        "country_col": country_col,
        "school_col": school_col,
        "outcome_col": outcome_col,
        "mem_col": mem_col,
        "pv_cols": pv_cols,
    }
    log(f"Detected columns: {required}")

    missing = [k for k, v in required.items() if (not v or (isinstance(v, list) and len(v) == 0))]
    if missing:
        error_msg = {
            "error": "Missing required columns for analysis",
            "missing": missing,
            "detected": required,
        }
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(error_msg, f, indent=2)
        log(str(error_msg))
        return

    # Fit models across PVs
    fit_res = fit_models(df, outcome_col, mem_col, school_col, country_col, pv_cols)

    ests = [r["coef"] for r in fit_res if r.get("success") and r.get("coef") is not None]
    ses = [r["se"] for r in fit_res if r.get("success") and r.get("se") is not None]

    combined = None
    if len(ests) >= 2:
        combined = combine_rubins(ests, ses)

    output = {
        "model": "3-level MixedLM (random intercepts for school and country via variance components)",
        "outcome": outcome_col,
        "moderator": mem_col,
        "predictor_school_average": "school mean of math plausible value (per PV)",
        "controls": ["individual math ability (z)", "individual math ability squared (z^2)"],
        "pv_columns": pv_cols,
        "fits": fit_res,
        "combined": combined,
        "notes": "Variables were standardized to approximate standardized betas. The interaction term is z(school-average ability) * z(memorization)."
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    log(f"Wrote results to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
