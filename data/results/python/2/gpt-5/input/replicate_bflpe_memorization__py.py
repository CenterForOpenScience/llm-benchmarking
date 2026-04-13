import os
import sys
import json
import re
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# Optional imports guarded
try:
    import pyreadr  # for .RDS
except Exception as e:
    pyreadr = None

import statsmodels.api as sm


def log(msg):
    print(msg, flush=True)


def find_existing_path(candidates):
    for p in candidates:
        if p is None:
            continue
        pth = Path(p)
        if pth.exists():
            return pth
    return None


def search_for_file(filename, start_dirs):
    for start in start_dirs:
        start = Path(start)
        if not start.exists():
            continue
        for root, dirs, files in os.walk(start):
            if filename in files:
                return Path(root) / filename
    return None


def load_dataset():
    # Primary expected locations inside container
    candidates = [
        "/app/data/PISA2012.replication.pkl",
        "/app/data/PISA2012.replication.RDS",
    ]
    # Fallbacks: relative to script dir and project tree
    script_dir = Path(__file__).resolve().parent
    candidates += [
        script_dir / "replication_data" / "PISA2012.replication.pkl",
        script_dir / "replication_data" / "PISA2012.replication.RDS",
        script_dir.parent / "replication_data" / "PISA2012.replication.pkl",
        script_dir.parent / "replication_data" / "PISA2012.replication.RDS",
    ]

    pth = find_existing_path(candidates)
    if pth is None:
        # As a last resort, search recursively under script_dir
        pth = search_for_file("PISA2012.replication.pkl", [script_dir, script_dir.parent, Path(".")])
        if pth is None:
            pth = search_for_file("PISA2012.replication.RDS", [script_dir, script_dir.parent, Path(".")])
    if pth is None:
        raise FileNotFoundError("Could not locate PISA2012.replication.{pkl|RDS} in /app/data or replication_data.")

    log(f"Loading dataset from: {pth}")
    if pth.suffix.lower() == ".pkl":
        df = pd.read_pickle(pth)
    elif pth.suffix.lower() == ".rds":
        if pyreadr is None:
            raise RuntimeError("pyreadr not available to read .RDS file. Please ensure pyreadr is installed.")
        res = pyreadr.read_r(str(pth))
        # Pick the largest data.frame object
        if len(res.keys()) == 1:
            df = next(iter(res.values()))
        else:
            best_key, best_len = None, -1
            for k, v in res.items():
                if isinstance(v, pd.DataFrame) and len(v) > best_len:
                    best_key, best_len = k, len(v)
            df = res[best_key]
    else:
        raise ValueError(f"Unsupported file type: {pth.suffix}")

    # Ensure pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df


def pick_best_column(df, priority_patterns, must_include_all=None, case_insensitive=True):
    cols = list(df.columns)
    def norm(s):
        return s.lower() if case_insensitive else s
    ncols = [norm(c) for c in cols]

    # Rank columns by how well they match the priority patterns (list of regex or substrings)
    best_idx, best_score = None, -1
    for i, c in enumerate(ncols):
        if must_include_all:
            if not all((m.lower() in c) for m in must_include_all):
                continue
        score = 0
        for pat in priority_patterns:
            if isinstance(pat, str):
                if pat.lower() in c:
                    score += 1
            else:
                if re.search(pat, c):
                    score += 2
        if score > best_score:
            best_idx, best_score = i, score
    return cols[best_idx] if best_idx is not None and best_score > 0 else None


def identify_variables(df):
    cols = list(df.columns)
    lower_cols = [c.lower() for c in cols]

    # Plausible values for math
    pv_cols = [c for c in cols if re.search(r"^pv\d+math", c, re.I) or re.search(r"^pv\d+_?math", c, re.I) or re.search(r"^pvmath\d+", c, re.I)]
    if not pv_cols:
        pv_cols = [c for c in cols if ("pv" in c.lower() and "math" in c.lower())]
    pv_cols = sorted(pv_cols)

    # Self-concept variable
    y_col = pick_best_column(
        df,
        priority_patterns=[r"scmat", r"scmath", "selfconcept", "self", "concept", "self_concept"],
    )
    # Prefer columns containing both 'self' and 'concept'
    if y_col is None:
        candidates = [c for c in cols if ("self" in c.lower() and "concept" in c.lower())]
        y_col = candidates[0] if candidates else None

    # Memorization strategy variable
    mem_col = pick_best_column(
        df,
        priority_patterns=["memor", "memory", "memorization", "memorisation"],
    )

    # School and country IDs
    school_col = pick_best_column(
        df,
        priority_patterns=["schoolid", "school_id", "idschool", "schid", "school"],
    )
    country_col = pick_best_column(
        df,
        priority_patterns=["country", "cnt", "cntry", "nation"],
    )

    info = {
        "pv_cols": pv_cols,
        "y_col": y_col,
        "mem_col": mem_col,
        "school_col": school_col,
        "country_col": country_col,
    }
    return info


def ensure_output_dir():
    out_dir = Path("/app/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / s.std(ddof=0)


def fit_models(df, info):
    out_dir = ensure_output_dir()
    results = {
        "per_pv": [],
        "combined": {}
    }

    pv_cols = info["pv_cols"]
    y_col = info["y_col"]
    mem_col = info["mem_col"]
    school_col = info["school_col"]
    country_col = info["country_col"]

    if not pv_cols:
        raise RuntimeError("No plausible value math columns detected.")
    if any(v is None for v in [y_col, mem_col, school_col, country_col]):
        raise RuntimeError(f"Missing required variables: {info}")

    # Create an individual ability composite (mean of PVs)
    pv_numeric = df[pv_cols].apply(pd.to_numeric, errors="coerce")
    df["pv_mean"] = pv_numeric.mean(axis=1)

    # Compute school-average ability using pv_mean as proxy
    # Note: includes the student's own score; for simplicity we document this choice.
    df[school_col] = df[school_col].astype(str)
    school_means = df.groupby(school_col)["pv_mean"].mean()
    df = df.join(school_means.rename("school_avg_ability"), on=school_col)

    # Center variables
    df["mem_z"] = zscore(pd.to_numeric(df[mem_col], errors="coerce"))
    df["school_avg_z"] = zscore(df["school_avg_ability"])

    # Prepare outputs
    summary_lines = []

    for pv in pv_cols:
        log(f"Fitting MixedLM for PV: {pv}")
        work = df[[y_col, "mem_z", "school_avg_z", pv, school_col, country_col]].copy()
        work = work.rename(columns={pv: "ability", y_col: "y", school_col: "school", country_col: "country"})
        # Drop missing
        work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["y", "mem_z", "school_avg_z", "ability", "school", "country"]).copy()

        # MixedLM with school random intercept; try country as variance component
        # Fallback to country fixed effects if variance component fails.
        result = None
        model = None
        used_vc = False
        try:
            model = sm.MixedLM.from_formula(
                "y ~ mem_z + school_avg_z + mem_z:school_avg_z + ability",
                groups="school",
                vc_formula={"country_vc": "0 + C(country)"},
                data=work
            )
            result = model.fit(method="lbfgs", reml=True, maxiter=200)
            used_vc = True
        except Exception as e:
            log(f"VC model failed for PV {pv} with error: {e}. Falling back to country fixed effects.")
            try:
                model = sm.MixedLM.from_formula(
                    "y ~ mem_z + school_avg_z + mem_z:school_avg_z + ability + C(country)",
                    groups="school",
                    data=work
                )
                result = model.fit(method="lbfgs", reml=True, maxiter=200)
            except Exception as e2:
                log(f"MixedLM with country FE also failed for PV {pv}: {e2}")
                continue

        params = result.params.to_dict()
        bse = result.bse.to_dict()
        pvalues = result.pvalues.to_dict() if hasattr(result, "pvalues") else {}

        key = "mem_z:school_avg_z"
        coef = params.get(key, float("nan"))
        se = bse.get(key, float("nan"))
        pv_val = pvalues.get(key, float("nan"))

        res_entry = {
            "pv": pv,
            "interaction_coef": coef,
            "interaction_se": se,
            "interaction_p": pv_val,
            "used_country_vc": used_vc,
            "n_obs": int(result.nobs) if hasattr(result, "nobs") else None
        }
        results["per_pv"].append(res_entry)

        # Append summary text
        summary_lines.append(f"===== {pv} =====\n")
        try:
            summary_lines.append(str(result.summary()) + "\n\n")
        except Exception:
            summary_lines.append("<summary unavailable>\n\n")

    # Combine results across PVs (simple average as fallback)
    if results["per_pv"]:
        inter_coefs = [r["interaction_coef"] for r in results["per_pv"] if pd.notnull(r["interaction_coef"])]
        inter_ses = [r["interaction_se"] for r in results["per_pv"] if pd.notnull(r["interaction_se"])]
        if inter_coefs:
            results["combined"]["interaction_coef_mean"] = float(np.mean(inter_coefs))
            results["combined"]["interaction_coef_sd"] = float(np.std(inter_coefs, ddof=1)) if len(inter_coefs) > 1 else 0.0
        if inter_ses:
            results["combined"]["interaction_se_mean"] = float(np.mean(inter_ses))

    # Save outputs
    out_json = out_dir / "model_results.json"
    out_txt = out_dir / "model_summaries.txt"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    with open(out_txt, "w") as f:
        f.write("\n".join(summary_lines))

    log(f"Saved results to {out_json} and summaries to {out_txt}")
    return results


def main():
    log("Starting replication: memorization x school-average ability -> math self-concept")
    df = load_dataset()
    log(f"Loaded dataset with shape: {df.shape}")
    info = identify_variables(df)
    log(f"Identified variables: {json.dumps(info, indent=2)}")

    results = fit_models(df, info)
    # Basic console report
    if results.get("per_pv"):
        log("Per-PV interaction coefficients:")
        for r in results["per_pv"]:
            log(f"  {r['pv']}: coef={r['interaction_coef']:.4f}, se={r['interaction_se']:.4f}, p={r['interaction_p']}")
        comb = results.get("combined", {})
        if comb:
            log(f"Combined (mean) interaction coef: {comb.get('interaction_coef_mean')} (sd={comb.get('interaction_coef_sd')})")


if __name__ == "__main__":
    main()
