import os
import json
import numpy as np
import pandas as pd

# Stats / models
import pingouin as pg
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy.stats import norm


def main():
    # Paths
    data_path = os.environ.get("DATA_PATH", "/app/data/Tremoliere_generalizability_score.csv")
    out_dir = "/app/data"
    os.makedirs(out_dir, exist_ok=True)

    # Fallback if default data path not present: use repo copy under /workspace
    if not os.path.exists(data_path):
        alt_path = "/workspace/replication_data/Tremoliere_generalizability_score.csv"
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Data file not found at {data_path} or fallback {alt_path}. Set DATA_PATH to the correct CSV path inside the container.")

    # Load data
    df = pd.read_csv(data_path)

    # Keep only needed columns (robust to missing optional columns)
    keep_cols = [
        "moral_accept", "moral_accept1", "ResponseId",
        "death_salience", "pain_salience", "satisfactory_manipulation_response1",
        "age", "gender", "politic_1", "politic_2", "politic_3", "race",
        "income", "education", "open", "cond", "salience", "participant_uid"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Long format
    long = df.melt(
        id_vars=[c for c in df.columns if c not in ["moral_accept", "moral_accept1"]],
        value_vars=[c for c in ["moral_accept", "moral_accept1"] if c in df.columns],
        var_name="variable",
        value_name="moral_acceptability"
    )

    # Filter to participants who passed manipulation check if column exists
    if "satisfactory_manipulation_response1" in long.columns:
        long = long[long["satisfactory_manipulation_response1"] == 1].copy()

    # Recode DV to binary 0/1 where 1 = utilitarian (as per R code mapping 2->1, 1->0)
    # Data may be numeric or string; handle both robustly
    def recode_util(x):
        try:
            xi = int(x)
        except Exception:
            return np.nan
        if xi == 1:
            return 0
        elif xi == 2:
            return 1
        else:
            return np.nan

    long["moral_acceptability_01"] = long["moral_acceptability"].apply(recode_util)

    # Scenario label
    scenario_map = {
        "moral_accept": "impartial_beneficience",
        "moral_accept1": "partial_beneficience",
    }
    long["moral_scenario"] = long["variable"].map(scenario_map)

    # Ensure salience exists and is categorical with levels like 'death'/'pain'
    if "salience" not in long.columns:
        raise RuntimeError("Expected column 'salience' not found in data.")
    long["salience"] = long["salience"].astype(str).str.lower()

    # Ensure subject id column exists
    subj_col = "participant_uid" if "participant_uid" in long.columns else ("ResponseId" if "ResponseId" in long.columns else None)
    if subj_col is None:
        raise RuntimeError("Expected a subject identifier column ('participant_uid' or 'ResponseId') not found.")

    # Drop rows with missing essential fields
    long = long.dropna(subset=["moral_acceptability_01", "moral_scenario", "salience", subj_col])

    # Basic descriptive counts
    desc_counts = {
        "n_rows": int(len(long)),
        "n_participants": int(long[subj_col].nunique()),
        "salience_counts": long["salience"].value_counts().to_dict(),
        "scenario_counts": long["moral_scenario"].value_counts().to_dict(),
    }

    # Mixed ANOVA (between: salience; within: moral_scenario)
    # Pingouin expects tidy data with subject id
    try:
        aov = pg.mixed_anova(data=long,
                             dv="moral_acceptability_01",
                             within="moral_scenario",
                             between="salience",
                             subject=subj_col)
        aov_path = os.path.join(out_dir, "tremoliere_mixed_anova_results.csv")
        aov.to_csv(aov_path, index=False)
    except Exception as e:
        aov = None
        aov_path = None
        desc_counts["mixed_anova_error"] = str(e)

    # Cluster-robust logistic regression approximating random intercept by clustering on subject
    # Build design matrix with interaction: salience (death=1), scenario (impartial=1), interaction
    # Reference levels: pain (0), partial_beneficience (0)
    long["salience_death"] = (long["salience"].astype(str).str.lower() == "death").astype(int)
    long["scenario_impartial"] = (long["moral_scenario"] == "impartial_beneficience").astype(int)
    long["interaction"] = long["salience_death"] * long["scenario_impartial"]

    y = long["moral_acceptability_01"].astype(int)
    X = sm.add_constant(long[["salience_death", "scenario_impartial", "interaction"]])

    logit_out = {}
    try:
        model = sm.Logit(y, X)
        res = model.fit(disp=False)
        # Compute cluster-robust covariance matrix clustered by subject
        groups = long[subj_col]
        cov = cov_cluster(res, groups)

        # Extract robust params and SEs
        coefs = res.params
        ses = np.sqrt(np.diag(cov))
        pvals = {}
        conf = {}
        zvals = {}
        for i, name in enumerate(coefs.index):
            z = coefs[i] / ses[i]
            zvals[name] = float(z)
            pvals[name] = float(2 * (1 - norm.cdf(abs(z))))
            lcl = coefs[i] - 1.96 * ses[i]
            ucl = coefs[i] + 1.96 * ses[i]
            conf[name] = (float(lcl), float(ucl))

        or_table = []
        for name in coefs.index:
            if name == 'const':
                continue
            beta = float(coefs[name])
            se = float(ses[coefs.index.get_loc(name)])
            p = float(pvals[name])
            lcl, ucl = conf[name]
            or_table.append({
                "term": name,
                "logit_coef": beta,
                "std_err": se,
                "z_value": zvals[name],
                "p_value": p,
                "odds_ratio": float(np.exp(beta)),
                "or_ci_lower": float(np.exp(lcl)),
                "or_ci_upper": float(np.exp(ucl)),
            })
        logit_out = {
            "n_obs": int(res.nobs),
            "llf": float(res.llf),
            "params": or_table,
        }
        # Save text summary
        with open(os.path.join(out_dir, "tremoliere_logit_summary.txt"), "w") as f:
            f.write(res.summary2().as_text())
            f.write("\n\nCluster-robust SEs (manual sandwich) by subject provided in JSON output.")
        # Save JSON
        with open(os.path.join(out_dir, "tremoliere_logit_results.json"), "w") as f:
            json.dump(logit_out, f, indent=2)
    except Exception as e:
        desc_counts["logit_error"] = str(e)

    # Save a small meta summary
    meta = {
        "input_path": data_path,
        "n_rows_input": int(df.shape[0]),
        "n_cols_input": int(df.shape[1]),
        "descriptives": desc_counts,
        "mixed_anova_results_path": aov_path,
        "logit_results_path": os.path.join(out_dir, "tremoliere_logit_results.json") if logit_out else None
    }
    with open(os.path.join(out_dir, "tremoliere_replication_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Also print key results to stdout
    print(json.dumps(meta, indent=2))
    if aov is not None:
        print("Mixed ANOVA results:")
        print(aov.to_string(index=False))


if __name__ == "__main__":
    main()
