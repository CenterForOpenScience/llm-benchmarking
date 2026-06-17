import os
import json
import re
import hashlib
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Helper to compute percent missing per row subset

def percent_miss(arr_like):
    arr = pd.Series(arr_like)
    return float(arr.isna().sum()) / float(len(arr)) * 100.0


def main():
    # IO paths
    data_path = os.environ.get("REPL_DATA_PATH", "/app/data/Kachanoff_Survey_deidentify.csv")
    out_results_path = "/app/data/replication_results.json"
    out_wide_path = "/app/data/DFwide.csv"

    # Load
    DF = pd.read_csv(data_path)

    # Clean times: keep sessions after the prereg cutoff
    # Using naive datetime to match input format
    DF["created_dt"] = pd.to_datetime(DF["created"], errors="coerce")
    cutoff = pd.to_datetime("2021-01-18 15:39:38")
    DF = DF.loc[DF["created_dt"] > cutoff].copy()

    # Exclude duplicate sessions per preregistration
    dup_sessions = [
        "yzjrqRlxP9w2VGfCc8oH8Ksx1Yy0dbIVzyYISA9cWO7usiklW1LN5cTrMUQ24ULh",
        "iz1H-cKfL9nNaOhPcG48r6qpur0ONIdF0I5de4DIqxE2zsJnB74YBh9zxgM_P7Ob",
        "EZZPS01lPHHFhMBsAb2VO037TFobEwTVd6W78z03LgW3OlznQ4dr8eXZZC03F1be",
        "sAetiZs9o7qBrRB057nP0BaGuKAT0Q6eNCUl3zQWBXCAsZP7gngqMvAtJ1GrNh7a",
        "d-a0BOe7-Z6nrW12Bsq2PhEzo9jIbc_3Ep5AQkLK5po-Bvy6b6fAIFOoEkyP7XBd",
    ]
    if "session" in DF.columns:
        DF = DF.loc[~DF["session"].isin(dup_sessions)].copy()

    # Exclude rows with empty participant_id (sha256("") in deidentified data)
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    if "participant_id" in DF.columns:
        DF = DF.loc[DF["participant_id"] != empty_sha256].copy()

    # Recode scales
    # Subtract 1 from all BAI items and attention_check1
    bai_cols = [c for c in DF.columns if re.search(r"^bai_\\d+|^attention_check1$", c)]
    DF.loc[:, bai_cols] = DF.loc[:, bai_cols] - 1

    # IES COVID: subtract 1 from intrusion and avoid items (some R code had a typo 'instrusion')
    avoid_cols = [c for c in DF.columns if c.startswith("avoid")]
    intrusion_cols = [c for c in DF.columns if c.startswith("intrusion")] + [c for c in DF.columns if "instrusion" in c]
    for cols in [avoid_cols, intrusion_cols]:
        if cols:
            DF.loc[:, cols] = DF.loc[:, cols] - 1

    # SDS reverse items 1 and 2: range 1-7 becomes 7->1 via 8 - x
    for c in ["sds1", "sds2"]:
        if c in DF.columns:
            DF.loc[:, c] = 8 - DF.loc[:, c]

    # Manipulation checks / Attention total
    DF["attention_total"] = 0
    # Must be moderately (value 2 after recode)
    if "attention_check1" in DF.columns:
        DF["attention_total"] = DF["attention_total"] + (DF["attention_check1"] == 2).astype(int)
    # Must be 50
    if "attention_check2" in DF.columns:
        DF["attention_total"] = DF["attention_total"] + (DF["attention_check2"] == 50).astype(int)
    # Must be less than previous check and divisible by 5
    if set(["attention_check3", "attention_check2"]).issubset(DF.columns):
        cond3 = (DF["attention_check3"] < DF["attention_check2"]) & ((DF["attention_check3"] % 5) == 0)
        DF["attention_total"] = DF["attention_total"] + cond3.fillna(False).astype(int)
    # Must be less than previous check
    if set(["attention_check4", "attention_check3"]).issubset(DF.columns):
        cond4 = DF["attention_check4"] < DF["attention_check3"]
        DF["attention_total"] = DF["attention_total"] + cond4.fillna(False).astype(int)
    # Dog-related word in free response
    if "attention_check5" in DF.columns:
        DF["attention_dog_scored"] = DF["attention_check5"].astype(str).str.lower().str.contains(
            r"dog|bark|bork|woof|bowwow|bow wow|ruff|roof|arf|wolf|whoof|woo|whoops|roo|boof"
        )
        DF["attention_total"] = DF["attention_total"] + DF["attention_dog_scored"].fillna(False).astype(int)

    # Apply exclusion: must get at least 4 correct
    DF = DF.loc[DF["attention_total"] >= 4].copy()

    # Composite scores
    # BAI total (sum)
    bai_item_cols = [c for c in DF.columns if re.match(r"^bai_\\d+$", c)]
    DF["BAI_total"] = DF.loc[:, bai_item_cols].sum(axis=1, skipna=True)

    # COVID Realistic and Symbolic averaged
    real_cols = [f"covid_real{i}" for i in range(1, 6) if f"covid_real{i}" in DF.columns]
    symb_cols = [f"covid_symbolic{i}" for i in range(1, 6) if f"covid_symbolic{i}" in DF.columns]
    DF["Realistic"] = DF.loc[:, real_cols].mean(axis=1, skipna=True)
    DF["Symbolic"] = DF.loc[:, symb_cols].mean(axis=1, skipna=True)

    # IES Intrusion and Avoidance summed, with 80% rule
    intr_cols = [c for c in DF.columns if c.startswith("intrusion")]
    avd_cols = [c for c in DF.columns if c.startswith("avoid")]
    # Intrusion
    if intr_cols:
        intr_pct_miss = DF.loc[:, intr_cols].isna().mean(axis=1) * 100
        DF["Intrusion"] = np.where(
            intr_pct_miss <= 20,
            DF.loc[:, intr_cols].sum(axis=1, skipna=True),
            np.nan,
        )
    else:
        DF["Intrusion"] = np.nan
    # Avoidance
    if avd_cols:
        avd_pct_miss = DF.loc[:, avd_cols].isna().mean(axis=1) * 100
        DF["Avoidance"] = np.where(
            avd_pct_miss <= 20,
            DF.loc[:, avd_cols].sum(axis=1, skipna=True),
            np.nan,
        )
    else:
        DF["Avoidance"] = np.nan

    # SWLS averaged
    swls_cols = [c for c in DF.columns if c.startswith("swls")]
    DF["SWLS"] = DF.loc[:, swls_cols].mean(axis=1, skipna=True) if swls_cols else np.nan

    # PANAS summed
    pos_cols = [c for c in DF.columns if c.startswith("positive")]
    neg_cols = [c for c in DF.columns if c.startswith("negative")]
    DF["Positive"] = DF.loc[:, pos_cols].sum(axis=1, skipna=True) if pos_cols else np.nan
    DF["Negative"] = DF.loc[:, neg_cols].sum(axis=1, skipna=True) if neg_cols else np.nan

    # Social Distance summed
    social_cols = [c for c in DF.columns if c.startswith("social")]
    DF["Social"] = DF.loc[:, social_cols].sum(axis=1, skipna=True) if social_cols else np.nan

    # SDS averaged
    sds_cols = [c for c in DF.columns if c.startswith("sds")]
    DF["SDS"] = DF.loc[:, sds_cols].mean(axis=1, skipna=True) if sds_cols else np.nan

    # Behaviors averaged
    norm_cols = [c for c in DF.columns if c.startswith("behave_norm")]
    am_cols = [c for c in DF.columns if c.startswith("behave_american")]
    DF["Norms"] = DF.loc[:, norm_cols].mean(axis=1, skipna=True) if norm_cols else np.nan
    DF["American"] = DF.loc[:, am_cols].mean(axis=1, skipna=True) if am_cols else np.nan

    # Long to wide
    DF = DF.sort_values(by=["created_dt"])  # ensure T1 first
    needed = [
        "participant_id", "created_dt", "BAI_total", "Realistic", "Symbolic",
        "Intrusion", "Avoidance", "SWLS", "Positive", "Negative", "Social",
        "SDS", "Norms", "American", "handwashing"
    ]
    # keep only rows with participant_id not null
    DF = DF.loc[DF["participant_id"].notna()].copy()
    DFreduced = DF.loc[:, [c for c in needed if c in DF.columns]].copy()

    # mark time2 as duplicate of participant_id
    DFreduced["time2"] = DFreduced.duplicated(subset=["participant_id"], keep="first")

    DFtime1 = DFreduced.loc[~DFreduced["time2"]].copy()
    DFtime2 = DFreduced.loc[DFreduced["time2"]].copy()

    # Merge on participant_id (inner join to ensure both waves present)
    DFwide = pd.merge(
        DFtime1.drop(columns=["time2"]).add_suffix(".x"),
        DFtime2.drop(columns=["time2"]).add_suffix(".y"),
        left_on="participant_id.x",
        right_on="participant_id.y",
        how="inner",
        suffixes=(".x", ".y"),
    )

    # Simplify: set a single participant_id column
    DFwide["participant_id"] = DFwide["participant_id.x"]

    # Save wide data for inspection
    DFwide.to_csv(out_wide_path, index=False)

    # OLS: Negative.y ~ Realistic.x + Symbolic.x
    # Drop rows with missing inputs
    model_df = DFwide[["Negative.y", "Realistic.x", "Symbolic.x"]].dropna().copy()
    y = model_df["Negative.y"].astype(float)
    X = sm.add_constant(model_df[["Realistic.x", "Symbolic.x"]].astype(float))

    model = sm.OLS(y, X).fit()

    # Collect results
    coef_realistic = model.params.get("Realistic.x", np.nan)
    p_realistic = model.pvalues.get("Realistic.x", np.nan)
    n_used = int(model.nobs)

    results = {
        "model": "OLS: Negative.y ~ Realistic.x + Symbolic.x",
        "n_used": n_used,
        "coefficients": {k: float(v) for k, v in model.params.items()},
        "p_values": {k: float(v) for k, v in model.pvalues.items()},
        "focal_path": {
            "predictor": "Realistic.x",
            "outcome": "Negative.y",
            "coef": float(coef_realistic) if pd.notna(coef_realistic) else None,
            "p_value": float(p_realistic) if pd.notna(p_realistic) else None,
            "direction": "positive" if coef_realistic is not None and coef_realistic > 0 else "non-positive",
        },
        "notes": "Replicates the focal path within a multivariate framework via separate OLS, equivalent to SEM path coefficient under observed variables.",
    }

    with open(out_results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also print a brief summary to stdout
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
