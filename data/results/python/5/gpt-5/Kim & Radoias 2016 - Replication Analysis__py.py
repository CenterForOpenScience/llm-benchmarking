import os
import sys
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit


def find_data_path():
    candidates = [
        "/app/data/processed_replication_data.csv",
        "/app/data/replication_data/processed_replication_data.csv",
        "/app/data/original/5/python/replication_data/processed_replication_data.csv",
        "/workspace/replication_data/processed_replication_data.csv",
        "/workspace/processed_replication_data.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"processed_replication_data.csv not found in any of: {candidates}")


def safe_numeric(col):
    if col.dtype == object:
        return pd.to_numeric(col, errors="coerce")
    return col


def parse_sex(series):
    # Handles codes like '1:Male', '3:Female', or numeric codes
    if series.dtype == object:
        codes = series.str.split(":", n=1, expand=True)[0]
        num = pd.to_numeric(codes, errors="coerce")
    else:
        num = pd.to_numeric(series, errors="coerce")
    # Define male=1 if code==1, female=0 if code==3
    male = np.where(num == 1, 1.0, np.where(num == 3, 0.0, np.nan))
    return pd.Series(male, index=series.index)


def reconstruct_yrs_school(df):
    yrs = pd.Series(np.nan, index=df.index, dtype=float)
    # clean dl07 code 98 -> missing
    if "dl07" in df.columns:
        dl07 = pd.to_numeric(df["dl07"], errors="coerce")
        dl07 = dl07.where(dl07 != 98, np.nan)
    else:
        dl07 = pd.Series(np.nan, index=df.index)

    dl06 = pd.to_numeric(df.get("dl06", pd.Series(np.nan, index=df.index)), errors="coerce")
    dl04 = pd.to_numeric(df.get("dl04", pd.Series(np.nan, index=df.index)), errors="coerce")

    # Never attended school
    yrs = np.where(dl04 == 3, 0.0, yrs)

    if "dl06" in df.columns and "dl07" in df.columns:
        # Elementary (dl06==2) -> yrs = dl07 (with special 7=>6)
        dl07_elem = dl07.copy()
        dl07_elem = np.where((dl07_elem == 7) & (dl06 == 2), 6, dl07_elem)
        yrs = np.where(dl06 == 2, dl07_elem, yrs)
        # Adult Education A (dl06==11): yrs = dl07 (7->1)
        dl07_adultA = np.where((dl07 == 7) & (dl06 == 11), 1, dl07)
        yrs = np.where(dl06 == 11, dl07_adultA, yrs)
        # Islamic Elementary (dl06==72): yrs = dl07 (7->6)
        dl07_mad_elem = np.where((dl07 == 7) & (dl06 == 72), 6, dl07)
        yrs = np.where(dl06 == 72, dl07_mad_elem, yrs)
        # Junior High (dl06 in {3,4}): yrs = dl07+6 (7->3)
        dl07_jr = np.where(dl07 == 7, 3, dl07)
        yrs = np.where(dl06.isin([3, 4]), dl07_jr + 6, yrs)
        # Adult Education B (dl06==12): yrs = dl07+1 (7->4)
        dl07_adultB = np.where(dl07 == 7, 4, dl07)
        yrs = np.where(dl06 == 12, dl07_adultB + 1, yrs)
        # Islamic Junior/High (dl06==73): yrs = dl07+6 (7->3)
        dl07_mad_jr = np.where(dl07 == 7, 3, dl07)
        yrs = np.where(dl06 == 73, dl07_mad_jr + 6, yrs)
        # Senior High (dl06==5): yrs = dl07+9 (7->3)
        dl07_sh = np.where(dl07 == 7, 3, dl07)
        yrs = np.where(dl06 == 5, dl07_sh + 9, yrs)
        # Senior High Vocational (dl06==6): yrs = dl07+9 (7->4)
        dl07_shv = np.where(dl07 == 7, 4, dl07)
        yrs = np.where(dl06 == 6, dl07_shv + 9, yrs)
        # Adult Education C (dl06==15): yrs = dl07+5 (7->3)
        dl07_adultC = np.where(dl07 == 7, 3, dl07)
        yrs = np.where(dl06 == 15, dl07_adultC + 5, yrs)
        # Islamic Senior (dl06==74): yrs = dl07+9 (7->3)
        dl07_mad_sh = np.where(dl07 == 7, 3, dl07)
        yrs = np.where(dl06 == 74, dl07_mad_sh + 9, yrs)
        # College (dl06==60): yrs = dl07+12 (7->3)
        dl07_d1 = np.where(dl07 == 7, 3, dl07)
        yrs = np.where(dl06 == 60, dl07_d1 + 12, yrs)
        # University S1 (dl06==61): yrs = dl07+12 (7->4)
        dl07_s1 = np.where(dl07 == 7, 4, dl07)
        yrs = np.where(dl06 == 61, dl07_s1 + 12, yrs)
        # University S2 (dl06==62): yrs = dl07+16 (7->3)
        dl07_s2 = np.where(dl07 == 7, 3, dl07)
        yrs = np.where(dl06 == 62, dl07_s2 + 16, yrs)
        # University S3 (dl06==63): yrs = dl07+16 (7->5)
        dl07_s3 = np.where(dl07 == 7, 5, dl07)
        yrs = np.where(dl06 == 63, dl07_s3 + 16, yrs)
        # Open University (dl06==13): yrs = dl07+12 (7->6)
        dl07_open = np.where(dl07 == 7, 6, dl07)
        yrs = np.where(dl06 == 13, dl07_open + 12, yrs)

    # Kindergarten (dl06==90) => yrs = 0
    yrs = np.where(dl06 == 90, 0.0, yrs)

    # Missing for certain dl06 codes
    yrs = np.where(dl06.isin([14, 98, 99]), np.nan, yrs)

    yrs = pd.to_numeric(yrs, errors="coerce")
    return pd.Series(yrs, index=df.index)


def fit_and_report(sample, model_vars, out_dir, data_path, label):
    y = sample["under_diag_final"].astype(float)
    X = sample[model_vars].astype(float)
    X = sm.add_constant(X, has_constant="add")

    try:
        model = Probit(y, X)
        res = model.fit(disp=False, cov_type="HC1")
        used = "Probit"
    except Exception as e:
        print(f"{label}: Probit fit failed with error: {e}")
        print(f"{label}: Falling back to GLM Binomial (Probit link) with robust SEs")
        glm = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.probit()))
        res = glm.fit(cov_type="HC1")
        used = "GLM-Binomial(Probit)"

    # Marginal effects
    try:
        margeff = res.get_margeff(at="overall")
        me_summary = margeff.summary_frame()
    except Exception as e:
        print(f"{label}: Marginal effects computation failed: {e}")
        me_summary = pd.DataFrame()

    # Save outputs
    res_txt = os.path.join(out_dir, f"results_replication_{label}.txt")
    with open(res_txt, "w") as f:
        f.write("Kim & Radoias 2016 - Replication Analysis (Python)\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Estimation sample (n): {len(sample)}\n")
        f.write(f"Model: under_diag ~ {' + '.join(model_vars)} (robust SEs) [{used}]\n\n")
        f.write(str(res.summary()))
        f.write("\n\n")
        if not me_summary.empty:
            f.write("Average Marginal Effects (overall):\n")
            f.write(me_summary.to_string())
            f.write("\n\n")
            # Focal effect row
            yrs_row = None
            for idx in me_summary.index if isinstance(me_summary.index, pd.Index) else []:
                if str(idx).endswith("yrs_school_final") or str(idx) == "yrs_school_final":
                    yrs_row = me_summary.loc[idx]
                    break
            if yrs_row is None:
                possible = [i for i in me_summary.index if "yrs_school" in str(i)]
                if possible:
                    yrs_row = me_summary.loc[possible[0]]
            if yrs_row is not None:
                try:
                    f.write("Focal effect (yrs_school):\n")
                    f.write(f"dy/dx={yrs_row['dy/dx']:.6f}, Std. Err.={yrs_row['Std. Err.']:.6f}, z={yrs_row['z']:.3f}, P>|z|={yrs_row['P>|z|']:.4f}, [ {yrs_row['[0.025']} , {yrs_row['0.975]']} ]\n")
                except Exception:
                    f.write(str(yrs_row) + "\n")

    if not me_summary.empty:
        me_csv = os.path.join(out_dir, f"marginal_effects_{label}.csv")
        me_summary.to_csv(me_csv, index=True)

    return res


def main():
    # Optional CLI override for data path
    data_path = sys.argv[1] if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) else find_data_path()
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    # Ensure key numeric variables are numeric
    for col in ["us07b1", "us07c1", "us07b2", "us07c2", "cd05", "ar07", "ar09", "rj11", "rj11x", "hypertension", "under_diag", "yrs_school"]:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    # Rebuild hypertension from BP measurements where possible
    for need in ["us07b1", "us07c1", "us07b2", "us07c2"]:
        if need not in df.columns:
            print("Warning: Missing BP column", need)
    if set(["us07b1", "us07c1", "us07b2", "us07c2"]).issubset(df.columns):
        systolic = (df["us07b1"] + df["us07c1"]) / 2.0
        diastolic = (df["us07b2"] + df["us07c2"]) / 2.0
        hyp = np.where((systolic > 140) | (diastolic > 90), 1.0, 0.0)
        hyp = np.where(df[["us07b1", "us07c1", "us07b2", "us07c2"]].isna().any(axis=1), np.nan, hyp)
        df["hypertension_rebuilt"] = hyp
        df["hypertension_final"] = df["hypertension_rebuilt"].where(~pd.isna(df["hypertension_rebuilt"]), df.get("hypertension"))
    else:
        df["hypertension_final"] = df.get("hypertension")

    # Build under_diag from cd05 and hypertension_final
    if "cd05" in df.columns:
        cd05 = df["cd05"]
        und = np.where((df["hypertension_final"] == 1) & (cd05 == 3), 1.0, 0.0)
        und = np.where((pd.isna(df["hypertension_final"])) | (pd.isna(cd05)) | (cd05 == 8), np.nan, und)
        df["under_diag_final"] = und
    else:
        df["under_diag_final"] = df.get("under_diag")

    # Poor health indicator from ar07: poor if >=3
    if "ar07" in df.columns:
        df["poor_health"] = (df["ar07"] >= 3).astype(float)
    else:
        df["poor_health"] = np.nan

    # Age and Agesq, with sentinel 998 treated as missing
    if "ar09" in df.columns:
        age = df["ar09"].where(df["ar09"] != 998, np.nan)
        df["age"] = age
        df["agesq"] = age ** 2
    else:
        df["age"] = np.nan
        df["agesq"] = np.nan

    # Distance with rj11x==8 set to missing
    if "rj11" in df.columns:
        df["distance"] = df["rj11"]
        if "rj11x" in df.columns:
            df.loc[df["rj11x"] == 8, "distance"] = np.nan
    else:
        df["distance"] = np.nan

    # Sex male indicator
    if "sex" in df.columns:
        df["male"] = parse_sex(df["sex"]).astype(float)
    else:
        df["male"] = np.nan

    # Years of schooling: prefer existing non-missing, else reconstruct
    if "yrs_school" in df.columns and df["yrs_school"].notna().sum() > 0:
        yrs_existing = pd.to_numeric(df["yrs_school"], errors="coerce")
    else:
        yrs_existing = pd.Series(np.nan, index=df.index)
    yrs_recon = reconstruct_yrs_school(df)
    df["yrs_school_final"] = yrs_existing.where(yrs_existing.notna(), yrs_recon)

    # Build core filtered sample
    sample = df[(df.get("age", np.nan) >= 45) & (df.get("hypertension_final", 0) == 1) & (df.get("poor_health", 0) == 1)].copy()

    # Define candidate variable sets
    base_vars = ["yrs_school_final", "age", "agesq", "male"]
    vars_with_dist = base_vars + ["distance"]

    # Drop NA for each candidate set
    sample_base = sample.dropna(subset=["under_diag_final"] + base_vars)
    sample_dist = sample.dropna(subset=["under_diag_final"] + vars_with_dist)

    n_total = len(df)
    n_core = len(sample)
    n_base = len(sample_base)
    n_dist = len(sample_dist)

    print(f"Total rows: {n_total}; Core filtered (age>=45 & hypertensive & poor health): {n_core}; Base usable: {n_base}; Dist usable: {n_dist}")

    out_dir = "/app/data"
    os.makedirs(out_dir, exist_ok=True)

    # Choose model: prefer distance if enough obs, else base; ensure minimum obs
    chosen_label = None
    chosen_vars = None
    chosen_sample = None

    if n_dist >= 30:
        chosen_label = "with_distance"
        chosen_vars = vars_with_dist
        chosen_sample = sample_dist
    elif n_base >= 30:
        chosen_label = "base"
        chosen_vars = base_vars
        chosen_sample = sample_base
    elif n_dist > 0:
        chosen_label = "with_distance_small"
        chosen_vars = vars_with_dist
        chosen_sample = sample_dist
    elif n_base > 0:
        chosen_label = "base_small"
        chosen_vars = base_vars
        chosen_sample = sample_base
    else:
        # No usable observations; write a note and exit gracefully
        note_path = os.path.join(out_dir, "results_replication.txt")
        with open(note_path, "w") as f:
            f.write("Kim & Radoias 2016 - Replication Analysis (Python)\n")
            f.write(f"Data path: {data_path}\n")
            f.write(f"Core filtered sample size is zero after applying age>=45, hypertensive==1, poor health==1.\n")
            f.write("Distance variable and/or key covariates are largely missing in this extract, preventing estimation.\n")
        print("No usable observations; wrote summary note.")
        return

    # Fit chosen model and report
    res = fit_and_report(chosen_sample, chosen_vars, out_dir, data_path, chosen_label)

    # Save metadata
    meta = {
        "n_total": int(n_total),
        "n_core_filtered": int(n_core),
        "n_base_usable": int(n_base),
        "n_dist_usable": int(n_dist),
        "model_used": chosen_label,
        "variables": chosen_vars,
        "data_path": data_path,
    }
    with open(os.path.join(out_dir, "execution_metadata.json"), "w") as f:
        json.dump(meta, f)

    print(f"Saved results for model '{chosen_label}' to /app/data")


if __name__ == "__main__":
    main()
