import json
import math
import os
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm

# All IO must use /app/data
DATA_PATH = "/app/data/replication_data.dta"
OUTPUT_JSON = "/app/data/replication_results.json"
OUTPUT_TXT = "/app/data/replication_model_summary.txt"
OUTPUT_DATA_CSV = "/app/data/replication_analysis_dataset.csv"

# Utility functions

def _extract_code_and_label(val):
    """Given a pandas categorical/string value like '1:Yes', return (1, 'Yes').
    If value is numeric, return (val, None). If NaN, return (None, None)."""
    if pd.isna(val):
        return (None, None)
    if isinstance(val, (int, float, np.integer, np.floating)):
        try:
            if np.isnan(val):
                return (None, None)
        except Exception:
            pass
        return (int(val) if float(val).is_integer() else float(val), None)
    s = str(val)
    if ":" in s:
        code, label = s.split(":", 1)
        code = code.strip()
        label = label.strip()
        try:
            code = int(float(code))
        except Exception:
            pass
        return (code, label)
    # no label
    try:
        return (int(float(s)), None)
    except Exception:
        return (None, s)


def build_years_of_education(df):
    """Approximate years of education using DL06 (highest level) and DL07 (grade/grad).
    This mapping is approximate, documented in replication_info notes.
    """
    lvl_vals = df.get("dl06")
    grd_vals = df.get("dl07")

    base_years = []
    max_years = []
    grad_flag = []
    grade_in_level = []

    for i in range(len(df)):
        lvl = lvl_vals.iat[i] if lvl_vals is not None else np.nan
        grd = grd_vals.iat[i] if grd_vals is not None else np.nan
        lvl_code, lvl_label = _extract_code_and_label(lvl)
        grd_code, grd_label = _extract_code_and_label(grd)

        # default
        base = np.nan
        maxy = np.nan

        label = (lvl_label or "").lower() if lvl_label is not None else ""
        # Map base and max for the level
        if label:
            if "elementary" in label or "ibtidaiyah" in label:
                base, maxy = 0, 6
            elif "junior high" in label or "tsanawiyah" in label or "vocational" in label and "junior" in label:
                base, maxy = 6, 3
            elif "senior high" in label or ("vocational" in label and "senior" in label):
                base, maxy = 9, 3
            elif "college" in label or ("d1" in label or "d2" in label or "d3" in label):
                base, maxy = 12, 3
            elif "university s1" in label or ("open university" in label):
                base, maxy = 12, 4
            elif "university s2" in label:
                base, maxy = 16, 2
            elif "university s3" in label:
                base, maxy = 18, 3
            elif "adult education a" in label:
                base, maxy = 0, 6
            elif "adult education b" in label:
                base, maxy = 6, 3
            elif "adult education c" in label:
                base, maxy = 9, 3
            elif "islamic school (pesantren)" in label:
                # treat as senior high completed if graduated
                base, maxy = 9, 3
            elif "school for disabled" in label:
                base, maxy = 0, 6
            else:
                # Fallback based on code ranges
                if lvl_code in [2]:  # Elementary
                    base, maxy = 0, 6
                elif lvl_code in [3, 4]:  # Junior high
                    base, maxy = 6, 3
                elif lvl_code in [5, 6]:  # Senior high
                    base, maxy = 9, 3
                elif lvl_code in [60]:  # College D1-D3
                    base, maxy = 12, 3
                elif lvl_code in [61]:  # University S1
                    base, maxy = 12, 4
                elif lvl_code in [62]:  # S2
                    base, maxy = 16, 2
                elif lvl_code in [63]:  # S3
                    base, maxy = 18, 3
        else:
            # No label, use codes
            if lvl_code in [2]:
                base, maxy = 0, 6
            elif lvl_code in [3, 4]:
                base, maxy = 6, 3
            elif lvl_code in [5, 6]:
                base, maxy = 9, 3
            elif lvl_code in [60]:
                base, maxy = 12, 3
            elif lvl_code in [61]:
                base, maxy = 12, 4
            elif lvl_code in [62]:
                base, maxy = 16, 2
            elif lvl_code in [63]:
                base, maxy = 18, 3

        base_years.append(base)
        max_years.append(maxy)

        # Graduation flag and grade within level
        grad = False
        grd_num = np.nan
        if isinstance(grd_label, str) and grd_label:
            gl = grd_label.lower()
            if "graduated" in gl:
                grad = True
            elif "don't know" in gl or "missing" in gl:
                grd_num = np.nan
            else:
                # try number within label
                m = re.search(r"(\d+)", gl)
                if m:
                    try:
                        grd_num = int(m.group(1))
                    except Exception:
                        grd_num = np.nan
        else:
            # use code if numeric
            if grd_code is None:
                grd_num = np.nan
            else:
                try:
                    grd_num = int(float(grd_code))
                except Exception:
                    grd_num = np.nan

        grad_flag.append(grad)
        grade_in_level.append(grd_num)

    # Compute years
    years = []
    for b, mx, gflag, gnum in zip(base_years, max_years, grad_flag, grade_in_level):
        if pd.isna(b) or pd.isna(mx):
            years.append(np.nan)
            continue
        if gflag:
            years.append(b + mx)
        else:
            if pd.isna(gnum):
                years.append(np.nan)
            else:
                # Cap at max
                y = b + max(0, min(int(gnum), int(mx)))
                years.append(y)

    ser = pd.Series(years, index=df.index, dtype=float)
    # plausible bounds 0..25
    ser = ser.where((ser >= 0) & (ser <= 25))
    return ser


def load_data(path):
    # Try preferred path, then fallbacks within /app/data
    candidate_paths = [path]
    # Fallback to nested original path if mounted dataset structure is preserved
    candidate_paths.append("/app/data/original/5/data-only/replication_data/replication_data.dta")
    # Fallback to same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths.append(os.path.join(script_dir, "replication_data.dta"))

    found = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            found = p
            break
    if found is None:
        # As a last resort, scan /app/data for a .dta file that looks right
        for root, dirs, files in os.walk("/app/data"):
            for f in files:
                if f.lower().endswith(".dta") and "replication_data" in f.lower():
                    found = os.path.join(root, f)
                    break
            if found:
                break
    if found is None:
        raise FileNotFoundError(
            "Could not locate replication_data.dta. Expected at /app/data/replication_data.dta or in the mounted dataset folder."
        )
    df = pd.read_stata(found, convert_categoricals=True)
    return df


def compute_variables(df):
    out = df.copy()

    # Blood pressure: use us07b1/us07b2 and us07c1/us07c2; drop first reading (not present), average the remaining two
    for col in ["us07b1", "us07b2", "us07c1", "us07c2"]:
        if col not in out.columns:
            out[col] = np.nan

    out["sys_avg"] = out[["us07b1", "us07c1"]].astype(float).mean(axis=1, skipna=True)
    out["dia_avg"] = out[["us07b2", "us07c2"]].astype(float).mean(axis=1, skipna=True)
    out["hypertensive"] = ((out["sys_avg"] >= 140) | (out["dia_avg"] >= 90)).astype(int)

    # Previous diagnosis: cd05 (1:Yes, 3:No, 8:Don't know)
    if "cd05" not in out.columns:
        out["cd05"] = np.nan
    cd_codes = pd.Series([_extract_code_and_label(v)[0] for v in out["cd05"]], index=out.index)
    out["prev_dx"] = np.where(cd_codes == 1, 1, np.where(cd_codes.isna(), np.nan, 0))

    # Undiagnosed among hypertensive
    out["undiagnosed"] = np.where(out["hypertensive"] == 1, np.where(out["prev_dx"] == 0, 1, np.where(pd.isna(out["prev_dx"]), np.nan, 0)), np.nan)

    # Age: prefer 'age' if available else 'ar09'
    age = out.get("age")
    if age is None or age.isna().all():
        age = out.get("ar09")
    if age is None:
        out["age_years"] = np.nan
    else:
        out["age_years"] = pd.to_numeric(age, errors="coerce")

    # Sex: map to female indicator
    def map_female(v):
        code, label = _extract_code_and_label(v)
        if isinstance(label, str) and label:
            return 1 if "female" in label.lower() else (0 if "male" in label.lower() else np.nan)
        if code is None:
            return np.nan
        # IFLS sometimes uses 1:Male, 2:Female OR 1:Male, 3:Female; treat code>=2 as female except unknown
        return 1 if int(code) in [2, 3] else 0

    if "sex" in out.columns:
        out["female"] = pd.Series([map_female(v) for v in out["sex"]], index=out.index)
    else:
        out["female"] = np.nan

    # General health status: prefer kk01 categories if available; else fallback to si01
    if "kk01" in out.columns:
        # kk01 labels: 1:Very healthy, 2:Somewhat healthy, 3:Somewhat unhealthy, 4:Very unhealthy
        kk_vals = out["kk01"].tolist()
        mapped = []
        for v in kk_vals:
            code, label = _extract_code_and_label(v)
            if code is None:
                mapped.append(np.nan)
            else:
                try:
                    c = int(float(code))
                except Exception:
                    mapped.append(np.nan)
                    continue
                if c in [3, 4]:
                    mapped.append(1)
                elif c in [1, 2]:
                    mapped.append(0)
                else:
                    mapped.append(np.nan)
        out["poor_health"] = pd.Series(mapped, index=out.index)
    elif "si01" in out.columns:
        # Fallback si01 numeric: 1-2 good, 3-5 poor, 8/9 unknown
        si01_num = pd.to_numeric(out["si01"], errors="coerce")
        out["poor_health"] = si01_num.apply(lambda x: (1 if x in [3,4,5] else (0 if x in [1,2] else np.nan)))
    else:
        out["poor_health"] = np.nan

    # Household size
    hh_size = pd.to_numeric(out.get("hh_size"), errors="coerce") if "hh_size" in out.columns else pd.Series([np.nan]*len(out), index=out.index)

    # Expenditures: prefer ks11aa (monthly). Fall back to ks10aa (weekly) * 4 if available.
    exp_month = pd.to_numeric(out.get("ks11aa"), errors="coerce") if "ks11aa" in out.columns else None
    if exp_month is None or (isinstance(exp_month, pd.Series) and exp_month.isna().all()):
        exp_week = pd.to_numeric(out.get("ks10aa"), errors="coerce") if "ks10aa" in out.columns else None
        if exp_week is not None:
            exp_month = exp_week * 4.0
    if exp_month is None:
        out["log_pce"] = np.nan
    else:
        denom = hh_size.replace(0, np.nan) if isinstance(hh_size, pd.Series) else hh_size
        pce = exp_month / denom if isinstance(exp_month, pd.Series) else np.nan
        out["log_pce"] = np.log1p(pce)

    # Distance to nearest health center: rj11 (impute missing to median and include missingness indicator)
    if "rj11" in out.columns:
        dist_raw = pd.to_numeric(out.get("rj11"), errors="coerce")
        out["dist_hc_missing"] = dist_raw.isna().astype(int)
        med = float(dist_raw.median(skipna=True)) if not pd.isna(dist_raw.median(skipna=True)) else 0.0
        out["dist_health_center"] = dist_raw.fillna(med)
    else:
        out["dist_hc_missing"] = 1
        out["dist_health_center"] = 0.0

    # Years of education
    out["years_education"] = build_years_of_education(out)

    return out


def run_probit_poor_health(df):
    # Restrict to: age>=45, hypertensive==1, poor_health==1, non-missing undiagnosed and years_education
    d = df.copy()

    mask = (
        (d["age_years"] >= 45) &
        (d["hypertensive"] == 1) &
        (d["poor_health"] == 1)
    )
    cols = ["undiagnosed", "years_education", "female", "age_years", "log_pce", "dist_health_center"]
    dmask = d.loc[mask, cols]
    # Debug diagnostics
    try:
        print("DEBUG: total_n=", len(d))
        print("DEBUG: nonmissing_age=", d["age_years"].notna().sum())
        print("DEBUG: hypertensive_sum=", int((d["hypertensive"] == 1).sum()))
        print("DEBUG: poor_health_sum=", int((d["poor_health"] == 1).sum()))
        print("DEBUG: mask_sum=", int(mask.sum()))
        print("DEBUG: mask_nonmissing_counts=", dmask.notna().sum().to_dict())
        print("DEBUG: mask_missing_counts=", dmask.isna().sum().to_dict())
    except Exception as _e:
        pass

    dsub = dmask.dropna()

    if dsub.empty:
        raise RuntimeError("Filtered analysis dataset is empty after applying criteria. Check variable construction and availability.")

    y = dsub["undiagnosed"].astype(int)
    X = dsub[["years_education", "female", "age_years", "log_pce", "dist_health_center"]].copy()
    X["age_sq"] = X["age_years"] ** 2
    X = sm.add_constant(X, has_constant="add")

    model = sm.Probit(y, X)
    res = model.fit(disp=False)

    # Average marginal effects
    margeff = res.get_margeff(at="overall")
    me_summary = margeff.summary_frame()

    # Extract effects for years_education
    if "years_education" in me_summary.index:
        me_row = me_summary.loc["years_education"]
        marginal_effect = float(me_row["dy/dx"]) if "dy/dx" in me_row else float(me_row[0])
        me_se = float(me_row["Std. Err."]) if "Std. Err." in me_row else None
        me_p = float(me_row["P>|z|"]) if "P>|z|" in me_row else None
    else:
        marginal_effect = None
        me_se = None
        me_p = None

    results = {
        "sample_size": int(len(dsub)),
        "n_total": int(len(df)),
        "filters": {
            "age_years_min": 45,
            "hypertensive": 1,
            "poor_health": 1
        },
        "variable_construction": {
            "hypertension_rule": "Avg systolic (us07b1, us07c1) > 140 or avg diastolic (us07b2, us07c2) > 90",
            "undiagnosed": "1 if hypertensive==1 and cd05 != 'Yes' (previously diagnosed)",
            "years_education": "Constructed from dl06 (level) and dl07 (grade/graduated) using approximate mapping",
            "log_pce": "log((monthly household expenditures ks11aa)/hh_size + 1); ks10aa*4 used if ks11aa missing",
            "dist_health_center": "rj11 numeric"
        },
        "probit_marginal_effects": {
            "years_education": {
                "dy_dx": marginal_effect,
                "std_err": me_se,
                "p_value": me_p
            }
        }
    }

    # Save outputs
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    with open(OUTPUT_TXT, "w") as f:
        f.write("Probit model (undiagnosed ~ years_education + female + age + age^2 + log_pce + dist_health_center)\n")
        f.write(res.summary2().as_text())
        f.write("\n\nAverage marginal effects:\n")
        f.write(me_summary.to_string())
        f.write("\n")

    # Also save the used analysis dataset for transparency
    dsub.to_csv(OUTPUT_DATA_CSV, index=False)

    return results


def main():
    df = load_data(DATA_PATH)
    df2 = compute_variables(df)
    results = run_probit_poor_health(df2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
