import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def find_data_path():
    candidates = [
        "/app/data/ReplicationData_Cohen_AmEcoRev_2015_2lb5.dta",
        "/app/data/replication_data/ReplicationData_Cohen_AmEcoRev_2015_2lb5.dta",
        # Fallbacks for local testing
        os.path.join(os.getcwd(), "ReplicationData_Cohen_AmEcoRev_2015_2lb5.dta"),
        os.path.join(os.getcwd(), "replication_data", "ReplicationData_Cohen_AmEcoRev_2015_2lb5.dta"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not locate ReplicationData_Cohen_AmEcoRev_2015_2lb5.dta in /app/data or /app/data/replication_data.")


def make_asset_dummy(series, code_str):
    def _f(x):
        if pd.isna(x):
            return np.nan
        s = str(x)
        return 1 if code_str in s else 0
    return series.apply(_f)


def main():
    out_dir = "/app/data"
    os.makedirs(out_dir, exist_ok=True)

    data_path = find_data_path()
    print(f"Loading data from: {data_path}")
    df = pd.read_stata(data_path, convert_categoricals=False)

    # Outcome: took_ACT from drugs_taken_AL
    if "drugs_taken_AL" not in df.columns:
        raise KeyError("Expected column 'drugs_taken_AL' not found in dataset.")
    df["took_ACT"] = df["drugs_taken_AL"]

    # Treatment: act_subsidy from maltest_chw_voucher_given
    if "maltest_chw_voucher_given" not in df.columns:
        raise KeyError("Expected column 'maltest_chw_voucher_given' not found in dataset.")
    df["act_subsidy"] = np.where(df["maltest_chw_voucher_given"] == 1, 1, 0)
    df.loc[df["maltest_chw_voucher_given"] == 98, "act_subsidy"] = np.nan

    # Assets from ses_hh_items; covariates used: refrigerator (code '3'), mobile (code '5')
    if "ses_hh_items" in df.columns:
        df["refrigerator"] = make_asset_dummy(df["ses_hh_items"], "3")
        df["mobile"] = make_asset_dummy(df["ses_hh_items"], "5")
    else:
        # If missing, create as NaN to be dropped later
        df["refrigerator"] = np.nan
        df["mobile"] = np.nan

    # Toilet type covariates from ses_toilet_type: vip(2), composting(5), other(8)
    if "ses_toilet_type" not in df.columns:
        raise KeyError("Expected column 'ses_toilet_type' not found in dataset.")
    df["vip_toilet"] = np.where(df["ses_toilet_type"] == 2, 1, np.where(df["ses_toilet_type"].isna(), np.nan, 0))
    df["composting_toilet"] = np.where(df["ses_toilet_type"] == 5, 1, np.where(df["ses_toilet_type"].isna(), np.nan, 0))
    df["other_toilet"] = np.where(df["ses_toilet_type"] == 8, 1, np.where(df["ses_toilet_type"].isna(), np.nan, 0))

    # Wall materials from ses_wall_material: stone(1), cement(7)
    if "ses_wall_material" not in df.columns:
        raise KeyError("Expected column 'ses_wall_material' not found in dataset.")
    df["stone_wall"] = np.where(df["ses_wall_material"] == 1, 1, np.where(df["ses_wall_material"].isna(), np.nan, 0))
    df["cement_wall"] = np.where(df["ses_wall_material"] == 7, 1, np.where(df["ses_wall_material"].isna(), np.nan, 0))

    # Number of sheep
    if "ses_no_sheep" not in df.columns:
        raise KeyError("Expected column 'ses_no_sheep' not found in dataset.")
    df["num_sheep"] = df["ses_no_sheep"]

    # Required variables for subsetting and weights
    for col in ["maltest_where", "wave", "weight", "cu_code"]:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in dataset.")

    # Subset: maltest_where==1 & wave!=0
    df_sub = df.loc[(df["maltest_where"] == 1) & (df["wave"] != 0)].copy()

    # Drop missing rows for model vars
    model_vars = [
        "took_ACT", "act_subsidy", "refrigerator", "mobile", "vip_toilet",
        "composting_toilet", "other_toilet", "stone_wall", "cement_wall",
        "num_sheep", "cu_code", "weight"
    ]
    before_n = len(df_sub)
    df_sub = df_sub.dropna(subset=model_vars)
    after_n = len(df_sub)
    print(f"Sample restricted from N={before_n} to N={after_n} after dropping missing model variables.")

    # Ensure numeric types where appropriate
    for col in ["took_ACT", "act_subsidy", "refrigerator", "mobile", "vip_toilet",
                "composting_toilet", "other_toilet", "stone_wall", "cement_wall",
                "num_sheep", "weight"]:
        df_sub[col] = pd.to_numeric(df_sub[col], errors="coerce")

    # Weighted LPM with strata FE (C(cu_code)); robust SE (HC1)
    formula = (
        "took_ACT ~ act_subsidy + C(cu_code) + refrigerator + mobile + "
        "vip_toilet + composting_toilet + other_toilet + stone_wall + cement_wall + num_sheep"
    )
    print("Fitting WLS model with HC1 robust standard errors...")
    wls_model = smf.wls(formula=formula, data=df_sub, weights=df_sub["weight"])  # weights are pweights
    wls_res = wls_model.fit()
    rob_res = wls_res.get_robustcov_results(cov_type="HC1")

    # Save summary
    summary_path = os.path.join(out_dir, "cohen2015_replication_summary.txt")
    with open(summary_path, "w") as f:
        f.write(str(rob_res.summary()))
    print(f"Model summary saved to: {summary_path}")
    # Prepare arrays and names for robust results (handles cases where params are NumPy arrays)
    names = getattr(rob_res.model, 'exog_names', None)
    if names is None:
        try:
            names = wls_res.model.exog_names
        except Exception:
            names = None
    if names is None:
        names = [f"x{i}" for i in range(len(np.atleast_1d(rob_res.params)))]

    params = np.atleast_1d(rob_res.params)
    bse = np.atleast_1d(rob_res.bse)
    tvals = np.atleast_1d(rob_res.tvalues)
    pvals = np.atleast_1d(rob_res.pvalues)
    ci_arr = np.asarray(rob_res.conf_int(alpha=0.05))

    # Save coefficients table using arrays/names
    res_table = pd.DataFrame({
        "param": names,
        "coef": params,
        "std_err": bse,
        "t": tvals,
        "p_value": pvals,
        "ci_lower": ci_arr[:, 0] if ci_arr.ndim == 2 and ci_arr.shape[1] == 2 else np.nan,
        "ci_upper": ci_arr[:, 1] if ci_arr.ndim == 2 and ci_arr.shape[1] == 2 else np.nan,
    })
    results_path = os.path.join(out_dir, "cohen2015_replication_results.csv")
    res_table.to_csv(results_path, index=False)
    print(f"Coefficients saved to: {results_path}")

    # Print focal estimate
    try:
        idx = names.index("act_subsidy")
        coef = params[idx]
        se = bse[idx]
        p = pvals[idx]
        print(f"Focal estimate - act_subsidy: coef={coef:.4f}, se={se:.4f}, p={p:.4g}")
    except ValueError:
        print("Warning: act_subsidy coefficient not found among parameter names.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
