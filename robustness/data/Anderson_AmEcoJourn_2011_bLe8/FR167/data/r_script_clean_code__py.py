import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# All IO must use /app/data
BASE_DATA_DIR = "/app/data"
OUT_FILE = os.path.join(BASE_DATA_DIR, "analysis_results.json")

np.set_printoptions(suppress=True)
pd.set_option('display.width', 200)


def read_data():
    data_dir = os.path.join(BASE_DATA_DIR, "AEJApp-2009-0289-data")
    hh_path = os.path.join(data_dir, "household.dta")
    t2_path = os.path.join(data_dir, "table2.dta")

    d = pd.read_stata(hh_path)
    table2 = pd.read_stata(t2_path)

    return d, table2


def prepare_data(d: pd.DataFrame, table2: pd.DataFrame) -> pd.DataFrame:
    # Keep relevant columns for consistency checks
    # Merge domhigh from table2 to main d using hhcode
    t2_sub = table2[["hhcode", "domhigh"]].rename(columns={"domhigh": "domhigh_from_table2"})
    d = pd.merge(d, t2_sub, on="hhcode", how="left")

    # Ensure village is a factor-like string as in R code for filtering later
    if "village" in d.columns:
        d["village_factor"] = d["village"].astype(str)
    else:
        raise ValueError("'village' column not found in household data.")

    return d


def compute_normality_and_md(d: pd.DataFrame):
    # Compute skewness and kurtosis for totinc (omit NA)
    totinc = d["totinc"].to_numpy()
    totinc = totinc[np.isfinite(totinc)]
    skew = stats.skew(totinc)
    kurt = stats.kurtosis(totinc, fisher=False)  # match e1071 default (non-Fisher) approx

    # Mahalanobis distance on domlow and totinc (rows with non-missing)
    md_df = d[["domlow", "totinc", "hhcode"]].dropna()
    # Ensure numeric
    X = md_df[["domlow", "totinc"]].astype(float).to_numpy()
    # Compute mean and covariance
    mu = X.mean(axis=0)
    S = np.cov(X, rowvar=False)
    # Inverse covariance
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        # add small ridge if singular
        S_inv = np.linalg.pinv(S)
    # Mahalanobis distance squared
    diff = X - mu
    mdsq = np.einsum('ij,jk,ik->i', diff, S_inv, diff)
    # p-values with df=k-1 where k = number of variables (R code used df=1)
    # Following the R code's comment: df = 1 for two variables? They used df=1.
    pvals = stats.chi2.sf(mdsq, df=1)

    outlier_flag = (pvals < 0.001).astype(int)
    md_out = md_df[["hhcode"]].copy()
    md_out["Mahalanobis"] = mdsq
    md_out["pvalue"] = pvals
    md_out["Outlier"] = outlier_flag

    # Merge back to main d
    d = pd.merge(d, md_out, on="hhcode", how="left")

    return {
        "skewness_totinc": float(skew) if np.isfinite(skew) else None,
        "kurtosis_totinc": float(kurt) if np.isfinite(kurt) else None,
        "n_md_rows": int(md_df.shape[0]),
        "n_md_outliers_p_lt_0_001": int(outlier_flag.sum()),
    }, d


def fit_models(d: pd.DataFrame):
    results = {}

    # Define datasets similar to R code
    # Full data: drop missing on totinc, domlow, village
    full = d.dropna(subset=["totinc", "domlow", "village"]).copy()

    # Dataset without six outlying villages (numeric comparison to avoid '70' vs '70.0' mismatch)
    outlier_villages = {70, 61, 60, 84, 59, 58}
    vill_num = pd.to_numeric(full["village"], errors="coerce").astype("Int64")
    noout = full[~vill_num.isin(list(outlier_villages))].copy()

    def mixedlm_summary(df, label):
        # Random intercept for village
        try:
            md = MixedLM.from_formula("totinc ~ domlow", groups=df["village"], data=df)
            mres = md.fit(reml=False, method='lbfgs', maxiter=100, disp=False)
            params = mres.params
            bse = mres.bse
            conf_int = mres.conf_int(alpha=0.05)
            # MixedLM uses z-values; record those
            zvalues = params / bse
            pvalues = mres.pvalues
            # Random intercept variance
            vc = mres.cov_re.iloc[0, 0] if hasattr(mres, 'cov_re') else None
            resid_var = mres.scale if hasattr(mres, 'scale') else None
            return {
                "n_obs": int(df.shape[0]),
                "coef_domlow": float(params.get("domlow", np.nan)),
                "se_domlow": float(bse.get("domlow", np.nan)),
                "z_domlow": float(zvalues.get("domlow", np.nan)),
                "p_domlow": float(pvalues.get("domlow", np.nan)),
                "ci95_domlow": [float(conf_int.loc["domlow", 0]), float(conf_int.loc["domlow", 1])],
                "random_intercept_var": float(vc) if vc is not None else None,
                "residual_var": float(resid_var) if resid_var is not None else None,
                "aic": float(mres.aic) if hasattr(mres, 'aic') else None,
                "bic": float(mres.bic) if hasattr(mres, 'bic') else None,
                "converged": bool(mres.converged),
                "method": "MixedLM (random intercept by village)",
                "label": label
            }
        except Exception as e:
            return {"error": f"MixedLM failed: {e}", "label": label}

    def ols_summary(df, label):
        try:
            X = sm.add_constant(df[["domlow"]].astype(float))
            y = df["totinc"].astype(float)
            ols = sm.OLS(y, X, missing='drop').fit()
            ci = ols.conf_int().loc["domlow"].to_numpy().tolist()
            return {
                "n_obs": int(df.shape[0]),
                "coef_domlow": float(ols.params.get("domlow", np.nan)),
                "se_domlow": float(ols.bse.get("domlow", np.nan)),
                "t_domlow": float(ols.tvalues.get("domlow", np.nan)),
                "p_domlow": float(ols.pvalues.get("domlow", np.nan)),
                "ci95_domlow": [float(ci[0]), float(ci[1])],
                "r2": float(ols.rsquared),
                "aic": float(ols.aic),
                "bic": float(ols.bic),
                "method": "OLS",
                "label": label
            }
        except Exception as e:
            return {"error": f"OLS failed: {e}", "label": label}

    results["mixedlm_full"] = mixedlm_summary(full, label="full_dataset")
    results["mixedlm_noout"] = mixedlm_summary(noout, label="without_six_outlying_villages")

    results["ols_full"] = ols_summary(full, label="full_dataset")
    results["ols_noout"] = ols_summary(noout, label="without_six_outlying_villages")

    # Some simple descriptive means for figure-like summary
    try:
        desc = full.groupby("domlow")["totinc"].agg(["count", "mean", "std"]).reset_index()
        desc_noout = noout.groupby("domlow")["totinc"].agg(["count", "mean", "std"]).reset_index()
        results["descriptives_full"] = desc.to_dict(orient="list")
        results["descriptives_noout"] = desc_noout.to_dict(orient="list")
    except Exception:
        pass

    return results


def main():
    out = {"steps": []}

    d, table2 = read_data()
    out["steps"].append({"read_data": {
        "household_cols": list(d.columns),
        "table2_cols": list(table2.columns),
        "n_households": int(d.shape[0]),
        "n_table2": int(table2.shape[0])
    }})

    d = prepare_data(d, table2)
    out["steps"].append({"prepare_data": {
        "columns_after_merge": list(d.columns),
        "example_villages": d["village"].dropna().astype(str).unique()[:5].tolist()
    }})

    norm_md, d = compute_normality_and_md(d)
    out["normality_md"] = norm_md

    model_results = fit_models(d)
    out["model_results"] = model_results

    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)

    # Also print a concise summary to stdout
    def fmt_res(key):
        res = model_results.get(key, {})
        if "error" in res:
            return f"{key}: ERROR {res['error']}"
        meth = res.get("method", "?")
        label = res.get("label", "?")
        coef = res.get("coef_domlow", np.nan)
        se = res.get("se_domlow", np.nan)
        p = res.get("p_domlow", np.nan)
        ci = res.get("ci95_domlow", [np.nan, np.nan])
        return f"{meth} [{label}] domlow: coef={coef:.3f}, SE={se:.3f}, p={p:.3g}, CI95=({ci[0]:.3f},{ci[1]:.3f}), n={res.get('n_obs','?')}"

    print("Results summary:")
    print(fmt_res("mixedlm_full"))
    print(fmt_res("mixedlm_noout"))
    print(fmt_res("ols_full"))
    print(fmt_res("ols_noout"))
    print(f"Skewness(totinc)={out['normality_md']['skewness_totinc']}, Kurtosis(totinc)={out['normality_md']['kurtosis_totinc']}")
    print(f"Mahalanobis outliers (p<0.001): {out['normality_md']['n_md_outliers_p_lt_0_001']}")


if __name__ == "__main__":
    main()
