import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_FILE = "/app/data/McCarter_OrgBehavior_2010_pILK data CLEAN.csv"
OUT_TEXT = "/app/data/mccarter_pILK_mixedlm_results.txt"
OUT_JSON = "/app/data/mccarter_pILK_mixedlm_results.json"


def to_named_dict(values, names=None):
    try:
        if hasattr(values, "index"):
            names = list(values.index)
            arr = np.asarray(values)
        else:
            arr = np.asarray(values)
        if names is None or len(names) != len(arr):
            names = [f"b{i}" for i in range(len(arr))]
        return {str(n): float(v) for n, v in zip(names, arr)}
    except Exception:
        try:
            return {str(i): float(v) for i, v in enumerate(values)}
        except Exception:
            return {}


def extract_regression(res):
    names = getattr(res.model, "exog_names", None)
    return {
        "params": to_named_dict(res.params, names),
        "bse": to_named_dict(res.bse, names),
        "pvalues": to_named_dict(res.pvalues, names),
        "nobs": int(getattr(res, "nobs", len(getattr(res.model, "endog", [])))),
        "df_resid": float(getattr(res, "df_resid", np.nan)),
    }


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["cont", "losses", "id", "sessioncode"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from dataset: {missing}")

    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["losses"] = pd.to_numeric(df["losses"], errors="coerce")
    df["cont"] = pd.to_numeric(df["cont"], errors="coerce")
    df["sessioncode"] = df["sessioncode"].astype(str)

    df = df.dropna(subset=["id", "losses", "cont", "sessioncode"]).copy()
    return df


def fit_ols_clustered(df: pd.DataFrame, y: str, x: str = "losses"):
    # OLS with session fixed effects and cluster-robust SEs by participant id
    formula = f"{y} ~ {x} + C(sessioncode)"
    ols = smf.ols(formula, data=df).fit()
    try:
        rob = ols.get_robustcov_results(cov_type='cluster', cov_kwds={'groups': df['id']})
        return rob
    except Exception:
        return ols.get_robustcov_results(cov_type='HC3')


def summarize_results(df: pd.DataFrame, res_cont, res_lossav=None, res_fear=None) -> str:
    lines = []
    lines.append("McCarter et al. (2010) replication analysis on pILK dataset")
    lines.append("")
    n_obs = len(df)
    n_subj = df["id"].nunique()
    n_sessions = df["sessioncode"].nunique()
    lines.append(f"Observations: {n_obs}")
    lines.append(f"Unique participants (id): {n_subj}")
    lines.append(f"Unique sessions: {n_sessions}")
    lines.append("")

    grp = df.groupby("losses")["cont"].agg(["count", "mean", "std"]).rename(index={0: "no_losses", 1: "losses"})
    lines.append("Contribution by losses indicator (0=gains-only; 1=losses-and-gains)")
    lines.append(grp.to_string(float_format=lambda v: f"{v:0.3f}"))
    lines.append("")

    lines.append("Primary model: cont ~ losses + C(sessioncode) with cluster-robust SEs by id")
    lines.append(res_cont.summary().as_text())
    lines.append("")
    if res_lossav is not None:
        lines.append("Secondary: loss_aversion ~ losses + C(sessioncode) (clustered by id)")
        lines.append(res_lossav.summary().as_text())
        lines.append("")
    if res_fear is not None:
        lines.append("Secondary: fear ~ losses + C(sessioncode) (clustered by id)")
        lines.append(res_fear.summary().as_text())
        lines.append("")

    return "\n".join(lines)


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Expected data file not found at {DATA_FILE}")

    df = load_data(DATA_FILE)
    df = prepare_df(df)

    # Primary model: OLS with session FE and cluster-robust SE by id
    res_cont = fit_ols_clustered(df, y="cont", x="losses")

    # Secondary models: loss_aversion and fear if present
    res_lossav = None
    res_fear = None
    if "loss_aversion" in df.columns:
        dfa = df.dropna(subset=["loss_aversion"])  # listwise for DV
        if len(dfa) > 0:
            res_lossav = fit_ols_clustered(dfa, y="loss_aversion", x="losses")
    if "fear" in df.columns:
        dff = df.dropna(subset=["fear"])  # listwise for DV
        if len(dff) > 0:
            res_fear = fit_ols_clustered(dff, y="fear", x="losses")

    # Write text summary
    text = summarize_results(df, res_cont, res_lossav, res_fear)
    with open(OUT_TEXT, "w") as f:
        f.write(text)

    # Extract key metrics to JSON
    out = {
        "data": {
            "n_obs": int(len(df)),
            "n_participants": int(df["id"].nunique()),
            "n_sessions": int(df["sessioncode"].nunique()),
            "group_means_cont": {str(k): float(v) for k, v in df.groupby("losses")["cont"].mean().to_dict().items()},
        },
        "models": {
            "cont_on_losses": extract_regression(res_cont)
        }
    }
    if res_lossav is not None:
        out["models"]["loss_aversion_on_losses"] = extract_regression(res_lossav)
    if res_fear is not None:
        out["models"]["fear_on_losses"] = extract_regression(res_fear)

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Analysis complete. Results written to:\n- {OUT_TEXT}\n- {OUT_JSON}")


if __name__ == "__main__":
    main()
