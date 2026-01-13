import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_cluster_2groups


def find_data():
    candidates = [
        "/app/data/Afghanistan_Election_Violence_2014.csv",
        "/app/data/Afghanistan_Election_Violence_2014.dta",
        "replication_data/Afghanistan_Election_Violence_2014.csv",
        "replication_data/Afghanistan_Election_Violence_2014.dta",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Afghanistan_Election_Violence_2014.{csv,dta} not found in /app/data or replication_data/")


def load_dataset():
    path = find_data()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".dta":
        df = pd.read_stata(path)
    else:
        raise ValueError(f"Unsupported data extension: {ext}")
    return df


def ensure_binary(series):
    # Coerce to 0/1 if not already
    s = series.copy()
    # If values are already 0/1, return as is; else threshold >0 -> 1
    uniq = pd.unique(s.dropna())
    if set(pd.Series(uniq).astype(float, errors='ignore')).issubset({0, 1}):
        return s.astype(int)
    return (s > 0).astype(int)




def ensure_binary_safe(series):
    s = pd.to_numeric(series, errors='coerce')
    # Map any positive numeric to 1, else 0; treat NaNs as 0 (they should be dropped earlier anyway)
    s = s.fillna(0)
    return (s > 0).astype(int)
def fit_logit_with_clusters(df, outcome, viol_var, controls, cluster1=None, cluster2=None):
    # Build design matrix with quadratic term
    data = df[[outcome] + [viol_var] + controls].copy()
    data = data.dropna()
    data[outcome] = ensure_binary_safe(data[outcome])

    data["viol"] = data[viol_var].astype(float)
    data["viol_sq"] = data["viol"] ** 2

    X = data[["viol", "viol_sq"] + controls].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = data[outcome]

    model = sm.Logit(y, X)
    res = model.fit(disp=0, maxiter=1000)

    # One-way clustered SEs
    table_one = None
    if cluster1 is not None and cluster1 in data.columns:
        g1 = data[cluster1]
        try:
            cov1 = cov_cluster(res, g1)
        except Exception:
            # Fallback: robust HC1
            cov1 = res.cov_params()
        table_one = build_coef_table(res.params, cov1, X.columns, label=f"{viol_var} one-way cluster by {cluster1}")
    else:
        # Fallback to default covariance
        cov_def = res.cov_params()
        table_one = build_coef_table(res.params, cov_def, X.columns, label=f"{viol_var} default covariance")

    # Two-way clustered SEs
    table_two = None
    if cluster1 is not None and cluster1 in data.columns and cluster2 is not None and cluster2 in data.columns:
        g1 = data[cluster1]
        g2 = data[cluster2]
        try:
            cov2 = cov_cluster_2groups(res, g1, g2)
            table_two = build_coef_table(res.params, cov2, X.columns, label=f"{viol_var} two-way cluster by {cluster1},{cluster2}")
        except Exception:
            # If two-way fails, keep None
            table_two = None

    meta = {
        "n_obs": int(res.nobs),
        "llf": float(res.llf),
        "prsquared": float(res.prsquared) if hasattr(res, "prsquared") else None,
        "turning_point": compute_turning_point(res.params.get("viol", np.nan), res.params.get("viol_sq", np.nan)),
    }

    return res, table_one, table_two, meta, data


def compute_turning_point(b1, b2):
    try:
        if np.isfinite(b1) and np.isfinite(b2) and b2 != 0:
            return -b1 / (2.0 * b2)
    except Exception:
        pass
    return None


def build_coef_table(params, cov, names, label=""):
    se = np.sqrt(np.diag(cov))
    idx = list(names)
    coefs = params.loc[idx].values if hasattr(params, 'loc') else np.array([params[name] for name in idx])
    z = coefs / se
    p = 2 * (1 - norm.cdf(np.abs(z)))
    out = pd.DataFrame({
        "term": idx,
        "coef": coefs,
        "std_err": se,
        "z": z,
        "p_value": p,
        "spec_label": label,
    })
    return out


def predict_margins(res, controls_means, viol_grid, viol_name):
    # Build design for grid
    df_grid = pd.DataFrame({
        "viol": viol_grid,
        "viol_sq": viol_grid ** 2,
    })
    for c, v in controls_means.items():
        df_grid[c] = v
    Xg = sm.add_constant(df_grid, has_constant='add')
    preds = res.predict(Xg)
    out = pd.DataFrame({
        viol_name: viol_grid,
        "pred_prob_fraud": preds,
    })
    return out


def save_csv(df, filename):
    out_dir = "/app/data"
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        df.to_csv(path, index=False)
        print(f"Saved: {path}")
        return path
    except Exception as e:
        # fallback to current directory
        path = filename
        df.to_csv(path, index=False)
        print(f"Saved (fallback): {path} -> {e}")
        return path


def save_plot(fig, filename):
    out_dir = "/app/data"
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f"Saved: {path}")
        return path
    except Exception as e:
        path = filename
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f"Saved (fallback): {path} -> {e}")
        return path


def main():
    print("Loading dataset...")
    df = load_dataset()
    print(f"Columns: {list(df.columns)}")

    # Expected columns
    outcome = "fraud"
    controls = ["pcx", "electric", "pcexpend", "dist", "elevation"]
    cluster1 = "regcom"
    cluster2 = "elect"

    # Verify presence of columns; filter controls to available
    controls_avail = [c for c in controls if c in df.columns]
    missing_ctrl = list(set(controls) - set(controls_avail))
    if missing_ctrl:
        print(f"Warning: Missing control variables dropped: {missing_ctrl}")
    if outcome not in df.columns:
        raise ValueError("Required outcome 'fraud' not found in dataset")

    models = [
        {"viol": "sigact_5r", "grid": np.arange(0.0, 0.4001, 0.01), "grid_name": "sigact_5r"},
        {"viol": "sigact_60r", "grid": np.arange(0.0, 3.0001, 0.075), "grid_name": "sigact_60r"},
    ]

    tables_one = []
    tables_two = []
    meta_rows = []

    for m in models:
        viol_var = m["viol"]
        if viol_var not in df.columns:
            print(f"Warning: {viol_var} not found; skipping this model.")
            continue

        print(f"Fitting Logit for {viol_var} with quadratic term ...")
        res, tab1, tab2, meta, used = fit_logit_with_clusters(
            df,
            outcome=outcome,
            viol_var=viol_var,
            controls=controls_avail,
            cluster1=cluster1 if cluster1 in df.columns else None,
            cluster2=cluster2 if cluster2 in df.columns else None,
        )
        meta.update({"viol_var": viol_var})
        meta_rows.append(meta)

        if tab1 is not None:
            tables_one.append(tab1)
        if tab2 is not None:
            tables_two.append(tab2)

        # Marginal predictions
        ctrl_means = used[controls_avail].mean().to_dict()
        margins = predict_margins(res, ctrl_means, m["grid"], viol_name=m["grid_name"])
        margins_file = f"margins_{m['grid_name'].replace('sigact_', '')}.csv"
        plot_file = f"margins_{m['grid_name'].replace('sigact_', '')}.png"
        save_csv(margins, margins_file)

        # Plot
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(margins[m["grid_name"]], margins["pred_prob_fraud"], label=f"Predicted Pr(fraud)")
        ax.set_xlabel(m["grid_name"])
        ax.set_ylabel("Predicted probability of fraud")
        ax.set_title(f"Margins over {m['grid_name']} (controls at means)")
        ax.legend()
        save_plot(fig, plot_file)
        plt.close(fig)

        tp = meta.get("turning_point")
        if tp is not None:
            print(f"Turning point for {viol_var}: {tp:.4f}")
        else:
            print(f"Turning point for {viol_var}: not defined")

    # Save coefficient tables
    if tables_one:
        t1 = pd.concat(tables_one, axis=0, ignore_index=True)
        save_csv(t1, "Table2_oneway.csv")
    else:
        print("No one-way clustered results to save.")

    if tables_two:
        t2 = pd.concat(tables_two, axis=0, ignore_index=True)
        save_csv(t2, "Table2_twoway.csv")
    else:
        print("Two-way clustered results unavailable or failed; skipping Table2_twoway.csv")

    # Save metadata summary
    if meta_rows:
        meta_df = pd.DataFrame(meta_rows)
        save_csv(meta_df, "model_meta_summary.csv")

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)
