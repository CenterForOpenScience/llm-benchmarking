import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from utils_io__py import (
    load_dataset,
    find_col,
    normalize_col,
    robust_ols,
    export_regression_csv,
    append_model_summary,
    ensure_dir,
)


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    # Identify columns by flexible matching
    covid_col = find_col(df, [
        "COVID.04.04_08.11",
        "covid_04_04_08_11",
        "covid04040811",
        "cases_2020_04_04_to_2020_08_11",
    ])
    if covid_col is None:
        raise KeyError("Could not locate the post-original COVID window column 'COVID.04.04_08.11'.")

    pop_col = find_col(df, ["popData2019", "population2019", "pop_2019"])
    if pop_col is None:
        raise KeyError("Could not locate population column (e.g., 'popData2019').")

    # Democracy could be 0-100 (Gapminder) or already 0-10 (EIU). We will try both common labels.
    dem_col = find_col(df, [
        "Democracy index (EIU)",
        "democracy_index_eiu",
        "democracy_eiu",
        "gapminder_democracy_index",
        "democracy_index",
        "democracy",
    ])
    if dem_col is None:
        raise KeyError("Could not locate a democracy index column (e.g., 'Democracy index (EIU)').")

    temp_col = find_col(df, ["Annual_temp", "temperature", "avg_temperature", "annual_temperature"])
    if temp_col is None:
        raise KeyError("Could not locate temperature column (e.g., 'Annual_temp').")

    open_col = find_col(df, ["trade.2016", "trade_2016", "openness", "trade_open"])
    if open_col is None:
        raise KeyError("Could not locate trade openness column (e.g., 'trade.2016').")

    out = df.copy()

    # Construct outcome: cases per million over the post-original window
    out["total_cases_window"] = pd.to_numeric(out[covid_col], errors="coerce")
    out["population_2019"] = pd.to_numeric(out[pop_col], errors="coerce")
    out["cases_per_million"] = (out["total_cases_window"] / out["population_2019"]) * 1e6

    # Democracy scaling: if value appears on 0-100 scale (median > 10), divide by 10.
    out["democracy_raw"] = pd.to_numeric(out[dem_col], errors="coerce")
    median_dem = out["democracy_raw"].median(skipna=True)
    if pd.notnull(median_dem) and median_dem > 10:
        out["democracy"] = out["democracy_raw"] / 10.0
        dem_note = "Democracy scaled from 0-100 to 0-10 by dividing by 10."
    else:
        out["democracy"] = out["democracy_raw"]
        dem_note = "Democracy appears already on 0-10 scale (no scaling applied)."

    # Controls
    out["temperature"] = pd.to_numeric(out[temp_col], errors="coerce")
    out["openness"] = pd.to_numeric(out[open_col], errors="coerce")

    # Keep relevant columns
    keep = [
        "cases_per_million",
        "democracy",
        "temperature",
        "openness",
    ]
    clean = out[keep].dropna()

    return clean, {
        "covid_col": covid_col,
        "pop_col": pop_col,
        "dem_col": dem_col,
        "temp_col": temp_col,
        "open_col": open_col,
        "dem_note": dem_note,
        "n_after_listwise": int(clean.shape[0]),
    }


def run_model(df: pd.DataFrame, cov_type: str = "HC1"):
    y = df["cases_per_million"]
    X = df[["democracy", "temperature", "openness"]]
    res = robust_ols(y, X, cov_type=cov_type)
    return res


def save_coef_plot(res, out_path: str):
    params = res.params.drop(labels=["const"], errors="ignore")
    conf = res.conf_int().rename(columns={0: "low", 1: "high"})
    conf = conf.loc[params.index]

    plot_df = pd.DataFrame({
        "term": params.index,
        "coef": params.values,
        "low": conf["low"].values,
        "high": conf["high"].values,
    })

    plt.figure(figsize=(6, 4))
    sns.pointplot(
        data=plot_df,
        y="term",
        x="coef",
        join=False,
        color="black",
    )
    for i, row in plot_df.iterrows():
        plt.plot([row["low"], row["high"]], [i, i], color="black", linewidth=1)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.title("OLS coefficients with 95% CI")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def save_coef_plot_v2(res, out_path: str):
    import numpy as _np
    import pandas as _pd
    # Derive parameter names from the model, fallback to generic names
    try:
        names = list(res.model.exog_names)
    except Exception:
        names = [f"param_{i}" for i in range(len(_np.asarray(res.params)))]

    params = _np.asarray(res.params)
    conf_raw = res.conf_int(alpha=0.05)
    conf_arr = _np.asarray(conf_raw)

    # Build full coefficient table then drop intercept if present
    df_all = _pd.DataFrame({
        "term": names,
        "coef": params,
        "low": conf_arr[:, 0],
        "high": conf_arr[:, 1],
    })
    plot_df = df_all[df_all["term"].str.lower() != "const"].reset_index(drop=True)

    plt.figure(figsize=(6, 4))
    sns.pointplot(data=plot_df, y="term", x="coef", join=False, color="black")
    for i, row in plot_df.iterrows():
        plt.plot([row["low"], row["high"]], [i, i], color="black", linewidth=1)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.title("OLS coefficients with 95% CI")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
def main():
    ensure_dir("/app/data")
    # Load dataset
    df, src_path = load_dataset()
    print(f"Loaded dataset from: {src_path}")

    # Prepare variables per preregistration/plan
    clean, meta = prepare_variables(df)
    print(json.dumps(meta, indent=2))

    if clean.shape[0] == 0:
        raise RuntimeError("No observations available after listwise deletion on required variables.")

    # Run OLS with robust SEs
    res = run_model(clean, cov_type="HC1")

    # Export
    out_csv = "/app/data/replication_results.csv"
    out_fig = "/app/data/coef_plot.png"
    out_df = export_regression_csv(res, out_csv)
    summ_path = append_model_summary(out_csv, r2=float(res.rsquared), nobs=int(res.nobs))
    save_coef_plot_v2(res, out_fig)

    # Also write a compact JSON for the focal coefficient
    foc = "democracy"
    foc_row = out_df[out_df["term"] == foc]
    if not foc_row.empty:
        foc_json = {
            "term": foc,
            "coef": float(foc_row.iloc[0]["coef"]),
            "std_err": float(foc_row.iloc[0]["std_err"]),
            "t": float(foc_row.iloc[0]["t"]),
            "p_value": float(foc_row.iloc[0]["p_value"]),
            "conf_low": float(foc_row.iloc[0]["conf_low"]),
            "conf_high": float(foc_row.iloc[0]["conf_high"]),
            "r_squared": float(res.rsquared),
            "nobs": int(res.nobs),
        }
        with open("/app/data/focal_democracy_result.json", "w") as f:
            json.dump(foc_json, f, indent=2)

    # Console summary
    print("Model results (robust HC1):")
    print(res.summary())
    print(f"Results saved to: {out_csv}, summary: {summ_path}, figure: {out_fig}")


if __name__ == "__main__":
    main()


def save_coef_plot_v2(res, out_path: str):
    import numpy as _np
    import pandas as _pd
    # Derive parameter names from the model, fallback to generic names
    try:
        names = list(res.model.exog_names)
    except Exception:
        names = [f"param_{i}" for i in range(len(_np.asarray(res.params)))]

    params = _np.asarray(res.params)
    conf_raw = res.conf_int(alpha=0.05)
    conf_arr = _np.asarray(conf_raw)

    # Build full coefficient table then drop intercept if present
    df_all = _pd.DataFrame({
        "term": names,
        "coef": params,
        "low": conf_arr[:, 0],
        "high": conf_arr[:, 1],
    })
    plot_df = df_all[df_all["term"].str.lower() != "const"].reset_index(drop=True)

    plt.figure(figsize=(6, 4))
    sns.pointplot(data=plot_df, y="term", x="coef", join=False, color="black")
    for i, row in plot_df.iterrows():
        plt.plot([row["low"], row["high"]], [i, i], color="black", linewidth=1)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.title("OLS coefficients with 95% CI")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
