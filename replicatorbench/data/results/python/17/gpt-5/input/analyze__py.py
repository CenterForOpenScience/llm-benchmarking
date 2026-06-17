import os
import sys
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = os.environ.get("DATA_PATH", "")
OUTPUT_DIR = "/app/data"
SEED = 2020

MODEL_VARS = [
    "gov_consumption",
    "sd_gov",
    "mean_gov",
    "africa",
    "laam",
    "asiae",
    "col_uka",
    "col_espa",
    "col_otha",
    "federal",
    "oecd",
    "log_gdp_per_capita",
    "trade_share",
    "age_15_64",
    "age_65_plus",
]


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_data_path() -> str:
    # Priority: environment variable
    env_path = os.environ.get("DATA_PATH", "").strip()
    candidates = []
    if env_path:
        candidates.append(env_path)
    # Common candidates in this orchestrator
    candidates.extend([
        "/app/data/original/17/0112_python_gpt5/replication_data/data.csv",
        "/workspace/replication_data/data.csv",
        "/workspace/data.csv",
        os.path.join(os.getcwd(), "replication_data", "data.csv"),
    ])
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not locate dataset. Tried: {candidates}")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def permute_columns(df: pd.DataFrame, start_col_idx: int = 3, seed: int = SEED) -> pd.DataFrame:
    # Permute columns independently from start_col_idx onward
    rng = np.random.default_rng(seed)
    out = df.copy()
    cols = list(out.columns)
    for c in cols[start_col_idx:]:
        vals = out[c].values.copy()
        rng.shuffle(vals)
        out[c] = vals
    return out


def run_regression(df: pd.DataFrame, drop_wave_7: bool, test_mode: bool) -> dict:
    # Optionally permute for test mode
    work = permute_columns(df, start_col_idx=3, seed=SEED) if test_mode else df.copy()

    # Filter strong democracies
    if "democ" in work.columns:
        work = work[work["democ"] >= 9]
    else:
        raise KeyError("Column 'democ' not found for filtering strong democracies")

    # Drop wave 7 if requested
    if drop_wave_7:
        if "wave" not in work.columns:
            raise KeyError("Column 'wave' not found for wave filtering")
        work = work[work["wave"] != 7]

    # Subset variables and drop missing
    missing_cols = [c for c in MODEL_VARS if c not in work.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns for model: {missing_cols}")

    model_df = work[MODEL_VARS].dropna().copy()

    if model_df.empty:
        raise ValueError("No data left after filtering and dropping missing values")

    y = model_df["gov_consumption"].astype(float)
    X = model_df.drop(columns=["gov_consumption"]).astype(float)
    X = sm.add_constant(X, has_constant='add')

    model = sm.OLS(y, X)
    res = model.fit(cov_type='HC1')

    # Extract focal coefficient details for sd_gov
    if "sd_gov" not in res.params.index:
        raise KeyError("Coefficient for 'sd_gov' not found in model results")

    coef = float(res.params["sd_gov"]) if "sd_gov" in res.params else np.nan
    se = float(res.bse["sd_gov"]) if "sd_gov" in res.bse else np.nan
    tval = float(res.tvalues["sd_gov"]) if "sd_gov" in res.tvalues else np.nan
    pval = float(res.pvalues["sd_gov"]) if "sd_gov" in res.pvalues else np.nan
    ci_low, ci_high = [float(v) for v in res.conf_int().loc["sd_gov"].tolist()]

    out = {
        "nobs": int(res.nobs),
        "r_squared": float(res.rsquared),
        "coef_sd_gov": coef,
        "se_sd_gov": se,
        "t_sd_gov": tval,
        "p_sd_gov": pval,
        "ci_sd_gov": [ci_low, ci_high],
        "params": res.params.to_dict(),
        "bse": res.bse.to_dict(),
        "pvalues": res.pvalues.to_dict(),
        "cov_type": res.cov_type,
        "drop_wave_7": drop_wave_7,
        "test_mode": test_mode,
    }
    return out


def format_text_result(res: dict) -> str:
    lines = []
    mode = "TEST" if res["test_mode"] else "REAL"
    wave = "DROP_WAVE_7" if res["drop_wave_7"] else "KEEP_WAVE_7"
    lines.append(f"Mode: {mode} | Wave handling: {wave}")
    lines.append(f"N (obs): {res['nobs']}")
    lines.append(f"R-squared: {res['r_squared']:.4f}")
    lines.append("")
    lines.append("Focal coefficient (sd_gov):")
    lines.append(f"  Coefficient: {res['coef_sd_gov']:.6f}")
    lines.append(f"  Robust SE (HC1): {res['se_sd_gov']:.6f}")
    lines.append(f"  t-stat: {res['t_sd_gov']:.6f}")
    lines.append(f"  p-value: {res['p_sd_gov']:.6f}")
    lines.append(f"  95% CI: [{res['ci_sd_gov'][0]:.6f}, {res['ci_sd_gov'][1]:.6f}]")
    lines.append("")
    lines.append("All coefficients (HC1 robust SEs):")
    # Align params and SEs
    params = res.get("params", {})
    bse = res.get("bse", {})
    pvals = res.get("pvalues", {})
    for name in params:
        se = bse.get(name, np.nan)
        pv = pvals.get(name, np.nan)
        lines.append(f"  {name}: coef={params[name]:.6f}, se={se:.6f}, p={pv:.6f}")
    return "\n".join(lines)


def save_outputs(results: dict, out_dir: str, test_mode: bool, drop_wave_7: bool) -> None:
    ensure_output_dir(out_dir)
    mode = "test" if test_mode else "real"
    wave = "drop_7" if drop_wave_7 else "keep_7"
    base = f"{mode}_{wave}"

    # Text summary
    text_path = os.path.join(out_dir, f"{base}.txt")
    with open(text_path, "w") as f:
        f.write(format_text_result(results) + "\n")

    # Full table as CSV
    params = pd.Series(results.get("params", {}), name="coef")
    bse = pd.Series(results.get("bse", {}), name="se")
    pvals = pd.Series(results.get("pvalues", {}), name="p_value")
    table = pd.concat([params, bse, pvals], axis=1)
    csv_path = os.path.join(out_dir, f"{base}_full_table.csv")
    table.to_csv(csv_path, index=True)

    # JSON metadata
    meta_path = os.path.join(out_dir, f"{base}_meta.json")
    meta = {
        "nobs": results["nobs"],
        "r_squared": results["r_squared"],
        "cov_type": results["cov_type"],
        "focal": {
            "name": "sd_gov",
            "coef": results["coef_sd_gov"],
            "se": results["se_sd_gov"],
            "t": results["t_sd_gov"],
            "p": results["p_sd_gov"],
            "ci95": results["ci_sd_gov"],
        },
        "mode": mode,
        "wave": wave,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def main():
    ensure_output_dir(OUTPUT_DIR)
    df = load_data(resolve_data_path())

    # Execute four combinations: test/real x drop_7/keep_7
    combos = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]

    for test_mode, drop_wave_7 in combos:
        try:
            res = run_regression(df, drop_wave_7=drop_wave_7, test_mode=test_mode)
            save_outputs(res, OUTPUT_DIR, test_mode=test_mode, drop_wave_7=drop_wave_7)
            print(format_text_result(res))
            print("-" * 60)
        except Exception as e:
            mode = "test" if test_mode else "real"
            wave = "drop_7" if drop_wave_7 else "keep_7"
            err_path = os.path.join(OUTPUT_DIR, f"{mode}_{wave}_ERROR.txt")
            with open(err_path, "w") as f:
                f.write(str(e) + "\n")
            print(f"Error in {mode} {wave}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
