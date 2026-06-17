import os
import warnings
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# Resolve data input path and output directory robustly for different mount strategies
DATA_CANDIDATES = [
    "/app/data/Full_long.dta",
    "/workspace/data/original/30/input/replication_data/Full_long.dta",
    "data/original/30/input/replication_data/Full_long.dta",
]
OUT_DIR_CANDIDATES = [
    "/app/data",
    "/workspace/data/original/30/input/replication_data",
    "data/original/30/input/replication_data",
]

def resolve_paths():
    data_path = None
    for p in DATA_CANDIDATES:
        if os.path.exists(p):
            data_path = p
            break
    if data_path is None:
        raise FileNotFoundError(f"Full_long.dta not found in any candidate paths: {DATA_CANDIDATES}")

    out_dir = None
    for d in OUT_DIR_CANDIDATES:
        try:
            os.makedirs(d, exist_ok=True)
            out_dir = d
            break
        except Exception:
            continue
    if out_dir is None:
        raise RuntimeError(f"Could not create or access any output directory from candidates: {OUT_DIR_CANDIDATES}")

    return data_path, out_dir


def prepare_data(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    # Keep rounds > 2, as in the Stata code
    df = df.copy()
    df = df[df["round"] > 2].copy()

    # Investment and appropriation measures
    df["nvst"] = df["playerinvestment"]
    df["dwnldpyff"] = df["playercollected_tokens"]  # share for appropriation uses tokens in the .do script

    # Position from playerrole (A-E -> 1-5)
    role_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    df["pstn"] = df["playerrole"].map(role_map).astype("Int64")

    # Adjusted round index (starts at 1 for actual round 3)
    df["rnd"] = (df["round"] - 2).astype("Int64")

    # Group identifier as in Stata: grp = Session*100 + groupid_in_subsession
    # Ensure Session is integer-like
    df["Session_int"] = df["Session"].astype("Int64")
    df["grp"] = (df["Session_int"] * 100 + df["groupid_in_subsession"].astype("Int64")).astype("Int64")

    # Group sums by (grp, rnd)
    grp_rnd = df.groupby(["grp", "rnd"], as_index=False).agg(
        grpnvst=("nvst", "sum"),
        grpdwnldpyff=("dwnldpyff", "sum"),
    )

    # Merge back
    df = df.merge(grp_rnd, on=["grp", "rnd"], how="left")

    # Shares with fallback to equal split (0.2 for 5 players) when denom==0
    df["shrnvst"] = np.where(df["grpnvst"] > 0, df["nvst"] / df["grpnvst"], 0.2)
    df["shrdwnld"] = np.where(df["grpdwnldpyff"] > 0, df["dwnldpyff"] / df["grpdwnldpyff"], 0.2)

    # Lagged share of downloads within individual (panel id = id, time = rnd)
    # Ensure proper sorting
    df = df.sort_values(["id", "rnd"]).copy()
    df["shr"] = df.groupby("id")["shrdwnld"].shift(1)

    # Drop rows with missing key variables (e.g., first available rnd per id after lag)
    model_df = df.dropna(subset=["nvst", "pstn", "shr", "rnd", "grp"]).copy()

    # Cast categorical types for formula handling; ensure pstn baseline is 1
    model_df["pstn"] = model_df["pstn"].astype(int)
    model_df["rnd"] = model_df["rnd"].astype(int)

    # Save processed dataset for transparency
    processed_path = os.path.join(out_dir, "processed_full_long_v2.csv")
    model_df.to_csv(processed_path, index=False)

    return model_df


def fit_mixed_model(df: pd.DataFrame):
    # Mixed effects model analogous to: mixed nvst b(1).pstn##c.shr b(10).rnd || grp:
    # We include round fixed effects as categorical controls (baseline is arbitrary for estimation equivalence)
    formula = "nvst ~ C(pstn)*shr + C(rnd)"
    model = smf.mixedlm(formula, data=df, groups=df["grp"])
    try:
        result = model.fit(reml=True, method="lbfgs", maxiter=200)
    except Exception:
        # Fallback optimizer if needed
        result = model.fit(reml=True, method="nm", maxiter=500)
    return result


def save_results(result, out_dir: str):
    # Save full summary
    txt_path = os.path.join(out_dir, "nvst_mixedlm_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result.summary().as_text())

    # Save coefficients for easy programmatic access
    coefs = result.params.rename("coef").to_frame()
    coefs["se"] = result.bse
    coefs["t"] = result.tvalues
    coefs["p"] = result.pvalues
    coefs_path = os.path.join(out_dir, "nvst_mixedlm_results.csv")
    coefs.to_csv(coefs_path)

    # Extract and print the focal interaction for tail-ender (pstn=5)
    # The parameter is typically named like 'C(pstn)[T.5]:shr'
    focal_key = None
    for k in coefs.index:
        if ":" in k and "C(pstn)[T.5]" in k and k.endswith(":shr"):
            focal_key = k
            break
        # Alternative naming order from patsy
        if ":" in k and k.startswith("shr:") and "C(pstn)[T.5]" in k:
            focal_key = k
            break
    if focal_key is None:
        focal_key = "C(pstn)[T.5]:shr"

    focal_out = os.path.join(out_dir, "focal_interaction_tailender.txt")
    with open(focal_out, "w", encoding="utf-8") as f:
        if focal_key in coefs.index:
            row = coefs.loc[focal_key]
            f.write(f"parameter,{focal_key}\ncoef,{row['coef']}\nse,{row['se']}\nt,{row['t']}\np,{row['p']}\n")
        else:
            f.write("Focal interaction coefficient not found in the model parameters.\n")


def main():
    data_path, out_dir = resolve_paths()
    raw = pd.read_stata(data_path)
    model_df = prepare_data(raw, out_dir)
    result = fit_mixed_model(model_df)
    save_results(result, out_dir)
    print(f"Analysis complete. Results saved to {out_dir}.")


if __name__ == "__main__":
    main()
