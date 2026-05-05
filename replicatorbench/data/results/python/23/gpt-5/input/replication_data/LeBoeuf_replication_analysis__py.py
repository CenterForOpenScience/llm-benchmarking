import os
import re
import json
import pandas as pd
import numpy as np

# Optional: plotting if needed
try:
    import pingouin as pg
except Exception as e:
    print(f"Warning: failed to import pingouin due to: {e}")
    pg = None

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA_DIR = "/app/data"
INPUT_DATA_FILE = os.path.join(DATA_DIR, "LeBeouf_replication_data.csv")
ITEMS_FILE = os.path.join(DATA_DIR, "ItemsList_Final.csv")
OUT_ANOVA_CSV = os.path.join(DATA_DIR, "mixed_anova_results.csv")
OUT_SUMMARY_JSON = os.path.join(DATA_DIR, "replication_results_summary.json")


def load_data():
    dat = pd.read_csv(INPUT_DATA_FILE)
    items = pd.read_csv(ITEMS_FILE)
    return dat, items


def apply_exclusions(dat: pd.DataFrame) -> pd.DataFrame:
    # Mirror the R filtering steps
    # Status == 0, screenQ not NA, Attention1 == 7, Attention2 == 1, screenQ == 2
    # Some columns might be floats due to CSV import, coerce safely
    def safe_eq(series, val):
        try:
            return series.astype(float) == float(val)
        except Exception:
            return series == val

    mask = (
        safe_eq(dat.get("Status"), 0).fillna(False)
        & dat.get("screenQ").notna()
        & safe_eq(dat.get("Attention1"), 7).fillna(False)
        & safe_eq(dat.get("Attention2"), 1).fillna(False)
        & safe_eq(dat.get("screenQ"), 2).fillna(False)
    )
    dat_f = dat.loc[mask].copy()

    # Ensure Cond is integer-like
    if "Cond" in dat_f.columns:
        dat_f["Cond"] = pd.to_numeric(dat_f["Cond"], errors="coerce").astype("Int64")
    else:
        raise ValueError("Required column 'Cond' not found in dataset.")

    # Create a sequential participant ID (string for pingouin/statsmodels compatibility)
    dat_f = dat_f.reset_index(drop=True)
    dat_f["ID"] = (dat_f.index + 1).astype(str)
    return dat_f


def get_block_columns(columns: list[str], prefix: str, block_index: int) -> list[str]:
    """
    Given a list of column names, return the list of 8 column names corresponding to
    items 1..8 for a particular prefix ("Util" or "Symbol") and block occurrence (1,2,3).
    The dataset contains three repeated blocks of item ratings per prefix.
    """
    # Build mapping from item number to list of matching columns in order of appearance
    pat = re.compile(r"^(\d+)_" + re.escape(prefix))
    hits_by_item = {i: [] for i in range(1, 9)}
    for col in columns:
        m = pat.match(str(col))
        if m:
            i = int(m.group(1))
            if 1 <= i <= 8:
                hits_by_item[i].append(col)
    # For each item, pick the nth occurrence (block_index)
    selected = []
    for i in range(1, 9):
        cols_i = hits_by_item[i]
        if len(cols_i) < block_index:
            raise ValueError(f"Could not find block {block_index} for item {i} and prefix {prefix}. Found {len(cols_i)} matches: {cols_i}")
        selected.append(cols_i[block_index - 1])
    return selected


def build_long_for_condition(dat_sub: pd.DataFrame, cond_value: int, block_index: int) -> pd.DataFrame:
    """Construct long-format dataframe with columns: ID, Cond, ItemCatID (1..8), Benefits, Symbols."""
    all_cols = list(dat_sub.columns)
    util_cols = get_block_columns(all_cols, prefix="Util", block_index=block_index)
    sym_cols = get_block_columns(all_cols, prefix="Symbol", block_index=block_index)

    # Sanity: maintain order by item
    rows = []
    for item_id in range(1, 9):
        b_col = util_cols[item_id - 1]
        s_col = sym_cols[item_id - 1]
        chunk = pd.DataFrame({
            "ID": dat_sub["ID"].values,
            "Cond": cond_value,
            "ItemCatID": item_id,
            "Benefits": pd.to_numeric(dat_sub[b_col], errors="coerce").values,
            "Symbols": pd.to_numeric(dat_sub[s_col], errors="coerce").values,
        })
        rows.append(chunk)
    long_df = pd.concat(rows, axis=0, ignore_index=True)
    return long_df


def prepare_long_data(dat: pd.DataFrame) -> pd.DataFrame:
    # Split by condition and select appropriate blocks
    # Cond: 1,2 => NoBrand (Category); 3 => Brand A; 4 => Brand B
    parts = []
    for cond_value, block_index in [(1, 1), (2, 1), (3, 2), (4, 3)]:
        sub = dat.loc[dat["Cond"] == cond_value].copy()
        if not sub.empty:
            parts.append(build_long_for_condition(sub, cond_value=cond_value, block_index=block_index))
    if not parts:
        raise ValueError("No data remained after filtering by conditions 1-4.")
    long_df = pd.concat(parts, axis=0, ignore_index=True)

    # Add human-readable Level/Condition label
    long_df["Condition"] = np.where(long_df["Cond"].isin([1, 2]), "Category", "Brand")
    return long_df


def merge_items(long_df: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    # Ensure types align for merge on Cond and ItemCatID
    items2 = items.copy()
    items2["Cond"] = pd.to_numeric(items2["Cond"], errors="coerce").astype("Int64")
    items2["ItemCatID"] = pd.to_numeric(items2["ItemCatID"], errors="coerce").astype("Int64")

    merged = pd.merge(long_df, items2, on=["ItemCatID", "Cond"], how="left")
    return merged


def aggregate_to_within_between(merged: pd.DataFrame) -> pd.DataFrame:
    # Compute ScoreDiff = Benefits - Symbols
    merged = merged.copy()
    merged["ScoreDiff"] = merged["Benefits"] - merged["Symbols"]

    # Collapse by participant (ID), product type (Utilitarian/Symbolic), and condition (Category/Brand)
    agg = (
        merged
        .groupby(["ID", "ProductType", "Condition"], dropna=False, as_index=False)
        ["ScoreDiff"].mean()
    )

    # Remove rows with missing product type or condition
    agg = agg.dropna(subset=["ProductType", "Condition"]).copy()

    # Enforce categorical types
    agg["ProductType"] = agg["ProductType"].astype(str)
    agg["Condition"] = agg["Condition"].astype(str)
    return agg


def _mixed_anova_fallback_via_diff_ttest(dat_collapsed: pd.DataFrame) -> pd.DataFrame:
    """Fallback for the mixed ANOVA using subject-wise differences and a two-sample t-test.

    Steps:
    - Pivot to have Utilitarian and Symbolic columns per subject and condition.
    - Compute within-subject difference: diff = Utilitarian - Symbolic.
    - Compare diff between Category and Brand groups via equal-variance two-sample t-test.
    - Return an ANOVA-like table with the interaction row populated.
    """
    pivot = dat_collapsed.pivot_table(index=["ID", "Condition"], columns="ProductType", values="ScoreDiff", aggfunc="mean")
    # Keep only rows where both product types are present
    pivot = pivot.dropna(subset=["Utilitarian", "Symbolic"], how="any").copy()
    pivot["diff"] = pivot["Utilitarian"] - pivot["Symbolic"]

    df_reset = pivot.reset_index()
    diffs_cat = df_reset.loc[df_reset["Condition"] == "Category", "diff"].dropna().values
    diffs_brand = df_reset.loc[df_reset["Condition"] == "Brand", "diff"].dropna().values

    n1, n2 = len(diffs_cat), len(diffs_brand)
    if n1 < 2 or n2 < 2:
        raise ValueError(f"Insufficient data for t-test fallback. n_category={n1}, n_brand={n2}")

    m1, m2 = float(np.mean(diffs_cat)), float(np.mean(diffs_brand))
    v1, v2 = float(np.var(diffs_cat, ddof=1)), float(np.var(diffs_brand, ddof=1))
    df2 = (n1 + n2 - 2)
    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df2
    se = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
    t_stat = (m1 - m2) / se if se > 0 else np.nan
    p_val = 2 * stats.t.sf(np.abs(t_stat), df2)
    Fval = t_stat ** 2
    # Partial eta-squared for 1 df numerator
    np2 = Fval / (Fval + df2) if np.isfinite(Fval) else np.nan

    out = pd.DataFrame([
        {
            "Source": "Condition",
            "DF1": np.nan,
            "DF2": np.nan,
            "F": np.nan,
            "p-unc": np.nan,
            "np2": np.nan,
        },
        {
            "Source": "ProductType",
            "DF1": np.nan,
            "DF2": np.nan,
            "F": np.nan,
            "p-unc": np.nan,
            "np2": np.nan,
        },
        {
            "Source": "Condition * ProductType",
            "DF1": 1.0,
            "DF2": float(df2),
            "F": float(Fval),
            "p-unc": float(p_val),
            "np2": float(np2),
        },
    ])
    return out


def run_mixed_anova(dat_collapsed: pd.DataFrame) -> pd.DataFrame:
    # Try pingouin if available; otherwise use t-test fallback
    try:
        if pg is not None:
            counts = dat_collapsed.groupby(["ID", "ProductType"]).size().unstack(fill_value=0)
            keep_ids = counts.index[(counts.get("Utilitarian", 0) > 0) & (counts.get("Symbolic", 0) > 0)]
            dat_use = dat_collapsed[dat_collapsed["ID"].isin(keep_ids)].copy()
            return pg.mixed_anova(dv="ScoreDiff", within="ProductType", between="Condition", subject="ID", data=dat_use)
        else:
            raise ImportError("pingouin not available")
    except Exception as e:
        print(f"Info: Falling back to subject-difference two-sample t-test due to: {e}")
        return _mixed_anova_fallback_via_diff_ttest(dat_collapsed)


def save_outputs(anova_df: pd.DataFrame, notes: dict):
    # Save full ANOVA table
    anova_df.to_csv(OUT_ANOVA_CSV, index=False)

    # Extract the interaction row if present
    interaction_row = None
    for i, row in anova_df.iterrows():
        src = str(row.get("Source", "")).strip().lower()
        if src in ("condition * producttype", "condition*producttype", "producttype * condition"):
            interaction_row = row
            break
    summary = {
        "notes": notes,
        "anova_table_path": OUT_ANOVA_CSV,
    }
    if interaction_row is not None:
        def to_float(x):
            try:
                return float(x)
            except Exception:
                return None
        summary["interaction"] = {
            "source": str(interaction_row.get("Source")),
            "ddof1": to_float(interaction_row.get("DF1")),
            "ddof2": to_float(interaction_row.get("DF2")),
            "F": to_float(interaction_row.get("F")),
            "p-unc": to_float(interaction_row.get("p-unc")),
            "np2": to_float(interaction_row.get("np2")),
        }
    else:
        summary["interaction"] = None

    with open(OUT_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved ANOVA table to:", OUT_ANOVA_CSV)
    print("Saved summary to:", OUT_SUMMARY_JSON)


def main():
    print("Loading data from:", INPUT_DATA_FILE)
    dat, items = load_data()

    print("Applying exclusions...")
    dat_f = apply_exclusions(dat)
    print(f"Remaining participants after exclusions: {dat_f.shape[0]}")

    print("Reshaping to long format by condition and item...")
    long_df = prepare_long_data(dat_f)

    print("Merging with items metadata...")
    merged = merge_items(long_df, items)

    print("Aggregating to participant x product type x condition...")
    dat_collapsed = aggregate_to_within_between(merged)

    print("Running 2x2 mixed ANOVA (Condition between, ProductType within)...")
    anova_df = run_mixed_anova(dat_collapsed)
    print(anova_df)

    print("Saving outputs...")
    notes = {
        "dv": "ScoreDiff = Benefits - Symbols",
        "between": "Condition (Category vs Brand)",
        "within": "ProductType (Utilitarian vs Symbolic)",
        "subject": "ID",
        "filters": {
            "Status": 0,
            "Attention1": 7,
            "Attention2": 1,
            "screenQ": 2
        }
    }
    save_outputs(anova_df, notes)


if __name__ == "__main__":
    main()
