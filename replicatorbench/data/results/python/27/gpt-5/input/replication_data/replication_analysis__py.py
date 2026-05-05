import os
import sys
import re
import pandas as pd
from scipy.stats import wilcoxon

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")

ROUND1 = os.path.join(DATA_DIR, "round1_raw.csv")
ROUND2 = os.path.join(DATA_DIR, "round2_raw.csv")

# Helper to load and reshape a round dataset
def load_and_prepare(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        sys.exit(1)
    df = pd.read_csv(path)

    # Select necessary columns
    keep_cols = [
        "session.code",
        "ipo_task.1.group.id_in_subsession",
        "participant.id_in_session",
        "ipo_task.20.player.total_missing_responses",
    ]

    # group.market_price columns across rounds 1..20
    gmp_cols = [c for c in df.columns if re.match(r"ipo_task\\.\\d+\\.group\\.market_price", c)]
    use_cols = [c for c in keep_cols if c in df.columns] + gmp_cols

    if len(gmp_cols) == 0:
        print("Error: No 'ipo_task.<round>.group.market_price' columns found.")
        sys.exit(1)

    # Filter to completed groups (group id != 1)
    if "ipo_task.1.group.id_in_subsession" not in df.columns:
        print("Error: 'ipo_task.1.group.id_in_subsession' not found in dataset.")
        sys.exit(1)

    df = df[use_cols].copy()
    df = df[df["ipo_task.1.group.id_in_subsession"] != 1]

    # Rename rounds_missed
    if "ipo_task.20.player.total_missing_responses" in df.columns:
        df = df.rename(columns={
            "ipo_task.1.group.id_in_subsession": "group_in_session",
            "ipo_task.20.player.total_missing_responses": "rounds_missed",
        })
    else:
        df = df.rename(columns={
            "ipo_task.1.group.id_in_subsession": "group_in_session",
        })
        df["rounds_missed"] = pd.NA

    # Reshape wide to long over group.market_price columns
    long = df.melt(
        id_vars=["session.code", "group_in_session", "participant.id_in_session", "rounds_missed"],
        value_vars=gmp_cols,
        var_name="var",
        value_name="market_price"
    )

    # Extract round number from var such as 'ipo_task.7.group.market_price'
    long["round"] = long["var"].str.extract(r"ipo_task\\.(\\d+)\\.group\\.market_price").astype(int)
    long = long.drop(columns=["var"]) 

    # Compute dropout/bankrupt flags per session-group
    grp = long.groupby(["session.code", "group_in_session"], as_index=False).agg(
        rounds_missed_max=("rounds_missed", "max"),
        rounds_missed_min=("rounds_missed", "min"),
    )
    grp["bankrupt"] = (grp["rounds_missed_min"] == -99).astype(int)
    grp["dropout"] = (grp["rounds_missed_max"] >= 6).astype(int)

    long = long.merge(grp, on=["session.code", "group_in_session"], how="left")

    # Average market price by session-group
    collapsed = (
        long
        .groupby(["session.code", "group_in_session", "bankrupt", "dropout"], as_index=False)
        .agg(market_price=("market_price", "mean"))
    )

    # Filter valid groups (not bankrupt or dropout) and non-missing prices
    collapsed = collapsed[(collapsed["bankrupt"] != 1) & (collapsed["dropout"] != 1)]
    collapsed = collapsed[collapsed["market_price"].notna()]

    return collapsed

# Load both rounds
r1 = load_and_prepare(ROUND1)
r2 = load_and_prepare(ROUND2)

# Combine
merged = pd.concat([r1, r2], ignore_index=True)

# Perform Wilcoxon signed-rank test vs mu=1.94
x = merged["market_price"].dropna().values
if x.size == 0:
    print("No observations available after filtering.")
    sys.exit(1)

stat, pval = wilcoxon(x - 1.94, alternative="two-sided", zero_method="wilcox", correction=False)

print("Wilcoxon signed-rank test on group-average market_price vs 1.94 (two-sided)")
print(f"n = {x.size}")
print(f"Statistic = {stat}")
print(f"p-value = {pval}")

median_diff = pd.Series(x - 1.94).median()
print(f"Median difference (market_price - 1.94) = {median_diff:.4f}")

# Save collapsed datasets for reference
OUT1 = os.path.join(DATA_DIR, "round1_analysis.csv")
OUTFULL = os.path.join(DATA_DIR, "fullsample_analysis.csv")

# Round 1 only and combined should be saved; rebuild round1 collapsed separately for output
r1.to_csv(OUT1, index=False)

# For full sample, we can also compute per-group average across sessions (mimicking Stata collapse by group)
# Here we just save the merged group-level data
merged.to_csv(OUTFULL, index=False)

print(f"Saved round1_analysis.csv and fullsample_analysis.csv to {DATA_DIR}")
