import pandas as pd
import scipy.stats as stats
import os, sys, glob, json

# Locate the FinalData.csv somewhere under /app/data
SEARCH_ROOT = os.environ.get("REPLICATION_DATA_ROOT", "/app/data")

csv_candidates = glob.glob(os.path.join(SEARCH_ROOT, "**", "FinalData.csv"), recursive=True)
if not csv_candidates:
    print("ERROR: Could not find FinalData.csv under /app/data. Checked recursively.")
    sys.exit(1)

csv_path = csv_candidates[0]
print(f"Using dataset at {csv_path}")

# Load dataset
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Failed to load CSV: {e}")
    sys.exit(1)

required_cols = ["treat", "effort_t1"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"Missing required columns: {missing}")
    sys.exit(1)

# Ensure treat coding is numeric
if not pd.api.types.is_numeric_dtype(df["treat"]):
    df["treat"] = pd.to_numeric(df["treat"], errors="coerce")

# Split groups
os_group = df[df["treat"] == 1]
ts_group = df[df["treat"] == 2]

if os_group.empty or ts_group.empty:
    print("One of the treatment groups is empty. Cannot run test.")
    sys.exit(1)

# Summary stats
summary = {
    "OS": {
        "n": len(os_group),
        "mean_effort_t1": os_group["effort_t1"].mean(),
        "median_effort_t1": os_group["effort_t1"].median()
    },
    "TS": {
        "n": len(ts_group),
        "mean_effort_t1": ts_group["effort_t1"].mean(),
        "median_effort_t1": ts_group["effort_t1"].median()
    }
}

# Mann-Whitney U test
u_stat, p_value = stats.mannwhitneyu(ts_group["effort_t1"], os_group["effort_t1"], alternative="two-sided")
summary["mannwhitney_u"] = {
    "u_statistic": u_stat,
    "p_value": p_value
}

# Print to console
print("Summary statistics and Mann-Whitney U-test results:")
print(json.dumps(summary, indent=4))

# Save results
out_path = os.path.join(SEARCH_ROOT, "replication_results.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=4)
print(f"Results saved to {out_path}")
