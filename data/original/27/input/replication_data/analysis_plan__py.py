import os
import sys
import pandas as pd
from scipy.stats import wilcoxon

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")

CSV_PATH = os.path.join(DATA_DIR, "zhang_avgprice.csv")

if not os.path.exists(CSV_PATH):
    print(f"Error: {CSV_PATH} not found. Ensure zhang_avgprice.csv is available in /app/data.")
    sys.exit(1)

# Load dataset
avg_df = pd.read_csv(CSV_PATH)
if "avgprice" not in avg_df.columns:
    print("Error: 'avgprice' column not found in zhang_avgprice.csv")
    sys.exit(1)

# Drop missing values if any
x = avg_df["avgprice"].dropna().values

# Wilcoxon signed-rank test against mu = 1.94 (two-sided)
# scipy's wilcoxon tests differences against zero; subtract 1.94 first
if x.size == 0:
    print("No observations available after dropping missing values.")
    sys.exit(1)

stat, pval = wilcoxon(x - 1.94, alternative="two-sided", zero_method="wilcox", correction=False)

print("Wilcoxon signed-rank test on avgprice vs 1.94 (two-sided)")
print(f"n = {x.size}")
print(f"Statistic = {stat}")
print(f"p-value = {pval}")

# Also report direction (median difference)
median_diff = pd.Series(x - 1.94).median()
print(f"Median difference (avgprice - 1.94) = {median_diff:.4f}")
