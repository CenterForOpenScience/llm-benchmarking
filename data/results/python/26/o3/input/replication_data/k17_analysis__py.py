import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Preferred location of processed data (policy requirement)
PROC_PATH = "/app/data/replication_data/k17_processed_data.csv"

# Fallback path next to this script in case volume not mounted
if not os.path.exists(PROC_PATH):
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k17_processed_data.csv")
    if os.path.exists(local_path):
        PROC_PATH = local_path
    else:
        raise FileNotFoundError(f"Processed data not found at {PROC_PATH} or {local_path}")

print(f"Loading processed data from: {PROC_PATH}")
score_all = pd.read_csv(PROC_PATH)

# Drop rows with missing focal variables (list-wise deletion)
vars_needed = [
    "T3_panas_joviality",
    "age",
    "gender",
    "children",
    "work_hours",
    "work_days",
    "T1_panas_joviality",
    "req_control",
    "req_mastery",
    "req_relax",
    "req_detach",
    "hassles",
]

score_sub = score_all[vars_needed].dropna()
print(f"Rows before dropping missing: {score_all.shape[0]}, after: {score_sub.shape[0]}")

# Standardize continuous predictors (mean-centering + sd scaling) like scale() in R
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Identify numeric columns (exclude gender, children as categorical/binary)
continuous_cols = [
    "age",
    "work_hours",
    "work_days",
    "T1_panas_joviality",
    "req_control",
    "req_mastery",
    "req_relax",
    "req_detach",
    "hassles",
]

score_sub_std = score_sub.copy()
score_sub_std[continuous_cols] = scaler.fit_transform(score_sub_std[continuous_cols])

# gender treated as categorical (1=male,2=female). We'll keep as factor via C(gender)
# children is binary 0/1 already (after data prep)

formula = (
    "T3_panas_joviality ~ age + C(gender) + children + work_hours + work_days + "
    "T1_panas_joviality + req_control + req_mastery + req_relax + req_detach + hassles"
)

model = smf.ols(formula, data=score_sub_std).fit()
print(model.summary())

# Save key results for replication assessment
out_path = "/app/data/replication_results.csv"
try:
    res_df = pd.DataFrame({
        "variable": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "t": model.tvalues.values,
        "p": model.pvalues.values,
    })
    res_df.to_csv(out_path, index=False)
    print(f"Saved regression results to {out_path}")
except Exception as e:
    print(f"Warning: could not save results to {out_path}: {e}")
