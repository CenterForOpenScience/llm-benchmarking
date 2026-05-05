#!/usr/bin/env python3
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# This script reproduces the analyses in the provided Stata .do file:
# - mixed cont losses || sessioncode: || id:
# - mixed loss_aversion losses || sessioncode: || id:
# - mixed fear losses || sessioncode: || id:
# It reads the CSV from /app/data and writes results to /app/data/mccarter_mixed_results.txt# It reads the CSV from /app/data and writes results to /app/data/mccarter_mixed_results.txt
# Resolve data path for container or local execution
import os

possible_paths = [
    '/app/data/McCarter_OrgBehavior_2010_pILK data CLEAN.csv',
    os.path.join(os.path.dirname(__file__), 'McCarter_OrgBehavior_2010_pILK data CLEAN.csv'),
    os.path.join(os.path.dirname(__file__), 'replication_data', 'McCarter_OrgBehavior_2010_pILK data CLEAN.csv'),
    os.path.join(os.getcwd(), 'replication_data', 'McCarter_OrgBehavior_2010_pILK data CLEAN.csv')
]
DATA_PATH = None
for p in possible_paths:
    if os.path.exists(p):
        DATA_PATH = p
        break
if DATA_PATH is None:
    # fallback to original container path
    DATA_PATH = '/app/data/McCarter_OrgBehavior_2010_pILK data CLEAN.csv'
OUTPUT_PATH = '/app/data/mccarter_mixed_results.txt'

def safe_read(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read data at {path}: {e}")


def fit_mixed(df, dep_var):
    # Fit mixed effects model with random intercept for sessioncode and variance component for id
    # This mirrors: mixed dep_var losses || sessioncode: || id:
    df = df.copy()
    df['sessioncode'] = df['sessioncode'].astype(str)
    # id used as a variance component; convert to string to avoid numeric grouping issues
    df['id'] = df['id'].astype(str)
    formula = f"{dep_var} ~ losses"
    try:
        md = smf.mixedlm(formula, df, groups=df['sessioncode'], vc_formula={"id": "0 + C(id)"})
        mfit = md.fit(reml=False)
        return mfit
    except Exception as e:
        raise


def main():
    df = safe_read(DATA_PATH)
    outputs = []
    outputs.append(f"Data loaded from: {DATA_PATH} -- rows={len(df)}\n")

    # Primary model: cont ~ losses
    try:
        m_cont = fit_mixed(df.dropna(subset=['cont','losses']), 'cont')
        outputs.append("\nMixed model: cont ~ losses\n")
        outputs.append(m_cont.summary().as_text())
    except Exception as e:
        outputs.append(f"Failed to fit cont model: {e}\n")

    # Exploratory: loss_aversion ~ losses
    for dep in ['loss_aversion', 'fear']:
        try:
            if dep not in df.columns:
                outputs.append(f"Column {dep} not in data.\n")
                continue
            dfd = df.dropna(subset=[dep,'losses'])
            m = fit_mixed(dfd, dep)
            outputs.append(f"\nMixed model: {dep} ~ losses\n")
            outputs.append(m.summary().as_text())
        except Exception as e:
            outputs.append(f"Failed to fit model for {dep}: {e}\n")

    # Write outputs
    with open(OUTPUT_PATH, 'w') as f:
        f.write('\n'.join(outputs))

    print(f"Wrote results to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
