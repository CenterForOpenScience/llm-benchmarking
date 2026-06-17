#!/usr/bin/env python3
import os
import sys
import json
import traceback
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

OUT_DIR = '/app/data'

def find_csv(root='/app/data'):
    matches = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith('.csv') and 'sandra_replicate' in fn.lower():
                matches.append(os.path.join(dirpath, fn))
    return matches

def main():
    try:
        candidates = find_csv()
        if not candidates:
            # fallback common path
            fallback = os.path.join('/app/data', 'replication_data', 'sandra_replicate.csv')
            if os.path.exists(fallback):
                candidates = [fallback]

        if not candidates:
            raise FileNotFoundError('Could not find sandra_replicate.csv under /app/data')

        data_path = candidates[0]
        print('Loading data from', data_path)
        df = pd.read_csv(data_path)

        # Ensure required columns exist
        required = ['logRT', 'NFC', 'trial', 'rewardlevel', 'blocknumber', 'SubjectID', 'accuracy']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError('Missing required columns: ' + ','.join(missing))

        # Filter correct trials
        df = df[df['accuracy'] == 1]

        # Drop rows with NA in key variables
        df = df.dropna(subset=['logRT', 'NFC', 'trial', 'rewardlevel', 'blocknumber', 'SubjectID'])

        # Scale logRT and NFC
        scaler = StandardScaler()
        df['scale_logRT'] = scaler.fit_transform(df[['logRT']])
        df['scale_NFC'] = scaler.fit_transform(df[['NFC']])

        # Ensure SubjectID is treated as group
        df['SubjectID'] = df['SubjectID'].astype(str)

        formula = 'scale_logRT ~ scale_NFC * trial * rewardlevel + blocknumber'
        print('Fitting MixedLM with formula:', formula)
        md = smf.mixedlm(formula, df, groups=df['SubjectID'])
        try:
            mdf = md.fit(reml=False, method='lbfgs')
        except Exception:
            # fallback to default
            mdf = md.fit(reml=False)

        summary_str = mdf.summary().as_text()
        print('Model fit complete.')

        # Attempt to find the three-way interaction parameter name
        param_name = None
        for name in mdf.params.index:
            lowered = name.lower()
            if 'scal_nfc' in lowered.replace('-', '_') and 'trial' in lowered and 'reward' in lowered:
                param_name = name
                break
            # Patsy might use different naming like 'scale_NFC:trial:rewardlevel'
            if ':' in name and 'scale' in name.lower() and 'nfc' in name.lower() and 'trial' in name.lower() and 'reward' in name.lower():
                param_name = name
                break

        extracted = {}
        if param_name is None:
            # Try to find any param that contains all three substrings
            for name in mdf.params.index:
                lowercase = name.lower()
                if all(k in lowercase for k in ['nfc', 'trial', 'reward']):
                    param_name = name
                    break

        if param_name:
            coef = float(mdf.params[param_name])
            se = float(mdf.bse[param_name]) if hasattr(mdf, 'bse') and param_name in mdf.bse.index else None
            pval = float(mdf.pvalues[param_name]) if hasattr(mdf, 'pvalues') and param_name in mdf.pvalues.index else None
            extracted = {'param_name': param_name, 'coef': coef, 'se': se, 'pvalue': pval}
        else:
            # No exact match found; save all params for inspection
            extracted = {'note': 'Could not locate exact three-way interaction param name. Saving all parameters under params_dump.',
                         'params_dump': {k: float(v) for k,v in mdf.params.items()}}

        # Write outputs
        os.makedirs(OUT_DIR, exist_ok=True)
        txt_out = os.path.join(OUT_DIR, 'replication_results.txt')
        with open(txt_out, 'w') as f:
            f.write('MixedLM Summary:\n')
            f.write(summary_str)
            f.write('\n\nExtracted:\n')
            f.write(json.dumps(extracted, indent=2))

        json_out = os.path.join(OUT_DIR, 'replication_extracted.json')
        with open(json_out, 'w') as f:
            json.dump(extracted, f, indent=2)

        print('Wrote results to', txt_out, 'and', json_out)
        sys.exit(0)

    except Exception as e:
        tb = traceback.format_exc()
        err_path = os.path.join(OUT_DIR, 'replication_error.txt')
        with open(err_path, 'w') as f:
            f.write('Error during replication script execution:\n')
            f.write(tb)
        print('Error occurred. Wrote traceback to', err_path)
        sys.exit(2)

if __name__ == '__main__':
    main()
