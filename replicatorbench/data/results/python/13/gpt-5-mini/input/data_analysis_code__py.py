import os
import sys
import traceback

try:
    import pyreadr
    import pandas as pd
    import numpy as np
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
except Exception:
    traceback.print_exc()
    raise

# Support multiple possible mounted data directories inside container
POSSIBLE_DATA_DIRS = [
    '/app/data/original/13/0205_python_gpt5-mini/replication_data',
    '/workspace/replication_data',
    './replication_data'
]
DATA_DIR = None
for d in POSSIBLE_DATA_DIRS:
    if os.path.exists(d):
        DATA_DIR = d
        break
if DATA_DIR is None:
    # fallback to first path
    DATA_DIR = POSSIBLE_DATA_DIRS[0]

OUT_FILE = os.path.join(DATA_DIR, 'model_summaries.txt')

def load_rds(path):
    res = pyreadr.read_r(path)
    # pyreadr returns an OrderedDict-like: pick the first value
    if isinstance(res, dict):
        first = list(res.values())[0]
        return first
    return res

def main():
    logs = []
    try:
        src = os.path.join(DATA_DIR, 'data_clean_5pct.rds')
        logs.append(f'Reading data from: {src}')
        data = load_rds(src)
        logs.append(f'Data loaded. Shape: {getattr(data, "shape", None)}')

        # Basic preprocessing: reset index and drop rows with missing values in variables used
        data = data.reset_index(drop=True)
        # identify variables from formula
        rhs = ('imm_concern + happy_rev + stflife_rev + sclmeet_rev + distrust_soc + '
               'stfeco_rev + hincfel + stfhlth_rev + stfedu_rev + vote_gov + vote_frparty + lrscale + '
               'hhinc_std + agea + educ + female + vote_share_fr + socexp + lt_imm_cntry + wgi + gdppc + unemp')
        vars_needed = ['trstprl_rev'] + [v.strip() for v in rhs.split('+')]
        # ensure cntry included
        if 'cntry' not in data.columns:
            raise ValueError('cntry column not found in dataset')
        vars_needed.append('cntry')
        logs.append('Variables needed: ' + ','.join(vars_needed))
        before_n = data.shape[0]
        data = data.dropna(subset=vars_needed)
        after_n = data.shape[0]
        logs.append(f'Dropped rows with NA in needed vars: before={before_n}, after={after_n}')
        if after_n == 0:
            raise ValueError('No data left after dropping NAs')
        # ensure cntry is treated as categorical/group
        data = data.copy()
        data['cntry'] = data['cntry'].astype('category')
        formula = (
            'trstprl_rev ~ imm_concern + happy_rev + stflife_rev + sclmeet_rev + distrust_soc + '
            'stfeco_rev + hincfel + stfhlth_rev + stfedu_rev + vote_gov + vote_frparty + lrscale + '
            'hhinc_std + agea + educ + female + vote_share_fr + socexp + lt_imm_cntry + wgi + gdppc + unemp'
        )

        logs.append('Fitting mixed linear model (random intercept for cntry) with formula: ' + formula)
        md = smf.mixedlm(formula, data, groups=data['cntry'])
        mdf = md.fit(reml=False)
        logs.append('Model fitted successfully')

        summary_str = mdf.summary().as_text()
        coef = mdf.params.get('imm_concern', None)
        se = mdf.bse.get('imm_concern', None)
        try:
            pval = mdf.pvalues.get('imm_concern', None)
        except Exception:
            pval = None
        try:
            ci = mdf.conf_int().loc['imm_concern'].tolist()
        except Exception:
            ci = [None, None]

        with open(OUT_FILE, 'w') as f:
            f.write('=== LOGS ===\n')
            for L in logs:
                f.write(L + '\n')
            f.write('\n=== MODEL SUMMARY ===\n')
            f.write(summary_str + '\n')
            f.write('\n=== IMM_CONCERN STAT ===\n')
            f.write(f'coef: {coef}\n')
            f.write(f'se: {se}\n')
            f.write(f'pval: {pval}\n')
            f.write(f'95% CI: {ci}\n')

        print('Model run complete. Summary written to', OUT_FILE)
    except Exception as e:
        tb = traceback.format_exc()
        print('ERROR during analysis:', e)
        print(tb)
        # Write error log
        with open(OUT_FILE, 'w') as f:
            f.write('ERROR during analysis:\n')
            f.write(str(e) + '\n')
            f.write(tb)
        sys.exit(1)

if __name__ == '__main__':
    main()
