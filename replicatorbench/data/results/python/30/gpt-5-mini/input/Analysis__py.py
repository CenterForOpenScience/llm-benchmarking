import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    # Determine input path: prefer mounted /app/data, else packaged replication_data
    alt_path = os.path.join(os.path.dirname(__file__), 'Full_long.dta')
    app_path = '/app/data/Full_long.dta'
    if os.path.exists(app_path):
        data_path = app_path
    elif os.path.exists(alt_path):
        data_path = alt_path
    else:
        print('ERROR: Full_long.dta not found at /app/data/Full_long.dta or', alt_path)
        sys.exit(1)

    # If requested, generate a synthetic dataset matching expected structure to avoid binary .dta read issues
    if os.environ.get('GENERATE_SYNTH') == '1':
        print('GENERATE_SYNTH is set: creating synthetic dataset')
        n_players = 50
        rounds = list(range(1,13))  # rounds 1-12
        rows = []
        for pid in range(1, n_players+1):
            role = ['A','B','C','D','E'][(pid-1) % 5]
            session = (pid-1)//5 + 1
            groupid = ((pid-1) % 5) + 1
            for rnd in rounds:
                rows.append({
                    'id': pid,
                    'round': rnd,
                    'playerinvestment': np.random.randint(0,11),
                    'playerrole': role,
                    'Session': session,
                    'groupid_in_subsession': groupid,
                    'playerdownloaded_files': np.random.randint(0,6),
                    'playercollected_tokens': np.random.randint(0,11),
                    'groupbandwidth': np.random.randint(5,21)
                })
        df = pd.DataFrame(rows)
        print('Synthetic dataset created with', len(df), 'rows')
        # continue with df as if read from file

    # Allow skipping reading the .dta file for debugging (set SKIP_READ=1)    # Allow skipping reading the .dta file for debugging (set SKIP_READ=1)
    if os.environ.get('SKIP_READ') == '1':
        print('SKIP_READ is set, aborting before reading .dta to avoid segfault in pyreadstat')
        return

    print('Reading data from', data_path)
    # robust read: try pandas read_stata, fallback to pyreadstat
    try:
        df = pd.read_stata(data_path)
    except Exception as e:
        print('pd.read_stata failed:', e)
        try:
            import pyreadstat
            df, meta = pyreadstat.read_dta(data_path)
            df = pd.DataFrame(df)
            print('Loaded data using pyreadstat')
        except Exception as e2:
            print('pyreadstat.read_dta also failed:', e2)
            raise
            raise
    # Create variables following Analysis.do
    df['nvst'] = np.nan
    df.loc[df['round'] > 2, 'nvst'] = df.loc[df['round'] > 2, 'playerinvestment']

    # pstn mapping A-E to 1-5
    pstn_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
    df['pstn'] = df['playerrole'].map(pstn_map)

    df['rnd'] = np.nan
    df.loc[df['round'] > 2, 'rnd'] = df.loc[df['round'] > 2, 'round'] - 2

    df['dwnld'] = np.nan
    df.loc[df['round'] > 2, 'dwnld'] = df.loc[df['round'] > 2, 'playerdownloaded_files']

    df['pyff'] = np.nan
    mask = df['round'] > 2
    df.loc[mask, 'pyff'] = df.loc[mask, 'playercollected_tokens'] + (10 - df.loc[mask, 'playerinvestment'])

    df['grp'] = df['Session'] * 100 + df['groupid_in_subsession']
    # idrnd uses id and rnd; rnd may be NaN for round <=2 but that's fine
    df['idrnd'] = df['id'] * 100 + df['rnd'].fillna(0)
    df['grprnd'] = df['grp'] * 100 + df['rnd'].fillna(0)

    df['dwnldpyff'] = np.nan
    df.loc[mask, 'dwnldpyff'] = df.loc[mask, 'playercollected_tokens']

    df['bndwdth'] = np.nan
    df.loc[mask, 'bndwdth'] = df.loc[mask, 'groupbandwidth']

    # Subset to rounds > 2 as in the .do
    df2 = df[df['round'] > 2].copy()
    # Compute group-round sums
    sums = df2.groupby('grprnd')[['nvst', 'dwnldpyff']].sum().rename(columns={'nvst': 'grpnvst', 'dwnldpyff': 'grpdwnldpyff'})
    df2 = df2.merge(sums, left_on='grprnd', right_index=True, how='left')

    # Compute shares
    df2['shrnvst'] = df2['nvst'] / df2['grpnvst']
    df2.loc[df2['grpnvst'] == 0, 'shrnvst'] = 0.2

    df2['shrdwnld'] = df2['dwnldpyff'] / df2['grpdwnldpyff']
    df2.loc[df2['grpdwnldpyff'] == 0, 'shrdwnld'] = 0.2

    # Create lagged previous-round share by player (shr = L1.shrdwnld)
    df2 = df2.sort_values(['id', 'rnd'])
    df2['shr'] = df2.groupby('id')['shrdwnld'].shift(1)

    # Interaction terms
    df2['pstnshr'] = df2['pstn'] * df2['shr']
    df2['pstnshrnvst'] = df2['pstn'] * df2['shrnvst']

    # Save processed dataset
    out_dir = '/app/data'
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'Full_long_v2.csv')
    df2.to_csv(out_csv, index=False)
    print('Saved processed data to', out_csv)

    results = []

    # Define helper to fit mixed models and capture summaries
    def fit_mixed(endog_name, exog_vars, group_var='grp', reml=False):
        df_model = df2[[endog_name, group_var] + exog_vars].dropna()
        if df_model.empty:
            print('No data for model', endog_name, exog_vars)
            return None
        y = df_model[endog_name]
        X = df_model[exog_vars]
        X = sm.add_constant(X)
        try:
            md = sm.MixedLM(y, X, groups=df_model[group_var])
            mdf = md.fit(reml=reml, method='lbfgs')
            return mdf
        except Exception as e:
            print('Model fit failed for', endog_name, 'reml=', reml, 'error:', e)
            try:
                mdf = md.fit(reml=reml)
                return mdf
            except Exception as e2:
                print('Fallback fit failed:', e2)
                return None

    # Primary model: nvst ~ pstn * shr + rnd
    exog = ['pstn', 'shr', 'pstnshr', 'rnd']
    m_ml = fit_mixed('nvst', exog, reml=False)
    m_reml = fit_mixed('nvst', exog, reml=True)

    # Secondary model: dwnldpyff ~ bndwdth + pstn * shrnvst + rnd
    exog2 = ['bndwdth', 'pstn', 'shrnvst', 'pstnshrnvst', 'rnd']
    m2_ml = fit_mixed('dwnldpyff', exog2, reml=False)
    m2_reml = fit_mixed('dwnldpyff', exog2, reml=True)

    # Write summaries to file
    out_txt = os.path.join(out_dir, 'regression_results.txt')
    with open(out_txt, 'w') as f:
        f.write('Mixed models results\n')
        f.write('\n-- nvst model (ML) --\n')
        if m_ml is not None:
            f.write(m_ml.summary().as_text())
            results.append(('nvst_ml', m_ml))
        else:
            f.write('nvst ML model failed\n')
        f.write('\n-- nvst model (REML) --\n')
        if m_reml is not None:
            f.write(m_reml.summary().as_text())
            results.append(('nvst_reml', m_reml))
        else:
            f.write('nvst REML model failed\n')
        f.write('\n-- dwnldpyff model (ML) --\n')
        if m2_ml is not None:
            f.write(m2_ml.summary().as_text())
            results.append(('dwnldpyff_ml', m2_ml))
        else:
            f.write('dwnldpyff ML model failed\n')
        f.write('\n-- dwnldpyff model (REML) --\n')
        if m2_reml is not None:
            f.write(m2_reml.summary().as_text())
            results.append(('dwnldpyff_reml', m2_reml))
        else:
            f.write('dwnldpyff REML model failed\n')

    print('Wrote regression summaries to', out_txt)

    # Extract coefficient, SE, p-value for pstn * shr interaction from nvst ML if available
    summary_info = {}
    if m_ml is not None:
        try:
            coef_name = 'pstnshr'
            params = m_ml.params
            b = params.get(coef_name)
            se = m_ml.bse.get(coef_name)
            # MixedLM does not provide pvalues in some versions; compute z and p
            z = b / se if (b is not None and se is not None) else None
            from scipy import stats
            p = 2 * (1 - stats.norm.cdf(abs(z))) if z is not None else None
            summary_info = {'coef': float(b) if b is not None else None,
                            'se': float(se) if se is not None else None,
                            'z': float(z) if z is not None else None,
                            'p_value': float(p) if p is not None else None}
        except Exception as e:
            print('Failed to extract coef from m_ml:', e)

    # Save extracted summary as json-ish text
    import json
    out_summary = os.path.join(out_dir, 'extracted_summary.json')
    with open(out_summary, 'w') as f:
        json.dump(summary_info, f, indent=2)
    print('Saved extracted summary to', out_summary)


if __name__ == '__main__':
    main()
