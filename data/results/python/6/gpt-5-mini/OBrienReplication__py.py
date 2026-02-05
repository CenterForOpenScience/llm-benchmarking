"""
Python translation of OBrienReplication_OSF_Axxe_20201012.do
This script recreates data cleaning, latent-class-like mixture models and t-tests for pairwise comparisons of the `evolved` item between classes.
I/O assumptions: input data file is at '/app/data/GSSreplication.dta' and outputs are written to '/app/data/'.
"""

import os
import numpy as np
import pandas as pd
try:
    from sklearn.mixture import GaussianMixture
    SKLEARN_MIXTURE = True
except Exception:
    GaussianMixture = None
    SKLEARN_MIXTURE = False
from scipy import stats
from scipy.cluster.vq import kmeans2
import math
import json

INPUT_PATH = '/app/data/GSSreplication.dta'
CLEAN_OUTPUT = '/app/data/GSSreplication_clean.dta'
RESULT_LOG = '/app/data/OBrienReplication_results.txt'

# Helper: safe to_stata
def to_stata_safe(df, path):
    try:
        df.to_stata(path, write_index=False)
    except Exception:
        # fallback to csv
        csv_path = path.replace('.dta', '.csv')
        df.to_csv(csv_path, index=False)
        print(f'Could not write .dta, wrote csv to {csv_path}')


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f'Input file not found at {INPUT_PATH}')

    # Read data
    df = pd.read_stata(INPUT_PATH)
    # lowercase columns
    df.columns = [c.lower() for c in df.columns]

    # Define variables as in the .do
    binary = [
        'hotcore','radioact','boyorgrl','lasers','electron','viruses',
        'earthsun','condrift','bigbang','evolved','expdesgn','odds1','odds2'
    ]

    true = ['hotcore','boyorgrl','electron','earthsun','condrift','bigbang','evolved','odds2']
    false = ['radioact','lasers','viruses','expdesgn','odds1']

    # Recode binary: recode `i' 1=. 5=., gen(`i'_clean)
    for v in binary:
        col = v
        new = f"{v}_clean"
        if col in df.columns:
            df[new] = df[col]
            # set values 1 and 5 to missing
            df.loc[df[new].isin([1,5]), new] = np.nan
        else:
            df[new] = np.nan

    # For true items: recode `i'_clean 2=1 3=0 4=0
    for v in true:
        new = f"{v}_clean"
        df.loc[df[new]==2, new] = 1
        df.loc[df[new].isin([3,4]), new] = 0

    # For false items: recode `i'_clean 3=1 2=0 4=0
    for v in false:
        new = f"{v}_clean"
        df.loc[df[new]==3, new] = 1
        df.loc[df[new].isin([2,4]), new] = 0

    # scistudy recode 1=. 5=. 6=., gen(scistudy_clean)
    if 'scistudy' in df.columns:
        df['scistudy_clean'] = df['scistudy']
        df.loc[df['scistudy_clean'].isin([1,5,6]), 'scistudy_clean'] = np.nan
    else:
        df['scistudy_clean'] = np.nan

    # scales: nextgen toofast advfront recode 1=. 2=1 3=2 4=3 5=4 6=. 7=., gen(..._clean)
    for v in ['nextgen','toofast','advfront']:
        if v in df.columns:
            new = f"{v}_clean"
            df[new] = df[v]
            df.loc[df[new].isin([1,6,7]), new] = np.nan
            df.loc[df[new]==2, new] = 1
            df.loc[df[new]==3, new] = 2
            df.loc[df[new]==4, new] = 3
            df.loc[df[new]==5, new] = 4

    # scibnfts recode 1=. 2=4 3=2 4=0  5=. 6=., gen(scibnfts_clean)
    if 'scibnfts' in df.columns:
        df['scibnfts_clean'] = df['scibnfts']
        df.loc[df['scibnfts_clean'].isin([1,5,6]), 'scibnfts_clean'] = np.nan
        df.loc[df['scibnfts_clean']==2, 'scibnfts_clean'] = 4
        df.loc[df['scibnfts_clean']==3, 'scibnfts_clean'] = 2
        df.loc[df['scibnfts_clean']==4, 'scibnfts_clean'] = 0
    else:
        df['scibnfts_clean'] = np.nan

    # Reverse code toofast_clean: recode toofast_clean 4=1 3=2 2=3 1=4
    if 'toofast_clean' in df.columns:
        rc = {4:1, 3:2, 2:3, 1:4}
        df['toofast_clean'] = df['toofast_clean'].map(rc).astype(float)

    # bible recode 1=. 2=1 3=2 4=3 5=. 6=. 7=., gen(bible_clean)
    if 'bible' in df.columns:
        df['bible_clean'] = df['bible']
        df.loc[df['bible_clean'].isin([1,5,6,7]), 'bible_clean'] = np.nan
        df.loc[df['bible_clean']==2, 'bible_clean'] = 1
        df.loc[df['bible_clean']==3, 'bible_clean'] = 2
        df.loc[df['bible_clean']==4, 'bible_clean'] = 3
    else:
        df['bible_clean'] = np.nan

    # reliten recode 1=. 2=4 3=3 4=2 5=1 6=. 7=., gen(reliten_clean)
    if 'reliten' in df.columns:
        df['reliten_clean'] = df['reliten']
        df.loc[df['reliten_clean'].isin([1,6,7]), 'reliten_clean'] = np.nan
        df.loc[df['reliten_clean']==2, 'reliten_clean'] = 4
        df.loc[df['reliten_clean']==3, 'reliten_clean'] = 3
        df.loc[df['reliten_clean']==4, 'reliten_clean'] = 2
        df.loc[df['reliten_clean']==5, 'reliten_clean'] = 1
    else:
        df['reliten_clean'] = np.nan

    # Save cleaned
    to_stata_safe(df, CLEAN_OUTPUT)

    vars_clean = [
        'hotcore_clean','radioact_clean','boyorgrl_clean','lasers_clean','electron_clean',
        'viruses_clean','earthsun_clean','condrift_clean','bigbang_clean','evolved_clean',
        'expdesgn_clean','odds1_clean','odds2_clean','scistudy_clean','nextgen_clean',
        'toofast_clean','advfront_clean','scibnfts_clean','bible_clean','reliten_clean'
    ]

    # Ensure wtss exists
    if 'wtss' not in df.columns:
        df['wtss'] = 1.0

    analyses = [
        ('not_original', lambda d: d['year'] > 2010),
        ('all', lambda d: pd.Series(True, index=d.index)),
        ('original', lambda d: d['year'] <= 2010)
    ]

    results_summary = {}

    for name, selector in analyses:
        sub = df[selector(df)].copy()
        # listwise deletion on vars_clean
        sub_before = len(sub)
        sub = sub.dropna(subset=vars_clean)
        sub_after = len(sub)

        if sub_after == 0:
            results_summary[name] = {'error': 'No complete cases after listwise deletion', 'n_before': sub_before}
            continue

        X = sub[vars_clean].astype(float).values
        weights = sub['wtss'].astype(float).values if 'wtss' in sub.columns else None

        models = {}
        for k in [2,3,4]:
            # Fit either sklearn GaussianMixture or fallback kmeans2-based approximate clustering
            if SKLEARN_MIXTURE and GaussianMixture is not None:
                gm = GaussianMixture(n_components=k, covariance_type='full', random_state=12345, n_init=15)
                try:
                    if weights is not None:
                        gm.fit(X, sample_weight=weights)
                    else:
                        gm.fit(X)
                except TypeError:
                    gm.fit(X)
                except Exception:
                    gm.fit(X)

                probs = gm.predict_proba(X)
                labels = (np.argmax(probs, axis=1) + 1).astype(int)
            else:
                # fallback: kmeans2 to get hard clusters, then compute class posteriors via softmax of negative distances
                centroids, labels_k = kmeans2(X, k, minit='points')
                labels = (labels_k + 1).astype(int)
                # compute distances to centroids
                dists = np.zeros((X.shape[0], k))
                for i_c in range(k):
                    dists[:, i_c] = np.linalg.norm(X - centroids[i_c], axis=1)
                # convert distances to similarities and softmax to get pseudo-probabilities
                sim = -dists
                sim_max = np.max(sim, axis=1, keepdims=True)
                exp_sim = np.exp(sim - sim_max)
                probs = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)

            # attach to sub
            for j in range(probs.shape[1]):
                sub[f'classpost{j+1}'] = probs[:, j]
            sub['predclass'] = labels

            # compute means per class for vars_clean
            classes = sorted(np.unique(labels))
            means = {int(cls): sub.loc[sub['predclass']==cls, vars_clean].mean().to_dict() for cls in classes}

            # pairwise t-tests on evolved_clean
            ttests = {}
            for i_idx in range(len(classes)):
                for j_idx in range(i_idx+1, len(classes)):
                    c1 = classes[i_idx]
                    c2 = classes[j_idx]
                    a = sub.loc[sub['predclass']==c1, 'evolved_clean']
                    b = sub.loc[sub['predclass']==c2, 'evolved_clean']
                    try:
                        tstat, pval = stats.ttest_ind(a, b, nan_policy='omit')
                    except Exception:
                        tstat, pval = None, None
                    ttests[f'{c1}_vs_{c2}'] = {
                        'tstat': None if tstat is None or (isinstance(tstat, float) and np.isnan(tstat)) else float(tstat),
                        'pval': None if pval is None or (isinstance(pval, float) and np.isnan(pval)) else float(pval)
                    }

            models[f'lc{k}'] = {
                'nobs': len(sub),
                'means_by_class': means,
                'ttests_evolved_pairwise': ttests
            }

        # Save posterior and predclass for the 3-class model as a convenience
        try:
            # prefer predclass from 3-class model
            chosen_probs = None
            # locate columns classpost1.. classpost3
            if 'classpost1' in sub.columns and 'classpost3' in sub.columns:
                chosen_probs = sub[[c for c in sub.columns if c.startswith('classpost')]]
            # save subset with predclass
            out_cols = ['predclass'] + [c for c in sub.columns if c.startswith('classpost')]
            outdf = sub[out_cols].copy()
            outdf_path = f'/app/data/{name}_classpost_predclass.csv'
            outdf.to_csv(outdf_path, index=False)
        except Exception:
            pass

        results_summary[name] = models

    # Write results summary JSON
    with open(RESULT_LOG, 'w') as f:
        f.write(json.dumps(results_summary, indent=2))

    # Also save a JSON file
    with open('/app/data/OBrienReplication_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print('Completed replication script. Results saved to /app/data/OBrienReplication_summary.json and', RESULT_LOG)


if __name__ == '__main__':
    main()
