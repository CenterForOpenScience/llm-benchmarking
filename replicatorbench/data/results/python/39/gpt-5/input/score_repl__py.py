import os
import json
import numpy as np
import pandas as pd

# Third-party packages
try:
    import pingouin as pg
except ImportError:
    raise ImportError("pingouin is required. Please install pingouin.")

try:
    import pyreadstat
except ImportError:
    raise ImportError("pyreadstat is required. Please install pyreadstat.")


def load_data():
    """Load SPSS .sav file from /app/data. The expected filename is 'SCORE_all data.sav'.
    You can override with env var SCORE_DATA_PATH.
    """
    default_path = "/app/data/SCORE_all data.sav"
    fpath = os.environ.get("SCORE_DATA_PATH", default_path)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Data file not found at {fpath}. Ensure the file is mounted to /app/data.")
    df, meta = pyreadstat.read_sav(fpath, apply_value_formats=False)
    return df


def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def compute_rh_measures(df):
    """Recreate the SPSS computations for deontological response proportions.
    - Recode APP*_main: 1->0, 2->1 (probability of rejecting harm)
    - Sum per dilemma type and divide by counts to get RH_PMD (20), RH_NMD (15), RH_CMD (15)
    Returns df with added columns and a long-format DataFrame for mixed ANOVA.
    """
    # Identify APP variables
    app_cols = [f"APP{i}_main" for i in range(1, 51)]
    present_app = [c for c in app_cols if c in df.columns]
    if len(present_app) < 50:
        missing = sorted(set(app_cols) - set(present_app))
        print(f"[WARN] Missing APP columns (expected 50): {missing}")
    # Coerce to numeric then map 1->0, 2->1
    df = coerce_numeric(df, present_app)
    for c in present_app:
        df[c] = df[c].map({1: 0, 2: 1}).astype('float')

    # Index groups based on the SPSS syntax
    pmd_idx = [f"APP{i}_main" for i in range(1, 21)]
    nmd_idx = [f"APP{i}_main" for i in range(21, 36)]
    cmd_idx = [f"APP{i}_main" for i in range(36, 51)]

    for block in [pmd_idx, nmd_idx, cmd_idx]:
        for c in block:
            if c not in df.columns:
                df[c] = np.nan

    df['RH_PMD'] = df[pmd_idx].mean(axis=1)
    df['RH_NMD'] = df[nmd_idx].mean(axis=1)
    df['RH_CMD'] = df[cmd_idx].mean(axis=1)
    df['RH_ALL'] = df[['RH_PMD', 'RH_NMD', 'RH_CMD']].mean(axis=1)

    # Ensure Subject column exists and is string (main() should already handle this)
    if 'Subject' not in df.columns:
        df = df.copy()
        df['Subject'] = np.arange(len(df)).astype(str)
    else:
        df['Subject'] = df['Subject'].astype(str)

    # Build long format without duplicating Subject via reset_index
    id_vars = ['Subject']
    if 'Condition' in df.columns:
        id_vars.append('Condition')

    melt_cols = ['RH_PMD', 'RH_NMD', 'RH_CMD']
    available_melt = [c for c in melt_cols if c in df.columns]
    long_df = df[id_vars + available_melt].melt(id_vars=id_vars, var_name='dilemma_type', value_name='RH')

    # Clean types
    if 'Condition' in long_df.columns:
        long_df['Condition'] = long_df['Condition'].astype(str)
    long_df['Subject'] = long_df['Subject'].astype(str)
    long_df['dilemma_type'] = long_df['dilemma_type'].replace({'RH_PMD': 'Personal', 'RH_NMD': 'Impersonal', 'RH_CMD': 'Nonmoral'})

    return df, long_df

def compute_confidence_rt(df):
    """Recreate confidence and RT aggregates if columns exist.
    Confidence recode: 1->2, 4->2, 2->1, 3->1; then average by block and overall.
    RT_ALL: mean of RT*_corr if available else RT*.
    """
    # Confidence
    conf_cols = [f"CONF{i}_main" for i in range(1, 51)]
    present_conf = [c for c in conf_cols if c in df.columns]
    if present_conf:
        df = coerce_numeric(df, present_conf)
        for c in present_conf:
            df[c] = df[c].map({1: 2, 4: 2, 2: 1, 3: 1})
        pmd_conf_idx = [f"CONF{i}_main" for i in range(1, 21)]
        nmd_conf_idx = [f"CONF{i}_main" for i in range(21, 36)]
        cmd_conf_idx = [f"CONF{i}_main" for i in range(36, 51)]
        for block in [pmd_conf_idx, nmd_conf_idx, cmd_conf_idx]:
            for c in block:
                if c not in df.columns:
                    df[c] = np.nan
        df['CONF_PMD'] = df[pmd_conf_idx].mean(axis=1)
        df['CONF_NMD'] = df[nmd_conf_idx].mean(axis=1)
        df['CONF_CMD'] = df[cmd_conf_idx].mean(axis=1)
        df['CONF_ALL'] = df[['CONF_PMD', 'CONF_NMD', 'CONF_CMD']].mean(axis=1)

    # Reaction times
    rt_corr_cols = [f"RT{i}_corr" for i in range(1, 51)]
    rt_raw_cols = [f"RT{i}" for i in range(1, 51)]
    if any(c in df.columns for c in rt_corr_cols):
        use_cols = [c for c in rt_corr_cols if c in df.columns]
    elif any(c in df.columns for c in rt_raw_cols):
        use_cols = [c for c in rt_raw_cols if c in df.columns]
    else:
        use_cols = []
    if use_cols:
        df = coerce_numeric(df, use_cols)
        df['RT_ALL'] = df[use_cols].mean(axis=1)

    return df


def run_mixed_anova(long_df):
    """Run 2 (Condition) x 3 (dilemma_type) mixed ANOVA on RH.
    First, try pingouin.mixed_anova. If it fails (e.g., due to sphericity/NaN issues),
    fall back to an OLS with subject fixed effects and a cluster-robust Wald test on
    the Condition*dilemma_type interaction.
    Returns (anova_table, interaction_summary_dict).
    """
    if 'Condition' not in long_df.columns:
        raise ValueError("Between-subject factor 'Condition' not found in dataset.")
    # Drop missing
    tmp = long_df.dropna(subset=['RH']).copy()
    # Ensure categorical
    tmp['Condition'] = tmp['Condition'].astype('category')
    tmp['dilemma_type'] = tmp['dilemma_type'].astype('category')
    tmp['Subject'] = tmp['Subject'].astype('category')

    # Try pingouin mixed_anova first
    try:
        aov = pg.mixed_anova(dv='RH', within='dilemma_type', between='Condition', subject='Subject', data=tmp)
        # Find interaction row (contains both factors)
        interaction_row = None
        if 'Source' in aov.columns:
            candidates = aov[aov['Source'].astype(str).str.contains('dilemma_type.*Condition|Condition.*dilemma_type', case=False, regex=True)]
            if not candidates.empty:
                interaction_row = candidates.iloc[0]
        summary = {}
        if interaction_row is not None:
            summary = {
                'F': float(interaction_row.get('F', np.nan)),
                'p': float(interaction_row.get('p-unc', np.nan)),
                'np2': float(interaction_row.get('np2', np.nan)),
                'SS': float(interaction_row.get('SS', np.nan)),
                'DF1': float(interaction_row.get('DF1', np.nan)),
                'DF2': float(interaction_row.get('DF2', np.nan)),
                'test': 'pingouin.mixed_anova'
            }
        return aov, summary
    except Exception as e:
        print(f"[WARN] pingouin.mixed_anova failed: {e}. Falling back to OLS+FE with cluster-robust Wald test.")

    # Fallback: OLS with subject fixed effects and cluster-robust (by Subject)
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    # Build formula with Subject fixed effects
    formula = 'RH ~ C(Condition) * C(dilemma_type) + C(Subject)'
    model = smf.ols(formula, data=tmp).fit(cov_type='cluster', cov_kwds={'groups': tmp['Subject']})

    # Identify interaction terms in the design matrix
    inter_terms = [p for p in model.params.index if (':C(dilemma_type)[' in p or 'C(dilemma_type)[' in p) and ':C(Condition)[' in p or (':C(Condition)[' in p and ':C(dilemma_type)[' in p)]
    # The above may miss reference-level interactions depending on coding. Safer: include any term that contains both 'C(Condition)' and 'C(dilemma_type)'.
    inter_terms = [p for p in model.params.index if ('C(Condition)' in p and 'C(dilemma_type)' in p)]
    if len(inter_terms) == 0:
        # Nothing to test (e.g., only one level of Condition). Return empty results.
        aov_fallback = pd.DataFrame([
            {'Source': 'Condition*dilemma_type', 'test': 'Wald chi2 (cluster by Subject)', 'stat': np.nan, 'df1': 0, 'p': np.nan}
        ])
        return aov_fallback, {'F': np.nan, 'p': np.nan, 'np2': np.nan, 'test': 'OLS FE (cluster)'}

    # Wald test on joint significance of interaction terms    # Wald test on joint significance of interaction terms
    R = np.zeros((len(inter_terms), len(model.params)))
    param_index = list(model.params.index)
    for i, term in enumerate(inter_terms):
        if term in param_index:
            R[i, param_index.index(term)] = 1.0
    # Request scalar outputs for compatibility across statsmodels versions
    wtest = model.wald_test(R, scalar=True)
    chi2 = float(np.atleast_1d(wtest.statistic).squeeze())
    df1 = int(len(inter_terms))
    pval = float(np.atleast_1d(wtest.pvalue).squeeze())

    # Build a small ANOVA-like table
    aov_fallback = pd.DataFrame([
        {'Source': 'Condition', 'test': 'FE OLS (cluster)', 'note': 'See coefficients'},
        {'Source': 'dilemma_type', 'test': 'FE OLS (cluster)', 'note': 'See coefficients'},
        {'Source': 'Condition*dilemma_type', 'test': 'Wald chi2 (cluster by Subject)', 'stat': chi2, 'df1': df1, 'p': pval}
    ])

    summary = {'F': float(chi2 / df1) if df1 > 0 else np.nan, 'p': pval, 'np2': np.nan, 'chi2': chi2, 'df1': df1, 'test': 'OLS FE (cluster)'}
    return aov_fallback, summary

def group_means(long_df):
    tbl = long_df.groupby(['Condition', 'dilemma_type'])['RH'].mean().reset_index()
    return tbl


def ttests_personal(df):
    """Independent t-test comparing RH_PMD between conditions, if possible."""
    if 'Condition' not in df.columns or 'RH_PMD' not in df.columns:
        return None
    # Prepare
    try:
        from scipy import stats
    except Exception:
        return None
    dfa = df[['Condition', 'RH_PMD']].dropna().copy()
    # Coerce condition to two groups
    groups = sorted(dfa['Condition'].astype(str).unique().tolist())
    if len(groups) != 2:
        return None
    g1, g2 = groups
    a = dfa.loc[dfa['Condition'].astype(str) == g1, 'RH_PMD'].values
    b = dfa.loc[dfa['Condition'].astype(str) == g2, 'RH_PMD'].values
    if len(a) < 2 or len(b) < 2:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
    # Compute Cohen's d (Hedges g not necessary)
    na, nb = len(a), len(b)
    sa2, sb2 = np.nanvar(a, ddof=1), np.nanvar(b, ddof=1)
    sp = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2))
    d = (np.nanmean(a) - np.nanmean(b)) / sp if sp > 0 else np.nan
    return {'t': float(t), 'p': float(p), 'cohens_d': float(d), 'group_labels': [g1, g2], 'means': [float(np.nanmean(a)), float(np.nanmean(b))]}


def main():
    df = load_data()
    # Add Subject index if none exists
    if 'Subject' not in df.columns and 'ID' not in df.columns:
        df = df.copy()
        df['Subject'] = np.arange(len(df)).astype(str)
    elif 'Subject' not in df.columns and 'ID' in df.columns:
        df['Subject'] = df['ID'].astype(str)

    df_rh, long_df = compute_rh_measures(df)
    df_rh = compute_confidence_rt(df_rh)

    # Run focal mixed ANOVA
    aov_table, interaction_summary = run_mixed_anova(long_df)
    means_tbl = group_means(long_df)

    # Optional t-test on personal dilemmas
    pmd_t = ttests_personal(df_rh)

    # Summarize results
    results = {
        'focal_test': '2 (Condition) x 3 (dilemma_type) mixed ANOVA on RH (probability of rejecting harm)',
        'interaction_summary': interaction_summary,
        'group_means': means_tbl.to_dict(orient='records')
    }
    if pmd_t is not None:
        results['personal_dilemma_ttest'] = pmd_t

    # Save outputs
    out_json = "/app/data/SCORE_results.json"
    out_long = "/app/data/SCORE_rh_long.csv"
    out_aov = "/app/data/SCORE_mixed_anova.csv"

    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    long_df.to_csv(out_long, index=False)
    aov_table.to_csv(out_aov, index=False)

    # Console print
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
