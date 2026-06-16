import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Task1 Python translation of analysisClaim.R
# IO policy: read from /app/data, write outputs to stdout

def main():
    # Load data
    df = pd.read_csv('/app/data/Benin2012survey.csv')

    # Keep only three experimental conditions
    df = df[df['passage'].isin(['FonFemme', 'Femme', 'Control'])].copy()

    # Exclude Yoruba (President's coethnics)
    df = df[df['ethnie'] != 'Yorouba'].copy()

    # Dependent variable: vote (1=yes, 0=no)
    # Note: dataset contains both 'BoniVote' and 'bonivote' columns; the analysis used 'BoniVote'
    vote_src = 'BoniVote' if 'BoniVote' in df.columns else ('bonivote' if 'bonivote' in df.columns else None)
    if vote_src is None:
        raise ValueError('Vote column not found. Expected BoniVote or bonivote.')
    df['vote'] = np.where(df[vote_src] == 1, 1, np.where(df[vote_src] == 0, 0, np.nan))

    # Drop missing vote
    df = df.dropna(subset=['vote']).copy()
    df['vote'] = df['vote'].astype(int)

    # Coethnic: Fon vs others
    df['coethnic'] = np.where(df['ethnie'] == 'Fon', 'co-ethnic(FON)', 'non-coethnic')

    # Condition: control, wife, wife(FON)
    df['condition'] = np.select(
        [df['passage'] == 'Femme', df['passage'] == 'FonFemme', df['passage'] == 'Control'],
        ['wife', 'wife(FON)', 'control'],
        default='control'
    )

    # Set categorical dtypes and baselines
    df['condition'] = pd.Categorical(df['condition'], categories=['control', 'wife', 'wife(FON)'])
    df['coethnic'] = pd.Categorical(df['coethnic'], categories=['non-coethnic', 'co-ethnic(FON)'])

    # Fit models (Binomial GLM with logit link)
    m0 = smf.glm('vote ~ C(condition) + C(coethnic)', data=df, family=sm.families.Binomial()).fit()
    m1 = smf.glm('vote ~ C(condition) * C(coethnic)', data=df, family=sm.families.Binomial()).fit()

    # Likelihood ratio test for interaction
    # Likelihood ratio test for interaction (manual computation)
    lr_stat = 2 * (m1.llf - m0.llf)
    df_diff = int(m1.df_model - m0.df_model)
    try:
        from scipy.stats import chi2
        lr_pvalue = float(chi2.sf(lr_stat, df_diff))
        lr_note = None
    except Exception:
        lr_pvalue = None
        lr_note = 'scipy not installed; p-value not computed'

# Outputs
    print('Sample size (after filtering and dropping missing vote):', len(df))
    print('\nCounts by condition:')
    print(df['condition'].value_counts().to_string())
    print('\nCounts by coethnic:')
    print(df['coethnic'].value_counts().to_string())

    print('\nModel with interaction coefficients (log-odds):')
    coefs = m1.summary2().tables[1]
    # Round selected columns
    for col in ['Coef.', 'Std.Err.', 'z', 'P>|z|']:
        if col in coefs.columns:
            coefs[col] = coefs[col].round(3)
    print(coefs.to_string())

    print('\nLikelihood-ratio test (interaction vs. no interaction):')
    print(f'LR stat={lr_stat:.3f}, df={int(df_diff)}, p={lr_pvalue:.3g}')

    # Print interaction terms explicitly
    inter_terms = [ix for ix in coefs.index if ':' in ix]
    if inter_terms:
        print('\nInteraction terms:')
        for term in inter_terms:
            row = m1.summary2().tables[1].loc[term]
            print(f"{term}: b={row['Coef.']:.3f}, SE={row['Std.Err.']:.3f}, z={row['z']:.2f}, p={row['P>|z|']:.3g}")

if __name__ == '__main__':
    main()
