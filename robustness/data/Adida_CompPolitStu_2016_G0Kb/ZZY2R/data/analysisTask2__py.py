import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Task2 Python translation of analysisTask2.R
# IO policy: read from /app/data, write outputs to stdout

def main():
    # Load data
    df = pd.read_csv('/app/data/Benin2012survey.csv')

    # Keep only three experimental conditions
    df = df[df['passage'].isin(['FonFemme', 'Femme', 'Control'])].copy()

    # Exclude Yoruba (President's coethnics)
    df = df[df['ethnie'] != 'Yorouba'].copy()

    # Dependent variable: vote (1=yes, 0=no)
    vote_src = 'BoniVote' if 'BoniVote' in df.columns else ('bonivote' if 'bonivote' in df.columns else None)
    if vote_src is None:
        raise ValueError('Vote column not found. Expected BoniVote or bonivote.')
    df['vote'] = np.where(df[vote_src] == 1, 1, np.where(df[vote_src] == 0, 0, np.nan))

    df = df.dropna(subset=['vote']).copy()
    df['vote'] = df['vote'].astype(int)

    # Coethnic: Fon vs others
    df['coethnic'] = np.where(df['ethnie'] == 'Fon', 'co-ethnic(FON)', 'non-coethnic')

    # Condition mapping
    df['condition'] = np.select(
        [df['passage'] == 'Femme', df['passage'] == 'FonFemme', df['passage'] == 'Control'],
        ['wife', 'wife(FON)', 'control'],
        default='control'
    )

    df['condition'] = pd.Categorical(df['condition'], categories=['control', 'wife', 'wife(FON)'])
    df['coethnic'] = pd.Categorical(df['coethnic'], categories=['non-coethnic', 'co-ethnic(FON)'])

    # Fit interaction model
    m1 = smf.glm('vote ~ C(condition) * C(coethnic)', data=df, family=sm.families.Binomial()).fit()

    print('Sample size (after filtering and dropping missing vote):', len(df))
    print('\nModel with interaction coefficients (log-odds):')
    coefs = m1.summary2().tables[1]
    for col in ['Coef.', 'Std.Err.', 'z', 'P>|z|']:
        if col in coefs.columns:
            coefs[col] = coefs[col].round(3)
    print(coefs.to_string())

    inter_terms = [ix for ix in coefs.index if ':' in ix]
    if inter_terms:
        print('\nInteraction terms:')
        for term in inter_terms:
            row = m1.summary2().tables[1].loc[term]
            print(f"{term}: b={row['Coef.']:.3f}, SE={row['Std.Err.']:.3f}, z={row['z']:.2f}, p={row['P>|z|']:.3g}")

if __name__ == '__main__':
    main()
