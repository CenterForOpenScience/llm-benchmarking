import os
import pandas as pd
import numpy as np
import pyreadstat
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Paths (assume /app/data is mounted to repository root)
DATA_PATH = '/app/data/replication_data/Ma-Kellams Replication Study Data.sav'
OUTPUT_PATH = '/app/data/replication_data/replication_results.txt'

def load_data(path):
    # Use pyreadstat to read the .sav file
    df, meta = pyreadstat.read_sav(path)
    return df


def recode_variables(df):
    # Ensure numeric conversion for relevant vars
    for col in ['PAS1','PAS2','PAS3','PAS4','PAS5','WordCount','SentenceCount','WritingCondition','CulturalBackground','BornCountry']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Wrote30WordsOrMore
    df['Wrote30WordsOrMore'] = np.where(df['WordCount']>29, 1, np.where(df['WordCount'].isna(), np.nan, 0))

    # EuropeanAmerican and EastAsian from CulturalBackground codes (per SPSS syntax: 1 -> European American, 4 -> East Asian)
    df['EuropeanAmerican'] = np.where(df['CulturalBackground']==1, 1, 0)
    df['EastAsian'] = np.where(df['CulturalBackground']==4, 1, 0)

    # Culture: 0 = EuropeanAmerican, 1 = EastAsian
    df['Culture'] = np.where(df['EuropeanAmerican']==1, 0, np.where(df['EastAsian']==1, 1, np.nan))

    # BornCountry derived indicators (USBorn and EastAsianBorn) - using codes from SPSS syntax
    df['USBorn'] = np.where(df['BornCountry'].isin([0,187]), 1, 0)
    east_asian_country_codes = [162,156,140,86,75,36,1358]
    df['EastAsianBorn'] = np.where(df['BornCountry'].isin(east_asian_country_codes), 1, 0)

    # PAS flipping for PAS2 and PAS5
    for col in ['PAS2','PAS5']:
        if col in df.columns:
            df[col+'_Positive'] = -1 * df[col]

    # PASAverage mean of PAS1, PAS2Positive, PAS3, PAS4, PAS5Positive
    pas_cols = []
    for col in ['PAS1','PAS2_Positive','PAS3','PAS4','PAS5_Positive']:
        # Note: depending on how columns are named after conversion, handle both
        if col in df.columns:
            pas_cols.append(col)
    # Fallback to known names in raw data
    if not pas_cols:
        # Using original column names and newly created ones
        df['PAS2Positive'] = -1 * pd.to_numeric(df.get('PAS2', pd.Series(dtype=float)), errors='coerce')
        df['PAS5Positive'] = -1 * pd.to_numeric(df.get('PAS5', pd.Series(dtype=float)), errors='coerce')
        pas_cols = ['PAS1','PAS2Positive','PAS3','PAS4','PAS5Positive']

    df[pas_cols] = df[pas_cols].apply(pd.to_numeric, errors='coerce')
    df['PASAverage'] = df[pas_cols].mean(axis=1)

    # PANASNetScore per syntax: sum of positive items minus sum negative items
    pos_items = ['PANAS1','PANAS3','PANAS5','PANAS9','PANAS10','PANAS12','PANAS14','PANAS16','PANAS17','PANAS19']
    neg_items = ['PANAS2','PANAS4','PANAS6','PANAS7','PANAS8','PANAS11','PANAS13','PANAS15','PANAS18','PANAS20']
    for col in pos_items + neg_items:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['PANASNetScore'] = df[pos_items].sum(axis=1) - df[neg_items].sum(axis=1)

    # Bail transformations
    if 'BailAmount' in df.columns:
        df['BailAmount'] = pd.to_numeric(df['BailAmount'], errors='coerce')
        df['BailAmountSqrt'] = np.sqrt(df['BailAmount'].clip(lower=0))
        df['BailAmountLog'] = np.log(df['BailAmount'].fillna(0) + 1)

    return df


def run_analyses(df):
    results = []
    # Primary 2x2 ANOVA on PASAverage by Culture x WritingCondition
    df_anova = df[['PASAverage','Culture','WritingCondition','PANASNetScore']].copy()
    df_anova = df_anova.dropna(subset=['PASAverage','Culture','WritingCondition'])
    # Ensure categorical
    df_anova['Culture'] = df_anova['Culture'].astype('category')
    df_anova['WritingCondition'] = df_anova['WritingCondition'].astype('category')

    # OLS model with interaction (equivalent to UNIANOVA design)
    model = smf.ols('PASAverage ~ C(Culture) * C(WritingCondition)', data=df_anova).fit()
    anova_table = sm.stats.anova_lm(model, typ=3)
    results.append('\nPrimary 2x2 ANOVA (Type III) on PASAverage ~ Culture * WritingCondition')
    results.append(anova_table.to_string())

    # Means table
    means = df.groupby(['Culture','WritingCondition'])['PASAverage'].agg(['mean','count','sem']).reset_index()
    results.append('\nGroup means (PASAverage) by Culture x WritingCondition:')
    results.append(means.to_string(index=False))

    # Follow-up: t-tests within cultures (WritingCondition effect for each culture)
    for culture_label, culture_code in [('EuropeanAmerican',0), ('EastAsian',1)]:
        subset = df[df['Culture']==culture_code]
        subset = subset.dropna(subset=['PASAverage','WritingCondition'])
        # Ensure writingcondition groups
        grp0 = subset[subset['WritingCondition']==0]['PASAverage'].dropna()
        grp1 = subset[subset['WritingCondition']==1]['PASAverage'].dropna()
        if len(grp0)>1 and len(grp1)>1:
            tstat, pval = stats.ttest_ind(grp1, grp0, equal_var=False)
            results.append(f"\nT-test for WritingCondition effect within {culture_label} (group1=WritingCondition==1 vs group0==0): t={tstat:.4f}, p={pval:.4f}, n1={len(grp1)}, n0={len(grp0)}")
        else:
            results.append(f"\nInsufficient data for t-test within {culture_label} (n0={len(grp0)}, n1={len(grp1)})")

    # Additional models: include PANASNetScore as covariate
    df_cov = df_anova.dropna(subset=['PANASNetScore'])
    if not df_cov.empty:
        model_cov = smf.ols('PASAverage ~ C(Culture) * C(WritingCondition) + PANASNetScore', data=df_cov).fit()
        anova_cov = sm.stats.anova_lm(model_cov, typ=3)
        results.append('\nANCOVA including PANASNetScore as covariate:')
        results.append(anova_cov.to_string())

    # Additional checks similar to SPSS syntax: exclude low word counts, filter by Prolific, etc.
    # Exclude participants who wrote fewer than 30 words
    df_30 = df[df['Wrote30WordsOrMore']==1].dropna(subset=['PASAverage','Culture','WritingCondition'])
    if not df_30.empty:
        model_30 = smf.ols('PASAverage ~ C(Culture) * C(WritingCondition)', data=df_30).fit()
        anova_30 = sm.stats.anova_lm(model_30, typ=3)
        results.append('\nANOVA after excluding participants who wrote fewer than 30 words:')
        results.append(anova_30.to_string())

    return '\n\n'.join(results)


def main():
    df = load_data(DATA_PATH)
    df = recode_variables(df)
    output = run_analyses(df)
    # Write results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH,'w') as f:
        f.write(output)
    print('Replication analysis completed. Results written to:', OUTPUT_PATH)

if __name__ == '__main__':
    main()
