"""Replication analysis translated to Python
Reads data from /app/data/Tremoliere_generalizability_score.csv
Performs data cleaning analogous to Tremoliere_generalizability_score.Rmd
Fits a binomial GLM with cluster-robust SEs (clusters = participant_uid)
Runs a mixed ANOVA (within = moral_scenario, between = salience) via pingouin
Saves textual summary to /app/data/analysis_results.txt
"""

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

try:
    import pingouin as pg
except Exception:
    pg = None

# Prefer /app/data location, but fall back to packaged input path if needed
INPUT_PATHS = ['/app/data/Tremoliere_generalizability_score.csv',
               'data/original/35/input/replication_data/Tremoliere_generalizability_score.csv']
OUTPUT_PATH = '/app/data/analysis_results.txt'
INPUT_PATH = None
for p in INPUT_PATHS:
    if os.path.exists(p):
        INPUT_PATH = p
        break
if INPUT_PATH is None:
    # As last resort, try filename in current directory
    local_path = os.path.join(os.path.dirname(__file__), 'Tremoliere_generalizability_score.csv')
    if os.path.exists(local_path):
        INPUT_PATH = local_path
    else:
        INPUT_PATH = '/app/data/Tremoliere_generalizability_score.csv'  # default; script will error if missing

def load_data(path):
    df = pd.read_csv(path)
    return df


def prepare_long(df):
    # Select relevant columns (some may be missing depending on file)
    cols = [c for c in df.columns]
    # Melt the two moral accept columns into long format
    id_vars = [c for c in ['ResponseId', 'death_salience','pain_salience','satisfactory_manipulation_response1',
                           'age','gender','gender_3_TEXT','politic_1','politic_2','politic_3','race',
                           'race_7_TEXT','race_8_TEXT','income','education','open','cond','salience','participant_uid'] if c in cols]
    value_vars = [v for v in ['moral_accept','moral_accept1'] if v in cols]
    df_long = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name='variable', value_name='moral_acceptability')

    # Keep only those who passed the manipulation check (value 1)
    if 'satisfactory_manipulation_response1' in df_long.columns:
        df_long = df_long[df_long['satisfactory_manipulation_response1'] == 1].copy()

    # Map moral acceptability to 0/1: original R code used '1'->0 (not moral), '2'->1 (moral)
    def map_response(x):
        try:
            if pd.isnull(x):
                return np.nan
            # handle numeric
            if isinstance(x, (int, float)):
                if int(x) == 1:
                    return 0
                elif int(x) == 2:
                    return 1
            # handle strings
            xs = str(x).strip()
            if xs == '1':
                return 0
            if xs == '2':
                return 1
        except Exception:
            pass
        return np.nan

    df_long['moral_acceptability_01'] = df_long['moral_acceptability'].apply(map_response)

    # Create moral_scenario labels
    df_long['moral_scenario'] = df_long['variable'].map({'moral_accept':'impartial_beneficience',
                                                          'moral_accept1':'partial_beneficience'})
    # Ensure salience is categorical (death/pain)
    if 'salience' in df_long.columns:
        df_long['salience'] = df_long['salience'].astype(str)
    else:
        # try to assemble salience from death_salience/pain_salience
        if 'death_salience' in df_long.columns:
            # If death_salience column exists with content, infer salience
            df_long['salience'] = np.where(df_long['death_salience'].notnull() & (df_long['death_salience']!=''), 'death','pain')
        else:
            df_long['salience'] = 'unknown'

    return df_long


def descriptive_stats(df_long):
    out = []
    out.append('Total observations (long): {}'.format(len(df_long)))
    out.append('Total unique participants: {}'.format(df_long['participant_uid'].nunique()))
    out.append('\nCounts by salience and scenario (n, % moral=1):')
    grp = df_long.groupby(['salience','moral_scenario'])['moral_acceptability_01'].agg(['count','mean']).reset_index()
    grp['pct_moral'] = grp['mean']*100
    out.append(grp.to_string(index=False))
    return '\n'.join(out)


def fit_glm_cluster(df_long):
    # Fit binomial GLM with interaction; use cluster-robust SEs to account for repeated measures within participant
    formula = 'moral_acceptability_01 ~ C(salience)*C(moral_scenario)'
    model = smf.glm(formula=formula, data=df_long, family=sm.families.Binomial())
    result = model.fit()
    try:
        result_clust = result.get_robustcov_results(cov_type='cluster', groups=df_long['participant_uid'])
    except Exception:
        result_clust = result
    return result, result_clust


def run_mixed_anova(df_long):
    if pg is None:
        return 'pingouin not installed; mixed ANOVA skipped. Install pingouin to run mixed_anova.'
    # pingouin requires no missing dv values
    df_anova = df_long.dropna(subset=['moral_acceptability_01', 'moral_scenario', 'salience'])
    try:
        aov = pg.mixed_anova(dv='moral_acceptability_01', within='moral_scenario', between='salience', subject='participant_uid', data=df_anova)
        return aov
    except Exception as e:
        return f'Error running pingouin.mixed_anova: {e}'


def save_output(text, path):
    with open(path, 'w') as f:
        f.write(text)


def main():
    if not os.path.exists(INPUT_PATH):
        print('Input data not found at', INPUT_PATH)
        return
    df = load_data(INPUT_PATH)
    df_long = prepare_long(df)

    outputs = []
    outputs.append('Descriptive statistics:\n')
    outputs.append(descriptive_stats(df_long))

    outputs.append('\n\nFitting binomial GLM (cluster-robust SEs) with formula: moral_acceptability_01 ~ C(salience)*C(moral_scenario)\n')
    res, res_clust = fit_glm_cluster(df_long)
    outputs.append('GLM (naive) summary:\n')
    outputs.append(res.summary().as_text())
    outputs.append('\nGLM (cluster-robust SEs) summary:\n')
    try:
        outputs.append(res_clust.summary().as_text())
    except Exception:
        outputs.append(str(res_clust))

    outputs.append('\n\nMixed ANOVA (pingouin) results:\n')
    aov = run_mixed_anova(df_long)
    outputs.append(str(aov))

    # Save outputs
    out_text = '\n'.join([str(x) for x in outputs])
    save_output(out_text, OUTPUT_PATH)
    print('Analysis finished. Results saved to', OUTPUT_PATH)

if __name__ == '__main__':
    main()
