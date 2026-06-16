import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm


def compute_contrasts(model, group_levels, et_levels, ref_group='Control'):
    params = model.params
    cov = model.cov_params()
    param_names = params.index.tolist()
    results = []
    for et in et_levels:
        for t in group_levels:
            if t == ref_group:
                continue
            L = np.zeros(len(params))
            name_main = f"C(Group, Treatment('{ref_group}'))[T.{t}]"
            if name_main in param_names:
                L[param_names.index(name_main)] = 1.0
            else:
                continue
            inter1 = f"C(Group, Treatment('{ref_group}'))[T.{t}]:C(et)[T.{et}]"
            inter2 = f"C(et)[T.{et}]:C(Group, Treatment('{ref_group}'))[T.{t}]"
            if inter1 in param_names:
                L[param_names.index(inter1)] += 1.0
            elif inter2 in param_names:
                L[param_names.index(inter2)] += 1.0
            est = float(np.dot(L, params.values))
            se = float(np.sqrt(np.dot(L, np.dot(cov.values, L))))
            z = est / se if se > 0 else np.nan
            p = 2 * (1 - norm.cdf(abs(z))) if se > 0 else np.nan
            ci_low = est - 1.96 * se
            ci_high = est + 1.96 * se
            or_est = float(np.exp(est))
            or_low = float(np.exp(ci_low))
            or_high = float(np.exp(ci_high))
            results.append({
                'et': et,
                'contrast': f"{t} vs {ref_group}",
                'log_or': est,
                'se': se,
                'z': z,
                'p': p,
                'or': or_est,
                'or_ci_low': or_low,
                'or_ci_high': or_high
            })
    return results


def main():
    data_path = "/app/data/Benin2012survey.csv"
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "Benin2012survey.csv")
    d = pd.read_csv(data_path)
    dexp = d[d['passage'].isin(['Control', 'Femme', 'FonFemme'])].copy()
    dexp['Group'] = pd.Categorical(dexp['passage'], categories=['Control', 'Femme', 'FonFemme'])

    def recode_et(x):
        if x == 'Fon':
            return 'Wife Coethnics'
        elif x in ['Yorouba', 'Bariba']:
            return 'President Coethnics'
        else:
            return 'Non Ethnics'

    dexp['et'] = dexp['ethnie'].apply(recode_et)
    df = dexp[['BoniVote', 'Group', 'et']].dropna().copy()
    formula = "BoniVote ~ C(Group, Treatment('Control'))*C(et)"
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()
    print(model.summary())

    group_levels = ['Control', 'Femme', 'FonFemme']
    et_levels = sorted(df['et'].dropna().unique().tolist())
    contrasts = compute_contrasts(model, group_levels, et_levels, ref_group='Control')

    print("\nTreatment vs Control contrasts within each ethnicity group:")
    for r in contrasts:
        print(f"et={r['et']}, {r['contrast']}: OR={r['or']:.3f}, 95% CI [{r['or_ci_low']:.3f}, {r['or_ci_high']:.3f}], p={r['p']:.3f}")

    out_path = "/app/data/task1_contrasts.csv"
    try:
        pd.DataFrame(contrasts).to_csv(out_path, index=False)
        print(f"Saved contrasts to {out_path}")
    except Exception as e:
        print(f"Could not save to /app/data: {e}")


if __name__ == '__main__':
    main()
