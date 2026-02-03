import json
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Paths
DATA_PATH = "/app/data/gelfand_replication_data.csv"
OUT_JSON = "/app/data/gelfand_replication_results.json"
OUT_TXT = "/app/data/gelfand_replication_results.txt"


def compute_country_growth_rates(df):
    # Keep only observations after country exceeds 1 case per million and up to 2020-03-30
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    mask = (df['total_covid_per_million'] > 1) & (df['date'] <= pd.Timestamp('2020-03-30'))
    df = df.loc[mask].sort_values(['country', 'date'])

    results = []
    for country, g in df.groupby('country', sort=False):
        if len(g) < 3:
            continue
        # Response: log of total cases per million (positive by construction of filter)
        y = np.log(g['total_covid_per_million'].values.astype(float))
        # Predictor: time index starting at 0
        t = np.arange(len(g), dtype=float)
        X = sm.add_constant(t)
        try:
            model = sm.OLS(y, X, missing='drop').fit()
            slope = model.params[1]
            results.append({
                'country': country,
                'growth_rate': float(slope),
                'n_obs': int(len(g)),
                'first_date': g['date'].iloc[0].strftime('%Y-%m-%d'),
                'last_date': g['date'].iloc[-1].strftime('%Y-%m-%d')
            })
        except Exception:
            continue
    return pd.DataFrame(results)


def standardize(series):
    s = pd.Series(series)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isclose(sd, 0):
        return (s - mu)
    return (s - mu) / sd


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Expected data at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Compute country-level growth rates
    gr = compute_country_growth_rates(df)

    # Merge predictors (take first row per country since country-level variables are constant over time)
    predictors = df.sort_values(['country', 'date']).groupby('country', as_index=False).first()[
        ['country', 'tightness', 'efficiency', 'gdp', 'median_age', 'gini_val', 'alternative_gini']
    ]

    # Prefer gini_val; if missing, use alternative_gini
    predictors['gini_any'] = predictors['gini_val']
    missing_mask = predictors['gini_any'].isna()
    predictors.loc[missing_mask, 'gini_any'] = predictors.loc[missing_mask, 'alternative_gini']

    data = pd.merge(gr, predictors, on='country', how='inner')

    # Drop missing on key predictors or outcome
    data = data.dropna(subset=['growth_rate', 'tightness', 'efficiency', 'gdp', 'median_age', 'gini_any', 'n_obs'])

    # Standardize continuous predictors (not the outcome)
    data['tightness_z'] = standardize(data['tightness'])
    data['efficiency_z'] = standardize(data['efficiency'])
    data['gdp_z'] = standardize(data['gdp'])
    data['median_age_z'] = standardize(data['median_age'])
    data['gini_z'] = standardize(data['gini_any'])

    # Interaction
    data['tight_x_eff_z'] = data['tightness_z'] * data['efficiency_z']

    # Weights: number of country-days contributing to growth-rate estimate
    weights = data['n_obs'].astype(float).values

    # Design matrix
    X = data[['tightness_z', 'efficiency_z', 'tight_x_eff_z', 'gdp_z', 'median_age_z', 'gini_z']]
    X = sm.add_constant(X)
    y = data['growth_rate'].values

    # Weighted least squares with robust covariance
    wls_model = sm.WLS(y, X, weights=weights)
    wls_res = wls_model.fit(cov_type='HC1')

    # Extract key results
    coef = wls_res.params.to_dict()
    pvals = wls_res.pvalues.to_dict()
    bse = wls_res.bse.to_dict()

    # Predicted growth rate at +/-1 SD of tightness and efficiency (controls at mean -> 0 after z-score)
    def pred(tz, ez):
        row = pd.DataFrame({
            'const': [1.0],
            'tightness_z': [tz],
            'efficiency_z': [ez],
            'tight_x_eff_z': [tz * ez],
            'gdp_z': [0.0],
            'median_age_z': [0.0],
            'gini_z': [0.0]
        })
        return float(np.dot(row.values, wls_res.params.values))

    preds = {
        'tight_-1_eff_-1': pred(-1, -1),
        'tight_+1_eff_+1': pred(+1, +1),
        'tight_-1_eff_+1': pred(-1, +1),
        'tight_+1_eff_-1': pred(+1, -1)
    }

    summary_text = []
    summary_text.append("Weighted OLS: growth_rate ~ tightness_z + efficiency_z + tight_x_eff_z + controls")
    summary_text.append(f"N countries: {len(data)}")
    summary_text.append("")
    summary_text.append(str(wls_res.summary()))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    with open(OUT_TXT, 'w') as f:
        f.write("\n".join(summary_text))

    out = {
        'model': 'WLS with HC1 SEs',
        'n_countries': int(len(data)),
        'weights': 'number of country-days with total_covid_per_million>1 up to 2020-03-30',
        'dependent_variable': 'country-level exponential growth rate of total cases per million (slope of log cases per million over time)',
        'predictors_standardized': True,
        'coefficients': coef,
        'standard_errors': bse,
        'p_values': pvals,
        'interaction_of_interest': 'tight_x_eff_z',
        'predicted_growth_rate_at_combinations': preds,
        'notes': 'Countries with missing efficiency or fewer than 3 days were excluded. Gini created as gini_val if available else alternative_gini.'
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"Saved results to {OUT_JSON} and {OUT_TXT}")


if __name__ == '__main__':
    main()
