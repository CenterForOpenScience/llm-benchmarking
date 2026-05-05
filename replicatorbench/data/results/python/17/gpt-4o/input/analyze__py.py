import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hc0

np.random.seed(2020)

def main(mode='test', drop_wave_7=True):
    data = pd.read_csv('/app/data/data.csv')
    
    if drop_wave_7:
        data = data[data['wave'] != 7]
    drop = 'drop_7' if drop_wave_7 else 'keep_7'
    
    if mode == 'test':
        # Randomize data in each column independently for testing purposes.
        for col in data.columns[2:]:
            data[col] = np.random.permutation(data[col].values)
    
    # Baseline "Long" Model
    X = data[['sd_gov', 'mean_gov', 'africa', 'laam', 'asiae',
              'col_uka', 'col_espa', 'col_otha', 'federal', 'oecd',
              'log_gdp_per_capita', 'trade_share', 'age_15_64', 'age_65_plus']]
    y = data['gov_consumption']
    X = sm.add_constant(X)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y[X.index]
    model = sm.OLS(y, X).fit(cov_type='HC1')
    
    # Output results
    out = model.summary2().tables[1]
    print(out.columns)
    out.loc['Obs.'] = [len(model.resid), '', '', '', '', '']
    out.loc['R-squared'] = [model.rsquared, '', '', '', '', '']
    out.to_csv(f'/app/data/{mode}_{drop}.txt')

# Test analysis code
main(mode='test', drop_wave_7=True)
main(mode='test', drop_wave_7=False)

# Run on real data w/o wave 7
main(mode='real', drop_wave_7=True)

# Run on real data w/ wave 7
main(mode='real', drop_wave_7=False)
