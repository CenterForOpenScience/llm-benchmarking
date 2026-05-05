import subprocess
import sys
import site

# Ensure required packages are installed
required = ['pandas','numpy','statsmodels']
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '--user'])

# Add user site-packages to sys.path to make newly installed packages importable
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

import pandas as pd
import numpy as np
import statsmodels.api as sm

def main():
    df = pd.read_csv('data/clean/dataset.csv')
    X = df[['violence']].copy()
    X['violence_sq'] = X['violence']**2
    X = sm.add_constant(X)
    y = df['fraud']
    model = sm.OLS(y, X).fit()
    print(model.summary())

if __name__ == '__main__':
    main()
