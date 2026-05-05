import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('artifacts', exist_ok=True)

# Read data
v = pd.read_csv('data/violent_elections.csv')
# ensure numeric
v['violence'] = pd.to_numeric(v['violence'], errors='coerce')
v['fraud'] = pd.to_numeric(v['fraud'], errors='coerce')

# Create squared term
v['viol_sq'] = v['violence'] ** 2

# Run regression
model = smf.ols('fraud ~ violence + viol_sq + control1 + control2', data=v).fit()
print(model.summary())

# Save coefficients
coefs = model.summary2().tables[1]
coefs.to_csv('artifacts/coefs.csv')

# Plot
sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
sns.scatterplot(data=v, x='violence', y='fraud')
# predicted curve
x = np.linspace(v['violence'].min(), v['violence'].max(), 100)
Xpred = pd.DataFrame({'violence': x, 'viol_sq': x**2, 'control1': v['control1'].mean(), 'control2': v['control2'].mean()})
yhat = model.predict(Xpred)
plt.plot(x, yhat, color='red')
plt.xlabel('Violence')
plt.ylabel('Fraud')
plt.tight_layout()
plt.savefig('artifacts/violence_fraud_plot.png')

# Save model summary text
with open('artifacts/model_summary.txt', 'w') as f:
    f.write(model.summary().as_text())
