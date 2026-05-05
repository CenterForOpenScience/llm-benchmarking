This code replicates the association between county-level Trump vote share and social distancing.

How it works:
1. Reads /app/data/transportation.csv and /app/data/county_variables.csv.
2. Computes average distance traveled per person per county-day using trip-distance bin midpoints and population engaged in mobility (pop_home + pop_not_home).
3. Calculates percent change relative to matched weekdays in February 2020 (pre-COVID baseline) for each county-day.
4. Averages this percent change for March 19-28, 2020 to create the outcome (negative values = more distancing).
5. Merges with county-level covariates and runs an OLS with state fixed effects and clustered SEs by state.
6. Reports the coefficient on trump_share and scales it by the original interquartile range (0.203) to compare to the original paper's 4.1 pp effect.

Outputs written to /app/data:
- regression_data.csv: analysis dataset used in the regression.
- replication_results.json: summary of key regression outputs and IQR-scaled effect.

To run inside Docker or a clean environment, ensure Python 3.10+, and install pandas, numpy, statsmodels, and patsy.