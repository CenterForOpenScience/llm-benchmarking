Execution entrypoint: replicate_gerhold__py.py

Runtime expectations:
- Data file must be present at /app/data/data_gerhold.csv
- Script outputs /app/data/replication_results.json with gender-difference tests for two anxiety items.

How to run:
python replication_data/replicate_gerhold__py.py

Notes:
- Uses WLS with normalized weights (weight_new preferred) and Welch's t-test unweighted.
- Filters to binary gender only (female_bin in {0,1}).
- Assumes CountryofLiving == 'Germany'.
