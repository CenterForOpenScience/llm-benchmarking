How to run (inside container):

1) Ensure the dataset file exists in /app/data with one of these names:
   - COVID replication.dta (preferred)
   - COVID_replication.dta
   - covid_replication.dta
   - OR the RDS/CSV equivalents with the same base name.

   The original Stata file includes columns:
   - country_name, country_code, Democracy index (EIU), popData2019,
   - COVID_12_31_04_03 (confirmed cases as of 2020-04-03), Annual_temp,
   - trade_recent, imputed_trade, trade_2016.

   The script will standardize columns to lowercase/snake_case and infer variables.

2) Run the replication script:
   python /app/replication_data/replicate_democracy_covid.py

3) Outputs will be written to /app/data:
   - variable_mapping.json
   - replication_results.json
   - replication_model_summary.txt

Model notes:
- Baseline OLS: cases per million ~ temperature + precipitation (if available) + openness + democracy + population density (if available). Robust SE (HC1).
- Refined OLS drops non-significant covariates at 10% level but keeps democracy.
- Openness is chosen from available columns in priority order: trade_recent, imputed_trade, trade_2016.
- Dependent variable is computed as COVID_12_31_04_03 / popData2019 * 1e6 if a per-million column is absent.
