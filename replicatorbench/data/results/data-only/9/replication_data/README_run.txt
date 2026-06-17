Run entrypoint: run_replication.py

This script expects the following input files to be present in /app/data:
- CMP_final.dta: Comparative Manifesto Project party-level dataset with policy variables and identifiers (country, year or election date).
- CPDS_final.dta: Control dataset with electoral system indicator (e.g., 'prop') keyed by country-year.

Outputs are written to /app/data/replication_outputs/:
- results_replication.json: Coefficient tables for OLS models with clustered SEs by country.
- analysis_dataset.csv: The constructed country-election dataset with dispersion and party count measures.

Method summary:
- If economic and social dimension variables are absent, the script builds two policy dimensions using PCA on CMP policy variables.
- Computes party dispersion (max-min) per country-election on both dimensions.
- Computes two measures of number of parties per election: effective number (ENP) and relevance count (>=1% share).
- Merges electoral system control from CPDS if available.
- Regresses log dispersion on log number-of-parties measures with electoral system control; clustered SEs by country.
