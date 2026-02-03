This folder contains Python code to replicate the focal claim from Kim & Radoias (2016) on IFLS Wave 4 data.

Entry point to execute inside the container:
  python /app/data/run_replication__py.py

This will run:
  - /app/data/underdiagnosis_poor_health__py.py

I/O paths (as required):
  - Input data must be available at /app/data/replication_data.dta (copy the provided replication_data.dta there before running).
  - Outputs are written to /app/data:
      * replication_results.json            -> key stats (sample size and marginal effect for years of education)
      * replication_model_summary.txt       -> full probit output and marginal effects table
      * replication_analysis_dataset.csv    -> the analysis-ready dataset used in the focal regression

Key variable construction notes:
  - Hypertension status is defined per paper: average of us07b1/us07c1 (systolic) and us07b2/us07c2 (diastolic), first reading excluded by design; hypertensive if systolic>140 or diastolic>90.
  - Under-diagnosed equals 1 if hypertensive and not previously diagnosed (cd05 != 'Yes').
  - General health status (si01) is split into good (1-2) vs poor (3-4). The focal model uses the poor health subsample.
  - Female is derived from 'sex' value labels; 1 for Female.
  - Log per capita expenditures uses ks11aa (monthly); if missing, ks10aa (weekly)*4 as fallback; divided by hh_size; log1p.
  - Distance to health center is rj11.
  - Years of education is reconstructed from dl06 (highest level) and dl07 (grade/graduated), using a transparent approximate mapping consistent with IFLS coding. This mirrors the paper's use of years of schooling.

Model estimated:
  - Probit on the poor-health, aged 45+, hypertensive sample:
      undiagnosed ~ years_education + female + age + age^2 + log_pce + dist_health_center
  - Average marginal effects reported; focal coefficient is the ME on years_education (expected negative, around -0.0087 as in the paper).

Dependencies:
  - Python 3.10+
  - pandas, numpy, statsmodels

