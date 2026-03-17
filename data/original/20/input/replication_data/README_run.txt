How to run the replication analysis (Python):

- Ensure the data file is available at /app/data/Kachanoff_Survey_deidentify.csv
- Execute the script:
    python /app/data/original/20/input/replication_data/Analysis_updated__py.py

Outputs:
- /app/data/DFwide.csv : the merged wide-format dataset with T1 (x) and T2 (y)
- /app/data/replication_results.json : OLS results for Negative.y ~ Realistic.x + Symbolic.x, including focal path stats
