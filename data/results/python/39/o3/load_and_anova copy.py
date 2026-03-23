# Wrapper script to locate and execute the analysis inside replication_data folder
import runpy, os, sys

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "replication_data", "load_and_anova.py")

if not os.path.exists(SCRIPT_PATH):
    sys.stderr.write(f"Cannot find analysis script: {SCRIPT_PATH}\n")
    sys.exit(1)

# Execute the target script in its own namespace
runpy.run_path(SCRIPT_PATH, run_name='__main__')
