#!/usr/bin/env python3
import runpy
import os

# Construct path to the analysis script in the mounted data directory
script_path = os.path.join('/app/data', 'replication_data', 'McCarter_OrgBehavior_2010_pILK _analysis__py.py')
print('Looking for script at:', script_path)
if not os.path.exists(script_path):
    raise FileNotFoundError(f"Analysis script not found at {script_path}")

# Execute the script
runpy.run_path(script_path, run_name='__main__')
