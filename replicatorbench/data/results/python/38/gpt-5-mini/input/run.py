#!/usr/bin/env python3
import subprocess
import sys

SCRIPT_PATH = "/app/data/replication_data/McCarter_OrgBehavior_2010_pILK _analysis__py.py"
print(f"Executing analysis script at: {SCRIPT_PATH}")
res = subprocess.run([sys.executable, SCRIPT_PATH], capture_output=False)
print(f"Script exited with return code: {res.returncode}")
