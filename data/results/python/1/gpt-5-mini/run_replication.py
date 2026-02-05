#!/usr/bin/env python3
import runpy
import sys
# Execute the translated python analysis script
sys.exit(runpy.run_path('replication_data/kavanagh_analysis__py.py', run_name='__main__'))