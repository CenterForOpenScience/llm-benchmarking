import os
import sys
print('Starting main entry')
entry = os.path.join('replication_data','k17_data_prep__py.py')
if os.path.exists(entry):
    print('Found entrypoint, executing')
    os.execvp(sys.executable, [sys.executable, entry])
else:
    print('Entrypoint not found:', entry)
    sys.exit(1)
