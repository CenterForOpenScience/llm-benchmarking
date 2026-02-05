import json
import os
import subprocess

POSSIBLE_ROOTS = [
    '/app/data/original/13/0205_python_gpt5-mini',
    '/workspace',
    '.'
]
info = None
for ROOT in POSSIBLE_ROOTS:
    info_path = os.path.join(ROOT, 'replication_info.json')
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        base_root = ROOT
        break
if info is None:
    print('replication_info.json not found in expected locations:', POSSIBLE_ROOTS)
    raise SystemExit(1)
entry = info.get('codebase', {}).get('entrypoint')
if not entry:
    print('No entrypoint found in replication_info.json')
    raise SystemExit(1)
entry_path = os.path.join(base_root, entry)
print('Running entrypoint:', entry_path)
# Ensure required Python packages are installed at runtime (fallback)
print('Installing required Python packages (pyreadr, pandas, statsmodels)')
rc_install = subprocess.call(['pip', 'install', 'pyreadr', 'pandas', 'statsmodels', 'numpy', 'scipy', 'patsy'])
print('pip install exit code:', rc_install)
# Execute the entry script with python
rc = subprocess.call(['python', entry_path])
print('Exit code:', rc)
if rc != 0:
    raise SystemExit(rc)
