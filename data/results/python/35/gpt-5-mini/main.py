import sys
import os
import subprocess

# Ensure replication_data is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'replication_data'))

# If user site-packages exists (where pip may install), add it to sys.path
user_site = os.path.expanduser('~/.local/lib/python3.9/site-packages')
if os.path.exists(user_site):
    sys.path.insert(0, user_site)

# Ensure required packages are installed (helpful if Docker image lacks them)
required = ['pandas','numpy','scipy','statsmodels','pingouin','openpyxl']
try:
    import importlib
    importlib.import_module('pandas')
except Exception:
    print('Required packages missing. Installing:', required)
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user'] + required)
        # After installing to user site, ensure it's on sys.path
        if os.path.exists(user_site) and user_site not in sys.path:
            sys.path.insert(0, user_site)
    except Exception as e:
        print('Failed to install packages at runtime:', e)
        raise

try:
    from tremoliere_generalizability_score__py import main as run_main
except Exception as e:
    print('Error importing analysis script:', e)
    raise

if __name__ == '__main__':
    run_main()
