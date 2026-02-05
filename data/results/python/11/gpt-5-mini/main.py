import sys
import subprocess
import site

# Install required packages at runtime if missing
required = [
    'pyreadr>=0.4.4',
    'pandas>=1.4.0',
    'numpy>=1.21.0',
    'statsmodels>=0.13.0',
    'patsy>=0.5.2'
]
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir'] + required)
    # Ensure user site-packages is on sys.path
    try:
        usersite = site.getusersitepackages()
        site.addsitedir(usersite)
    except Exception:
        pass
except Exception as e:
    print('Runtime pip install failed:', e)

from replication_data.replication_analysis import main

if __name__ == '__main__':
    main()
