import sys
import subprocess

if __name__ == '__main__':
    cmd = [sys.executable, 'run_replication.py']
    returncode = subprocess.call(cmd)
    sys.exit(returncode)
