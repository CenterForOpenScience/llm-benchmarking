import os
import subprocess
import sys

# Wrapper to run the focal replication script. Ensures single entrypoint.

def main():
    script_candidates = [
        "/app/data/underdiagnosis_poor_health__py.py",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "underdiagnosis_poor_health__py.py"),
    ]

    script_path = None
    for p in script_candidates:
        if os.path.exists(p):
            script_path = p
            break
    if script_path is None:
        print("Could not locate underdiagnosis_poor_health__py.py. Ensure it is available in /app/data or alongside this runner.")
        sys.exit(1)

    cmd = [sys.executable, script_path]
    try:
        subprocess.check_call(cmd)
        print("Replication script completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Replication script failed with return code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
