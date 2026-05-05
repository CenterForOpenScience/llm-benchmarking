import subprocess, sys

cmd = ["Rscript", "replication_data/Popper Replication R Script File.R"]
print("Executing:", " ".join(cmd))
res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr, file=sys.stderr)
    sys.exit(res.returncode)
else:
    print("R script completed successfully.")
