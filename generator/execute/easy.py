# generate/execute/easy.py
import os, json
from datetime import datetime

def run_execute_easy(study_path: str):
    """
    Placeholder for Execute (Easy): 
    expected to:
      - read preregistration_design.json
      - run replication code on the dataset
      - write execution outputs (results, logs, artifacts)
    """
    out = {
        "status": "not_implemented_yet",
        "study_path": study_path,
        "expects": {
            "input": "preregistration_design.json + replication dataset/code",
            "output": ["execution_results.json", "figures/", "tables/"]
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    path = os.path.join(study_path, "execute_results_stub.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Execute/Easy] wrote stub to {path}")
    return out

