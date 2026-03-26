"""
Wrapper script to execute the replication analysis.
This file sits at the repository root so that the orchestrator can use
`replicate_analysis.py` as the entrypoint.  It delegates to the actual
implementation located in `replication_data/replicate_analysis.py`.
"""

import sys
from pathlib import Path

# Add the replication_data directory to the Python path
CURRENT_DIR = Path(__file__).resolve().parent
REPLICATION_DIR = CURRENT_DIR / "replication_data"
if str(REPLICATION_DIR) not in sys.path:
    sys.path.insert(0, str(REPLICATION_DIR))

# Import and run the main() function from the original script
from replicate_analysis import main as rep_main  # type: ignore

if __name__ == "__main__":
    rep_main()
