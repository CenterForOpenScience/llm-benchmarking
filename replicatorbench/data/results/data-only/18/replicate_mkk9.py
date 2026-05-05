# Wrapper to call the actual script in replication_data so orchestrator can run `python replicate_mkk9.py`
from replication_data.replicate_mkk9 import main

if __name__ == "__main__":
    main()
