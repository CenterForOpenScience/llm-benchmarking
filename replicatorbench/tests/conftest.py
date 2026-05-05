# tests/conftest.py
import os
import sys
from pathlib import Path

# Put the project root on sys.path so `import logger` work.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# make sure we never accidentally call the real API during tests
os.environ.setdefault("API_KEY", "")

