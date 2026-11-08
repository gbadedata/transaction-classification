import sys
from pathlib import Path

# Make the repo root importable so tests can import analysis.py and the txnclass package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
