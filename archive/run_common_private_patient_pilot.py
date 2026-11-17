"""Explicit isolated role-target pilot; defaults to a nonexecuting plan."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_models.common_private_synergy.patient_pilot import main

if __name__ == '__main__':
    main()
