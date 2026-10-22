"""BLCA / UNI2-h / fold0 eleven-arm queue; no patient work without --run."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survot_rank.research.slotspe_proof.redundancy_queue import main

if __name__ == "__main__":
    main()
