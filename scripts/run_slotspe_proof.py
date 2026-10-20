"""Independent SlotSPE proof task; no patient work unless --run is supplied."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survot_rank.research.slotspe_proof.queue import main

if __name__ == "__main__":
    main()
