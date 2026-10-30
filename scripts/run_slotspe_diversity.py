"""Plan/audit are read-only. Only --run starts the new SlotSPE loss arms."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slotspe_proof.diversity_queue import main

if __name__ == "__main__":
    main()
