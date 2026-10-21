"""Plan by default; --run compares native Top2 and dense inference controls."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survot_rank.research.slotspe_proof.sparsity_ablation import main

if __name__ == "__main__":
    raise SystemExit(main())
