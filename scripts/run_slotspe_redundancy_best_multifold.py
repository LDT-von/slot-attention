"""Best-checkpoint eleven-arm queue entry point; explicit execution uses --run."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slotspe_proof.redundancy_best_multifold import main


if __name__ == "__main__":
    main()
