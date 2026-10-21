"""Original P1-P7 mechanism experiments. Default action only prints a plan."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survot_rank.research.slotspe_proof.problem_experiments import main

if __name__ == "__main__":
    raise SystemExit(main())
