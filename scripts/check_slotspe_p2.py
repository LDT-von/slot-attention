"""Plan by default; --run probes already retained native final slots only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from survot_rank.research.slotspe_proof.p2_diagnostics import main

if __name__ == "__main__":
    raise SystemExit(main())
