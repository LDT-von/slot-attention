"""Independent cross-view queue; default command only prints a data-free plan."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slotspe_proof.crossview_queue import main

if __name__ == '__main__':
    raise SystemExit(main())
