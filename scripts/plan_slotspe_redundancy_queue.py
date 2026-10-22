"""Metadata-only compatibility planner; explicit execution uses run_slotspe_redundancy.py --run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from survot_rank.research.slotspe_proof.redundancy_queue import make_plan

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.parse_args()
    print(json.dumps(make_plan(), ensure_ascii=False, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
