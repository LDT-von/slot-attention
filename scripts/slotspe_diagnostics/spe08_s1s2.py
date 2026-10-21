"""Run only S1 and S2 (eval-only on existing last.pt), and merge into
the existing addendum.json that already contains S3 + S4.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path("/data1/DCT-Reg")
sys.path.insert(0, str(REPO))

from _s2026_1020_addendum import (
    _load_model_and_data,
    s1_p6_rng,
    s2_p2_slots,
    OUT_DIR,
)

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_path = OUT_DIR / "addendum.json"
    existing = json.loads(existing_path.read_text())

    model, manifest, fold_data, val_ids = _load_model_and_data()

    print("[run S1 then S2 only]", flush=True)
    existing["S1_p6_rng_controls"] = s1_p6_rng(
        model, manifest, fold_data, val_ids
    )
    existing["S2_p2_slot_details"] = s2_p2_slots(
        model, manifest, fold_data, val_ids
    )

    existing["recorded_at_final"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    existing_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    print(f"\n[done] {existing_path}", flush=True)
