"""Run only the unfinished sections: S3 + S4.

S1 and S2 outputs already on disk from prior run are read back in.
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
    s3_p4_full_perm,
    s4_risk_direction,
    OUT_DIR,
)

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_path = OUT_DIR / "addendum.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())
    else:
        existing = {
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "blca_fold0_seed3_3epoch_eval_only": {
                "epoch_loaded": 3,
                "checkpoint_sha256": "1c6daa01969879c40612bc67dd17ab7b9d3ed9e5ea56671e7f8278dd4c30aec5",
                "training_runs": 0,
                "backward_passes": 0,
            },
        }

    model, manifest, fold_data, val_ids = _load_model_and_data()
    existing["blca_fold0_seed3_3epoch_eval_only"]["model_class"] = type(model).__name__
    existing["blca_fold0_seed3_3epoch_eval_only"]["patches"] = manifest["patches"]
    existing["blca_fold0_seed3_3epoch_eval_only"]["val_ids_count"] = len(val_ids)

    print("[resume] S3 only", flush=True)
    existing["S3_p4_full_perm"] = s3_p4_full_perm(
        model, manifest, fold_data, val_ids
    )
    print("[resume] S4 offline", flush=True)
    existing["S4_risk_direction_offline"] = s4_risk_direction(
        manifest, fold_data, val_ids
    )

    existing["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    existing_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    print(f"\n[done] {existing_path}", flush=True)
