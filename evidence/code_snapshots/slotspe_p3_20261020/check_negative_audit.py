"""Rehashed metadata attacks against read-only hard-linked result copies."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(r"E:\DCT-Reg")
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from survot_rank.research.slotspe_proof.p3_diagnostics import audit, _read, _sha, _write

original = ROOT / ".review/slotspe_p3_20261020/seed3_native_v1"
probe = original.parent / "probe_source_v1.py"
before = {p.name: _sha(p) for p in original.iterdir() if p.is_file()}
checks = []
for attack in ("wrong_checkpoint_seed", "empty_rng_digests", "false_summary"):
    target = original.parent / "negative_audit" / attack
    target.mkdir(parents=True, exist_ok=False)
    for path in original.iterdir():
        if path.name in ("observations.json", "receipt.json"):
            shutil.copy2(path, target / path.name)
        else:
            os.link(path, target / path.name)  # never written; no raw-array byte duplication
    report = _read(target / "observations.json")
    if attack == "wrong_checkpoint_seed":
        report["seed"] = 999
    elif attack == "empty_rng_digests":
        report["controlled_rng_before"] = report["controlled_rng_after"] = {}
    else:
        report["summaries"]["native"]["patient_risk_range"]["max"] = 0
    (target / "observations.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt = _read(target / "receipt.json")
    receipt["outputs"]["observations.json"] = _sha(target / "observations.json")
    (target / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        audit(target, probe)
    except ValueError as exc:
        checks.append({"attack": attack, "rehashed_report": True, "rejected": True, "reason": str(exc)})
    else:
        raise AssertionError("False metadata accepted: " + attack)
assert before == {p.name: _sha(p) for p in original.iterdir() if p.is_file()}
_write(ROOT / "docs/research/slotspe_p3_20261020/raw_negative_audit.json", {"scope": "copies of real result metadata, no new patient inference",
    "checks": checks, "original_outputs_unchanged": True, "count_not_added_to_synthetic_tests": True})
print(json.dumps(checks, ensure_ascii=False, indent=2))
