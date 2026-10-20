"""Tamper only copied reports; linked raw arrays are always read-only."""
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(r"E:\DCT-Reg")
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from survot_rank.research.slotspe_proof.p4_diagnostics import audit, _read, _sha, _write

original = ROOT / ".review/slotspe_p4_20261020/seed3_native_v1"
before = {p.relative_to(original).as_posix(): _sha(p) for p in original.rglob("*") if p.is_file()}
checks = []
for attack in ("unsupported_semantic_claim", "false_matching"):
    target = original.parent / "negative_audit" / attack
    target.mkdir(parents=True, exist_ok=False)
    for path in original.rglob("*"):
        if path.is_dir():
            continue
        dest = target / path.relative_to(original)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.name in ("observations.json", "receipt.json"):
            shutil.copy2(path, dest)
        else:
            os.link(path, dest)
    report = _read(target / "observations.json")
    if attack == "unsupported_semantic_claim":
        report["semantic_drift_or_harm_proved"] = True
    else:
        report["records"][0]["wsi"]["repeats_against_reference"][0]["attention_match"]["matched_mean_cost"] = 0
    (target / "observations.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt = _read(target / "receipt.json")
    receipt["outputs"]["observations.json"] = _sha(target / "observations.json")
    (target / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        audit(target)
    except ValueError as exc:
        checks.append({"attack": attack, "rehashed_report": True, "rejected": True, "reason": str(exc)})
    else:
        raise AssertionError("False claim accepted")
assert before == {p.relative_to(original).as_posix(): _sha(p) for p in original.rglob("*") if p.is_file()}
_write(ROOT / "docs/research/slotspe_p4_20261020/raw_negative_audit.json", {"scope": "copies of actual result metadata; no patient inference",
    "checks": checks, "original_outputs_unchanged": True, "not_added_to_synthetic_test_count": True})
print(json.dumps(checks, ensure_ascii=False, indent=2))
