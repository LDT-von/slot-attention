"""Original-only server contracts, using invented cases and tensors."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_clean_audit_fixtures", ROOT / "tests/test_slotspe_proof_queue.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def test_original_audit_plan_has_no_repair_or_abc(tmp_path):
    from survot_rank.research.slotspe_proof import queue
    config = fixtures.configuration(tmp_path, experiment="original_audit")
    plan = queue.make_plan(config, tmp_path / "out")
    assert plan["training_arms"] == ["original"] and plan["training_runs"] == 1
    assert plan["same_weights_eval_controls"] == 0
    assert [s["kind"] for s in plan["stages"]] == ["protocol", "prepare", "pair", "summarize"]
    required = queue._stage_files(tmp_path, {"kind": "pair", "fold": 0, "seed": 3}, config)
    assert any(p.name == "native_model_audit.json" for p in required)
    assert not any("readout_fix" in str(p) or "same_weights_eval_fix" in str(p) for p in required)


def test_clean_delivery_runs_native_and_offline_audits(tmp_path):
    torch = pytest.importorskip("torch")
    archive = tmp_path / "clean.zip"
    built = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/build_slotspe_proof_delivery.py"),
        "--original-only", "--clean-source-root", str(ROOT / "third_party/SlotSPE"), "--output", str(archive)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert built.returncode == 0, built.stdout + built.stderr
    with zipfile.ZipFile(archive) as z:
        manifest_name = next(n for n in z.namelist() if n.endswith("/DELIVERY_MANIFEST.json"))
        delivery = json.loads(z.read(manifest_name))
        assert delivery["requested_server_reproduction"]["training_runs"] == 1
        z.extractall(tmp_path / "standalone")
    task = tmp_path / "standalone" / delivery["prefix"].rstrip("/")
    default_plan = subprocess.run([sys.executable, "-B", str(task / "scripts/run_slotspe_proof.py"), "--plan"],
        cwd=task, capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
    safe_plan = json.loads(default_plan.stdout)
    assert safe_plan["training_runs"] == 1 and safe_plan["config"]["epochs"] == 3
    assert safe_plan["config"]["patches"] == 256 and safe_plan["training_arms"] == ["original"]
    fixtures.metadata_fixture(tmp_path, rna=True)
    features = tmp_path / "features"
    features.mkdir()
    generator = torch.Generator().manual_seed(931)
    for i in range(7):
        torch.save(torch.randn(10, 16, generator=generator), features / f"SYNTHETIC_CASE_{i}.pt")
    config = fixtures.configuration(tmp_path, experiment="original_audit", source_root=str(task / "third_party/SlotSPE"))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / "results"
    run = subprocess.run([sys.executable, "-B", str(task / "scripts/run_slotspe_proof.py"),
        "--config", str(profile), "--run", "--output-root", str(out)],
        cwd=task, capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert run.returncode == 0, run.stdout + run.stderr
    pair = json.loads((out / "pairs/fold_0/seed_3/summary.json").read_text())
    assert [r["arm"] for r in pair["arms"]] == ["original"]
    assert pair["same_weights_eval_fix"] is None
    assert not (out / "pairs/fold_0/seed_3/readout_fix").exists()
    native = json.loads((out / "pairs/fold_0/seed_3/original/native_model_audit.json").read_text())
    assert all(native["purity"]["checks"].values())
    assert native["checkpoint_state_unchanged_by_diagnostics"]
    assert native["rng_replay"]["within_1e_minus_6"]
    gate = json.loads((out / "pairs/fold_0/seed_3/original/natural_gate_observations.json").read_text())
    assert gate["observations"] == 2 * 2 * 2
    accepted = subprocess.run([sys.executable, "-B", str(task / "scripts/audit_slotspe_return.py"),
        "--root", str(out), "--delivery", str(archive)], cwd=task,
        capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    # Absence of the proposed natural gate failure must be accepted as a result.
    assert gate["near_zero_gate"] in ("observed", "not_observed")
    from survot_rank.research.slotspe_proof.acceptance import audit_return
    artifact = out / "pairs/fold_0/seed_3/original/native_model_audit.json"
    receipt = out / "receipts/fold_0_seed_3.json"
    original_bytes, receipt_bytes = artifact.read_bytes(), receipt.read_bytes()
    for tamper, message in (("padding", "coverage differs"), ("rng", "RNG replay")):
        value = json.loads(original_bytes)
        if tamper == "padding":
            value["padding"]["records"][0]["patient_id"] = "WRONG_PATIENT"
        else:
            value["rng_replay"]["eval_seed"] += 1
        artifact.write_text(json.dumps(value), encoding="utf-8")
        updated = json.loads(receipt_bytes)
        updated["outputs"][str(artifact.relative_to(out))] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        receipt.write_text(json.dumps(updated), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            audit_return(out)
        artifact.write_bytes(original_bytes)
        receipt.write_bytes(receipt_bytes)
    artifact.write_text(artifact.read_text().replace('"status": "measured"', '"status": "altered"'))
    with pytest.raises(ValueError, match="hash differs"):
        audit_return(out)
