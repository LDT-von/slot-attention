"""Data-free queue contract checks, with invented summaries only."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from slotspe_proof import diversity_queue as queue


def config():
    return queue.read(queue.DEFAULT_CONFIG)


def test_default_plan_never_reads_reference_or_imports_torch(monkeypatch):
    monkeypatch.setattr(queue, "audit", lambda *args: pytest.fail("plan audited private inputs"))
    plan = queue.make_plan(config())
    assert plan["new_training_runs"] == 20
    assert plan["reused_arm_fold_results"] == 15
    assert {s["arm"] for s in plan["stages"]} == set(queue.NEW_ARMS)
    assert not {"S0", "R4", "R3"} & {s["arm"] for s in plan["stages"]}
    assert not plan["patient_work_executed"]
    result = subprocess.run([sys.executable, "-B", "-c",
        "import sys; from slotspe_proof.diversity_queue import make_plan,read,DEFAULT_CONFIG; "
        "make_plan(read(DEFAULT_CONFIG)); assert 'torch' not in sys.modules"],
        cwd=queue.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name,value", [("cosine_margin", 1), ("cosine_margin", -0.1),
    ("read_overlap_margin", float("nan")), ("weight", -1), ("weight", True),
    ("warmup_epochs", 1.2), ("warmup_epochs", -1), ("warmup_epochs", True)])
def test_invalid_loss_parameters_are_rejected(name, value):
    current = config()
    current["diversity"][name] = value
    with pytest.raises(ValueError):
        queue.make_plan(current)


@pytest.mark.parametrize("change", [
    {"folds": [0]}, {"new_arms": ["SCM", "SRO", "S0"]}, {"reuse_arms": ["S0"]},
    {"content_weight": 4}, {"devices": ["cpu"]}, {"devices": ["cuda:0", "cuda:0"]}])
def test_no_implicit_scope_expansion_or_reference_training(change):
    current = config()
    current.update(change)
    with pytest.raises(ValueError):
        queue.make_plan(current)


def test_new_output_cannot_overlap_reference_or_leave_private_root():
    current = config()
    for output in (current["reference_root"], current["reference_root"] + "/new", "docs/output"):
        current["output_root"] = output
        with pytest.raises(ValueError):
            queue.make_plan(current)


def test_audit_failure_does_not_create_output_or_retrain(monkeypatch, tmp_path):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    current = config_from_disk()
    current["output_root"] = "local_evidence/new"
    def fail(*_):
        raise RuntimeError("reference mismatch")
    monkeypatch.setattr(queue, "audit", fail)
    with pytest.raises(RuntimeError, match="reference mismatch"):
        queue.execute(current)
    assert not (tmp_path / "local_evidence/new").exists()


def config_from_disk():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/slotspe_diversity_queue.json").read_text(encoding="utf-8"))


def test_runtime_retains_split_and_preparation_paths():
    receipt = {"runtime": {"source_root": "old/source", "metadata_root": "old/meta",
                           "splits_dir": "5fold", "variants": ["S0", "R4"]}}
    original = copy.deepcopy(receipt)
    runtime = queue._runtime(config(), receipt, "cuda:1")
    assert runtime["source_root"] == "old/source"
    assert runtime["splits_dir"] == "5fold"
    assert runtime["variants"] == list(queue.NEW_ARMS)
    assert receipt == original


def test_identity_pins_reference_runtime_and_paths(monkeypatch):
    monkeypatch.setattr(queue, "implementation_hashes", lambda: {"x.py": "x"})
    receipt = {"runtime": {"epochs": 30}, "reference_root": "old", "manifest_path": "old/manifest.json",
               "file_hashes": {"a": "abc"}, "expected_initial_state_hashes": {"0": {"core": "c", "probe": "p"}}}
    first = queue._identity(config(), receipt)[0]
    receipt["runtime"]["epochs"] = 1
    assert queue._identity(config(), receipt)[0] != first


def test_worker_requires_explicit_active_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    root = tmp_path / "local_evidence/new"
    queue.write(root / "queue_manifest.json", {"status": "completed", "records": {}})
    with pytest.raises(RuntimeError, match="active"):
        queue.worker(root, "fold0_SCM", "cuda:0")


def test_missing_folds_cannot_be_called_five_fold_summary():
    with pytest.raises(RuntimeError):
        queue._stats([0.7] * 4)
    with pytest.raises(RuntimeError):
        queue._stats([0.7] * 4 + [None])


def test_summary_keeps_reference_origin_and_geometry(monkeypatch, tmp_path):
    current = config()
    old, new = tmp_path / "old", tmp_path / "new"
    for fold in range(5):
        for index, arm in enumerate(queue.REUSE_ARMS + queue.NEW_ARMS):
            root = old if arm in queue.REUSE_ARMS else new
            folder = root / "folds" / str(fold) / "arms" / arm
            queue.write(folder / "summary.json", {"status": "completed", "best_epoch": 7,
                "evaluation_summary": {"val_cindex_mean_exploratory": 0.6 + index * 0.01 + fold * 0.001,
                                       "repeats": 20},
                "slot_diagnostics": {m: {"cosine": 0.9, "read_overlap": 0.8, "slot_count": 8,
                                          "effective_rank": 1.5} for m in ("wsi", "omic")}})
            queue.write(folder / "curve.json", [])
    result = queue.summarize(new, current, {"reference_root": str(old), "runtime": {"splits_dir": "5fold"}})
    assert result["new_training_runs"] == 20
    assert result["per_arm"]["S0"]["result_origin"] == "reused_verified_reference"
    assert result["per_arm"]["SCM"]["geometry"]["omic"]["read_overlap"]["mean"] == 0.8
    assert result["paired_contrasts"]["R4_SCM_minus_R4"]["fold_mean"] == pytest.approx(0.04)


def _fake_execution(monkeypatch, tmp_path, fail=False):
    """Exercise the real scheduler with zero model/data work."""
    from slotspe_proof import patient
    current = config_from_disk()
    current["output_root"] = "local_evidence/new"
    current["reference_root"] = "local_evidence/old"
    current["devices"] = ["cuda:0", "cuda:1"]
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    old = tmp_path / "local_evidence/old"
    old.mkdir(parents=True)
    receipt = {"runtime": {"splits_dir": "5fold", "source_root": "unused"},
               "reference_root": str(old), "manifest_path": str(old / "prepared/manifest.json"),
               "file_hashes": {}, "checks": {}, "expected_initial_state_hashes": {}}
    monkeypatch.setattr(queue, "audit", lambda _: receipt)
    monkeypatch.setattr(queue, "implementation_hashes", lambda: {"generated": "fixed"})
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    verified = []
    monkeypatch.setattr(patient, "_manifest", lambda *a, **k: verified.append(k["verify_cache"]))
    monkeypatch.setattr(queue, "summarize", lambda *a: {"status": "completed"})
    spawned = []
    class FakeProcess:
        returncode = None
        def __init__(self, command, **_):
            self.stage = command[command.index("--worker-stage") + 1]
            spawned.append(self.stage)
            self.root = Path(command[command.index("--output-root") + 1])
        def poll(self):
            if self.returncode is None:
                self.returncode = 1 if fail else 0
                if not fail:
                    first, arm = self.stage.split("_", 1)
                    folder = self.root / "folds" / first[4:] / "arms" / arm
                    for name in ("summary.json", "curve.json", "eval_repeats.json", "epoch_validation.json"):
                        queue.write(folder / name, {"status": "completed"})
                    for name in ("last.pt", "best.pt", "diagnostics.pt", "validation_predictions.csv"):
                        (folder / name).write_bytes(b"invented-placeholder-no-patient-tensors")
            return self.returncode
        def terminate(self):
            self.returncode = -1
        def wait(self):
            return self.returncode
    monkeypatch.setattr(queue.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(queue.time, "sleep", lambda _: None)
    return current, spawned, verified


def test_scheduler_runs_only_new_arms_and_verified_resume_skips_them(monkeypatch, tmp_path):
    current, spawned, verified = _fake_execution(monkeypatch, tmp_path)
    assert queue.execute(current)["status"] == "completed"
    assert len(spawned) == 20
    assert not any(name.endswith(("_S0", "_R4", "_R3")) for name in spawned)
    assert verified == [True]
    root = tmp_path / current["output_root"]
    assert not (root / ".queue.lock").exists()
    assert queue.execute(current, resume=True)["status"] == "completed"
    assert len(spawned) == 20
    (root / "folds/0/arms/SCM/best.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="artifacts changed"):
        queue.execute(current, resume=True)
    assert len(spawned) == 20


def test_scheduler_failure_keeps_receipt_and_releases_lock(monkeypatch, tmp_path):
    current, spawned, _ = _fake_execution(monkeypatch, tmp_path, fail=True)
    with pytest.raises(RuntimeError, match="failed"):
        queue.execute(current)
    root = tmp_path / current["output_root"]
    assert queue.read(root / "queue_manifest.json")["status"] == "failed"
    assert not (root / ".queue.lock").exists()
    assert 1 <= len(spawned) <= 2
    first_logs = {p.name: p.read_bytes() for p in (root / "logs").glob("*.log")}
    with pytest.raises(RuntimeError, match="failed"):
        queue.execute(current, resume=True)
    assert (root / "logs/fold0_SCM.attempt2.log").exists()
    for name, data in first_logs.items():
        assert (root / "logs" / name).read_bytes() == data


def test_scheduler_checks_disk_before_launch(monkeypatch, tmp_path):
    current, spawned, _ = _fake_execution(monkeypatch, tmp_path)
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match="disk"):
        queue.execute(current)
    assert spawned == []
