"""Eleven-arm queue contracts, using generated files and injected executors only.

These tests never import a patient worker, load features, or execute a model.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

from survot_rank.research.slotspe_proof import redundancy_queue as queue


REPO = Path(__file__).resolve().parents[1]
ARMS = ["S0", "S4", "R1", "R2", "R3", "R4", "R5", "S1", "S2", "S3", "S5"]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    config = copy.deepcopy(queue.load_config())
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(queue, "verify_config_pins", lambda *args, **kwargs: None)
    monkeypatch.setattr(queue, "implementation_hashes", lambda: {"generated_engine.py": "a" * 64})
    monkeypatch.setattr(queue, "doctor", lambda *args, **kwargs: {"status": "ready"})
    return config, tmp_path / "local_evidence" / "generated_queue"


def writer(calls):
    def execute(stage, runtime, root):
        calls.append(stage["id"])
        for relative in queue.expected_artifacts(stage):
            path = root / relative
            assert path.resolve().is_relative_to(root.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"synthetic": True, "stage": stage["id"]}), encoding="utf-8")
    return execute


def diagnostic(config):
    return next(stage for stage in config["stages"] if stage["id"] == "real_data_diagnostic")


def test_default_plan_imports_no_training_or_data_packages(tmp_path):
    output = REPO / "local_evidence" / ("generated_plan_test_" + uuid.uuid4().hex)
    program = (
        "import json,sys; from survot_rank.research.slotspe_proof import redundancy_queue as q; "
        f"p=q.make_plan(output_root={str(output)!r}); "
        "assert not ({'torch','pandas','numpy','h5py'} & set(sys.modules)); "
        "print(json.dumps(p))"
    )
    result = subprocess.run([sys.executable, "-B", "-c", program], cwd=REPO,
                            capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["training_runs"] == 11
    assert [stage["arm"] for stage in plan["stages"] if stage["kind"] == "arm"] == ARMS
    assert not output.exists()


def test_cli_defaults_to_plan_without_creating_an_output(tmp_path):
    output = REPO / "local_evidence" / ("generated_cli_test_" + uuid.uuid4().hex)
    script = REPO / "scripts/run_slotspe_redundancy.py"
    result = subprocess.run([sys.executable, "-B", str(script), "--output-root", str(output)],
                            cwd=REPO, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["training_runs"] == 11
    assert not output.exists()


@pytest.mark.parametrize("field,value", [
    ("study", "brca"), ("feature_encoder", "UNI"), ("folds", [1]),
    ("folds", [0, 1]), ("seeds", [17]), ("seeds", [3, 17]),
    ("automatic_cross_fold_expansion", True),
    ("selected_variants", ARMS[:-1]),
    ("selected_variants", ARMS + ["S6"]),
    ("selected_variants", ARMS + ["S0"]),
])
def test_scope_is_locked_to_requested_eleven_arm_blca_uni2h_fold0_seed3(field, value):
    config = copy.deepcopy(queue.load_config())
    config["requested_scope"][field] = value
    with pytest.raises(ValueError):
        queue.normalize_runtime(config)


@pytest.mark.parametrize("field,value", [
    ("study", "brca"), ("feature_encoder", "another_encoder"),
    ("folds", [0, 1]), ("seeds", [3, 17]), ("variants", ARMS[:-1]),
    ("epochs", 0), ("real_patches", True), ("batch_size", 0),
    ("eval_repeats", 1),
])
def test_executable_stage_cannot_silently_widen_scope_or_use_invalid_budget(field, value):
    config = copy.deepcopy(queue.load_config())
    diagnostic(config)[field] = value
    with pytest.raises(ValueError):
        queue.normalize_runtime(config)


def test_execution_rejects_public_output_location(harness):
    config, output = harness
    calls = []
    with pytest.raises((ValueError, RuntimeError)):
        queue.execute(config, output.parent.parent / "public_results", executor=writer(calls), runtime_check=False)
    assert calls == []


def test_preflight_failure_precedes_output_creation_and_workers(harness, monkeypatch):
    config, output = harness
    calls = []
    def missing_inputs(*args, **kwargs):
        raise ValueError("unresolved synthetic metadata root")
    monkeypatch.setattr(queue, "doctor", missing_inputs)
    with pytest.raises(ValueError, match="unresolved"):
        queue.execute(config, output, executor=writer(calls), runtime_check=False)
    assert calls == []
    assert not output.exists()


def test_execution_and_resume_require_completed_artifact_hashes(harness):
    config, output = harness
    calls = []
    queue.execute(config, output, executor=writer(calls), runtime_check=False)
    assert calls == [stage["id"] for stage in queue.make_plan(config, output)["stages"]]
    assert len(calls) == 14
    calls.clear()
    queue.execute(config, output, resume=True, executor=writer(calls), runtime_check=False)
    assert calls == []
    plan = queue.make_plan(config, output)
    relative = next(iter(queue.expected_artifacts(plan["stages"][0])))
    (output / relative).write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed|hash|integrity"):
        queue.execute(config, output, resume=True, executor=writer(calls), runtime_check=False)
    assert calls == []


def test_new_run_refuses_existing_nonempty_output(harness):
    config, output = harness
    output.mkdir(parents=True)
    retained = output / "retain.txt"
    retained.write_text("keep this evidence", encoding="utf-8")
    with pytest.raises((FileExistsError, RuntimeError)):
        queue.execute(config, output, executor=writer([]), runtime_check=False)
    assert retained.read_text(encoding="utf-8") == "keep this evidence"


def test_resume_refuses_changed_configuration(harness):
    config, output = harness
    queue.execute(config, output, executor=writer([]), runtime_check=False)
    modified = copy.deepcopy(config)
    diagnostic(modified)["epochs"] += 1
    calls = []
    with pytest.raises(RuntimeError, match="changed|configuration|identity|drift"):
        queue.execute(modified, output, resume=True, executor=writer(calls), runtime_check=False)
    assert calls == []


def test_resume_refuses_changed_implementation(harness, monkeypatch):
    config, output = harness
    queue.execute(config, output, executor=writer([]), runtime_check=False)
    monkeypatch.setattr(queue, "implementation_hashes", lambda: {"generated_engine.py": "b" * 64})
    calls = []
    with pytest.raises(RuntimeError, match="changed|implementation|identity|drift"):
        queue.execute(config, output, resume=True, executor=writer(calls), runtime_check=False)
    assert calls == []


def test_resume_refuses_changed_metadata_hashes(harness, monkeypatch):
    config, output = harness
    fixture = output.parent.parent / "generated_metadata.csv"
    fixture.write_text("invented_input,version_a\n", encoding="utf-8")
    monkeypatch.setattr(queue, "doctor", lambda *args, **kwargs: {
        "status": "ready", "source_and_metadata_sha256": {str(fixture): queue.sha(fixture)}})
    queue.execute(config, output, executor=writer([]), runtime_check=False)
    fixture.write_text("invented_input,version_b\n", encoding="utf-8")
    calls = []
    with pytest.raises(RuntimeError, match="changed|metadata|identity|drift"):
        queue.execute(config, output, resume=True, executor=writer(calls), runtime_check=False)
    assert calls == []


def test_midstage_implementation_edit_cannot_receive_completion_receipt(harness, monkeypatch):
    config, output = harness
    pins = {"generated_engine.py": "a" * 64}
    monkeypatch.setattr(queue, "implementation_hashes", lambda: dict(pins))
    calls = []
    write_stage = writer(calls)
    def changed(stage, runtime, root):
        write_stage(stage, runtime, root)
        pins["generated_engine.py"] = "b" * 64
    with pytest.raises(RuntimeError, match="implementation changed"):
        queue.execute(config, output, executor=changed, runtime_check=False)
    manifest = json.loads((output / "queue_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["records"][calls[0]]["status"] == "failed"
    assert len(calls) == 1


def test_midstage_saved_configuration_edit_cannot_receive_completion_receipt(harness):
    config, output = harness
    calls = []
    write_stage = writer(calls)
    def changed(stage, runtime, root):
        write_stage(stage, runtime, root)
        path = root / "resolved_config.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["epochs"] += 1
        path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(RuntimeError, match="configuration|runtime|changed"):
        queue.execute(config, output, executor=changed, runtime_check=False)
    manifest = json.loads((output / "queue_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["records"][calls[0]]["status"] == "failed"
    assert len(calls) == 1


def test_midstage_metadata_edit_cannot_receive_completion_receipt(harness, monkeypatch):
    config, output = harness
    fixture = output.parent.parent / "generated_metadata.csv"
    fixture.write_text("invented_input,version_a\n", encoding="utf-8")
    monkeypatch.setattr(queue, "doctor", lambda *args, **kwargs: {
        "status": "ready", "source_and_metadata_sha256": {str(fixture): queue.sha(fixture)}})
    calls = []
    write_stage = writer(calls)
    def changed(stage, runtime, root):
        write_stage(stage, runtime, root)
        fixture.write_text("invented_input,version_b\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="metadata changed"):
        queue.execute(config, output, executor=changed, runtime_check=False)
    manifest = json.loads((output / "queue_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["records"][calls[0]]["status"] == "failed"
    assert len(calls) == 1


def test_resume_refuses_edited_saved_runtime(harness):
    config, output = harness
    queue.execute(config, output, executor=writer([]), runtime_check=False)
    path = output / "resolved_config.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["epochs"] += 1
    path.write_text(json.dumps(saved), encoding="utf-8")
    calls = []
    with pytest.raises(RuntimeError, match="configuration|drift"):
        queue.execute(config, output, resume=True, executor=writer(calls), runtime_check=False)
    assert calls == []


def test_extra_cache_file_is_also_bound_by_completed_stage_receipt(harness):
    config, output = harness
    calls = []
    write_stage = writer(calls)
    def with_extra(stage, runtime, root):
        write_stage(stage, runtime, root)
        if stage["kind"] == "prepare":
            (root / "prepared/generated_cache.bin").write_bytes(b"generated fixture cache")
    queue.execute(config, output, executor=with_extra, runtime_check=False)
    (output / "prepared/generated_cache.bin").write_bytes(b"changed fixture cache")
    calls.clear()
    with pytest.raises(RuntimeError, match="artifacts changed"):
        queue.execute(config, output, resume=True, executor=writer(calls), runtime_check=False)
    assert calls == []


def test_failure_retains_partial_evidence_and_does_not_continue_or_resume(harness):
    config, output = harness
    calls = []
    def fail(stage, runtime, root):
        calls.append(stage["id"])
        (root / "synthetic_failure.txt").write_text("retained synthetic failure", encoding="utf-8")
        raise RuntimeError("deliberate generated failure")
    with pytest.raises(RuntimeError, match="deliberate"):
        queue.execute(config, output, executor=fail, runtime_check=False)
    assert len(calls) == 1
    assert (output / "synthetic_failure.txt").is_file()
    with pytest.raises(RuntimeError, match="incomplete|failed|partial|resume"):
        queue.execute(config, output, resume=True, executor=writer(calls), runtime_check=False)
    assert len(calls) == 1


def test_missing_declared_output_cannot_receive_success(harness):
    config, output = harness
    calls = []
    def incomplete(stage, runtime, root):
        calls.append(stage["id"])
    with pytest.raises(RuntimeError, match="missing|artifact|output"):
        queue.execute(config, output, executor=incomplete, runtime_check=False)
    assert len(calls) == 1
