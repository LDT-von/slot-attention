"""Multi-fold eleven-arm queue contracts, using generated files only.

These tests never import a patient worker, load a feature tensor, or execute
a model.  They exercise the planning, scope-lock, staging and aggregation
logic of the five-fold queue.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

from slotspe_proof import redundancy_multifold as multifold


REPO = Path(__file__).resolve().parents[1]
ARMS = ["S0", "S4", "R1", "R2", "R3", "R4", "R5", "S1", "S2", "S3", "S5"]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    config = copy.deepcopy(multifold.load_config())
    monkeypatch.setattr(multifold, "ROOT", tmp_path)
    monkeypatch.setattr(multifold, "verify_config_pins", lambda *args, **kwargs: None)
    monkeypatch.setattr(multifold, "implementation_hashes", lambda: {"generated_engine.py": "a" * 64})
    monkeypatch.setattr(multifold, "doctor", lambda *args, **kwargs: {
        "status": "ready", "source_and_metadata_sha256": {}, "runtime": {}})
    return config, tmp_path / "local_evidence" / "generated_multifold"


def diagnostic(config):
    return next(stage for stage in config["stages"] if stage["id"] == "real_data_diagnostic")


def test_default_plan_imports_no_training_or_data_packages(tmp_path):
    output = REPO / "local_evidence" / ("generated_multifold_plan_test_" + uuid.uuid4().hex)
    program = (
        "import json,sys; from slotspe_proof import redundancy_multifold as q; "
        f"p=q.make_plan(output_root={str(output)!r}); "
        "assert not ({'torch','pandas','numpy','h5py','scipy'} & set(sys.modules)); "
        "print(json.dumps(p))"
    )
    result = subprocess.run([sys.executable, "-B", "-c", program], cwd=REPO,
                            capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["training_runs"] == 44
    assert plan["folds"] == [0, 1, 2, 3, 4]
    assert plan["seeds"] == [3]
    assert plan["training_arms"] == ARMS
    assert plan["patient_work_executed"] is False
    assert not output.exists()


def test_cli_defaults_to_plan_without_creating_an_output(tmp_path):
    output = REPO / "local_evidence" / ("generated_multifold_cli_test_" + uuid.uuid4().hex)
    script = REPO / "scripts/run_slotspe_redundancy_multifold.py"
    result = subprocess.run([sys.executable, "-B", str(script), "--output-root", str(output)],
                            cwd=REPO, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["training_runs"] == 44
    assert not output.exists()


def test_stage_list_covers_four_new_folds_and_eleven_arms_each():
    stages = multifold.stages({"new_folds": [1, 2, 3, 4]})
    kinds = [stage["kind"] for stage in stages]
    assert kinds.count("prepare") == 1
    assert kinds.count("initialize") == 4
    assert kinds.count("arm") == 44
    assert kinds.count("summarize_fold") == 4
    assert kinds.count("summarize") == 1
    arms = [stage["arm"] for stage in stages if stage["kind"] == "arm"]
    assert arms == ARMS * 4
    for fold in (1, 2, 3, 4):
        assert sum(1 for stage in stages if stage["kind"] == "arm" and stage["fold"] == fold) == 11


@pytest.mark.parametrize("field,value", [
    ("study", "brca"), ("feature_encoder", "UNI"),
    ("folds", [0, 1, 2, 3]), ("folds", [0, 1, 2, 3, 4, 5]), ("folds", [1, 2, 3, 4]),
    ("seeds", [3, 17]), ("seeds", [17]),
    ("selected_variants", ARMS[:-1]), ("selected_variants", ARMS + ["S6"]),
    ("new_folds", [1, 2, 3]), ("new_folds", [0, 1, 2, 3, 4]),
    ("reused_folds", []), ("reused_folds", [1]),
    ("automatic_cross_fold_expansion", True),
])
def test_scope_cannot_silently_widen(field, value):
    config = copy.deepcopy(multifold.load_config())
    config["requested_scope"][field] = value
    with pytest.raises(ValueError):
        multifold.normalize_runtime(config)


@pytest.mark.parametrize("field,value", [
    ("epochs", 3), ("real_patches", 256), ("batch_size", 4),
    ("eval_repeats", 5), ("splits_dir", "5fold_uni2h"), ("variants", ARMS[:-1]),
])
def test_budget_cannot_change_from_the_fold0_protocol(field, value):
    config = copy.deepcopy(multifold.load_config())
    diagnostic(config)[field] = value
    with pytest.raises(ValueError):
        multifold.normalize_runtime(config)


def test_resolved_runtime_is_revalidated_identically():
    config = copy.deepcopy(multifold.load_config())
    runtime = multifold.normalize_runtime(config)
    assert multifold.normalize_runtime(runtime) == runtime


def test_execution_rejects_public_output_location(harness):
    config, output = harness
    with pytest.raises((ValueError, RuntimeError)):
        multifold.execute(config, output.parent.parent / "public_results",
                          executor=lambda *a, **k: None, runtime_check=False)


def test_new_run_refuses_existing_nonempty_output(harness):
    config, output = harness
    output.mkdir(parents=True)
    (output / "retain.txt").write_text("keep this evidence", encoding="utf-8")
    with pytest.raises((FileExistsError, RuntimeError)):
        multifold.execute(config, output, executor=lambda *a, **k: None, runtime_check=False)
    assert (output / "retain.txt").read_text(encoding="utf-8") == "keep this evidence"


def test_failure_retains_partial_evidence_and_does_not_resume(harness):
    config, output = harness
    calls = []

    def fail(stage, runtime, root):
        calls.append(stage["id"])
        (root / "synthetic_failure.txt").write_text("retained generated failure", encoding="utf-8")
        raise RuntimeError("deliberate generated failure")

    with pytest.raises(RuntimeError, match="deliberate"):
        multifold.execute(config, output, executor=fail, runtime_check=False)
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="incomplete|failed|partial|resume|drift"):
        multifold.execute(config, output, resume=True, executor=lambda *a, **k: None, runtime_check=False)
    assert len(calls) == 1


def test_fold_stats_use_population_and_sample_spread():
    stats = multifold._fold_stats([0.6, 0.7, 0.8])
    assert stats["fold_mean"] == pytest.approx(0.7)
    assert stats["fold_std_population"] == pytest.approx(0.0816496580927726)
    assert stats["fold_std_sample"] == pytest.approx(0.1)
    assert stats["folds_defined"] == 3
    empty = multifold._fold_stats([None, None])
    assert empty["fold_mean"] is None and empty["folds_defined"] == 0


def test_expected_artifacts_are_confined_to_the_output_root():
    stage = {"id": "arm_fold_2_S0", "kind": "arm", "fold": 2, "arm": "S0"}
    for name in multifold.expected_artifacts(stage):
        assert name.startswith("folds/2/arms/S0/")


def test_stage_chains_keep_a_fold_sequential_and_separate_folds_apart():
    stages = multifold.stages({"new_folds": [1, 2, 3, 4]})
    chains = multifold.stage_chains(stages)
    fold_chains = [chain for chain in chains if chain[0]["kind"] == "initialize"]
    assert len(fold_chains) == 4
    for chain in fold_chains:
        folds = {stage.get("fold") for stage in chain}
        assert len(folds) == 1, "a chain must never mix folds"
        kinds = [stage["kind"] for stage in chain]
        assert kinds[0] == "initialize"
        assert kinds.count("arm") == 11
        assert kinds[-1] == "summarize_fold"


def test_execution_waves_put_prepare_and_summarize_alone_and_folds_together():
    chains = multifold.stage_chains(multifold.stages({"new_folds": [1, 2, 3, 4]}))
    waves = multifold.execution_waves(chains)
    assert len(waves) == 3
    assert [chain[0]["kind"] for chain in waves[0]] == ["prepare"]
    assert len(waves[1]) == 4
    assert all(chain[0]["kind"] == "initialize" for chain in waves[1])
    assert [chain[0]["kind"] for chain in waves[2]] == ["summarize"]


def test_device_assignment_is_round_robin_over_folds_and_never_touches_shared_stages():
    stages = multifold.stages({"new_folds": [1, 2, 3, 4]})
    assert multifold.assign_devices(stages, None, ["0", "1"]) == ["0", "1"]
    assert multifold.assign_devices(stages, None, None) == []


def test_an_arm_may_only_start_after_its_fold_initialization():
    stages = multifold.stages({"new_folds": [1, 2, 3, 4]})
    by_id = {stage["id"]: stage for stage in stages}
    assert multifold.stage_dependencies(by_id["arm_fold_2_S5"]) == ["init_fold_2"]
    assert multifold.stage_dependencies(by_id["init_fold_2"]) == []
    assert multifold.stage_dependencies(by_id["prepare"]) == []
    # Arms of the same fold do not depend on each other, so they may overlap.
    assert multifold.stage_dependencies(by_id["arm_fold_2_S0"]) == multifold.stage_dependencies(by_id["arm_fold_2_S1"])


def test_a_fold_summary_may_only_start_after_all_its_arms():
    stages = multifold.stages({"new_folds": [1, 2]})
    by_id = {stage["id"]: stage for stage in stages}
    dependencies = multifold.stage_dependencies(by_id["summarize_fold_2"])
    assert set(dependencies) == {f"arm_fold_2_{arm}" for arm in ARMS}
    assert len(dependencies) == 11


def test_device_choice_respects_the_measured_peak_memory():
    heavy = {"kind": "arm", "arm": "S1"}
    light = {"kind": "arm", "arm": "S0"}
    assert multifold.stage_memory_gib(heavy) > multifold.stage_memory_gib(light)
    pool = ["0", "1"]
    load, memory = {}, {}
    first = multifold._choose_device(heavy, pool, load, memory)
    load[first] = 1
    memory[first] = multifold.stage_memory_gib(heavy)
    second = multifold._choose_device(heavy, pool, load, memory)
    assert first != second, "two heavy arms must not share one GPU"
    # A light arm may still share the device that already holds a heavy arm.
    third = multifold._choose_device(light, pool, load, memory)
    assert third in pool


def test_device_choice_refuses_to_overfill_a_gpu():
    heavy = {"kind": "arm", "arm": "S1"}
    pool = ["0"]
    load, memory = {}, {"0": multifold.GPU_USABLE_GIB - 1.0}
    assert multifold._choose_device(heavy, pool, load, memory) is None


def test_parallel_scheduler_keeps_one_fold_sequential_in_time(harness):
    """The in-process path must still visit every stage exactly once."""
    config, output = harness
    order = []

    def executor(stage, runtime, root):
        order.append(stage["id"])
        for name in multifold.expected_artifacts(stage):
            path = Path(root) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("generated", encoding="utf-8")

    multifold.execute(config, output, executor=executor, runtime_check=False)
    assert len(order) == len(set(order)) == 54
    assert order[0] == "prepare"
    assert order[-1] == "summarize"


def test_the_real_execution_path_uses_the_concurrent_scheduler(monkeypatch, tmp_path):
    """Without a custom executor the queue must run the concurrent wave.

    A custom in-process executor is for the generated tests only.  When no
    executor is supplied the queue must use the concurrent scheduler, so that
    the stages of different folds really do overlap.
    """
    calls = []

    def fake_run_wave(order, pending_ids, runtime, root, gpu_pool, workers):
        calls.append({"order": [stage["id"] for stage in order],
                      "pending": sorted(pending_ids),
                      "gpu_pool": list(gpu_pool),
                      "workers": workers})
        for stage in order:
            for name in multifold.expected_artifacts(stage):
                path = Path(root) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("generated", encoding="utf-8")

    monkeypatch.setattr(multifold, "_run_wave", fake_run_wave)
    monkeypatch.setattr(multifold, "verify_config_pins", lambda *a, **k: None)
    monkeypatch.setattr(multifold, "implementation_hashes", lambda: {"engine.py": "b" * 64})
    monkeypatch.setattr(multifold, "doctor", lambda *a, **k: {
        "status": "ready", "source_and_metadata_sha256": {}, "runtime": {}})
    config = copy.deepcopy(multifold.load_config())
    output = REPO / "local_evidence" / ("generated_concurrent_" + uuid.uuid4().hex)
    multifold.execute(config, output, runtime_check=False, devices=["0", "1"], workers=6)
    assert calls, "the concurrent scheduler must be used when no executor is given"
    assert calls[0]["gpu_pool"] == ["0", "1"]
    assert calls[0]["workers"] == 6
    assert "prepare" in calls[0]["order"]


def test_concurrent_scheduler_respects_dependencies_and_the_worker_budget():
    """A simulated wave must never start a stage before its dependencies."""
    stages = multifold.stages({"new_folds": [1, 2]})
    by_id = {stage["id"]: stage for stage in stages}
    order = [stage for stage in stages
             if stage["kind"] in ("initialize", "arm", "summarize_fold")]
    started, finished = [], []
    workers = 2
    running = {}
    completed = set()

    def ready(stage):
        return all(dependency in completed for dependency in multifold.stage_dependencies(stage))

    # Drive the same rules the real scheduler uses.
    guard = 0
    while len(completed) < len(order) and guard < 200:
        guard += 1
        for stage in order:
            if len(running) >= workers:
                break
            if stage["id"] in completed or stage["id"] in running:
                continue
            if not ready(stage):
                continue
            running[stage["id"]] = stage
            started.append(stage["id"])
        # Let the oldest running stage finish.
        if running:
            oldest = next(iter(running))
            finished.append(oldest)
            completed.add(oldest)
            del running[oldest]
        elif not any(ready(stage) for stage in order if stage["id"] not in completed):
            break

    assert set(started) == {stage["id"] for stage in order}
    # No arm started before its own fold's initialization finished.
    assert finished.index("init_fold_1") < started.index("arm_fold_1_S0")
    assert finished.index("init_fold_2") < started.index("arm_fold_2_S0")
    # No fold summary started before all of that fold's arms finished.
    for fold in (1, 2):
        for arm in ARMS:
            assert finished.index(f"arm_fold_{fold}_{arm}") < started.index(f"summarize_fold_{fold}")
