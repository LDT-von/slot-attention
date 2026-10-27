"""Queue contracts for the best-checkpoint eleven-arm diagnostic.

These mirror the final-epoch queue's own queue tests: the plan must import no
training or data package, the stage list must cover all five folds and eleven
arms, a new run must refuse an existing non-empty output, a failure must
retain its evidence and must not resume, and the real execution path must use
the concurrent scheduler.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from slotspe_proof import redundancy_best_multifold as best
from slotspe_proof import redundancy_multifold as multifold

ARMS = list(multifold.ARMS)
BEST_SELECTION = "per_epoch_validation_best_then_reload_and_reevaluate"

METADATA_ROOT = "/data1/dataset_csv"
FEATURES_ROOT = "/data1/TCGA-UNI2-h-features/blca/uni2-h/pt_files"


def _resolved_config():
    """The queue configuration with its environment-resolved input roots."""
    config = copy.deepcopy(best.load_config())
    config["execution"] = {**config.get("execution", {}),
                            "metadata_root": METADATA_ROOT,
                            "features_root": FEATURES_ROOT}
    return config


def test_default_plan_imports_no_training_or_data_packages(tmp_path):
    output = REPO / "local_evidence" / ("generated_best_plan_test_" + uuid.uuid4().hex)
    program = (
        "import json,sys; from slotspe_proof import redundancy_best_multifold as q; "
        f"p=q.make_plan(output_root={str(output)!r}); "
        "assert not ({'torch','pandas','numpy','h5py','scipy'} & set(sys.modules)); "
        "print(json.dumps(p))"
    )
    result = subprocess.run([sys.executable, "-B", "-c", program], cwd=REPO,
                            capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["training_runs"] == 55
    assert plan["folds"] == [0, 1, 2, 3, 4]
    assert plan["seeds"] == [3]
    assert plan["training_arms"] == ARMS
    assert plan["patient_work_executed"] is False
    assert plan["reused_folds"] == []
    assert plan["checkpoint_selection"] == BEST_SELECTION
    assert not output.exists()


def test_cli_defaults_to_plan_without_creating_an_output(tmp_path):
    output = REPO / "local_evidence" / ("generated_best_cli_test_" + uuid.uuid4().hex)
    script = REPO / "scripts/run_slotspe_redundancy_best_multifold.py"
    result = subprocess.run([sys.executable, "-B", str(script), "--output-root", str(output)],
                            cwd=REPO, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["training_runs"] == 55
    assert not output.exists()


def test_stage_list_covers_all_five_folds_and_eleven_arms_each():
    stages = best.base.stages({"new_folds": [0, 1, 2, 3, 4]})
    kinds = [stage["kind"] for stage in stages]
    assert kinds.count("prepare") == 1
    assert kinds.count("initialize") == 5
    assert kinds.count("arm") == 55
    assert kinds.count("summarize_fold") == 5
    assert kinds.count("summarize") == 1
    for fold in (0, 1, 2, 3, 4):
        assert sum(1 for s in stages if s["kind"] == "arm" and s["fold"] == fold) == 11


def test_the_best_queue_never_reuses_the_final_epoch_fold0():
    """A fold trained under one checkpoint rule cannot be mixed into the
    statistic of a fold trained under another, so all five folds are retrained:
    5 folds x 11 arms = 55 new training runs."""
    assert best.NEW_FOLDS == (0, 1, 2, 3, 4)
    assert best.REUSED_FOLDS == ()
    config = best.load_config()
    scope = config["requested_scope"]
    assert scope["new_folds"] == [0, 1, 2, 3, 4]
    assert scope["reused_folds"] == []
    assert best.make_plan()["training_runs"] == 55


def test_the_best_queue_declares_the_original_checkpoint_rule():
    config = best.load_config()
    assert config["execution"]["checkpoint_selection"] == BEST_SELECTION
    assert config["checkpoint_rule"]["source"] == "third_party/SlotSPE/utils/core_utils.py:421-475"


def test_the_best_queue_pins_the_same_original_source_bytes():
    """The original source must be pinned exactly as the final-epoch queue
    pins it; a best-checkpoint rule does not license touching the original."""
    best_pins = best.load_config()["protected_original_source_sha256"]
    final_pins = multifold.load_config()["protected_original_source_sha256"]
    assert best_pins == final_pins
    for name, expected in best_pins.items():
        assert best.base.sha(REPO / name) == expected


def test_the_best_queue_budget_matches_the_final_epoch_queue():
    """Both queues must declare the same budget and hyperparameters; only the
    checkpoint rule may differ.  The input roots are environment-resolved, so
    they are compared after the same resolution."""
    import os
    environment = dict(os.environ)
    environment.setdefault("SLOTSPE_METADATA_ROOT", "/data1/dataset_csv")
    environment.setdefault("SLOTSPE_FEATURES_ROOT", "/data1/TCGA-UNI2-h-features/blca/uni2-h/pt_files")
    best_runtime = best.normalize_runtime(best.load_config(),
                                          environment["SLOTSPE_METADATA_ROOT"],
                                          environment["SLOTSPE_FEATURES_ROOT"])
    final_runtime = multifold.normalize_runtime(multifold.load_config(),
                                                environment["SLOTSPE_METADATA_ROOT"],
                                                environment["SLOTSPE_FEATURES_ROOT"])
    for key in ("epochs", "patches", "batch_size", "eval_repeats", "lr", "probe_lr",
                "projection_dim", "slot_iters", "content_target_dim", "content_clusters",
                "content_reference_seed", "dias_distance", "slotbert_temperature",
                "splits_dir", "label_col", "signature", "torch_threads", "device",
                "metadata_root", "features_root", "source_root"):
        assert best_runtime[key] == final_runtime[key], key
    assert best_runtime["checkpoint_selection"] == BEST_SELECTION


def test_expected_artifacts_include_the_best_checkpoint_and_epoch_record():
    stage = {"id": "arm_fold_2_S0", "kind": "arm", "fold": 2, "arm": "S0"}
    names = best.expected_artifacts(stage)
    assert "folds/2/arms/S0/best.pt" in names
    assert "folds/2/arms/S0/epoch_validation.json" in names
    for name in names:
        assert name.startswith("folds/2/arms/S0/")


def test_new_run_refuses_existing_nonempty_output():
    config = _resolved_config()
    output = REPO / "local_evidence" / ("generated_best_nonempty_" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    (output / "retain.txt").write_text("keep this evidence", encoding="utf-8")
    with pytest.raises((FileExistsError, RuntimeError)):
        best.execute(config, output, executor=lambda *a, **k: None, runtime_check=False)
    assert (output / "retain.txt").read_text(encoding="utf-8") == "keep this evidence"


def test_failure_retains_partial_evidence_and_does_not_resume():
    config = _resolved_config()
    output = REPO / "local_evidence" / ("generated_best_fail_" + uuid.uuid4().hex)
    calls = []

    def fail(stage, runtime, root):
        calls.append(stage["id"])
        (root / "synthetic_failure.txt").write_text("retained generated failure", encoding="utf-8")
        raise RuntimeError("deliberate generated failure")

    with pytest.raises(RuntimeError, match="deliberate"):
        best.execute(config, output, executor=fail, runtime_check=False)
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="incomplete|failed|partial|resume|drift"):
        best.execute(config, output, resume=True, executor=lambda *a, **k: None, runtime_check=False)
    assert len(calls) == 1


def test_the_disk_budget_accounts_for_the_cache_the_arms_and_the_headroom():
    """A full disk truncates a checkpoint, so the budget must be computed
    from the measured sizes of this cohort, not guessed."""
    budget = best._estimate_disk_budget()
    assert budget["prepared_cache_bytes"] > 30 * 1024 ** 3
    assert budget["arm_bytes"] == 11 * 5 * best.ARM_BYTES
    assert budget["fold_initial_bytes"] == 5 * best.FOLD_INITIAL_BYTES
    assert budget["required_bytes"] == (budget["prepared_cache_bytes"] + budget["arm_bytes"]
                                        + budget["fold_initial_bytes"] + best.DISK_HEADROOM_BYTES)
    assert budget["required_bytes"] < 100 * 1024 ** 3


def test_the_disk_budget_uses_the_manifest_patch_count_when_it_exists():
    """Before preparation the estimate uses the measured cohort figure; after
    preparation it must use the manifest's own patch count."""
    manifest = {"encoding_dim": 1536,
                "patients": {"p1": {"full_patch_count": 1000}, "p2": {"full_patch_count": 2000}}}
    budget = best._estimate_disk_budget(manifest)
    assert budget["patches"] == 3000
    assert budget["patients"] == 2
    assert budget["prepared_cache_bytes"] == 3000 * 1536 * 4


def test_a_full_filesystem_refuses_to_start_before_writing_anything(monkeypatch):
    """The guard must run before the preparation stage, so a refused run
    leaves no half-built preparation and no truncated checkpoint."""
    output = REPO / "local_evidence" / ("generated_best_diskfull_" + uuid.uuid4().hex)
    monkeypatch.setattr(best, "_free_bytes", lambda path: 1)
    with pytest.raises(RuntimeError, match="Insufficient disk"):
        best.execute(_resolved_config(), output, executor=lambda *a, **k: None, runtime_check=False)
    assert not output.exists()


def test_the_disk_guard_reports_the_output_path_and_the_free_space():
    output = REPO / "local_evidence" / ("generated_best_diskinfo_" + uuid.uuid4().hex)
    budget = best.check_disk_budget(output)
    assert budget["required_bytes"] > 0
    assert output not in [Path("/tmp")]


def test_the_free_space_probe_works_on_a_directory_that_does_not_exist_yet():
    """The output directory is created by the queue, so the first budget check
    runs on a path that is not there yet."""
    missing = REPO / "local_evidence" / ("generated_best_missing_" + uuid.uuid4().hex) / "deeper"
    assert not missing.exists()
    assert best._free_bytes(missing) > 0


def test_the_doctor_reports_the_disk_budget_before_execution():
    # The doctor must be given a configuration whose input roots are resolved:
    # an unresolved ${...} placeholder is a configuration error, and the
    # budget is computed from the real output path, not from a template.
    result = best.doctor(_resolved_config(), runtime_check=False)
    assert "disk_budget" in result
    assert result["disk_budget"]["output_root"].startswith(str(REPO / "local_evidence"))
    assert result["disk_budget"]["sufficient"] in (True, False)


def test_the_real_execution_path_uses_the_concurrent_scheduler(monkeypatch):
    """Without a custom executor the queue must run the concurrent wave."""
    calls = []

    def fake_run_wave(order, pending_ids, runtime, root, gpu_pool, workers):
        calls.append({"order": [stage["id"] for stage in order],
                      "pending": sorted(pending_ids),
                      "gpu_pool": list(gpu_pool),
                      "workers": workers})
        for stage in order:
            for name in best.expected_artifacts(stage):
                path = Path(root) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("generated", encoding="utf-8")

    monkeypatch.setattr(best, "_run_wave", fake_run_wave)
    monkeypatch.setattr(best, "verify_config_pins", lambda *a, **k: None)
    monkeypatch.setattr(best, "implementation_hashes", lambda: {"engine.py": "b" * 64})
    monkeypatch.setattr(best, "doctor", lambda *a, **k: {
        "status": "ready", "source_and_metadata_sha256": {}, "runtime": {}})
    config = _resolved_config()
    output = REPO / "local_evidence" / ("generated_best_concurrent_" + uuid.uuid4().hex)
    best.execute(config, output, runtime_check=False, devices=["0", "1"], workers=6)
    assert calls, "the concurrent scheduler must be used when no executor is given"
    assert calls[0]["gpu_pool"] == ["0", "1"]
    assert calls[0]["workers"] == 6
    assert "prepare" in calls[0]["order"]


def test_the_worker_spawner_names_this_module():
    """An arm stage must train under the best rule; the shared spawner of the
    final-epoch queue would train it under the final-epoch rule instead."""
    source = (REPO / "slotspe_proof" / "redundancy_best_multifold.py").read_text(encoding="utf-8")
    assert "slotspe_proof.redundancy_best_multifold" in source
    assert "-m\", \"slotspe_proof.redundancy_multifold" not in source
    assert "--worker-stage" in source


def test_the_best_summary_covers_all_five_folds_without_reuse():
    """The summary must state the five-fold scope, the 55-run count and the
    selection boundary rather than silently mixing protocols or dropping a
    fold."""
    source = (REPO / "slotspe_proof" / "redundancy_best_multifold.py").read_text(encoding="utf-8")
    assert "validation-selected peak" in source
    assert "no_fold_is_reused_from_the_final_epoch_queue" in source
    assert "cindex_five_fold" in source
    assert "cindex_four_fold" not in source
    assert "folds 1-4" not in source
    result = best.make_plan()
    assert result["reused_folds"] == []
    assert result["training_runs"] == 55
    assert result["relationship_to_final_epoch_queue"].startswith("separate protocol")
