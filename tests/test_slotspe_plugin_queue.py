"""Plugin plans and explicit CLI gates; no patient or model execution."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from slotspe_proof import plugin_queue as queue


def config():
    return json.loads(queue.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def metadata_fixture(tmp_path):
    metadata, features = tmp_path / "metadata", tmp_path / "features"
    features.mkdir()
    for relative in queue._METADATA_FILES:
        path = metadata / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("invented_fixture_metadata\n", encoding="utf-8")
    return metadata, features


def test_default_plan_locks_all_nine_runs_but_only_v1_is_selected(monkeypatch):
    monkeypatch.delenv("SLOTSPE_METADATA_ROOT", raising=False)
    monkeypatch.delenv("SLOTSPE_FEATURES_ROOT", raising=False)
    plan = queue.make_plan(config())
    assert plan["stage"] == "v1"
    assert plan["stage_presets"] == list(queue.PRESETS[:5])
    assert plan["total_training_runs"] == 9 and plan["stage_training_runs"] == 5
    assert plan["total_training_epoch_budget"] == 270
    assert plan["stage_training_epoch_budget"] == 150
    assert plan["config"]["metadata_root"] == "${SLOTSPE_METADATA_ROOT}"
    assert plan["unresolved_environment_variables"] == ["${SLOTSPE_FEATURES_ROOT}", "${SLOTSPE_METADATA_ROOT}"]
    assert not plan["patient_work_executed"] and not plan["later_stages_run_automatically"]
    assert plan["checkpoint_policy"] == {"initial_best": 0.0, "comparison": ">=", "tie": "latest"}


@pytest.mark.parametrize("key,value", [
    ("epochs", 3), ("patches", 256), ("folds", [0, 1]), ("seeds", [4]),
    ("folds", [False]), ("epochs", 30.), ("batch_size", 1), ("eval_repeats", 1),
    ("splits_dir", "5fold"), ("lr", 0.001), ("native_alpha", 0.1),
    ("native_reconstruction_weight", 0.2), ("gradient_clip_norm", 1.),
    ("checkpoint_selection", "last"), ("validation_patch_policy", "subsample"),
    ("projection_dim", 8), ("slot_iters", 2), ("feature_encoder", "other"),
    ("encoding_dim", 16), ("torch_threads", 4), ("unknown_flag", True),
])
def test_invalid_or_shrunken_patient_protocol_is_rejected(key, value):
    raw = config()
    raw[key] = value
    with pytest.raises(ValueError):
        queue.normalize_config(raw)


def test_source_is_fixed_and_input_variables_resolve_without_loading(monkeypatch, tmp_path):
    monkeypatch.setenv("SLOTSPE_METADATA_ROOT", str(tmp_path / "metadata"))
    normalized = queue.normalize_config(config(), features_root=tmp_path / "features", device="cpu")
    assert Path(normalized["source_root"]) == queue.SOURCE_ROOT.resolve()
    assert normalized["metadata_root"] == str((tmp_path / "metadata").resolve())
    raw = config()
    raw["source_root"] = str(tmp_path / "different_source")
    with pytest.raises(ValueError, match="fixed"):
        queue.normalize_config(raw)
    normalized["folds"].append(1)
    assert queue.normalize_config(config())["folds"] == [0]


def test_require_inputs_checks_only_file_existence(tmp_path, monkeypatch):
    metadata, features = metadata_fixture(tmp_path)
    monkeypatch.setattr(importlib, "import_module", lambda *args: pytest.fail("input validation must not import runtimes"))
    normalized = queue.normalize_config(config(), metadata_root=metadata, features_root=features,
                                        require_inputs=True)
    assert normalized["patches"] == 4096
    (metadata / queue._METADATA_FILES[-1]).unlink()
    with pytest.raises(ValueError, match="required metadata file"):
        queue.normalize_config(config(), metadata_root=metadata, features_root=features, require_inputs=True)


def test_unresolved_inputs_fail_only_when_required(monkeypatch):
    monkeypatch.delenv("SLOTSPE_METADATA_ROOT", raising=False)
    monkeypatch.delenv("SLOTSPE_FEATURES_ROOT", raising=False)
    assert queue.normalize_config(config())["metadata_root"].startswith("${")
    with pytest.raises(ValueError, match="unresolved"):
        queue.normalize_config(config(), require_inputs=True)


@pytest.mark.parametrize("stage,groups,prior", [("v1", 5, []), ("v2", 2, ["v1"]), ("v3", 2, ["v1", "v2"])])
def test_stage_plan_never_automatically_expands(stage, groups, prior):
    plan = queue.make_plan(config(), stage=stage)
    assert plan["stage_training_runs"] == groups
    assert plan["required_prior_stages"] == prior


@pytest.mark.parametrize("path", [".", "local_evidence", "public_results/new", "local_evidence/../../outside"])
def test_output_rejects_public_or_escaping_paths(path):
    with pytest.raises(ValueError):
        queue.output_path(path)


def test_new_output_plan_does_not_create_a_directory():
    output = queue.REPO / "local_evidence/plugin_queue_uncreated_unit_test_destination"
    assert not output.exists()
    assert queue.make_plan(config(), output)["output_root"] == str(output.resolve())
    assert not output.exists()


def test_cli_default_plan_imports_no_torch_pandas_or_engine():
    code = ("import sys; from slotspe_proof.plugin_queue import main; "
            "assert main([])==0; assert 'torch' not in sys.modules; "
            "assert 'pandas' not in sys.modules; assert 'slotspe_proof.plugin_engine' not in sys.modules")
    completed = subprocess.run([sys.executable, "-B", "-c", code], cwd=queue.REPO,
                               capture_output=True, text=True, encoding="utf-8", check=True)
    assert json.loads(completed.stdout)["mode"] == "plan"


def test_script_default_plan_is_safe():
    completed = subprocess.run([sys.executable, "-B", str(queue.REPO / "scripts/run_slotspe_plugins.py")],
                               cwd=queue.REPO, capture_output=True, text=True, encoding="utf-8", check=True)
    assert json.loads(completed.stdout)["patient_work_executed"] is False


def test_explicit_run_alone_reaches_engine_with_locked_config(tmp_path, monkeypatch, capsys):
    metadata, features = metadata_fixture(tmp_path)
    calls = []
    stub = SimpleNamespace(execute=lambda runtime, output, stage: calls.append((runtime, output, stage)) or {"stub": True})
    monkeypatch.setattr(importlib, "import_module", lambda name: stub if name == "slotspe_proof.plugin_engine" else pytest.fail(name))
    arguments = ["--metadata-root", str(metadata), "--features-root", str(features), "--device", "cpu"]
    assert queue.main(arguments) == 0 and not calls
    assert queue.main(["--run", "--stage", "v2", *arguments]) == 0
    assert len(calls) == 1 and calls[0][2] == "v2"
    assert calls[0][0]["epochs"] == 30 and calls[0][0]["patches"] == 4096


def test_generated_check_needs_no_server_inputs_or_config(monkeypatch):
    calls = []
    stub = SimpleNamespace(generated_smoke=lambda output: calls.append(output) or {"generated": True})
    monkeypatch.setattr(importlib, "import_module", lambda name: stub)
    assert queue.main(["--check-generated", "--config", "nonexistent_configuration.json"]) == 0
    assert calls == [queue.DEFAULT_GENERATED_OUTPUT.resolve()]
    assert queue.DEFAULT_GENERATED_OUTPUT != queue.DEFAULT_OUTPUT


def test_run_rejects_missing_inputs_before_importing_engine(monkeypatch):
    monkeypatch.delenv("SLOTSPE_METADATA_ROOT", raising=False)
    monkeypatch.delenv("SLOTSPE_FEATURES_ROOT", raising=False)
    monkeypatch.setattr(importlib, "import_module", lambda name: pytest.fail("must reject before engine import"))
    assert queue.main(["--run"]) == 1


def test_doctor_hashes_metadata_but_does_not_construct_models_or_load_arrays(tmp_path, monkeypatch):
    metadata, features = metadata_fixture(tmp_path)
    runtime = queue.normalize_config(config(), metadata_root=metadata, features_root=features, device="cpu")
    fake_torch = SimpleNamespace(__version__="fixture", version=SimpleNamespace(cuda=None),
                                 cuda=SimpleNamespace(is_available=lambda: False))
    imported = []
    monkeypatch.setattr(importlib, "import_module", lambda name: imported.append(name) or fake_torch)
    report = queue.doctor(runtime)
    assert imported == ["torch"]
    assert len(report["metadata_files"]) == 4
    assert all(len(row["sha256"]) == 64 for row in report["metadata_files"])
    assert not report["patient_arrays_loaded"] and not report["models_constructed"]
