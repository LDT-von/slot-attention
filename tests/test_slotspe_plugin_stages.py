"""Stage publication gates with invented metadata and stub experiment arms.

No native model is built and no patient data is accessed in these tests.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slotspe_proof import plugin_engine as engine
from slotspe_proof import plugin_queue as queue


@pytest.fixture
def staged(tmp_path, monkeypatch):
    metadata, features = tmp_path / "metadata", tmp_path / "features"
    features.mkdir()
    for relative in queue._METADATA_FILES:
        path = metadata / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("invented_metadata_only\n", encoding="utf-8")
    runtime = queue.normalize_config({"metadata_root": str(metadata),
                                      "features_root": str(features), "device": "cpu"})
    state = SimpleNamespace(config=runtime, root=tmp_path / "queue", metadata=metadata,
                            presets=[], prepare_calls=0, fail_prepare=False,
                            fail_preset=None, after_arm=None,
                            code={"invented_plugin_source.py": "original_fixture_hash"})
    prepared = {"patients": {"invented_train": {"cache_path": "prepared/cache.pt"}},
                "folds": {"0": {"train": ["invented_train"], "val": []}},
                "omic_sizes": [1], "encoding_dim": 2, "patches": 4096}

    def prepare(config, root):
        state.prepare_calls += 1
        directory = Path(root) / "prepared"
        directory.mkdir(parents=True)
        if state.fail_prepare:
            (directory / "partial_marker.txt").write_text("keep incomplete preparation\n", encoding="utf-8")
            raise RuntimeError("injected preparation failure")
        engine._dump(directory / "manifest.json", prepared)
        torch.save({"patches": torch.zeros(1, 2), "omics": [torch.zeros(1)]}, directory / "cache.pt")

    def manifest(config, root, verify_cache):
        assert verify_cache is True
        return copy.deepcopy(prepared)

    class Reference:
        def __init__(self, omic_sizes, encoding_dim):
            assert omic_sizes == [1] and encoding_dim == 2

        def fit(self, inputs):
            # The engine is allowed to read only the tiny invented cache above.
            rows = list(inputs)
            assert len(rows) == 1 and rows[0]["x_wsi"].shape == (1, 1, 2)
            return self

        def state_dict(self):
            return {"invented_reference_fitted": True}

        def load_state_dict(self, state_dict):
            assert state_dict == {"invented_reference_fitted": True}

    def run_arm(config, root, manifest, preset, initial, reference, verify_live):
        verify_live()
        state.presets.append(preset)
        assert initial == {"invented_native_state": 1}
        directory = Path(root) / "arms" / preset
        directory.mkdir(parents=True)
        if preset == state.fail_preset:
            (directory / "partial_marker.txt").write_text("keep incomplete arm\n", encoding="utf-8")
            raise RuntimeError("injected arm failure")
        summary = {"status": "completed", "preset": preset, "fixture_only": True}
        engine._dump(directory / "summary.json", summary)
        if state.after_arm is not None:
            state.after_arm(preset)
        return summary

    monkeypatch.setattr(engine, "output_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(engine, "implementation_hashes", lambda: dict(state.code))
    monkeypatch.setattr(engine.patient, "_csv_paths", lambda config: [metadata / p for p in queue._METADATA_FILES])
    monkeypatch.setattr(engine.patient, "prepare", prepare)
    monkeypatch.setattr(engine.patient, "_manifest", manifest)
    monkeypatch.setattr(engine, "ContentReference", Reference)
    monkeypatch.setattr(engine, "_build", lambda *args, **kwargs: SimpleNamespace(backbone=object()))
    monkeypatch.setattr(engine, "_cpu_state", lambda model: {"invented_native_state": 1})
    monkeypatch.setattr(engine, "_run_arm", run_arm)
    original_threads = torch.get_num_threads()
    original_backend = (torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32,
                        torch.backends.cuda.matmul.allow_tf32)
    try:
        yield state
    finally:
        torch.set_num_threads(original_threads)
        (torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32,
         torch.backends.cuda.matmul.allow_tf32) = original_backend


def read_queue(state):
    return json.loads((state.root / "plugin_queue_manifest.json").read_text(encoding="utf-8"))


def test_stages_execute_only_selected_presets_in_order_without_repreparing(staged):
    expected = []
    for stage, presets in queue.STAGES.items():
        result = engine.execute(staged.config, staged.root, stage=stage)
        expected.extend(presets)
        assert staged.presets == expected
        assert result["stage"] == stage and result["presets"] == list(presets)
        record = read_queue(staged)["stages"][stage]
        assert record["status"] == "completed"
        assert record["artifacts"] == engine._artifact_hashes(staged.root, stage)
    assert staged.presets == list(queue.PRESETS)
    assert staged.prepare_calls == 1


@pytest.mark.parametrize("stage", ["v2", "v3"])
def test_later_stage_without_fresh_v1_is_rejected_before_preparation(staged, stage):
    with pytest.raises(RuntimeError, match="start stage v1"):
        engine.execute(staged.config, staged.root, stage=stage)
    assert not staged.root.exists() and not staged.presets
    assert staged.prepare_calls == 0


def test_existing_queue_without_completed_prior_stage_is_rejected(staged):
    engine.execute(staged.config, staged.root, stage="v1")
    manifest = staged.root / "plugin_queue_manifest.json"
    record = read_queue(staged)
    record["stages"] = {}
    engine._dump(manifest, record)
    before = manifest.read_bytes()
    with pytest.raises(RuntimeError, match="previous stage missing"):
        engine.execute(staged.config, staged.root, stage="v2")
    assert manifest.read_bytes() == before
    assert staged.presets == list(queue.STAGES["v1"])


@pytest.mark.parametrize("change", ["modify", "delete"])
def test_changed_completed_arm_artifact_blocks_next_stage(staged, change):
    engine.execute(staged.config, staged.root, stage="v1")
    artifact = staged.root / "arms" / queue.STAGES["v1"][0] / "summary.json"
    if change == "modify":
        artifact.write_text("tampered invented artifact\n", encoding="utf-8")
    else:
        artifact.unlink()
    manifest = staged.root / "plugin_queue_manifest.json"
    before = manifest.read_bytes()
    with pytest.raises(RuntimeError, match="previous stage missing, incomplete or changed"):
        engine.execute(staged.config, staged.root, stage="v2")
    assert manifest.read_bytes() == before
    assert staged.presets == list(queue.STAGES["v1"])


@pytest.mark.parametrize("change", ["config", "code", "csv"])
def test_runtime_identity_drift_blocks_next_stage_without_overwriting(staged, change):
    engine.execute(staged.config, staged.root, stage="v1")
    runtime = dict(staged.config)
    if change == "config":
        alternative = staged.root.parent / "different_features"
        alternative.mkdir()
        runtime["features_root"] = str(alternative)
    elif change == "code":
        staged.code["invented_plugin_source.py"] = "changed_fixture_hash"
    else:
        (staged.metadata / queue._METADATA_FILES[0]).write_text("changed invented metadata\n", encoding="utf-8")
    manifest = staged.root / "plugin_queue_manifest.json"
    before = manifest.read_bytes()
    with pytest.raises(RuntimeError, match="code/config/metadata/runtime changed"):
        engine.execute(runtime, staged.root, stage="v2")
    assert manifest.read_bytes() == before
    assert staged.prepare_calls == 1 and staged.presets == list(queue.STAGES["v1"])


@pytest.mark.parametrize("relative", ["prepared/manifest.json", "initial.pt"])
def test_shared_initialization_or_prepared_manifest_drift_is_rejected(staged, relative):
    engine.execute(staged.config, staged.root, stage="v1")
    (staged.root / relative).write_bytes(b"changed invented shared artifact")
    manifest = staged.root / "plugin_queue_manifest.json"
    before = manifest.read_bytes()
    with pytest.raises(RuntimeError, match="preparation or shared initialization changed"):
        engine.execute(staged.config, staged.root, stage="v2")
    assert manifest.read_bytes() == before
    assert staged.presets == list(queue.STAGES["v1"])


def test_completed_stage_cannot_be_repeated(staged):
    engine.execute(staged.config, staged.root, stage="v1")
    manifest = staged.root / "plugin_queue_manifest.json"
    before = manifest.read_bytes()
    with pytest.raises(RuntimeError, match="stage already started"):
        engine.execute(staged.config, staged.root, stage="v1")
    assert manifest.read_bytes() == before
    assert staged.presets == list(queue.STAGES["v1"])


def test_preparation_failure_is_persisted_and_cannot_be_overwritten(staged):
    staged.fail_prepare = True
    with pytest.raises(RuntimeError, match="injected preparation failure"):
        engine.execute(staged.config, staged.root, stage="v1")
    record = read_queue(staged)
    failure = record["preparation_failure"]
    assert failure["status"] == "failed" and failure["type"] == "RuntimeError"
    assert failure["message"] == "injected preparation failure"
    marker = staged.root / "prepared/partial_marker.txt"
    assert marker.read_text(encoding="utf-8") == "keep incomplete preparation\n"
    manifest = staged.root / "plugin_queue_manifest.json"
    before = manifest.read_bytes()
    with pytest.raises(RuntimeError, match="prior preparation incomplete"):
        engine.execute(staged.config, staged.root, stage="v1")
    assert manifest.read_bytes() == before and marker.exists()
    assert staged.prepare_calls == 1 and not staged.presets


def test_failed_stage_retains_prior_arms_and_cannot_resume(staged):
    staged.fail_preset = queue.STAGES["v1"][1]
    with pytest.raises(RuntimeError, match="injected arm failure"):
        engine.execute(staged.config, staged.root, stage="v1")
    record = read_queue(staged)
    assert record["stages"]["v1"]["status"] == "failed"
    completed = staged.root / "arms" / queue.STAGES["v1"][0] / "summary.json"
    assert completed.exists()
    manifest = staged.root / "plugin_queue_manifest.json"
    before = manifest.read_bytes()
    with pytest.raises(RuntimeError, match="stage already started"):
        engine.execute(staged.config, staged.root, stage="v1")
    assert manifest.read_bytes() == before
    assert staged.presets == list(queue.STAGES["v1"][:2])


def test_live_implementation_drift_marks_stage_failed_before_next_arm(staged):
    staged.after_arm = lambda preset: staged.code.update({"invented_plugin_source.py": "changed_mid_stage"})
    with pytest.raises(RuntimeError, match="live implementation/metadata drift"):
        engine.execute(staged.config, staged.root, stage="v1")
    assert staged.presets == [queue.STAGES["v1"][0]]
    assert read_queue(staged)["stages"]["v1"]["status"] == "failed"
