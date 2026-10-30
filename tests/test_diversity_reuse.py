"""Metadata-only generated fixtures; no tensors or model execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from slotspe_proof import diversity_reuse as reuse


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _edit(path, change):
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    _write(path, value)


@pytest.fixture
def reference(tmp_path):
    repo = tmp_path / "repo"
    root = repo / "local_evidence/reference"
    source = repo / "third_party/SlotSPE"
    metadata = tmp_path / "metadata"
    for name in ("models/SlotSPE.py", "dataset/dataset_survival.py", "utils/loss_func.py"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# generated source fixture\n", encoding="utf-8")
    implementation = {}
    for name in reuse.REQUIRED_IMPLEMENTATION:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# generated implementation fixture\n", encoding="utf-8")
        implementation[name] = _hash(path)
    csv_names = ["clinical/all/blca.csv", "raw_rna_data_inter/blca_rna_inter.csv",
                 "signatures/combine_signatures.csv"]
    csv_names += [f"splits/5fold/blca/fold_{fold}.csv" for fold in reuse.FOLDS]
    csv_hashes = {}
    for name in csv_names:
        path = metadata / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("invented,metadata\n", encoding="utf-8")
        csv_hashes[str(path)] = _hash(path)
    runtime = {**reuse.EXPECTED_RUNTIME, "variants": list(reuse.ALL_ARMS),
               "new_folds": list(reuse.FOLDS), "reused_folds": [],
               "source_root": str(source), "metadata_root": str(metadata),
               "features_root": str(tmp_path / "features"), "label_col": "survival_months_dss",
               "signature": "combine", "encoding_dim": None, "splits_dir": "5fold"}
    _write(root / "resolved_config.json", runtime)
    source_hashes = {p.relative_to(source).as_posix(): _hash(p) for p in source.rglob("*.py")}
    records = {}

    def record(stage, names):
        records[stage] = {"status": "completed", "artifact_sha256": {
            name: _hash(root / name) for name in names}}

    patients = {}
    for index in range(10):
        pid = f"invented_{index}"
        relative = f"prepared/cache/patients/{index}.pt"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Intentionally not a serialized torch object. Metadata audit must not
        # deserialize this or any of the fake checkpoints below.
        path.write_bytes(f"fake cache bytes {index}".encode())
        patients[pid] = {"cache_path": relative, "cache_sha256": _hash(path),
                         "full_patch_count": 5000}
    folds = {}
    for fold in reuse.FOLDS:
        ids = list(patients)
        val = ids[fold * 2:fold * 2 + 2]
        folds[str(fold)] = {"train": [pid for pid in ids if pid not in val], "val": val}
    preparation = {key: runtime[key] for key in reuse.PREPARATION_KEYS}
    preparation["splits_dir"] = "5fold"
    manifest = {"schema_version": 2, "source_hashes": source_hashes,
                "preparation_config": preparation, "patches": 4096, "encoding_dim": 1536,
                "input_csv_sha256": csv_hashes, "patients": patients,
                "folds": folds, "omic_sizes": [4, 5]}
    _write(root / "prepared/manifest.json", manifest)
    record("prepare", ["prepared/manifest.json"] + [p["cache_path"] for p in patients.values()])
    per_arm = {arm: {"status": "completed", "cindex_per_fold": {}}
               for arm in reuse.ALL_ARMS}
    stages = [{"id": "prepare", "kind": "prepare"}]
    for fold in reuse.FOLDS:
        initial = f"folds/{fold}/initial"
        file_hashes = {}
        for name in ("core.pt", "probe.pt", "reference.pt"):
            path = root / initial / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fake initial {fold} {name}".encode())
            file_hashes[name] = _hash(path)
        _write(root / initial / "receipt.json", {"fold": fold, "training_examples": 8,
               "reference_fit_split": "train_only", "frozen_during_evaluation": True,
               "files_sha256": file_hashes})
        stage_id = f"init_fold_{fold}"
        stages.append({"id": stage_id, "kind": "initialize", "fold": fold})
        record(stage_id, [initial + "/" + name for name in [*file_hashes, "receipt.json"]])
        fold_arms = {}
        for arm in reuse.ALL_ARMS:
            prefix = f"folds/{fold}/arms/{arm}"
            contract = {"adapter": "S0", "gram_weight": 0.0,
                        "slotbert_weight": 1.0 if arm == "R3" else 0.0,
                        "content_weight": 0.0 if arm == "S0" else 5.0,
                        "slotbert_temperature": 0.5}
            protocol = {"epochs": 30, "patches": 4096, "batch_size": 8,
                        "checkpoint_selection": reuse.BEST_SELECTION,
                        "source_hashes": source_hashes, "contract": contract,
                        "initial_core_hash": "a" * 64, "initial_probe_hash": "b" * 64,
                        "initial_stream_seed": 3 * 1_000_003 + fold * 7919 + 13}
            score = 0.65 + fold * .01
            evaluation = {"repeats": 20, "defined_cindex_repeats": 20,
                          "val_cindex_mean_exploratory": score}
            best = root / prefix / "best.pt"
            best.parent.mkdir(parents=True, exist_ok=True)
            best.write_bytes(b"not a checkpoint")
            value = {"status": "completed", "arm": arm, "fold": fold, "seed": 3,
                     "epochs": 30, "source_unchanged": True, "training_cases": 8,
                     "validation_cases": 2, "protocol": protocol,
                     "evaluation_summary": evaluation, "best_epoch": 10,
                     "best_checkpoint_sha256": _hash(best),
                     "epoch_validation": [{"epoch": i, "cindex": score} for i in range(1, 31)],
                     "slot_diagnostics": {"wsi": {"cosine": 0.9}, "omic": {"cosine": .8}}}
            _write(root / prefix / "summary.json", value)
            # Invalid JSON intentionally: audit hashes these input-derived
            # diagnostic files and does not read their per-patient arrays.
            for name in ("eval_repeats.json", "curve.json", "epoch_validation.json"):
                (root / prefix / name).write_bytes(b"hash only, no patient array parsing")
            stage_id = f"arm_fold_{fold}_{arm}"
            stages.append({"id": stage_id, "kind": "arm", "fold": fold, "arm": arm})
            record(stage_id, [prefix + "/" + name for name in
                             ("summary.json", "best.pt", "eval_repeats.json", "curve.json", "epoch_validation.json")])
            per_arm[arm]["cindex_per_fold"][str(fold)] = score
            fold_arms[arm] = {"cindex_mean_over_repeats": score}
        relative = f"folds/{fold}/summary.json"
        _write(root / relative, {"status": "completed", "arms": fold_arms})
        stage_id = f"summarize_fold_{fold}"
        stages.append({"id": stage_id, "kind": "summarize_fold", "fold": fold})
        record(stage_id, [relative])
    summary = {"schema": "slotspe_native_redundancy_best_multifold_summary_v1", "status": "completed",
               "scope": {"study": "blca", "feature_encoder": "UNI2-h", "folds": list(reuse.FOLDS),
                         "seed": 3, "arms": list(reuse.ALL_ARMS), "executed_folds": list(reuse.FOLDS),
                         "reused_folds": []},
               "protocol": {key: runtime[key] for key in ("epochs", "patches", "batch_size", "eval_repeats",
                            "splits_dir", "lr", "probe_lr", "projection_dim", "slot_iters", "content_target_dim",
                            "content_clusters", "checkpoint_selection")}, "per_arm": per_arm}
    _write(root / "summary.json", summary)
    stages.append({"id": "summarize", "kind": "summarize"})
    record("summarize", ["summary.json"])
    identity = {"runtime": runtime, "implementation": implementation, "stages": stages, "inputs": csv_hashes}
    _write(root / "queue_manifest.json", {
        "schema": "slotspe_native_redundancy_best_multifold_queue_v1", "status": "completed",
        "config": runtime, "identity_payload": identity,
        "identity": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        "stages": stages, "records": records})
    return root, repo


def _audit(reference, **kwargs):
    root, repo = reference
    return reuse.audit_reference(root, repo_root=repo, **kwargs)


def test_complete_reference_is_reused_without_loading_tensors(reference):
    root, _ = reference
    before = {str(p): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    receipt = _audit(reference)
    assert receipt["checks"]["reused_training_runs"] == 15
    assert receipt["checks"]["completed_reference_stages"] == 67
    assert receipt["checks"]["checkpoint_deserializations"] == 0
    assert receipt["checks"]["cache_content_hashes_verified"] is False
    assert receipt["splits_dir"] == "5fold"  # Inherit actual old family, do not guess UNI2-h.
    assert receipt["expected_initial_state_hashes"]["0"] == {"core": "a" * 64, "probe": "b" * 64}
    assert receipt["reused_results"]["R4"]["4"]["cindex"] == pytest.approx(.69)
    assert receipt["file_hashes"]["folds/0/arms/R4/curve.json"] == _hash(root / "folds/0/arms/R4/curve.json")
    assert before == {str(p): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    public = json.dumps(reuse.public_receipt(receipt))
    assert "invented_" not in public and str(root) not in public


@pytest.mark.parametrize("relative", ["prepared/manifest.json", "folds/0/initial/core.pt",
                                     "folds/2/arms/R4/curve.json", "folds/3/arms/R3/eval_repeats.json"])
def test_artifact_drift_refuses_reuse(reference, relative):
    root, _ = reference
    path = root / relative
    # Whitespace preserves JSON syntax where the artifact is JSON metadata.
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="artifact hash"):
        _audit(reference)


def test_no_baseline_retraining_fallback_when_reference_is_missing(tmp_path):
    with pytest.raises(RuntimeError, match="root is absent"):
        reuse.audit_reference(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


def test_changed_implementation_refuses_reuse(reference):
    _, repo = reference
    (repo / "slotspe_proof/patient.py").write_text("changed\n")
    with pytest.raises(RuntimeError, match="historical implementation"):
        _audit(reference)


def test_changed_source_refuses_reuse(reference):
    _, repo = reference
    (repo / "third_party/SlotSPE/models/SlotSPE.py").write_text("changed\n")
    with pytest.raises(RuntimeError, match="protected original source"):
        _audit(reference)


def test_requested_split_must_match_actual_split(reference):
    with pytest.raises(RuntimeError, match="requested runtime splits_dir"):
        _audit(reference, expected_runtime={"splits_dir": "5fold_uni2h"})


def test_changed_metadata_refuses_reuse(reference):
    root, _ = reference
    runtime = json.loads((root / "resolved_config.json").read_text())
    (Path(runtime["metadata_root"]) / "splits/5fold/blca/fold_0.csv").write_text("changed\n")
    with pytest.raises(RuntimeError, match="clinical/RNA/signature/split hashes"):
        _audit(reference)


def test_cache_content_verification_is_explicitly_deferred(reference):
    root, _ = reference
    (root / "prepared/cache/patients/0.pt").write_bytes(b"changed cache content")
    receipt = _audit(reference)
    assert receipt["checks"]["cache_content_hashes_verified"] is False
    assert receipt["cache_metadata"]["prepared/cache/patients/0.pt"]["size_bytes"] == 21


def test_optional_checkpoint_byte_hashing(reference):
    root, _ = reference
    _audit(reference, hash_checkpoints=True)
    (root / "folds/0/arms/R4/best.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="artifact hash"):
        _audit(reference, hash_checkpoints=True)


def test_incomplete_stage_refuses_reuse(reference):
    root, _ = reference
    _edit(root / "queue_manifest.json", lambda q: q["records"]["arm_fold_0_S0"].update(status="failed"))
    with pytest.raises(RuntimeError, match="stage is incomplete"):
        _audit(reference)


def test_module_import_is_standard_library_only():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-B", "-c",
        "import sys; from slotspe_proof import diversity_reuse; "
        "assert not ({'torch','numpy','pandas','scipy','h5py'} & set(sys.modules))"],
        cwd=repo, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
