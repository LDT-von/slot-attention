"""Read-only provenance gate for reusing the completed best-checkpoint queue.

Only JSON metadata and file bytes for hashes are read. This module imports no
model/data package and never deserializes a checkpoint or a patient tensor.
The returned receipt is private: it contains paths and prepared patient IDs.
Use ``public_receipt`` for a path/identifier-free console or public report.
Full prepared-cache hashing is deferred to ``patient._manifest(...,
verify_cache=True)`` in an explicitly authorized execution, before any model
is loaded. An audit mismatch never falls back to retraining a baseline.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
BEST_SELECTION = "per_epoch_validation_best_then_reload_and_reevaluate"
ALL_ARMS = ("S0", "S4", "R1", "R2", "R3", "R4", "R5", "S1", "S2", "S3", "S5")
REUSED_ARMS = ("S0", "R4", "R3")
FOLDS = (0, 1, 2, 3, 4)
EXPECTED_RUNTIME = {
    "study": "blca", "feature_encoder": "UNI2-h", "folds": list(FOLDS),
    "seeds": [3], "epochs": 30, "patches": 4096, "batch_size": 8,
    "eval_repeats": 20, "lr": 0.0005, "probe_lr": 0.0005,
    "projection_dim": 256, "slot_iters": 10, "content_target_dim": 32,
    "content_clusters": 8, "content_reference_seed": 731,
    "dias_distance": 0.05, "slotbert_temperature": 0.5,
    "checkpoint_selection": BEST_SELECTION,
}
# Files whose bytes determine the reused model/data/optimizer/evaluation path.
# Extra files in the old identity are also checked; absence of a core dependency
# cannot make a hand-written or incomplete identity look like a valid receipt.
REQUIRED_IMPLEMENTATION = (
    "slotspe_proof/patient.py", "slotspe_proof/redundancy_engine.py",
    "slotspe_proof/redundancy_best_engine.py",
    "slotspe_proof/redundancy_content.py", "slotspe_proof/native_allocation.py",
    "slotspe_proof/slot_regularization.py", "slotspe_proof/assignment_router.py",
)
PREPARATION_KEYS = ("source_root", "metadata_root", "features_root", "study",
                    "label_col", "signature", "folds", "patches", "encoding_dim")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _require(condition, message):
    if not condition:
        raise RuntimeError("reference reuse audit: " + message)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _inside(root, relative):
    relative = Path(relative)
    _require(not relative.is_absolute() and ".." not in relative.parts,
             "artifact path must be relative and contained")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root), "artifact path escapes its root")
    _require(path.is_file(), "required artifact is absent: " + relative.as_posix())
    return path


def _same(value, expected, label):
    _require(value == expected, label + " differs")


def _score(value, label):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and 0 <= value <= 1,
             label + " must be a finite C-index")
    return float(value)


def _close(value, expected, label):
    _require(math.isclose(_score(value, label), _score(expected, label),
                         rel_tol=0.0, abs_tol=1e-12), label + " differs")


def _stage_ids():
    result = ["prepare"]
    for fold in FOLDS:
        result.append(f"init_fold_{fold}")
        result.extend(f"arm_fold_{fold}_{arm}" for arm in ALL_ARMS)
        result.append(f"summarize_fold_{fold}")
    return [*result, "summarize"]


def audit_reference(reference_root, expected_runtime=None, repo_root=None, *,
                    hash_checkpoints=False):
    """Validate and return a private reuse receipt, or fail without mutation.

    The old split directory is inherited, never guessed from the encoder name.
    ``expected_runtime`` adds exact comparisons against the original resolved
    runtime; it cannot weaken the fixed study/budget/single-seed contract.
    Full cache tensors, predictions and training curves are not parsed.
    """
    root = Path(reference_root).resolve()
    repo = Path(repo_root or ROOT).resolve()
    _require(root.is_dir(), "completed reference root is absent")
    files = {}

    def read_file(relative):
        path = _inside(root, relative)
        files[relative] = sha256(path)
        return _read(path)

    runtime = read_file("resolved_config.json")
    queue = read_file("queue_manifest.json")
    summary = read_file("summary.json")
    manifest = read_file("prepared/manifest.json")
    for key, value in EXPECTED_RUNTIME.items():
        _same(runtime.get(key), value, "runtime " + key)
    for key, value in (expected_runtime or {}).items():
        _same(runtime.get(key), value, "requested runtime " + key)
    _same(runtime.get("variants"), list(ALL_ARMS), "original arm list")
    _same(runtime.get("new_folds"), list(FOLDS), "original new folds")
    _same(runtime.get("reused_folds"), [], "original reused folds")
    splits = runtime.get("splits_dir")
    _require(isinstance(splits, str) and re.fullmatch(r"[A-Za-z0-9_-]+", splits),
             "explicit split directory is required")
    _same(queue.get("schema"), "slotspe_native_redundancy_best_multifold_queue_v1", "queue schema")
    _same(queue.get("status"), "completed", "queue completion")
    _same(queue.get("config"), runtime, "queue runtime")
    identity = queue.get("identity_payload", {})
    _same(identity.get("runtime"), runtime, "identity runtime")
    _same(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
          queue.get("identity"), "queue identity digest")
    stages = queue.get("stages", [])
    expected_stages = _stage_ids()
    _same([stage.get("id") for stage in stages], expected_stages, "67-stage graph")
    _same(identity.get("stages"), stages, "identity stages")
    records = queue.get("records", {})
    _same(set(records), set(expected_stages), "completed stage coverage")
    _require(all(record.get("status") == "completed" for record in records.values()),
             "a stage is incomplete")
    artifact_hashes = {}
    for record in records.values():
        hashes = record.get("artifact_sha256", {})
        _require(isinstance(hashes, dict) and hashes, "missing stage artifact hashes")
        for name, digest in hashes.items():
            _require(_digest(digest), "invalid artifact digest")
            if name in artifact_hashes:
                _same(artifact_hashes[name], digest, "duplicate artifact digest")
            artifact_hashes[name] = digest

    def verify_file(relative, *, hash_bytes=True):
        path = _inside(root, relative)
        expected = artifact_hashes.get(relative)
        _require(_digest(expected), "missing recorded artifact hash: " + relative)
        if hash_bytes:
            actual = files.get(relative) or sha256(path)
            _same(actual, expected, "artifact hash " + relative)
            files[relative] = actual
        return path

    _same(summary.get("schema"), "slotspe_native_redundancy_best_multifold_summary_v1", "summary schema")
    _same(summary.get("status"), "completed", "summary completion")
    scope = summary.get("scope", {})
    for key, value in {"study": "blca", "feature_encoder": "UNI2-h",
                       "folds": list(FOLDS), "seed": 3,
                       "arms": list(ALL_ARMS), "executed_folds": list(FOLDS),
                       "reused_folds": []}.items():
        _same(scope.get(key), value, "summary scope " + key)
    for key in ("epochs", "patches", "batch_size", "eval_repeats", "splits_dir",
                "lr", "probe_lr", "projection_dim", "slot_iters", "content_target_dim",
                "content_clusters", "checkpoint_selection"):
        _same(summary.get("protocol", {}).get(key), runtime[key], "summary protocol " + key)
    verify_file("summary.json")
    verify_file("prepared/manifest.json")

    implementation = identity.get("implementation", {})
    _require(isinstance(implementation, dict) and set(REQUIRED_IMPLEMENTATION) <= set(implementation),
             "historical implementation identity is incomplete")
    for name, digest in implementation.items():
        _require(_digest(digest), "invalid historical implementation digest")
        _same(sha256(_inside(repo, name)), digest, "historical implementation " + name)
    source = (repo / "third_party/SlotSPE").resolve()
    _same(Path(runtime["source_root"]).resolve(), source,
          "source checkout (relocation requires an explicit provenance mapping)")
    source_files = {p.relative_to(source).as_posix(): sha256(p)
                    for directory in ("models", "utils", "dataset")
                    for p in sorted((source / directory).rglob("*.py"))
                    if "__pycache__" not in p.parts}
    _require("models/SlotSPE.py" in source_files and "dataset/dataset_survival.py" in source_files,
             "original source is incomplete")
    _same(manifest.get("source_hashes"), source_files, "protected original source")
    _same(manifest.get("schema_version"), 2, "prepared schema")
    preparation = {key: runtime[key] for key in PREPARATION_KEYS}
    preparation["splits_dir"] = splits
    _same(manifest.get("preparation_config"), preparation, "prepared configuration")
    _same(manifest.get("patches"), runtime["patches"], "prepared patch budget")
    _same(manifest.get("encoding_dim"), 1536, "UNI2-h feature dimension")
    metadata = Path(runtime["metadata_root"])
    signature = ("signatures.csv" if runtime["signature"] == "six" else
                 "combine_signatures.csv" if runtime["signature"] == "all" else
                 str(runtime["signature"]) + "_signatures.csv")
    csv_paths = [metadata / "clinical/all/blca.csv", metadata / "raw_rna_data_inter/blca_rna_inter.csv",
                 metadata / "signatures" / signature]
    csv_paths.extend(metadata / "splits" / splits / "blca" / f"fold_{fold}.csv" for fold in FOLDS)
    csv_hashes = {str(path): sha256(path) for path in csv_paths}
    _same(manifest.get("input_csv_sha256"), csv_hashes, "clinical/RNA/signature/split hashes")
    inputs = identity.get("inputs", {})
    _require(isinstance(inputs, dict) and inputs, "queue input hashes are absent")
    for path, digest in inputs.items():
        _require(_digest(digest), "invalid queue input digest")
        _same(sha256(Path(path)), digest, "queue input metadata hash")
    _require(set(csv_hashes) <= set(inputs), "queue input identity omits prepared metadata")
    _require(all(inputs[path] == digest for path, digest in csv_hashes.items()),
             "queue/prepared metadata hashes disagree")

    patients = manifest.get("patients", {})
    _require(isinstance(patients, dict) and patients, "prepared cohort is empty")
    cache_metadata = {}
    for patient in patients.values():
        relative = patient.get("cache_path", "")
        path = _inside(root, relative)
        _require(_digest(patient.get("cache_sha256")), "missing prepared cache digest")
        _same(artifact_hashes.get(relative), patient["cache_sha256"], "prepared cache identity")
        _require(isinstance(patient.get("full_patch_count"), int) and patient["full_patch_count"] > 0,
                 "full real-patch count is required")
        stat = path.stat()
        _require(stat.st_size > 0, "prepared cache is empty")
        cache_metadata[relative] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                    "recorded_sha256": patient["cache_sha256"]}
    _same(set(manifest.get("folds", {})), {str(fold) for fold in FOLDS}, "prepared fold coverage")
    folds, reused = {}, {arm: {} for arm in REUSED_ARMS}
    for fold in FOLDS:
        key = str(fold)
        fold_data = manifest["folds"][key]
        train, val = fold_data.get("train", []), fold_data.get("val", [])
        _require(train and val and not (set(train) & set(val)), "empty or leaking fold")
        _require(len(train) == len(set(train)) and len(val) == len(set(val)), "duplicated fold IDs")
        _require(set(train + val) <= set(patients), "fold refers to absent prepared cases")
        initial = f"folds/{fold}/initial"
        receipt = read_file(initial + "/receipt.json")
        verify_file(initial + "/receipt.json")
        _same(receipt.get("fold"), fold, "initial fold")
        _same(receipt.get("training_examples"), len(train), "reference training count")
        _same(receipt.get("reference_fit_split"), "train_only", "reference fit scope")
        _same(receipt.get("frozen_during_evaluation"), True, "reference freeze contract")
        initial_paths = {}
        for name in ("core.pt", "probe.pt", "reference.pt"):
            relative = initial + "/" + name
            path = verify_file(relative)
            _same(receipt.get("files_sha256", {}).get(name), files[relative], "initial receipt " + name)
            initial_paths[name] = str(path)
        initial_states = None
        fold_summary = read_file(f"folds/{fold}/summary.json")
        verify_file(f"folds/{fold}/summary.json")
        _same(fold_summary.get("status"), "completed", "fold completion")
        for arm in REUSED_ARMS:
            prefix = f"folds/{fold}/arms/{arm}"
            value = read_file(prefix + "/summary.json")
            verify_file(prefix + "/summary.json")
            for name in ("eval_repeats.json", "epoch_validation.json", "curve.json"):
                verify_file(prefix + "/" + name)
            for field, expected in {"status": "completed", "arm": arm, "fold": fold,
                                    "seed": 3, "epochs": 30, "source_unchanged": True,
                                    "training_cases": len(train), "validation_cases": len(val)}.items():
                _same(value.get(field), expected, "arm summary " + field)
            protocol = value.get("protocol", {})
            for field in ("epochs", "patches", "batch_size", "checkpoint_selection"):
                _same(protocol.get(field), runtime[field], "arm protocol " + field)
            _same(protocol.get("source_hashes"), source_files, "arm source hashes")
            contract = {"adapter": "S0", "gram_weight": 0.0,
                        "slotbert_weight": 1.0 if arm == "R3" else 0.0,
                        "content_weight": 0.0 if arm == "S0" else 5.0,
                        "slotbert_temperature": 0.5}
            _same(protocol.get("contract"), contract, "reused arm loss contract")
            current_states = {name: protocol.get(name) for name in ("initial_core_hash", "initial_probe_hash")}
            _require(all(_digest(v) for v in current_states.values()), "initial state hashes are absent")
            if initial_states is None:
                initial_states = current_states
            _same(current_states, initial_states, "paired initial state hashes")
            _same(protocol.get("initial_stream_seed"), 3 * 1_000_003 + fold * 7919 + 13,
                  "initial sampling stream")
            evaluation = value.get("evaluation_summary", {})
            _same(evaluation.get("repeats"), 20, "evaluation repeat count")
            _same(evaluation.get("defined_cindex_repeats"), 20, "defined evaluation repeat count")
            score = _score(evaluation.get("val_cindex_mean_exploratory"), "arm C-index")
            epoch_rows = value.get("epoch_validation", [])
            _same([row.get("epoch") for row in epoch_rows], list(range(1, 31)), "epoch validation coverage")
            _require(isinstance(value.get("best_epoch"), int) and 1 <= value["best_epoch"] <= 30,
                     "best epoch is invalid")
            checkpoint = verify_file(prefix + "/best.pt", hash_bytes=hash_checkpoints)
            _same(value.get("best_checkpoint_sha256"), artifact_hashes[prefix + "/best.pt"],
                  "best checkpoint recorded digest")
            per_arm = summary.get("per_arm", {}).get(arm, {})
            _same(per_arm.get("status"), "completed", "aggregate arm completion")
            _close(per_arm.get("cindex_per_fold", {}).get(key), score, "root/fold score")
            _close(fold_summary.get("arms", {}).get(arm, {}).get("cindex_mean_over_repeats"), score,
                   "fold/arm score")
            reused[arm][key] = {"status": "completed", "cindex": score,
                                "best_epoch": value["best_epoch"],
                                "evaluation_summary": evaluation,
                                "slot_diagnostics": value.get("slot_diagnostics", {}),
                                "summary_path": str(root / prefix / "summary.json"),
                                "best_checkpoint_path": str(checkpoint),
                                "source": "reused_completed_reference"}
        folds[key] = {"core_path": initial_paths["core.pt"], "probe_path": initial_paths["probe.pt"],
                      "reference_path": initial_paths["reference.pt"], "initial_path": str(root / initial),
                      "output_root": str(root / "folds" / key),
                      "expected_initial_state_hashes": initial_states}
    # Detect metadata mutation during the audit instead of returning a mixed
    # snapshot if a completed queue was changed by another process.
    for relative, digest in files.items():
        _same(sha256(_inside(root, relative)), digest, "artifact changed during audit")
    return {"schema": "slotspe_diversity_reference_reuse_v1", "status": "verified_metadata",
            "reference_root": str(root), "prepared_root": str(root / "prepared"),
            "manifest_path": str(root / "prepared/manifest.json"), "manifest": manifest,
            "runtime": runtime, "folds": folds, "reused_results": reused,
            "expected_initial_state_hashes": {
                key: {"core": data["expected_initial_state_hashes"]["initial_core_hash"],
                      "probe": data["expected_initial_state_hashes"]["initial_probe_hash"]}
                for key, data in folds.items()},
            "file_hashes": files, "implementation_sha256": implementation,
            "cache_metadata": cache_metadata, "split_policy": "inherit_verified_reference",
            "splits_dir": splits,
            "checks": {"reused_training_runs": len(REUSED_ARMS) * len(FOLDS),
                       "completed_reference_stages": len(expected_stages),
                       "protected_source_files": len(source_files),
                       "initial_file_hashes_verified": 15, "checkpoint_deserializations": 0,
                       "cache_content_hashes_verified": False,
                       "best_checkpoint_bytes_hashed": bool(hash_checkpoints),
                       "patient_arrays_parsed": False, "model_forwards": 0},
            "claim_boundary": "Metadata/hash compatibility only; no patient arrays or models evaluated; "
                              "full cache content verification is required at authorized execution."}


def public_receipt(receipt):
    """Return aggregate audit status without patient IDs, paths or raw arrays."""
    return {"schema": receipt["schema"], "status": receipt["status"],
            "splits_dir": receipt["splits_dir"], "split_policy": receipt["split_policy"],
            "reused_arms": list(REUSED_ARMS), "folds": list(FOLDS),
            "checks": dict(receipt["checks"]), "claim_boundary": receipt["claim_boundary"]}
