"""Best-checkpoint eleven-arm BLCA five-fold queue.

This is the checkpoint-rule counterpart of
``slotspe_proof/redundancy_multifold.py``.  The stage graph, the scheduler,
the device assignment, the concurrency rules, the evaluation contract and the
summary shape are the same; the single difference is that every arm trains
under ``slotspe_proof.redundancy_best_engine.run_arm_best``, which reproduces
the original published rule of evaluating every epoch on the validation fold,
retaining the best epoch and reloading it for the final evaluation.

The final-epoch queue and this queue are two separate protocols with separate
output roots.  Neither overwrites the other, and the completed final-epoch
queue is never resumed, resumed-adjacent or modified by this module.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slotspe_proof import redundancy_multifold as base

DEFAULT_CONFIG = ROOT / "configs/slotspe_redundancy_best_multifold_queue.json"
DEFAULT_OUTPUT = "local_evidence/redundancy_blca_uni2h_best_folds0to4_seed3_run01"
ARMS = base.ARMS
ALL_FOLDS = base.ALL_FOLDS
# Every fold is trained fresh under the best-checkpoint rule.  fold0 was
# executed by the final-epoch protocol and cannot be reused here, because a
# fold trained under one checkpoint rule cannot be mixed into the statistic of
# a fold trained under another.
NEW_FOLDS = (0, 1, 2, 3, 4)
REUSED_FOLDS = ()
FOLD0_ROOT = base.FOLD0_ROOT

# The implementation identity of this queue covers the best engine and the
# best queue module in addition to the shared modules, so that a change to
# either one invalidates a running queue exactly as the final-epoch queue
# does.
IMPLEMENTATION_FILES = tuple(base.implementation_hashes().keys()) + (
    "slotspe_proof/redundancy_best_engine.py",
    "slotspe_proof/redundancy_best_multifold.py",
    "scripts/run_slotspe_redundancy_best_multifold.py",
)


def implementation_hashes():
    files = []
    for name in IMPLEMENTATION_FILES:
        path = (ROOT / name).resolve()
        if path.is_file():
            files.append(path)
    return {p.relative_to(ROOT).as_posix(): base.sha(p) for p in sorted(files)}


def load_config(path=None):
    return base.read(path or DEFAULT_CONFIG)


def verify_config_pins(config):
    """Every pinned file must still carry the recorded bytes."""
    for group in base.PIN_GROUPS:
        for name, expected in config.get(group, {}).items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or base.sha(path) != expected:
                raise RuntimeError(f"pinned file differs: {name}")
    if not config.get("protected_original_source_sha256"):
        raise RuntimeError("the original SlotSPE source must be pinned byte for byte")


def normalize_runtime(config, metadata_root=None, features_root=None, device=None):
    """Validate the scope, then enforce the best-checkpoint checkpoint rule.

    The shared validator pins the final-epoch queue's fold split, so this
    module validates the scope itself and then reuses the shared budget and
    hyperparameter checks.
    """
    if "requested_scope" in config:
        scope = config["requested_scope"]
        stage = next((s for s in config["stages"] if s["id"] == "real_data_diagnostic"), None)
        if stage is None:
            raise ValueError("the real_data_diagnostic stage is required")
        if scope.get("automatic_cross_fold_expansion") is not False:
            raise ValueError("automatic cross-fold expansion is outside this queue")
        for key in ("study", "feature_encoder", "folds", "seeds"):
            if stage.get(key) != scope.get(key):
                raise ValueError(f"stage and requested scope disagree: {key}")
        if stage.get("variants") != scope.get("selected_variants"):
            raise ValueError("stage and requested variants disagree")
        runtime = {
            "study": scope["study"], "feature_encoder": scope["feature_encoder"],
            "folds": scope["folds"], "seeds": scope["seeds"],
            "variants": scope["selected_variants"], "epochs": stage["epochs"],
            "patches": stage["real_patches"], "batch_size": stage["batch_size"],
            "eval_repeats": stage["eval_repeats"], "splits_dir": stage.get("splits_dir", "5fold"),
            **config.get("execution", {}),
        }
        new_folds = list(scope["new_folds"])
        reused_folds = list(scope["reused_folds"])
    else:
        runtime = dict(config)
        new_folds = list(runtime.get("new_folds") or [])
        reused_folds = list(runtime.get("reused_folds") or [])
    defaults = {
        "source_root": "third_party/SlotSPE", "metadata_root": "${SLOTSPE_METADATA_ROOT}",
        "features_root": "${SLOTSPE_FEATURES_ROOT}", "label_col": "survival_months_dss",
        "signature": "combine", "device": "cuda:0", "lr": 0.0005,
        "probe_lr": 0.0005, "torch_threads": 2, "encoding_dim": None,
        "projection_dim": 256, "slot_iters": 10, "content_target_dim": 32,
        "content_clusters": 8, "content_reference_seed": 731, "dias_distance": 0.05,
        "slotbert_temperature": 0.5, "experiment": "original_audit", "splits_dir": "5fold",
        "checkpoint_selection": "per_epoch_validation_best_then_reload_and_reevaluate",
    }
    runtime = {**defaults, **runtime}
    for key, value in (("metadata_root", metadata_root), ("features_root", features_root), ("device", device)):
        if value is not None:
            runtime[key] = value
    runtime["new_folds"] = new_folds
    runtime["reused_folds"] = reused_folds
    # Scope lock: exactly BLCA / UNI2-h / the five declared folds / seed3 /
    # the declared eleven arms, and every fold trained fresh under this
    # checkpoint rule.  Nothing here may widen itself.
    if (runtime.get("study") != "blca" or runtime.get("feature_encoder") != "UNI2-h"):
        raise ValueError("scope is exactly BLCA / UNI2-h")
    if list(runtime.get("folds") or []) != list(ALL_FOLDS):
        raise ValueError("scope is exactly folds 0,1,2,3,4")
    if list(runtime.get("seeds") or []) != [3]:
        raise ValueError("scope is exactly seed3")
    if list(runtime.get("variants") or []) != list(ARMS):
        raise ValueError("scope is exactly the declared eleven arms")
    if tuple(runtime["new_folds"]) != NEW_FOLDS or tuple(runtime["reused_folds"]) != REUSED_FOLDS:
        raise ValueError("all five folds are trained here under the best-checkpoint rule; nothing is reused")
    for key in ("epochs", "patches", "batch_size", "eval_repeats", "torch_threads", "projection_dim",
                "slot_iters", "content_target_dim", "content_clusters", "content_reference_seed"):
        value = runtime[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if runtime["projection_dim"] != 256 or runtime["slot_iters"] != 10:
        raise ValueError("patient queue must use native 256 dimensions and 10 slot iterations")
    if runtime["eval_repeats"] < 2 or runtime["content_clusters"] != 8:
        raise ValueError("need repeated evaluation and eight reference clusters")
    if (runtime["content_target_dim"] != 32 or runtime["dias_distance"] != 0.05 or
            runtime["slotbert_temperature"] != 0.5):
        raise ValueError("use the declared 32-dimension content, DIAS .05 and paper temperature .5 presets")
    if runtime["epochs"] != 30 or runtime["patches"] != 4096 or runtime["batch_size"] != 8:
        raise ValueError("the best queue keeps the declared budget: 30 epochs / 4096 patches / batch 8")
    if runtime["eval_repeats"] != 20:
        raise ValueError("the best queue keeps the 20-repeat evaluation contract")
    if runtime.get("splits_dir") != "5fold":
        raise ValueError("the best queue uses the 5fold split family")
    for key in ("lr", "probe_lr", "slotbert_temperature"):
        if isinstance(runtime[key], bool) or not isinstance(runtime[key], (int, float)) or \
                not math.isfinite(runtime[key]) or runtime[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if runtime["encoding_dim"] is not None and (isinstance(runtime["encoding_dim"], bool) or
                                                not isinstance(runtime["encoding_dim"], int) or runtime["encoding_dim"] < 1):
        raise ValueError("encoding_dim must be positive or null")
    if not isinstance(runtime["device"], str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", runtime["device"]):
        raise ValueError("device must be cpu or cuda[:index]")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(runtime["splits_dir"])):
        raise ValueError("splits_dir must be a directory name under <metadata_root>/splits, not a path")
    for key in ("source_root", "metadata_root", "features_root"):
        runtime[key] = base._resolve(runtime[key])
    if Path(runtime["source_root"]).resolve() != (ROOT / "third_party/SlotSPE").resolve():
        raise ValueError("use the pinned original SlotSPE source in this checkout")
    if runtime.get("checkpoint_selection") != "per_epoch_validation_best_then_reload_and_reevaluate":
        raise ValueError("the best queue must declare the per-epoch validation checkpoint rule")
    return runtime


def make_plan(config=None, output_root=None):
    config = load_config() if config is None else config
    verify_config_pins(config)
    runtime = normalize_runtime(config)
    root = base.private_output(output_root or ROOT / DEFAULT_OUTPUT)
    stages = base.stages(config)
    return {"schema": "slotspe_native_redundancy_best_multifold_execution_plan_v1", "config": runtime,
            "output_root": str(root), "stages": stages,
            "training_runs": len(NEW_FOLDS) * len(ARMS), "reused_folds": list(REUSED_FOLDS),
            "training_arms": list(ARMS), "feature_encoder": "UNI2-h",
            "folds": list(ALL_FOLDS), "seeds": [3], "default_action": "plan_only",
            "patient_work_executed": False, "execution_implemented": True,
            "checkpoint_selection": "per_epoch_validation_best_then_reload_and_reevaluate",
            "evaluation_contract": "20 repeats aggregated inside each fold before the five-fold statistic",
            "unresolved_variables": [k for k in ("metadata_root", "features_root") if "$" in runtime[k]],
            "content_contract": "train-only frozen reference projection and cluster targets; rectangular matching with unmatched coverage costs",
            "batching": "same per-example compact native forward and one core update per logical batch for all arms",
            "fold_independence": "bins and content reference are fitted from each fold's own training patients only",
            "relationship_to_final_epoch_queue": "separate protocol, separate output root; the completed final-epoch queue is neither resumed nor modified"}


def _spawn_stage(stage, runtime, root, device=None):
    """Start one stage in its own process with its own retained log.

    The worker command names THIS module, not the final-epoch queue, so that
    an arm stage trains under the best-checkpoint rule.
    """
    log = root / "logs" / (stage["id"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", "-m", "slotspe_proof.redundancy_best_multifold",
               "--run", "--worker-stage", stage["id"], "--output-root", str(root)]
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
    stream = log.open("x", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT)
    return stage, process, stream


def _subprocess_stage(stage, runtime, root, device=None):
    log = root / "logs" / (stage["id"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", "-m", "slotspe_proof.redundancy_best_multifold",
               "--run", "--worker-stage", stage["id"], "--output-root", str(root)]
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    if device:
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"stage {stage['id']} failed with exit {result.returncode}; retained log: {log}")


def _run_wave(order, pending_ids, runtime, root, gpu_pool, workers):
    """Run the pending stages concurrently, respecting their dependencies.

    This is the shared scheduler.  It is re-bound here so that the spawned
    worker is this module's worker; the concurrency, the device choice and
    the dependency rule are unchanged from the final-epoch queue.
    """
    import statistics
    import subprocess
    import time

    completed = set()
    running = {}
    failures = []
    device_load = {}
    device_memory = {}
    stage_by_id = {stage["id"]: stage for stage in order}

    def reap():
        for stage_id, (process, stream, device) in list(running.items()):
            if process.poll() is None:
                continue
            stream.close()
            del running[stage_id]
            if device is not None:
                device_load[device] = device_load.get(device, 0) - 1
                device_memory[device] = max(0.0, device_memory.get(device, 0.0)
                                            - base.stage_memory_gib(stage_by_id.get(stage_id, {})))
            if process.returncode:
                failures.append(f"{stage_id} exited {process.returncode}")
            else:
                completed.add(stage_id)

    while len(completed) + len(failures) < len(pending_ids):
        reap()
        if len(completed) + len(failures) >= len(pending_ids):
            break
        for stage in order:
            if len(running) >= workers:
                break
            if stage["id"] in completed or stage["id"] in running:
                continue
            if stage["id"] not in pending_ids:
                continue
            if any(dependency not in completed for dependency in base.stage_dependencies(stage)):
                continue
            device = None
            if gpu_pool:
                device = base._choose_device(stage, gpu_pool, device_load, device_memory)
                if device is None:
                    break
                device_load[device] = device_load.get(device, 0) + 1
                device_memory[device] = device_memory.get(device, 0.0) + base.stage_memory_gib(stage)
            spawned, process, stream = _spawn_stage(stage, runtime, root, device)
            running[stage["id"]] = (process, stream, device)
        if not running:
            time.sleep(0.5)
            continue
        time.sleep(0.5)
    reap()
    if failures:
        raise RuntimeError("parallel stage group failed: " + "; ".join(sorted(failures)))


def expected_artifacts(stage):
    """The best protocol adds the retained best checkpoint and the per-epoch
    validation record to the artifact set."""
    kind = stage["kind"]
    if kind == "prepare":
        return ["prepared/manifest.json"]
    if kind == "initialize":
        return [f"folds/{stage['fold']}/initial/{name}" for name in
                ("core.pt", "probe.pt", "reference.pt", "receipt.json")]
    if kind == "arm":
        return [f"folds/{stage['fold']}/arms/{stage['arm']}/{name}" for name in
                ("last.pt", "best.pt", "curve.json", "summary.json", "eval_repeats.json",
                 "epoch_validation.json", "diagnostics.pt", "validation_predictions.csv")]
    if kind == "summarize_fold":
        return [f"folds/{stage['fold']}/summary.json"]
    return ["summary.json", "summary.txt"]


def doctor(config, runtime_check=True):
    runtime = normalize_runtime(config)
    from .queue import preflight
    result = preflight(runtime, checks_only=False, check_runtime=runtime_check)
    if runtime_check:
        import h5py
        import scipy
        result["runtime"].update(h5py=h5py.__version__, scipy=scipy.__version__)
    result.update(feature_encoder="UNI2-h",
                  encoder_provenance="declared existing root; actual feature source must match server records",
                  feature_extraction_started=False, patient_training_executed=False)
    return result


def _stage_folder(stage):
    kind = stage["kind"]
    if kind == "prepare":
        return "prepared"
    if kind == "initialize":
        return f"folds/{stage['fold']}/initial"
    if kind == "arm":
        return f"folds/{stage['fold']}/arms/{stage['arm']}"
    if kind == "summarize_fold":
        return f"folds/{stage['fold']}"
    return ""


def _artifact_hashes(root, stage):
    required = expected_artifacts(stage)
    for name in required:
        if not (root / name).is_file():
            raise RuntimeError(f"stage {stage['id']} missing artifact: {name}")
    folder = _stage_folder(stage)
    files = sorted((root / folder).rglob("*")) if folder else [root / n for n in required]
    return {p.relative_to(root).as_posix(): base.sha(p) for p in files if p.is_file()}


def _verify_completed(root, stage, record):
    current = _artifact_hashes(root, stage)
    if current != record.get("artifact_sha256"):
        raise RuntimeError(f"completed stage artifacts changed: {stage['id']}")


def _summarize_fold(root, runtime, fold):
    rows = {}
    for arm in ARMS:
        folder = base._fold_arm_paths(root, fold, arm)
        summary = base.read(folder / "summary.json")
        curve = base.read(folder / "curve.json")
        view = base._arm_view(summary, curve)
        view["best_epoch"] = summary.get("best_epoch")
        view["best_epoch_cindex_single_pass"] = summary.get("best_epoch_cindex_single_pass")
        view["best_checkpoint_sha256"] = summary.get("best_checkpoint_sha256")
        view["epoch_validation"] = summary.get("epoch_validation")
        rows[arm] = view
    result = {"schema": "slotspe_native_redundancy_best_multifold_fold_v1", "fold": fold, "seed": 3,
              "status": "completed", "arms": rows, "epochs": runtime["epochs"],
              "patches": runtime["patches"], "eval_repeats": runtime["eval_repeats"],
              "checkpoint_selection": "per_epoch_validation_best_then_reload_and_reevaluate"}
    base.write(root / "folds" / str(fold) / "summary.json", result)
    return result


def _summarize_all(root, runtime):
    fold_views = {}
    for fold in NEW_FOLDS:
        path = root / "folds" / str(fold) / "summary.json"
        value = base.read(path)
        if value.get("status") != "completed":
            raise RuntimeError(f"fold {fold} is not completed")
        fold_views[fold] = {"source": "executed_here", "root": f"folds/{fold}", "arms": value["arms"]}
    per_arm = {}
    for arm in ARMS:
        cindex = {fold: fold_views[fold]["arms"][arm]["cindex_mean_over_repeats"] for fold in NEW_FOLDS}
        best_epoch = {fold: fold_views[fold]["arms"][arm].get("best_epoch") for fold in NEW_FOLDS}
        entry = {
            "cindex_per_fold": cindex,
            "cindex_four_fold": base._fold_stats([cindex[fold] for fold in NEW_FOLDS]),
            "best_epoch_per_fold": best_epoch,
            "cindex_repeat_std_per_fold": {fold: fold_views[fold]["arms"][arm]["cindex_repeat_std_population"]
                                           for fold in NEW_FOLDS},
            "evaluation_repeats_per_fold": {fold: fold_views[fold]["arms"][arm]["evaluation_repeats"]
                                            for fold in NEW_FOLDS},
            "status_per_fold": {fold: fold_views[fold]["arms"][arm]["status"] for fold in NEW_FOLDS},
            "elapsed_seconds_per_fold": {fold: fold_views[fold]["arms"][arm]["elapsed_seconds"]
                                         for fold in NEW_FOLDS},
            "cuda_peak_bytes_per_fold": {fold: fold_views[fold]["arms"][arm]["cuda_peak_bytes"]
                                          for fold in NEW_FOLDS},
        }
        entry["status"] = ("completed" if all(value == "completed" for value in entry["status_per_fold"].values())
                           else "incomplete")
        per_arm[arm] = entry
    result = {
        "schema": "slotspe_native_redundancy_best_multifold_summary_v1", "status": "completed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scope": {"study": "blca", "feature_encoder": "UNI2-h", "folds": list(NEW_FOLDS), "seed": 3,
                  "arms": list(ARMS), "executed_folds": list(NEW_FOLDS),
                  "excluded_folds": list(REUSED_FOLDS),
                  "excluded_folds_reason": "fold0 was executed by the final-epoch protocol and is not re-executed "
                                          "here; a fold trained under a different checkpoint rule cannot be mixed "
                                          "into one five-fold statistic"},
        "protocol": {"epochs": runtime["epochs"], "patches": runtime["patches"],
                     "batch_size": runtime["batch_size"], "eval_repeats": runtime["eval_repeats"],
                     "splits_dir": runtime["splits_dir"], "lr": runtime["lr"],
                     "probe_lr": runtime["probe_lr"], "projection_dim": runtime["projection_dim"],
                     "slot_iters": runtime["slot_iters"], "content_target_dim": runtime["content_target_dim"],
                     "content_clusters": runtime["content_clusters"],
                     "checkpoint_selection": "per_epoch_validation_best_then_reload_and_reevaluate",
                     "checkpoint_rule_source": "third_party/SlotSPE/utils/core_utils.py:421-475",
                     "per_epoch_validation": "one full validation pass after each epoch; the global random streams "
                                             "are restored afterwards so the training trajectory equals the "
                                             "final-epoch protocol for the same arm, fold and seed",
                     "evaluation_aggregation": "the 20 repeats are averaged inside each fold first; "
                                               "the four-fold statistic is then computed over the four per-fold values",
                     "std_convention": "population and sample standard deviations are both reported for the between-fold spread"},
        "folds": fold_views, "per_arm": per_arm,
        "caveats": [
            "The reported C-index is a validation-selected peak. A validation-selected peak is not an independently tested generalization result.",
            "Selecting the best of 30 validation evaluations inflates the reported value relative to an independent test set by an amount that depends on the epoch-to-epoch spread of the validation metric.",
            "Only the four newly executed folds are summarized. fold0 is not re-executed under this checkpoint rule and is therefore excluded from the statistic rather than mixed in.",
            "The 20 evaluation repeats inside a fold are not independent patients and are averaged before the fold value is formed.",
            "C-index differences between arms are exploratory single-seed observations and are not corrected for multiplicity here."],
        "claim_boundary": "Four-fold single-seed BLCA diagnostic of one original SlotSPE implementation family under the "
                          "original per-epoch best-checkpoint rule; no generalization, clinical benefit or "
                          "multiplicity-corrected significance is asserted"}
    base.write(root / "summary.json", result)
    lines = ["SlotSPE BLCA / UNI2-h / folds 1-4 / seed3 / eleven arms: per-epoch best-checkpoint rule, all folds completed.",
             "",
             "arm  cindex_per_fold(1..4)  fold_mean  fold_std_population  best_epochs"]
    for arm in ARMS:
        entry = per_arm[arm]
        values = " ".join(f"{entry['cindex_per_fold'][fold]:.4f}" if entry["cindex_per_fold"][fold] is not None
                          else "  n/a " for fold in NEW_FOLDS)
        epochs = " ".join(str(entry["best_epoch_per_fold"][fold]) for fold in NEW_FOLDS)
        lines.append(f"{arm:>3}  {values}  {entry['cindex_four_fold']['fold_mean']:.4f}  "
                     f"{entry['cindex_four_fold']['fold_std_population']:.4f}  {epochs}")
    (root / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def worker(stage_id, root):
    root = base.private_output(root)
    manifest = base.read(root / "queue_manifest.json")
    active = manifest.get("active_stage")
    active_ids = [active] if isinstance(active, str) else list(active or [])
    if manifest.get("status") != "running" or stage_id not in active_ids:
        raise RuntimeError("worker must belong to the active explicitly-run queue stage")
    runtime = base.read(root / "resolved_config.json")
    expected_identity = hashlib.sha256(json.dumps(manifest["identity_payload"], sort_keys=True).encode()).hexdigest()
    if (runtime != manifest["config"] or runtime != manifest["identity_payload"]["runtime"] or
            expected_identity != manifest["identity"] or normalize_runtime(runtime) != runtime or
            implementation_hashes() != manifest["identity_payload"]["implementation"]):
        raise RuntimeError("worker runtime or implementation differs from its queued identity")
    stage = next((s for s in manifest["stages"] if s["id"] == stage_id), None)
    if stage is None:
        raise ValueError("unknown stage")
    kind = stage["kind"]
    if kind == "prepare":
        from .patient import prepare
        prepare(runtime, root)
        return
    from .patient import _manifest
    prepared = _manifest(runtime, root, verify_cache=True)
    if kind == "initialize":
        base._initialize_fold(runtime, root, prepared, stage["fold"])
    elif kind == "arm":
        import torch
        from .redundancy_content import ContentReference
        from .redundancy_best_engine import run_arm_best
        folder = root / "folds" / str(stage["fold"]) / "initial"
        reference = ContentReference(prepared["omic_sizes"], prepared["encoding_dim"],
                                     target_dim=runtime["content_target_dim"],
                                     clusters=runtime["content_clusters"],
                                     seed=runtime["content_reference_seed"])
        reference.load_state_dict(torch.load(folder / "reference.pt", map_location="cpu", weights_only=True))
        core = torch.load(folder / "core.pt", map_location="cpu", weights_only=True)
        probe = torch.load(folder / "probe.pt", map_location="cpu", weights_only=True)
        run_arm_best(runtime, root, prepared, stage["arm"], core, probe, reference,
                     fold=stage["fold"], output_root=root / "folds" / str(stage["fold"]))
    elif kind == "summarize_fold":
        _summarize_fold(root, runtime, stage["fold"])
    else:
        _summarize_all(root, runtime)


def execute(config, output_root, resume=False, executor=None, runtime_check=True,
            devices=None, workers=None):
    """Run the best-checkpoint queue, reusing the final-epoch scheduler.

    The scheduler, the device assignment, the concurrency rules and the
    identity verification are the shared implementation.  Only the arm runner
    and the artifact set differ.
    """
    plan = make_plan(config, output_root)
    runtime, root = plan["config"], Path(plan["output_root"])
    checked = doctor(runtime, runtime_check=runtime_check)
    code = implementation_hashes()
    identity_payload = {"runtime": runtime, "implementation": code, "stages": plan["stages"],
                        "inputs": checked.get("source_and_metadata_sha256", {}),
                        "runtime_versions": checked.get("runtime", {})}
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    device_pool = base.assign_devices(plan["stages"], runtime, devices)
    if workers is None:
        workers = 3 * len(device_pool) if device_pool else 1
    if workers < 1:
        raise ValueError("workers must be positive")
    if device_pool and workers > 4 * len(device_pool):
        raise ValueError("workers must not oversubscribe the GPU pool")
    manifest_path = root / "queue_manifest.json"
    if root.exists() and any(root.iterdir()):
        if not resume or not manifest_path.is_file():
            raise RuntimeError("nonempty output: preserve it; use a fresh directory or verified --resume")
        manifest = base.read(manifest_path)
        if manifest.get("identity") != identity or base.read(root / "resolved_config.json") != runtime:
            raise RuntimeError("configuration, input metadata or implementation drift; cannot resume")
    else:
        root.mkdir(parents=True, exist_ok=True)
        base.write(root / "resolved_config.json", runtime)
        manifest = {"schema": "slotspe_native_redundancy_best_multifold_queue_v1", "identity": identity,
                    "identity_payload": identity_payload, "config": runtime,
                    "stages": plan["stages"], "status": "running", "records": {},
                    "device_assignment": device_pool,
                    "max_concurrent_stages": workers,
                    "started_utc": datetime.now(timezone.utc).isoformat()}
        base.write(manifest_path, manifest)
    custom_executor = executor is not None
    executor = executor or _subprocess_stage

    def verify_live_identity():
        saved = base.read(manifest_path)
        if (base.read(root / "resolved_config.json") != runtime or saved.get("config") != runtime or
                saved.get("identity") != identity or saved.get("identity_payload") != identity_payload):
            raise RuntimeError("saved runtime or queue identity changed during execution")
        if implementation_hashes() != code:
            raise RuntimeError("implementation changed while queue was running")
        if any(base.sha(Path(name)) != value for name, value in identity_payload["inputs"].items()):
            raise RuntimeError("source or metadata changed while queue was running")

    for wave in base.execution_waves(base.stage_chains(plan["stages"])):
        verify_live_identity()
        for prior in plan["stages"]:
            record = manifest["records"].get(prior["id"], {})
            if record.get("status") == "completed":
                _verify_completed(root, prior, record)
        order = [stage for chain in wave for stage in chain]
        pending = []
        for stage in order:
            record = manifest["records"].get(stage["id"], {})
            if record.get("status") == "completed":
                continue
            if record.get("status") in ("running", "failed"):
                raise RuntimeError(f"partial/failed stage retained: {stage['id']}; use a fresh batch, never overwrite it")
            folder = _stage_folder(stage)
            if folder and (root / folder).exists() and any((root / folder).iterdir()):
                raise RuntimeError(f"unrecorded partial stage folder retained: {folder}")
            pending.append(stage)
        if not pending:
            continue
        manifest["status"] = "running"
        manifest["active_stage"] = [stage["id"] for stage in pending]
        for stage in pending:
            manifest["records"][stage["id"]] = {"status": "running"}
        base.write(manifest_path, manifest)
        try:
            if custom_executor:
                for stage in pending:
                    executor(stage, runtime, root)
            else:
                # This module's own scheduler, whose spawned worker is this
                # module's worker.  The shared scheduler of the final-epoch
                # queue would train the arms under the final-epoch rule.
                _run_wave(order, {stage["id"] for stage in pending}, runtime, root,
                          device_pool, workers)
            verify_live_identity()
            for stage in pending:
                manifest["records"][stage["id"]] = {"status": "completed",
                                                    "artifact_sha256": _artifact_hashes(root, stage)}
            manifest.pop("active_stage", None)
            base.write(manifest_path, manifest)
            print("completed stage(s) " + ", ".join(stage["id"] for stage in pending), flush=True)
        except BaseException as error:
            manifest["status"] = "failed"
            for stage in pending:
                manifest["records"][stage["id"]] = {"status": "failed", "error": str(error),
                                                    "type": type(error).__name__}
            base.write(manifest_path, manifest)
            raise
    manifest["status"] = "completed"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    base.write(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--metadata-root")
    parser.add_argument("--features-root")
    parser.add_argument("--device")
    parser.add_argument("--devices", help="Comma-separated physical GPU ids used to run "
                                          "stages concurrently; execution only.")
    parser.add_argument("--workers", type=int,
                        help="Maximum number of concurrent training stages; execution only.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--doctor", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-stage", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_stage:
        if not args.run or not args.output_root:
            parser.error("internal worker requires --run and a queued output directory")
        worker(args.worker_stage, args.output_root)
        return
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    config = load_config(args.config)
    runtime = normalize_runtime(config, args.metadata_root, args.features_root, args.device)
    resolved = dict(config)
    resolved["execution"] = {**config.get("execution", {}), **runtime}
    devices = None
    if args.devices:
        devices = [value.strip() for value in args.devices.split(",") if value.strip()]
        if not devices:
            parser.error("--devices needs at least one GPU id")
    if args.run:
        result = execute(resolved, args.output_root or ROOT / DEFAULT_OUTPUT, resume=args.resume,
                         devices=devices, workers=args.workers)
        print(json.dumps({"status": result["status"], "output_root": str(base.private_output(args.output_root))}))
    elif args.doctor:
        verify_config_pins(config)
        print(json.dumps(doctor(runtime), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(make_plan(resolved, args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
