"""Extend the eleven-arm BLCA queue from fold0 to the remaining four folds.

fold0 is reused from the completed single-fold queue; folds 1-4 are trained
here.  Model computation, hyperparameters, initialization, batching, patch
budget, checkpoint rule and evaluation contract are identical to the fold0
queue.  Every fold fits its own survival bins and its own content reference
from that fold's training patients only.

Planning is standard-library-only: no torch, pandas, numpy or patient array is
imported while a plan is produced.  Each stage runs in its own process with its
own retained log.  Completed stages resume only when configuration, code,
metadata and artifact hashes still match; a partial stage is never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/slotspe_redundancy_multifold_queue.json"
ARMS = ("S0", "S4", "R1", "R2", "R3", "R4", "R5", "S1", "S2", "S3", "S5")
ALL_FOLDS = (0, 1, 2, 3, 4)
NEW_FOLDS = (1, 2, 3, 4)
REUSED_FOLDS = (0,)
FOLD0_ROOT = "local_evidence/redundancy_blca_uni2h_fold0_seed3_run01"
DEFAULT_OUTPUT = "local_evidence/redundancy_blca_uni2h_folds1to4_seed3_run01"
PIN_GROUPS = ("evidence_sha256", "protected_existing_queue_sha256",
              "candidate_implementation_sha256", "protected_original_source_sha256")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            value.update(part)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def load_config(path=None):
    return read(path or DEFAULT_CONFIG)


def _resolve(value):
    value = os.path.expandvars(os.path.expanduser(str(value)))
    return str((ROOT / value).resolve()) if not Path(value).is_absolute() else str(Path(value).resolve())


def normalize_runtime(config, metadata_root=None, features_root=None, device=None):
    """Validate the multi-fold scope without reading any patient input.

    Accepts either the queue configuration (which carries ``requested_scope``
    and ``stages``) or an already resolved runtime, so that a worker can
    re-validate its own saved runtime.
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
    }
    runtime = {**defaults, **runtime}
    for key, value in (("metadata_root", metadata_root), ("features_root", features_root), ("device", device)):
        if value is not None:
            runtime[key] = value
    runtime["new_folds"] = new_folds
    runtime["reused_folds"] = reused_folds
    # Scope lock: exactly BLCA / UNI2-h / the five declared folds / seed3 /
    # the declared eleven arms.  Nothing here may widen itself.
    if (runtime.get("study") != "blca" or runtime.get("feature_encoder") != "UNI2-h"):
        raise ValueError("scope is exactly BLCA / UNI2-h")
    if list(runtime.get("folds") or []) != list(ALL_FOLDS):
        raise ValueError("scope is exactly folds 0,1,2,3,4")
    if list(runtime.get("seeds") or []) != [3]:
        raise ValueError("scope is exactly seed3")
    if list(runtime.get("variants") or []) != list(ARMS):
        raise ValueError("scope is exactly the declared eleven arms")
    if tuple(runtime["new_folds"]) != NEW_FOLDS or tuple(runtime["reused_folds"]) != REUSED_FOLDS:
        raise ValueError("folds 1-4 are trained here and fold0 is reused; nothing else")
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
        raise ValueError("the multi-fold queue keeps the fold0 budget: 30 epochs / 4096 patches / batch 8")
    if runtime["eval_repeats"] != 20:
        raise ValueError("the multi-fold queue keeps the fold0 evaluation contract: 20 repeats")
    if runtime.get("splits_dir") != "5fold":
        raise ValueError("the multi-fold queue uses the 5fold split family, which is the cohort "
                         "that the completed fold0 queue actually executed")
    for key in ("lr", "probe_lr", "slotbert_temperature"):
        if isinstance(runtime[key], bool) or not math.isfinite(runtime[key]) or runtime[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if runtime["encoding_dim"] is not None and (isinstance(runtime["encoding_dim"], bool) or
                                                not isinstance(runtime["encoding_dim"], int) or runtime["encoding_dim"] < 1):
        raise ValueError("encoding_dim must be positive or null")
    if not isinstance(runtime["device"], str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", runtime["device"]):
        raise ValueError("device must be cpu or cuda[:index]")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(runtime["splits_dir"])):
        raise ValueError("splits_dir must be a directory name under <metadata_root>/splits, not a path")
    for key in ("source_root", "metadata_root", "features_root"):
        runtime[key] = _resolve(runtime[key])
    if Path(runtime["source_root"]).resolve() != (ROOT / "third_party/SlotSPE").resolve():
        raise ValueError("use the pinned original SlotSPE source in this checkout")
    return runtime


def verify_config_pins(config):
    """Every pinned file must still carry the recorded bytes."""
    for group in PIN_GROUPS:
        for name, expected in config.get(group, {}).items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or sha(path) != expected:
                raise RuntimeError(f"pinned file differs: {name}")
    if not config.get("protected_original_source_sha256"):
        raise RuntimeError("the original SlotSPE source must be pinned byte for byte")


def implementation_hashes():
    package = Path(__file__).parent
    files = [package / name for name in (
        "redundancy_multifold.py", "redundancy_queue.py", "redundancy_engine.py",
        "redundancy_content.py", "patient.py", "model.py", "assignment_router.py",
        "native_allocation.py", "native_pruning.py", "native_variants.py",
        "prototype_bank.py", "slot_regularization.py")]
    files += sorted((ROOT / "third_party/SlotSPE").rglob("*.py"))
    files += [ROOT / "scripts/run_slotspe_redundancy_multifold.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def private_output(path):
    path = Path(path or ROOT / DEFAULT_OUTPUT)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    base = (ROOT / "local_evidence").resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError("outputs must use a dedicated ignored local_evidence subdirectory")
    return path


def stages(config):
    new_folds = list(config.get("new_folds") or config.get("requested_scope", {}).get("new_folds") or NEW_FOLDS)
    result = [{"id": "prepare", "kind": "prepare"}]
    for fold in new_folds:
        result.append({"id": f"init_fold_{fold}", "kind": "initialize", "fold": fold})
        for arm in ARMS:
            result.append({"id": f"arm_fold_{fold}_{arm}", "kind": "arm", "fold": fold, "arm": arm})
        result.append({"id": f"summarize_fold_{fold}", "kind": "summarize_fold", "fold": fold})
    result.append({"id": "summarize", "kind": "summarize"})
    return result


def expected_artifacts(stage):
    kind = stage["kind"]
    if kind == "prepare":
        return ["prepared/manifest.json"]
    if kind == "initialize":
        return [f"folds/{stage['fold']}/initial/{name}" for name in
                ("core.pt", "probe.pt", "reference.pt", "receipt.json")]
    if kind == "arm":
        return [f"folds/{stage['fold']}/arms/{stage['arm']}/{name}" for name in (
            "last.pt", "curve.json", "summary.json", "eval_repeats.json",
            "diagnostics.pt", "validation_predictions.csv")]
    if kind == "summarize_fold":
        return [f"folds/{stage['fold']}/summary.json"]
    return ["summary.json", "summary.txt"]


def make_plan(config=None, output_root=None):
    config = load_config() if config is None else config
    verify_config_pins(config)
    runtime = normalize_runtime(config)
    root = private_output(output_root)
    plan_stages = stages(config)
    return {"schema": "slotspe_native_redundancy_multifold_execution_plan_v1", "config": runtime,
            "output_root": str(root), "stages": plan_stages,
            "training_runs": len(NEW_FOLDS) * len(ARMS), "reused_folds": list(REUSED_FOLDS),
            "training_arms": list(ARMS), "feature_encoder": "UNI2-h",
            "folds": list(ALL_FOLDS), "seeds": [3], "default_action": "plan_only",
            "patient_work_executed": False, "execution_implemented": True,
            "checkpoint_selection": "fixed_final_epoch_no_validation_selection",
            "evaluation_contract": "20 repeats aggregated inside each fold before the five-fold statistic",
            "unresolved_variables": [k for k in ("metadata_root", "features_root") if "$" in runtime[k]],
            "content_contract": "train-only frozen reference projection and cluster targets; rectangular matching with unmatched coverage costs",
            "batching": "same per-example compact native forward and one core update per logical batch for all arms",
            "fold_independence": "bins and content reference are fitted from each fold's own training patients only"}


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
    return {p.relative_to(root).as_posix(): sha(p) for p in files if p.is_file()}


def _verify_completed(root, stage, record):
    current = _artifact_hashes(root, stage)
    if current != record.get("artifact_sha256"):
        raise RuntimeError(f"completed stage artifacts changed: {stage['id']}")


def _subprocess_stage(stage, runtime, root, device=None):
    log = root / "logs" / (stage["id"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", "-m", "slotspe_proof.redundancy_multifold",
               "--run", "--worker-stage", stage["id"], "--output-root", str(root)]
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    if device:
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"stage {stage['id']} failed with exit {result.returncode}; retained log: {log}")
def execution_waves(chains):
    """Group chains into waves that may run concurrently.

    The shared bookends (preparation and the final summary) are their own
    waves.  Every fold chain then forms one wave, because the folds do not
    share a model state, a data cache write or a survival bin, and the arms
    inside a fold are ordered only by their shared initialization.
    """
    waves = []
    fold_chains = []
    for chain in chains:
        if len(chain) == 1 and chain[0]["kind"] in ("prepare", "summarize"):
            if fold_chains:
                waves.append(fold_chains)
                fold_chains = []
            waves.append([chain])
        else:
            fold_chains.append(chain)
    if fold_chains:
        waves.append(fold_chains)
    return waves


def _spawn_stage(stage, runtime, root, device=None):
    """Start one stage in its own process with its own retained log."""
    log = root / "logs" / (stage["id"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", "-m", "slotspe_proof.redundancy_multifold",
               "--run", "--worker-stage", stage["id"], "--output-root", str(root)]
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
    stream = log.open("x", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT)
    return stage, process, stream


def stage_dependencies(stage):
    """The stages that must finish before ``stage`` may start.

    An arm of a fold needs that fold's shared initialization.  A fold's
    summary needs all eleven arms of that fold.  Nothing else is shared
    between arms: every arm re-reads the same frozen initialization and the
    same frozen content reference, and the sampling schedule is derived from
    the fold, epoch and position, never from the arm.
    """
    kind = stage.get("kind")
    if kind == "arm":
        return [f"init_fold_{stage['fold']}"]
    if kind == "summarize_fold":
        return [f"arm_fold_{stage['fold']}_{arm}" for arm in ARMS]
    return []


# Measured peak CUDA allocation of each arm on one RTX 5090, taken from the
# completed fold0 queue.  The scheduler uses these figures only to decide how
# many arms may share one GPU; they never enter the model computation.
ARM_PEAK_GIB = {
    "S0": 2.09, "S1": 8.87, "S2": 5.35, "S3": 5.61, "S4": 8.87, "S5": 2.38,
    "R1": 2.09, "R2": 2.09, "R3": 2.09, "R4": 2.09, "R5": 2.09,
}
GPU_CAPACITY_GIB = 31.8
# Each worker process also holds a CUDA context and allocator reserve that the
# torch peak figure does not include; measured at about 1.0 GiB per process.
CUDA_CONTEXT_GIB = 1.0
# The usable budget leaves headroom on the 31.8 GiB device.
GPU_USABLE_GIB = 28.0


def stage_memory_gib(stage):
    """The device memory one stage needs, including its own CUDA context."""
    if stage.get("kind") != "arm":
        return 2.2 + CUDA_CONTEXT_GIB
    return ARM_PEAK_GIB.get(stage.get("arm"), 8.9) + CUDA_CONTEXT_GIB


def _choose_device(stage, gpu_pool, device_load, device_memory):
    """Pick a GPU that can still hold this stage next to what it already runs.

    The choice is a scheduling decision only.  It cannot change the model
    computation, the data, the seeds or the numbers.
    """
    best = None
    best_key = None
    need = stage_memory_gib(stage)
    for device in gpu_pool:
        used = device_memory.get(device, 0.0)
        if used + need > GPU_USABLE_GIB:
            continue
        key = (device_load.get(device, 0), used, device)
        if best_key is None or key < best_key:
            best, best_key = device, key
    return best


def _run_wave(order, pending_ids, runtime, root, gpu_pool, workers):
    """Run the pending stages concurrently, respecting their dependencies.

    Stages of different folds, and stages of the same fold that do not depend
    on each other, may run at the same time.  The dependency rule is the only
    ordering that is enforced: an arm starts after its fold's shared
    initialization, and nothing else.  Every stage still runs in its own
    process with its own retained log, so the concurrency changes the wall
    time only, never the model computation.

    The GPU is chosen per stage from the pool with the measured peak memory
    of that arm, so that a heavy arm never shares a device with another heavy
    arm and the pool is kept busy.
    """
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
                                            - stage_memory_gib(stage_by_id.get(stage_id, {})))
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
            if any(dependency not in completed for dependency in stage_dependencies(stage)):
                continue
            device = None
            if gpu_pool:
                device = _choose_device(stage, gpu_pool, device_load, device_memory)
                if device is None:
                    # No GPU can hold this arm next to the running ones.
                    break
                device_load[device] = device_load.get(device, 0) + 1
                device_memory[device] = device_memory.get(device, 0.0) + stage_memory_gib(stage)
            spawned, process, stream = _spawn_stage(stage, runtime, root, device)
            running[stage["id"]] = (process, stream, device)
        if not running:
            time.sleep(0.5)
            continue
        time.sleep(0.5)
    reap()
    if failures:
        raise RuntimeError("parallel stage group failed: " + "; ".join(sorted(failures)))


def stage_chains(plan_stages):
    """Split the stage list into per-fold chains plus the shared bookends.

    Within one fold the eleven arms stay strictly sequential, because they
    deliberately share one initialization and one sampling schedule.  Chains of
    different folds are independent by construction: each fold fits its own
    survival bins and its own content reference from its own training patients.
    """
    chains = []
    current = []
    current_fold = None
    for stage in plan_stages:
        kind = stage["kind"]
        if kind in ("prepare", "summarize"):
            if current:
                chains.append(current)
                current, current_fold = [], None
            chains.append([stage])
            continue
        fold = stage.get("fold")
        if current and fold != current_fold:
            chains.append(current)
            current = []
        current.append(stage)
        current_fold = fold
    if current:
        chains.append(current)
    return chains


def assign_devices(plan_stages, runtime, devices=None):
    """The physical GPUs this queue may use.

    Device assignment never changes the model computation, the seeds or the
    data; it only decides which GPU executes a stage.  The available set is
    recorded in the queue manifest so that a reader can verify it.
    """
    if not devices:
        return []
    return [str(value) for value in devices]


def execute(config, output_root, resume=False, executor=None, runtime_check=True,
            devices=None, workers=None):
    plan = make_plan(config, output_root)
    runtime, root = plan["config"], Path(plan["output_root"])
    checked = doctor(runtime, runtime_check=runtime_check)
    code = implementation_hashes()
    identity_payload = {"runtime": runtime, "implementation": code, "stages": plan["stages"],
                        "inputs": checked.get("source_and_metadata_sha256", {}),
                        "runtime_versions": checked.get("runtime", {})}
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    device_pool = assign_devices(plan["stages"], runtime, devices)
    if workers is None:
        # Three concurrent stages per GPU was measured to keep both devices
        # busy while staying inside the memory budget of the heaviest arm.
        workers = 3 * len(device_pool) if device_pool else 1
    if workers < 1:
        raise ValueError("workers must be positive")
    if device_pool and workers > 4 * len(device_pool):
        raise ValueError("workers must not oversubscribe the GPU pool")
    manifest_path = root / "queue_manifest.json"
    if root.exists() and any(root.iterdir()):
        if not resume or not manifest_path.is_file():
            raise RuntimeError("nonempty output: preserve it; use a fresh directory or verified --resume")
        manifest = read(manifest_path)
        if manifest.get("identity") != identity or read(root / "resolved_config.json") != runtime:
            raise RuntimeError("configuration, input metadata or implementation drift; cannot resume")
    else:
        root.mkdir(parents=True, exist_ok=True)
        write(root / "resolved_config.json", runtime)
        manifest = {"schema": "slotspe_native_redundancy_multifold_queue_v1", "identity": identity,
                    "identity_payload": identity_payload, "config": runtime,
                    "stages": plan["stages"], "status": "running", "records": {},
                    "device_assignment": device_pool,
                    "max_concurrent_stages": workers,
                    "started_utc": datetime.now(timezone.utc).isoformat()}
        write(manifest_path, manifest)
    custom_executor = executor is not None
    executor = executor or _subprocess_stage

    def verify_live_identity():
        saved = read(manifest_path)
        if (read(root / "resolved_config.json") != runtime or saved.get("config") != runtime or
                saved.get("identity") != identity or saved.get("identity_payload") != identity_payload):
            raise RuntimeError("saved runtime or queue identity changed during execution")
        if implementation_hashes() != code:
            raise RuntimeError("implementation changed while queue was running")
        if any(sha(Path(name)) != value for name, value in identity_payload["inputs"].items()):
            raise RuntimeError("source or metadata changed while queue was running")

    for wave in execution_waves(stage_chains(plan["stages"])):
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
        write(manifest_path, manifest)
        try:
            if custom_executor:
                # In-process path used by the generated tests.
                for stage in pending:
                    executor(stage, runtime, root)
            else:
                _run_wave(order, {stage["id"] for stage in pending}, runtime, root,
                          device_pool, workers)
            verify_live_identity()
            for stage in pending:
                manifest["records"][stage["id"]] = {"status": "completed",
                                                    "artifact_sha256": _artifact_hashes(root, stage)}
            manifest.pop("active_stage", None)
            write(manifest_path, manifest)
            print("completed stage(s) " + ", ".join(stage["id"] for stage in pending), flush=True)
        except BaseException as error:
            manifest["status"] = "failed"
            for stage in pending:
                manifest["records"][stage["id"]] = {"status": "failed", "error": str(error),
                                                    "type": type(error).__name__}
            write(manifest_path, manifest)
            raise
    manifest["status"] = "completed"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write(manifest_path, manifest)
    return manifest


def _initialize_fold(runtime, root, manifest, fold):
    """One shared initialization and one train-only content reference per fold."""
    import torch
    from .patient import _batch
    from .redundancy_content import ContentReference
    from .redundancy_engine import make_initial_states
    torch.set_num_threads(runtime["torch_threads"])
    out = root / "folds" / str(fold) / "initial"
    out.mkdir(parents=True, exist_ok=True)
    core, probe = make_initial_states(manifest, runtime["source_root"], runtime["seeds"][0],
                                      projection_dim=runtime["projection_dim"], slot_iters=runtime["slot_iters"])
    reference = ContentReference(manifest["omic_sizes"], manifest["encoding_dim"],
                                 target_dim=runtime["content_target_dim"],
                                 clusters=runtime["content_clusters"],
                                 seed=runtime["content_reference_seed"])
    fold_data = manifest["folds"][str(fold)]
    # The fold0 queue drew these reference patches from the unseeded global
    # numpy stream, so the fitted moments were not reproducible.  A fixed
    # per-patient draw makes the reference exactly reproducible without
    # changing what it is: a train-fold-only feature-content contract.
    reference.fit(_batch(root, manifest, fold_data, [pid], "cpu", torch,
                         sample_seed=runtime["content_reference_seed"]) for pid in fold_data["train"])
    torch.save(core, out / "core.pt")
    torch.save(probe, out / "probe.pt")
    torch.save(reference.state_dict(), out / "reference.pt")
    write(out / "receipt.json", {
        "fold": fold, "training_examples": len(fold_data["train"]),
        "reference_fit_split": "train_only", "frozen_during_evaluation": True,
        "encoding_dim": manifest["encoding_dim"], "omic_sizes": manifest["omic_sizes"],
        "reference_patch_draw": "fixed per-patient seed derived from content_reference_seed",
        "files_sha256": {n: sha(out / n) for n in ("core.pt", "probe.pt", "reference.pt")},
        "target_scope": "train-standardized fixed random projection, input-only per-example clusters; not biological ground truth"})


def _stability(curve):
    """Training-stability summary derived from the retained per-batch curve."""
    norms = [float(row["core_gradient_norm"]) for row in curve if row.get("core_gradient_norm") is not None]
    losses = [float(row["total_loss"]) for row in curve if row.get("total_loss") is not None]
    finite = [value for value in norms if math.isfinite(value)]
    epoch_losses = {}
    for row in curve:
        epoch_losses.setdefault(int(row["epoch"]), []).append(float(row["total_loss"]))
    epoch_mean = {epoch: sum(values) / len(values) for epoch, values in epoch_losses.items()}
    best_epoch = min(epoch_mean, key=lambda key: epoch_mean[key]) if epoch_mean else None
    final_epoch = max(epoch_mean) if epoch_mean else None
    return {
        "batches": len(norms), "nonfinite_gradient_batches": len(norms) - len(finite),
        "grad_norm_median": statistics.median(finite) if finite else None,
        "grad_norm_max": max(finite) if finite else None,
        "grad_norm_mean": (sum(finite) / len(finite)) if finite else None,
        "batches_over_1e3": sum(1 for value in finite if value > 1e3),
        "batches_over_1e4": sum(1 for value in finite if value > 1e4),
        "train_total_loss_final_epoch_mean": epoch_mean.get(final_epoch),
        "train_total_loss_best_epoch": epoch_mean.get(best_epoch),
        "best_epoch": best_epoch, "final_epoch": final_epoch,
        "last_epoch_train_loss_degradation": (
            epoch_mean[final_epoch] - epoch_mean[best_epoch]
            if final_epoch is not None and best_epoch is not None else None),
    }


def _arm_view(summary, curve):
    evaluation = summary.get("evaluation_summary", {})
    view = {
        "status": summary.get("status"),
        "cindex_mean_over_repeats": evaluation.get("val_cindex_mean_exploratory"),
        "cindex_repeat_std_population": evaluation.get("val_cindex_std_population"),
        "cindex_repeat_min": evaluation.get("val_cindex_min_exploratory"),
        "cindex_repeat_max": evaluation.get("val_cindex_max_exploratory"),
        "cindex_defined_repeats": evaluation.get("defined_cindex_repeats"),
        "evaluation_repeats": evaluation.get("repeats"),
        "patient_risk_std_mean": evaluation.get("patient_risk_std_mean"),
        "patient_risk_range_max": evaluation.get("patient_risk_range_max"),
        "evaluation_randomness_observed": evaluation.get("evaluation_randomness_observed"),
        "slot_diagnostics": summary.get("slot_diagnostics"),
        "counters": summary.get("counters"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "cuda_peak_bytes": summary.get("cuda_peak_bytes"),
        "initial_core_hash": (summary.get("protocol") or {}).get("initial_core_hash"),
        "initial_probe_hash": (summary.get("protocol") or {}).get("initial_probe_hash"),
        "source_unchanged": summary.get("source_unchanged"),
        "training_stability": _stability(curve) if curve else None,
    }
    return view


def _fold_arm_paths(root, fold, arm):
    return root / "folds" / str(fold) / "arms" / arm


def _summarize_fold(root, runtime, fold):
    rows = {}
    for arm in ARMS:
        folder = _fold_arm_paths(root, fold, arm)
        summary = read(folder / "summary.json")
        curve = read(folder / "curve.json")
        rows[arm] = _arm_view(summary, curve)
    result = {"schema": "slotspe_native_redundancy_multifold_fold_v1", "fold": fold, "seed": 3,
              "status": "completed", "arms": rows, "epochs": runtime["epochs"],
              "patches": runtime["patches"], "eval_repeats": runtime["eval_repeats"]}
    write(root / "folds" / str(fold) / "summary.json", result)
    return result


def _fold0_views():
    """Read the completed fold0 queue without re-executing it."""
    root = ROOT / FOLD0_ROOT
    if not root.is_dir():
        raise RuntimeError(f"the reused fold0 queue is missing: {FOLD0_ROOT}")
    rows, provenance = {}, {}
    for arm in ARMS:
        folder = root / "arms" / arm
        summary = read(folder / "summary.json")
        curve = read(folder / "curve.json")
        rows[arm] = _arm_view(summary, curve)
        provenance[arm] = {"summary_sha256": sha(folder / "summary.json"),
                           "last_checkpoint_sha256": summary.get("last_checkpoint_sha256")}
    manifest = read(root / "queue_manifest.json")
    engine_hash = (manifest.get("identity_payload", {}).get("implementation", {}) or {}).get(
        "slotspe_proof/redundancy_engine.py")
    return rows, {"queue_manifest_sha256": sha(root / "queue_manifest.json"),
                  "engine_sha256_at_production": engine_hash,
                  "summary_sha256": {arm: provenance[arm]["summary_sha256"] for arm in ARMS},
                  "last_checkpoint_sha256": {arm: provenance[arm]["last_checkpoint_sha256"] for arm in ARMS}}


def _mean(values):
    finite = [float(value) for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def _fold_stats(values):
    """Mean and between-fold spread of the five per-fold values."""
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return {"fold_mean": None, "fold_std_population": None, "fold_std_sample": None, "folds_defined": 0}
    return {"fold_mean": sum(finite) / len(finite),
            "fold_std_population": statistics.pstdev(finite),
            "fold_std_sample": statistics.stdev(finite) if len(finite) > 1 else None,
            "folds_defined": len(finite)}


def _summarize_all(root, runtime):
    fold_views, provenance = {}, {}
    fold0_rows, fold0_provenance = _fold0_views()
    fold_views[0] = {"source": "reused_completed_fold0_queue", "root": FOLD0_ROOT, "arms": fold0_rows}
    provenance[0] = fold0_provenance
    for fold in NEW_FOLDS:
        path = root / "folds" / str(fold) / "summary.json"
        value = read(path)
        if value.get("status") != "completed":
            raise RuntimeError(f"fold {fold} is not completed")
        fold_views[fold] = {"source": "executed_here", "root": f"folds/{fold}", "arms": value["arms"]}
    per_arm = {}
    for arm in ARMS:
        cindex = {fold: fold_views[fold]["arms"][arm]["cindex_mean_over_repeats"] for fold in ALL_FOLDS}
        entry = {
            "cindex_per_fold": cindex,
            "cindex_five_fold": _fold_stats([cindex[fold] for fold in ALL_FOLDS]),
            "cindex_repeat_std_per_fold": {fold: fold_views[fold]["arms"][arm]["cindex_repeat_std_population"]
                                           for fold in ALL_FOLDS},
            "evaluation_repeats_per_fold": {fold: fold_views[fold]["arms"][arm]["evaluation_repeats"]
                                            for fold in ALL_FOLDS},
            "status_per_fold": {fold: fold_views[fold]["arms"][arm]["status"] for fold in ALL_FOLDS},
            "elapsed_seconds_per_fold": {fold: fold_views[fold]["arms"][arm]["elapsed_seconds"]
                                         for fold in ALL_FOLDS},
            "elapsed_seconds_total": sum(value for value in
                                         (fold_views[fold]["arms"][arm]["elapsed_seconds"] for fold in ALL_FOLDS)
                                         if value is not None),
            "cuda_peak_bytes_per_fold": {fold: fold_views[fold]["arms"][arm]["cuda_peak_bytes"]
                                          for fold in ALL_FOLDS},
        }
        entry["status"] = ("completed" if all(value == "completed" for value in entry["status_per_fold"].values())
                           else "incomplete")
        for modality in ("wsi", "omic"):
            geometry = {}
            for key in ("cosine", "absolute_cosine", "rms_cosine", "effective_rank", "read_overlap",
                        "slot_count", "nonzero_count", "min_norm"):
                per_fold = {fold: (fold_views[fold]["arms"][arm]["slot_diagnostics"] or {})
                            .get(modality, {}).get(key) for fold in ALL_FOLDS}
                geometry[key] = {"per_fold": per_fold, **_fold_stats(list(per_fold.values()))}
            valid = {fold: ((fold_views[fold]["arms"][arm]["slot_diagnostics"] or {})
                            .get(modality, {}).get("valid_pair_draws")) for fold in ALL_FOLDS}
            draws = {fold: ((fold_views[fold]["arms"][arm]["slot_diagnostics"] or {})
                            .get(modality, {}).get("case_draws")) for fold in ALL_FOLDS}
            geometry["valid_pair_draws"] = {"per_fold": valid, **_fold_stats(list(valid.values()))}
            geometry["case_draws"] = {"per_fold": draws, **_fold_stats(list(draws.values()))}
            entry[f"slot_geometry_{modality}"] = geometry
            content = {}
            for key in ("own_mse", "donor_mse", "static_mse", "donor_to_own_mse_ratio",
                        "same_noise_changed_input_distance", "same_input_changed_noise_distance"):
                per_fold = {fold: (fold_views[fold]["arms"][arm]["slot_diagnostics"] or {})
                            .get(modality, {}).get(key) for fold in ALL_FOLDS}
                content[key] = {"per_fold": per_fold, **_fold_stats(list(per_fold.values()))}
            entry[f"content_probe_{modality}"] = content
        stability = {}
        for key in ("grad_norm_median", "grad_norm_max", "grad_norm_mean", "batches_over_1e3",
                    "batches_over_1e4", "nonfinite_gradient_batches", "batches",
                    "train_total_loss_final_epoch_mean", "train_total_loss_best_epoch",
                    "last_epoch_train_loss_degradation"):
            per_fold = {fold: ((fold_views[fold]["arms"][arm]["training_stability"] or {}).get(key))
                        for fold in ALL_FOLDS}
            stability[key] = {"per_fold": per_fold, **_fold_stats(list(per_fold.values()))}
        entry["training_stability"] = stability
        per_arm[arm] = entry
    result = {
        "schema": "slotspe_native_redundancy_multifold_summary_v1", "status": "completed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scope": {"study": "blca", "feature_encoder": "UNI2-h", "folds": list(ALL_FOLDS), "seed": 3,
                  "arms": list(ARMS), "reused_folds": list(REUSED_FOLDS), "executed_folds": list(NEW_FOLDS)},
        "protocol": {"epochs": runtime["epochs"], "patches": runtime["patches"],
                     "batch_size": runtime["batch_size"], "eval_repeats": runtime["eval_repeats"],
                     "splits_dir": runtime["splits_dir"], "lr": runtime["lr"],
                     "probe_lr": runtime["probe_lr"], "projection_dim": runtime["projection_dim"],
                     "slot_iters": runtime["slot_iters"], "content_target_dim": runtime["content_target_dim"],
                     "content_clusters": runtime["content_clusters"],
                     "checkpoint_selection": "fixed_final_epoch_no_validation_selection",
                     "evaluation_aggregation": "the 20 repeats are averaged inside each fold first; "
                                               "the five-fold statistic is then computed over the five per-fold values",
                     "std_convention": "population and sample standard deviations are both reported for the between-fold spread"},
        "folds": fold_views, "fold_provenance": provenance, "per_arm": per_arm,
        "caveats": [
            "fold0 was executed by the earlier single-fold queue and is reused unchanged; folds 1-4 were executed by this queue.",
            "The reused fold0 queue drew the first training logical batch and the content-reference patches from an unseeded global numpy stream, so those draws were not reproducible and were not identical across arms. Folds 1-4 use a fixed, reproducible draw, so the first logical batch and the content-reference patch subset differ from fold0 by that mechanism alone.",
            "The five-fold statistic treats the five per-fold values as the units; the 20 evaluation repeats inside a fold are not independent patients and are averaged before the fold value is formed.",
            "C-index differences between arms are exploratory single-seed observations and are not corrected for multiplicity here."],
        "claim_boundary": "Five-fold single-seed BLCA diagnostic of one original SlotSPE implementation family; "
                          "no generalization, clinical benefit or multiplicity-corrected significance is asserted",
    }
    write(root / "summary.json", result)
    lines = ["SlotSPE BLCA / UNI2-h / folds 0-4 / seed3 / eleven arms: all folds completed.",
             f"fold0 reused from {FOLD0_ROOT}; folds 1-4 executed here.",
             "",
             "arm  cindex_per_fold(0..4)  fold_mean  fold_std_population"]
    for arm in ARMS:
        entry = per_arm[arm]
        values = " ".join(f"{entry['cindex_per_fold'][fold]:.4f}" if entry["cindex_per_fold"][fold] is not None
                          else "  n/a " for fold in ALL_FOLDS)
        lines.append(f"{arm:>3}  {values}  {entry['cindex_five_fold']['fold_mean']:.4f}  "
                     f"{entry['cindex_five_fold']['fold_std_population']:.4f}")
    (root / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def worker(stage_id, root):
    root = private_output(root)
    manifest = read(root / "queue_manifest.json")
    active = manifest.get("active_stage")
    active_ids = [active] if isinstance(active, str) else list(active or [])
    if manifest.get("status") != "running" or stage_id not in active_ids:
        raise RuntimeError("worker must belong to the active explicitly-run queue stage")
    runtime = read(root / "resolved_config.json")
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
        _initialize_fold(runtime, root, prepared, stage["fold"])
    elif kind == "arm":
        import torch
        from .redundancy_content import ContentReference
        from .redundancy_engine import run_arm
        folder = root / "folds" / str(stage["fold"]) / "initial"
        reference = ContentReference(prepared["omic_sizes"], prepared["encoding_dim"],
                                     target_dim=runtime["content_target_dim"],
                                     clusters=runtime["content_clusters"],
                                     seed=runtime["content_reference_seed"])
        reference.load_state_dict(torch.load(folder / "reference.pt", map_location="cpu", weights_only=True))
        core = torch.load(folder / "core.pt", map_location="cpu", weights_only=True)
        probe = torch.load(folder / "probe.pt", map_location="cpu", weights_only=True)
        run_arm(runtime, root, prepared, stage["arm"], core, probe, reference,
                fold=stage["fold"], output_root=root / "folds" / str(stage["fold"]))
    elif kind == "summarize_fold":
        _summarize_fold(root, runtime, stage["fold"])
    else:
        _summarize_all(root, runtime)


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
        print(json.dumps({"status": result["status"], "output_root": str(private_output(args.output_root))}))
    elif args.doctor:
        verify_config_pins(config)
        print(json.dumps(doctor(runtime), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(make_plan(resolved, args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
