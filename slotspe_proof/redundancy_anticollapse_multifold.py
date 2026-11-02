"""Anti-collapse queue for the SlotSPE slot-geometry investigation.

This queue trains the two anti-collapse arms on all five declared folds
under the same protocol as the completed best-checkpoint redundancy queue:
BLCA, UNI2-h, seed 3, 30 epochs, 4096 real patches, batch size 8, 20
repeated evaluations, the 5fold split family and the original per-epoch
best-validation checkpoint rule.

It reuses the completed queue's preparation, fold initialization, device
assignment, disk budget guard, wave scheduler and identity verification,
so the arms differ from the completed R5 arm only in the two documented
interventions: core gradient clipping and the Gram weight decay.

It does not modify any file pinned by the completed queues.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from . import redundancy_best_multifold as best
from . import redundancy_multifold as base
from . import redundancy_anticollapse_engine as engine
from .redundancy_anticollapse_engine import (ANTI_COLLAPSE_ARMS, ARM_CONTRACTS_ANTI,
                                             GRAM_DECAY_FLOOR, run_arm_anti_collapse)

ALL_FOLDS = (0, 1, 2, 3, 4)
NEW_FOLDS = (0, 1, 2, 3, 4)
REUSED_FOLDS = ()
DEFAULT_OUTPUT = "local_evidence/anticollapse_blca_uni2h_folds0to4_seed3_run01"
CONFIG_PATH = "configs/slotspe_anticollapse_multifold_queue.json"
CLIP_NORM = 1.0
WORKER_MODULE = "slotspe_proof.redundancy_anticollapse_multifold"


def implementation_hashes():
    """Hash the modules this queue's behaviour depends on."""
    names = ("redundancy_anticollapse_engine.py", "redundancy_best_engine.py",
             "redundancy_engine.py", "redundancy_content.py", "slot_regularization.py",
             "patient.py")
    root = Path(__file__).resolve().parent
    result = {}
    for name in names:
        path = root / name
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def normalize_runtime(config, metadata_root=None, features_root=None, device=None):
    """Resolve and lock the runtime values for this queue.

    The scope is deliberately narrower than the eleven-arm queue: exactly
    the two anti-collapse arms, exactly the five declared folds, exactly
    seed 3, and exactly the declared budget.  Nothing here may widen
    itself.
    """
    if "requested_scope" in config:
        scope = config["requested_scope"]
        stage = next((s for s in config["stages"] if s["id"] == "anti_collapse_diagnostic"), None)
        if stage is None:
            raise ValueError("the anti_collapse_diagnostic stage is required")
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
    else:
        runtime = dict(config)
    defaults = {
        "source_root": "third_party/SlotSPE", "metadata_root": "${SLOTSPE_METADATA_ROOT}",
        "features_root": "${SLOTSPE_FEATURES_ROOT}", "label_col": "survival_months_dss",
        "signature": "combine", "device": "cuda:0", "lr": 0.0005,
        "probe_lr": 0.0005, "torch_threads": 2, "encoding_dim": None,
        "projection_dim": 256, "slot_iters": 10, "content_target_dim": 32,
        "content_clusters": 8, "content_reference_seed": 731, "dias_distance": 0.05,
        "slotbert_temperature": 0.5, "experiment": "original_audit", "splits_dir": "5fold",
        "checkpoint_selection": "per_epoch_validation_best_then_reload_and_reevaluate",
        "gradient_clip_norm": CLIP_NORM,
    }
    runtime = {**defaults, **runtime}
    for key, value in (("metadata_root", metadata_root), ("features_root", features_root),
                       ("device", device)):
        if value is not None:
            runtime[key] = value
    if runtime.get("study") != "blca" or runtime.get("feature_encoder") != "UNI2-h":
        raise ValueError("scope is exactly BLCA / UNI2-h")
    if list(runtime.get("folds") or []) != list(ALL_FOLDS):
        raise ValueError("scope is exactly folds 0,1,2,3,4")
    if list(runtime.get("seeds") or []) != [3]:
        raise ValueError("scope is exactly seed3")
    if list(runtime.get("variants") or []) != list(ANTI_COLLAPSE_ARMS):
        raise ValueError("scope is exactly the declared anti-collapse arms")
    for key in ("epochs", "patches", "batch_size", "eval_repeats", "torch_threads",
                "projection_dim", "slot_iters", "content_target_dim", "content_clusters",
                "content_reference_seed"):
        value = runtime[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if runtime["projection_dim"] != 256 or runtime["slot_iters"] != 10:
        raise ValueError("patient queue must use native 256 dimensions and 10 slot iterations")
    if runtime["epochs"] != 30 or runtime["patches"] != 4096 or runtime["batch_size"] != 8:
        raise ValueError("the queue keeps the declared budget: 30 epochs / 4096 patches / batch 8")
    if runtime["eval_repeats"] != 20:
        raise ValueError("the queue keeps the 20-repeat evaluation contract")
    if runtime.get("splits_dir") != "5fold":
        raise ValueError("the queue uses the 5fold split family")
    if runtime.get("checkpoint_selection") != "per_epoch_validation_best_then_reload_and_reevaluate":
        raise ValueError("the queue keeps the original per-epoch best checkpoint rule")
    clip = runtime.get("gradient_clip_norm")
    if isinstance(clip, bool) or not isinstance(clip, (int, float)) or clip <= 0:
        raise ValueError("gradient_clip_norm must be a positive number for this queue")
    for key in ("lr", "probe_lr", "slotbert_temperature"):
        if isinstance(runtime[key], bool) or not isinstance(runtime[key], (int, float)) or \
                not (0 < runtime[key] < 1):
            raise ValueError(f"{key} must be a positive fraction")
    return runtime


def make_plan(config, output_root):
    runtime = normalize_runtime(config)
    root = base.private_output(output_root or base.ROOT / DEFAULT_OUTPUT)
    stages = [{"id": "prepare", "kind": "prepare", "depends_on": []}]
    for fold in NEW_FOLDS:
        stages.append({"id": f"init_fold_{fold}", "kind": "initialize", "fold": fold,
                       "depends_on": ["prepare"]})
        for arm in runtime["variants"]:
            stages.append({"id": f"arm_fold_{fold}_{arm}", "kind": "arm", "fold": fold,
                           "arm": arm, "depends_on": [f"init_fold_{fold}"]})
        stages.append({"id": f"summarize_fold_{fold}", "kind": "summarize_fold", "fold": fold,
                       "depends_on": [f"arm_fold_{fold}_{arm}" for arm in runtime["variants"]]})
    stages.append({"id": "summarize", "kind": "summarize",
                   "depends_on": [f"summarize_fold_{fold}" for fold in NEW_FOLDS]})
    return {"config": runtime, "output_root": str(root), "stages": stages}


def expected_artifacts(stage):
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


def doctor(config, runtime_check=True, output_root=None):
    runtime = normalize_runtime(config)
    from .queue import preflight
    result = preflight(runtime, checks_only=False, check_runtime=runtime_check)
    if runtime_check:
        import h5py
        import scipy
        result["runtime"].update(h5py=h5py.__version__, scipy=scipy.__version__)
    result.update(feature_encoder="UNI2-h", feature_extraction_started=False,
                  patient_training_executed=False,
                  anti_collapse_arms={arm: ARM_CONTRACTS_ANTI[arm] for arm in ANTI_COLLAPSE_ARMS},
                  gram_decay_floor=GRAM_DECAY_FLOOR)
    return result


def worker(stage_id, root, device=None):
    """Execute one queued stage in this process."""
    import torch
    from .patient import _manifest
    from .redundancy_content import ContentReference
    root = base.private_output(root)
    manifest = base.read(root / "queue_manifest.json")
    active = manifest.get("active_stage")
    active_ids = [active] if isinstance(active, str) else list(active or [])
    if manifest.get("status") != "running" or stage_id not in active_ids:
        raise RuntimeError("worker must belong to the active explicitly-run queue stage")
    runtime = base.read(root / "resolved_config.json")
    stage = next((s for s in manifest["stages"] if s["id"] == stage_id), None)
    if stage is None:
        raise ValueError("unknown stage")
    kind = stage["kind"]
    if kind == "prepare":
        from .patient import prepare
        prepare(runtime, root)
        return
    prepared = _manifest(runtime, root, verify_cache=True)
    if kind == "initialize":
        base._initialize_fold(runtime, root, prepared, stage["fold"])
        return
    if kind == "arm":
        folder = root / "folds" / str(stage["fold"]) / "initial"
        reference = ContentReference(prepared["omic_sizes"], prepared["encoding_dim"],
                                     target_dim=runtime["content_target_dim"],
                                     clusters=runtime["content_clusters"],
                                     seed=runtime["content_reference_seed"])
        reference.load_state_dict(torch.load(folder / "reference.pt", map_location="cpu",
                                             weights_only=True))
        core = torch.load(folder / "core.pt", map_location="cpu", weights_only=True)
        probe = torch.load(folder / "probe.pt", map_location="cpu", weights_only=True)
        run_arm_anti_collapse(runtime, root, prepared, stage["arm"], core, probe, reference,
                              fold=stage["fold"],
                              output_root=root / "folds" / str(stage["fold"]))
        return
    if kind == "summarize_fold":
        _summarize_fold(root, runtime, stage["fold"])
        return
    _summarize_all(root, runtime)


def _arm_view_anti(root, fold, arm):
    """Build the per-arm view for THIS queue's schema.

    The completed queue's shared view helper cannot be reused here.  That
    helper reads the C-index aggregation out of ``evaluation_summary``, but
    this queue's engine writes the slot-diagnostics summary into
    ``evaluation_summary`` and leaves the C-index aggregation out of the
    summary entirely.  Reading the shared helper would therefore report a
    null C-index for every arm.

    The numbers are therefore taken from the same retained artifacts the
    completed queue uses, but from this queue's own schema:
    ``eval_repeats.json`` holds the twenty repeated evaluations with their
    C-index, ``epoch_validation.json`` holds the per-epoch validation
    passes, and ``summary.json`` holds the slot diagnostics and counters.
    The field names match the completed queue so that the fold and root
    summaries stay comparable with it.
    """
    folder = base._fold_arm_paths(root, fold, arm)
    summary = base.read(folder / "summary.json")
    curve = base.read(folder / "curve.json")
    repeats = base.read(folder / "eval_repeats.json")
    epoch_validation = base.read(folder / "epoch_validation.json")

    scores = [float(value["cindex"]) for value in repeats if value.get("cindex") is not None]
    risks = [[float(x) for x in value["risks"]] for value in repeats]
    view = {
        "status": summary.get("status"),
        "cindex_mean_over_repeats": (sum(scores) / len(scores)) if scores else None,
        "cindex_repeat_std_population": (float(base.statistics.pstdev(scores))
                                          if len(scores) > 1 else 0.0 if scores else None),
        "cindex_repeat_min": min(scores) if scores else None,
        "cindex_repeat_max": max(scores) if scores else None,
        "cindex_defined_repeats": len(scores),
        "evaluation_repeats": len(repeats),
        "patient_risk_std_mean": None,
        "patient_risk_range_max": None,
        "evaluation_randomness_observed": None,
        "slot_diagnostics": summary.get("slot_diagnostics"),
        "counters": summary.get("counters"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "cuda_peak_bytes": summary.get("cuda_peak_bytes"),
        "initial_core_hash": (summary.get("protocol") or {}).get("initial_core_hash"),
        "initial_probe_hash": (summary.get("protocol") or {}).get("initial_probe_hash"),
        "source_unchanged": summary.get("source_unchanged"),
        "training_stability": base._stability(curve) if curve else None,
        "best_epoch": summary.get("best_epoch"),
        "best_epoch_cindex_single_pass": summary.get("best_epoch_cindex_single_pass"),
        "best_checkpoint_sha256": base.sha(folder / "best.pt"),
        "epoch_validation": epoch_validation,
    }
    if risks and all(len(row) == len(risks[0]) for row in risks):
        spread_max = []
        for index in range(len(risks[0])):
            column = [risks[repeat][index] for repeat in range(len(risks))]
            spread_max.append(max(column) - min(column))
        view["patient_risk_std_mean"] = None
        view["patient_risk_range_max"] = max(spread_max)
        view["evaluation_randomness_observed"] = any(value != 0.0 for value in spread_max)    # The gradient norm in the curve is the value AFTER clipping, so the
    # pre-clip norm is reported separately to keep the spike evidence.
    pre_clip = [float(row["pre_clip_gradient_norm"]) for row in curve
                if row.get("pre_clip_gradient_norm") is not None]
    if pre_clip:
        view["pre_clip_gradient_norm_max"] = max(pre_clip)
        view["pre_clip_gradient_norm_median"] = base.statistics.median(pre_clip)
        view["pre_clip_batches_over_1e3"] = sum(1 for value in pre_clip if value > 1e3)
        view["pre_clip_batches_over_1e4"] = sum(1 for value in pre_clip if value > 1e4)
    clipped = [row for row in curve if row.get("gradient_was_clipped")]
    if clipped:
        view["clipped_steps"] = len(clipped)
        view["clipped_steps_fraction"] = len(clipped) / len(curve)
    return view


def _reconcile_failed_records(root, records, stages):
    """Recover the stages that failed on a launcher error but finished.

    The first wave of this queue failed in its launcher, before the stage
    bodies could run, because the retained log of an earlier attempt was
    opened in exclusive mode.  The launcher marked every stage of that
    wave as failed, including the stages whose training had in fact
    completed and whose artifacts are complete on disk.

    A stage is therefore reconciled to completed when its full artifact
    set is present and every artifact can be hashed.  A stage whose
    artifacts are incomplete or missing stays failed, so a half-trained
    arm is never mistaken for a completed one.  Nothing is deleted and no
    artifact is rewritten: the artifacts are hashed exactly as they are,
    and the reconciliation is recorded in the stage's own record.
    """
    index = {stage["id"]: stage for stage in stages}
    result = {}
    for stage_id, record in records.items():
        if record.get("status") != "failed":
            result[stage_id] = record
            continue
        stage = index.get(stage_id)
        if stage is None:
            result[stage_id] = record
            continue
        try:
            hashes = best._artifact_hashes(root, stage)
        except Exception:
            result[stage_id] = record
            continue
        result[stage_id] = {"status": "completed", "reconciled_from": "failed",
                            "reconciled_error_type": record.get("type"),
                            "reconciled_error": record.get("error"),
                            "artifact_sha256": hashes}
    return result


def _clear_failed_stage(root, stage, record):
    """Release a genuinely failed stage so that it can be re-run.

    A stage that failed with incomplete artifacts cannot simply be marked
    as pending, because the queue refuses to write into a folder that
    already holds content.  For an arm or an initialization the incomplete
    folder is therefore moved aside rather than deleted: the original files
    are preserved under a quarantine name, so nothing is lost and the
    failure can still be inspected.

    A fold summary is different.  Its stage folder is the fold directory
    itself, which legitimately holds that fold's ``initial`` and ``arms``
    subfolders, so the folder must never be moved and nothing inside it
    is removed.  The summary file that the stage itself would have written
    is simply overwritten by the re-run.
    """
    if stage.get("kind") == "summarize_fold":
        return None
    folder = best._stage_folder(stage)
    if not folder:
        return None
    path = root / folder
    if not path.exists():
        return None
    if not any(path.iterdir()):
        return None
    quarantine = root / "quarantined_stages" / stage["id"]
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    if quarantine.exists():
        index = 1
        while (quarantine.parent / f"{stage['id']}_{index}").exists():
            index += 1
        quarantine = quarantine.parent / f"{stage['id']}_{index}"
    path.rename(quarantine)
    return quarantine


def _summarize_fold(root, runtime, fold):
    """Summarize one fold over THIS queue's arms.

    The completed queue's helper hard-codes its own eleven arms, so it
    cannot be reused here: this queue trains only the two anti-collapse
    arms and the eleven-arm folders do not exist under this output root.
    """
    rows = {}
    for arm in ANTI_COLLAPSE_ARMS:
        rows[arm] = _arm_view_anti(root, fold, arm)
    result = {"schema": "slotspe_anticollapse_multifold_fold_v1", "fold": fold, "seed": 3,
              "status": "completed", "arms": rows, "epochs": runtime["epochs"],
              "patches": runtime["patches"], "eval_repeats": runtime["eval_repeats"],
              "checkpoint_selection": "per_epoch_validation_best_then_reload_and_reevaluate"}
    base.write(root / "folds" / str(fold) / "summary.json", result)
    return result


def _summarize_all(root, runtime):
    """Summarize every fold of this queue over this queue's two arms.

    The completed queue's helper hard-codes its own eleven arms and its own
    fold list, so it is replaced here rather than reused.  Only the arms and
    the folds that this queue actually declares are read, and a fold that is
    missing or incomplete stops the summary instead of being silently
    skipped.
    """
    fold_views = {}
    for fold in NEW_FOLDS:
        path = root / "folds" / str(fold) / "summary.json"
        value = base.read(path)
        if value.get("status") != "completed":
            raise RuntimeError(f"fold {fold} is not completed")
        fold_views[fold] = {"source": "executed_here", "root": f"folds/{fold}", "arms": value["arms"]}
    per_arm = {}
    for arm in ANTI_COLLAPSE_ARMS:
        cindex = {fold: fold_views[fold]["arms"][arm]["cindex_mean_over_repeats"] for fold in NEW_FOLDS}
        best_epoch = {fold: fold_views[fold]["arms"][arm].get("best_epoch") for fold in NEW_FOLDS}
        entry = {
            "cindex_per_fold": cindex,
            "cindex_five_fold": base._fold_stats([cindex[fold] for fold in NEW_FOLDS]),
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
        "schema": "slotspe_anticollapse_multifold_summary_v1", "status": "completed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scope": {"study": "blca", "feature_encoder": "UNI2-h", "folds": list(NEW_FOLDS), "seed": 3,
                  "arms": list(ANTI_COLLAPSE_ARMS), "executed_folds": list(NEW_FOLDS),
                  "reused_folds": list(REUSED_FOLDS),
                  "anti_collapse_arms": {arm: ARM_CONTRACTS_ANTI[arm] for arm in ANTI_COLLAPSE_ARMS},
                  "fold_scope_reason": "Every one of the five folds is trained from scratch under the "
                                       "best-checkpoint rule, so this is 5 folds x 2 arms = 10 new "
                                       "training runs. The intervention is the core gradient clip and "
                                       "the optional Gram weight decay; the model computation, the "
                                       "initialization, the patch budget, the loss structure of the "
                                       "completed R5 arm, the evaluation contract and the checkpoint "
                                       "rule are unchanged.",
                  },
        "protocol": {"epochs": runtime["epochs"], "patches": runtime["patches"],
                     "batch_size": runtime["batch_size"], "eval_repeats": runtime["eval_repeats"],
                     "splits_dir": runtime["splits_dir"], "lr": runtime["lr"],
                     "probe_lr": runtime["probe_lr"], "projection_dim": runtime["projection_dim"],
                     "slot_iters": runtime["slot_iters"], "content_target_dim": runtime["content_target_dim"],
                     "content_clusters": runtime["content_clusters"],
                     "gradient_clip_norm": runtime.get("gradient_clip_norm"),
                     "gram_decay_floor": (engine.GRAM_DECAY_FLOOR
                                          if any(ARM_CONTRACTS_ANTI[arm]["decay"]
                                                 for arm in ANTI_COLLAPSE_ARMS) else None),
                     "checkpoint_selection": "per_epoch_validation_best_then_reload_and_reevaluate",
                     "checkpoint_rule_source": "third_party/SlotSPE/utils/core_utils.py:421-475",
                     "per_epoch_validation": "one full validation pass after each epoch; the global random "
                                             "streams are restored afterwards so the training trajectory "
                                             "equals the final-epoch protocol for the same arm, fold and seed",
                     "evaluation_aggregation": "the 20 repeats are averaged inside each fold first; "
                                               "the five-fold statistic is then computed over the five per-fold values",
                     "std_convention": "population and sample standard deviations are both reported for the between-fold spread"},
        "folds": fold_views, "per_arm": per_arm,
        "caveats": [
            "The reported C-index is a validation-selected peak. A validation-selected peak is not an independently tested generalization result.",
            "Selecting the best of 30 validation evaluations inflates the reported value relative to an independent test set by an amount that depends on the epoch-to-epoch spread of the validation metric.",
            "The twenty evaluation repeats are repeats of one trained model on the same validation fold and are not independent patients.",
            "Arm differences are exploratory single-seed observations with no multiple-comparison correction.",
            "The fold standard deviation reflects the fold split only and contains no training randomness.",
            "Slot geometry and content retention are descriptive measurements and not clinical endpoints.",
        ],
        "claim_boundary": "single-seed validation-selected comparison on BLCA; no independent-test, "
                          "generalization or clinical claim",
    }
    base.write(root / "summary.json", result)
    lines = ["SlotSPE BLCA / UNI2-h / folds 0-4 / seed3 / two anti-collapse arms: "
             "per-epoch best-checkpoint rule, all folds completed.",
             "",
             "arm  cindex_per_fold(0..4)  fold_mean  fold_std_population  best_epochs"]
    for arm in ANTI_COLLAPSE_ARMS:
        entry = per_arm[arm]
        values = " ".join(f"{entry['cindex_per_fold'][fold]:.4f}"
                          if entry["cindex_per_fold"][fold] is not None else "  n/a "
                          for fold in NEW_FOLDS)
        epochs = " ".join(str(entry["best_epoch_per_fold"][fold]) for fold in NEW_FOLDS)
        lines.append(f"{arm:>3}  {values}  {entry['cindex_five_fold']['fold_mean']:.4f}  "
                     f"{entry['cindex_five_fold']['fold_std_population']:.4f}  {epochs}")
    (root / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _spawn_stage(stage, runtime, root, device=None):
    """Start one stage in its own process with its own retained log.

    The worker command names THIS module, so the arm stages train under the
    anti-collapse protocol.

    The log is opened in append mode.  The exclusive mode that the completed
    queues use is correct for a first attempt but raises FileExistsError when
    a stage is retried after an earlier attempt, because the retained log of
    that attempt is still present.  Appending keeps the earlier attempt
    visible instead of overwriting or deleting it.
    """
    log = root / "logs" / (stage["id"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", "-m", WORKER_MODULE,
               "--run", "--worker-stage", stage["id"], "--output-root", str(root)]
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
    stream = log.open("a", encoding="utf-8")
    process = subprocess.Popen(command, cwd=base.ROOT, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT)
    return stage, process, stream


def _run_wave(order, pending_ids, runtime, root, gpu_pool, workers, satisfied=None):
    """Run one wave of stages concurrently.

    The shared scheduler is used with this module's own spawn function, so
    the worker command names this module and the arms train under the
    anti-collapse protocol.  Everything else about the wave, the device
    assignment and the dependency rule is the shared implementation.

    The shared dependency helper names the eleven completed arms, so this
    queue's own dependency rule is installed for the duration of the wave.

    ``satisfied`` names the stages that were already complete before this
    wave started.  The shared scheduler only knows about the stages that
    it spawns itself, so on a resume it would wait forever for a
    dependency that is already finished: the fold 3 initialization was
    completed by an earlier attempt, but the arms that depend on it are
    not, and the dependency would never be satisfied inside the wave.
    The already-satisfied dependencies are therefore treated as met.
    """
    original_spawn = base._spawn_stage
    original_dependencies = base.stage_dependencies
    base._spawn_stage = _spawn_stage
    already = set(satisfied or ())
    if already:
        def dependencies_for(stage):
            """This queue's rule, minus the dependencies already met."""
            return [name for name in stage_dependencies(stage) if name not in already]
        base.stage_dependencies = dependencies_for
    else:
        base.stage_dependencies = stage_dependencies
    original_memory = base.stage_memory_gib

    def memory_for(stage):
        """Use the measured peak of the completed R5 arm for these arms.

        The anti-collapse arms keep the completed R5 loss structure and the
        same model, so the measured R5 peak applies.  It is a scheduling
        input only and never enters the model computation.
        """
        if stage.get("kind") == "arm" and stage.get("arm") in ANTI_COLLAPSE_ARMS:
            return base.ARM_PEAK_GIB["R5"] + base.CUDA_CONTEXT_GIB
        return original_memory(stage)

    base.stage_memory_gib = memory_for
    try:
        base._run_wave(order, pending_ids, runtime, root, gpu_pool, workers)
    finally:
        base._spawn_stage = original_spawn
        base.stage_dependencies = original_dependencies
        base.stage_memory_gib = original_memory


def stage_dependencies(stage):
    """This queue's dependency rule.

    An arm needs that fold's shared initialization, a fold summary needs
    that fold's anti-collapse arms, and nothing else is shared.
    """
    kind = stage.get("kind")
    if kind == "arm":
        return [f"init_fold_{stage['fold']}"]
    if kind == "summarize_fold":
        return [f"arm_fold_{stage['fold']}_{arm}" for arm in ANTI_COLLAPSE_ARMS]
    return []


def execute(config, output_root, resume=False, runtime_check=True, devices=None,
            workers=None):
    """Run the anti-collapse queue with the shared scheduler and guards."""
    plan = make_plan(config, output_root)
    runtime, root = plan["config"], Path(plan["output_root"])
    disk_budget = best.check_disk_budget(root)
    checked = doctor(runtime, runtime_check=runtime_check, output_root=root)
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
        # A directory that holds only a complete, verifiable preparation is
        # an acceptable starting point: the preparation is the expensive
        # part and it is verified against the input hashes before reuse.
        # Anything else must be preserved and re-run in a fresh directory.
        entries = [entry for entry in root.iterdir() if entry.name != "prepared"]
        if entries or not (root / "prepared" / "manifest.json").is_file():
            if not resume or not manifest_path.is_file():
                raise RuntimeError("nonempty output: preserve it; use a fresh directory or verified --resume")
            manifest = base.read(manifest_path)
            if manifest.get("identity") != identity or base.read(root / "resolved_config.json") != runtime:
                raise RuntimeError("configuration, input metadata or implementation drift; cannot resume")
        else:
            root.mkdir(parents=True, exist_ok=True)
            base.write(root / "resolved_config.json", runtime)
            manifest = {"schema": "slotspe_anticollapse_multifold_queue_v1",
                        "identity": identity, "identity_payload": identity_payload,
                        "config": runtime, "stages": plan["stages"], "status": "running",
                        "records": {}, "device_assignment": device_pool,
                        "max_concurrent_stages": workers, "disk_budget": disk_budget,
                        "anti_collapse_arms": {arm: ARM_CONTRACTS_ANTI[arm]
                                               for arm in ANTI_COLLAPSE_ARMS},
                        "started_utc": datetime.now(timezone.utc).isoformat()}
            base.write(manifest_path, manifest)
    else:
        root.mkdir(parents=True, exist_ok=True)
        base.write(root / "resolved_config.json", runtime)
        manifest = {"schema": "slotspe_anticollapse_multifold_queue_v1",
                    "identity": identity, "identity_payload": identity_payload,
                    "config": runtime, "stages": plan["stages"], "status": "running",
                    "records": {}, "device_assignment": device_pool,
                    "max_concurrent_stages": workers, "disk_budget": disk_budget,
                    "anti_collapse_arms": {arm: ARM_CONTRACTS_ANTI[arm]
                                           for arm in ANTI_COLLAPSE_ARMS},
                    "started_utc": datetime.now(timezone.utc).isoformat()}
        base.write(manifest_path, manifest)

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
        disk_budget = best.check_disk_budget(root)
        # A stage that failed in the launcher may still have produced its
        # complete artifacts.  Those are reconciled to completed here, so
        # that the resume reuses the finished training instead of
        # rejecting the queue or overwriting the finished arms.
        manifest["records"] = _reconcile_failed_records(root, manifest["records"], plan["stages"])
        base.write(manifest_path, manifest)
        for prior in plan["stages"]:
            record = manifest["records"].get(prior["id"], {})
            if record.get("status") == "completed":
                best._verify_completed(root, prior, record)
        order = [stage for chain in wave for stage in chain]
        pending = []
        for stage in order:
            record = manifest["records"].get(stage["id"], {})
            if record.get("status") == "completed":
                continue
            if record.get("status") == "running":
                # An interrupted stage whose process may still have been
                # alive is never released automatically.  A stage that was
                # recorded as failed, on the other hand, has a terminated
                # process and may be released.
                raise RuntimeError(f"partial stage retained: {stage['id']}; use a fresh batch")
            if record.get("status") == "failed":
                # A stage that failed with incomplete artifacts is released
                # so that it can be re-run from scratch.  The incomplete
                # folder is quarantined rather than deleted, so the failed
                # attempt stays inspectable.
                quarantine = _clear_failed_stage(root, stage, record)
                manifest["records"][stage["id"]] = {
                    "status": "released_for_rerun",
                    "released_from": record.get("status"),
                    "released_error_type": record.get("type"),
                    "released_error": record.get("error"),
                    "quarantined_to": str(quarantine.relative_to(root)) if quarantine else None,
                }
                base.write(manifest_path, manifest)
                record = manifest["records"][stage["id"]]
            folder = best._stage_folder(stage)
            if folder and (root / folder).exists() and any((root / folder).iterdir()):
                # A complete, verifiable preparation is the one partial
                # folder that may be present before the queue starts: the
                # prepare stage verifies it against the input hashes and
                # reuses it instead of rebuilding it.
                #
                # A fold summary is written into the fold directory, which
                # legitimately holds that fold's finished initialization
                # and arms, so its presence is not a partial stage.
                if stage.get("kind") == "summarize_fold":
                    pass
                elif not (folder == "prepared" and (root / "prepared" / "manifest.json").is_file()):
                    raise RuntimeError(f"unrecorded partial stage folder retained: {folder}")
            pending.append(stage)
        if not pending:
            continue
        # The stages that were already complete before this wave are the
        # dependencies that the wave may rely on.  They are passed to the
        # scheduler so that a resumed wave does not wait forever for a
        # dependency that an earlier attempt already produced.
        satisfied = {stage["id"] for stage in plan["stages"]
                     if manifest["records"].get(stage["id"], {}).get("status") == "completed"}
        manifest["status"] = "running"
        manifest["active_stage"] = [stage["id"] for stage in pending]
        for stage in pending:
            manifest["records"][stage["id"]] = {"status": "running"}
        base.write(manifest_path, manifest)
        try:
            _run_wave(order, {stage["id"] for stage in pending}, runtime, root,
                      device_pool, workers, satisfied=satisfied)
            verify_live_identity()
            for stage in pending:
                manifest["records"][stage["id"]] = {"status": "completed",
                                                    "artifact_sha256": best._artifact_hashes(root, stage)}
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
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--metadata-root")
    parser.add_argument("--features-root")
    parser.add_argument("--device")
    parser.add_argument("--devices", help="Comma-separated physical GPU ids; execution only.")
    parser.add_argument("--workers", type=int, help="Concurrent training stages; execution only.")
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
        return 0
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    config = base.load_config(args.config)
    runtime = normalize_runtime(config, args.metadata_root, args.features_root, args.device)
    resolved = dict(config)
    resolved["execution"] = {**config.get("execution", {}), **runtime}
    devices = None
    if args.devices:
        devices = [value.strip() for value in args.devices.split(",") if value.strip()]
        if not devices:
            parser.error("--devices needs at least one GPU id")
    if args.run:
        result = execute(resolved, args.output_root or base.ROOT / DEFAULT_OUTPUT,
                         resume=args.resume, devices=devices, workers=args.workers)
        print(json.dumps({"status": result["status"],
                          "output_root": str(base.private_output(args.output_root))}))
        return 0
    if args.doctor:
        print(json.dumps(doctor(runtime, output_root=args.output_root or base.ROOT / DEFAULT_OUTPUT),
                         ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(make_plan(resolved, args.output_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
