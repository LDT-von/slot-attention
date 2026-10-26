"""Best-checkpoint variant of the eleven-arm SlotSPE diagnostic.

This module reproduces the original published checkpoint rule of
``third_party/SlotSPE/utils/core_utils.py``: every epoch is evaluated on the
validation fold, the epoch with the highest validation concordance index is
retained, and after the final epoch the retained best weights are loaded and
evaluated once more with the full repeated-evaluation contract.

It deliberately does NOT modify ``slotspe_proof/redundancy_engine.py`` or any
other file pinned by the completed final-epoch queue.  The final-epoch queue
and this module are two separate, separately recorded protocols.

Two properties make the comparison against the final-epoch queue meaningful:

1.  The per-epoch validation pass restores every global random stream it
    touched (python ``random``, numpy legacy global, torch CPU and CUDA).
    The training-time real-patch draw consumes the numpy global stream, so an
    unguarded evaluation would silently change the training trajectory of
    every later epoch.  With the guard, the training trajectory of this
    module is identical to the final-epoch queue for the same arm, fold and
    seed, and the only difference is which checkpoint is reported.

2.  The best weight is written to its own file next to ``last.pt`` so the two
    protocols never share a checkpoint.

Reported C-index under this protocol is a validation-selected peak.  A
validation-selected peak is not an independently tested generalization
result; see the ``latest_official_epoch_selection_review`` entry of
``docs/PROGRESS.json`` for that recorded boundary.
"""

from __future__ import annotations

import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from . import patient
from .redundancy_engine import (ARM_CONTRACTS, ARMS, SlotCapture, _check_capture,
                                _construct, _cpu_state, _evaluate_repeat,
                                _finite, _item, _summarize_diagnostics,
                                case_seed, execute_logical_batch, make_native_adapter,
                                state_hash)

CHECKPOINT_SELECTION = "per_epoch_validation_best_then_reload_and_reevaluate"
CHECKPOINT_RULE_SOURCE = "third_party/SlotSPE/utils/core_utils.py:421-475"


class _StreamGuard:
    """Save and restore every global random stream a validation pass touches.

    The training loop draws the real-patch subset from the numpy global
    stream, and the evaluation helper reseeds python, numpy and torch.  A
    validation pass between two epochs would therefore change the patch
    subsets drawn by the following epochs.  This guard makes the validation
    pass observationally inert with respect to the training trajectory.
    """

    def __init__(self):
        self._py = None
        self._np = None
        self._torch = None
        self._cuda = None

    def __enter__(self):
        self._py = random.getstate()
        self._np = np.random.get_state()
        self._torch = torch.get_rng_state()
        if torch.cuda.is_available():
            self._cuda = torch.cuda.get_rng_state_all()
        return self

    def __exit__(self, *exc):
        random.setstate(self._py)
        np.random.set_state(self._np)
        torch.set_rng_state(self._torch)
        if self._cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self._cuda)
        return False


def _validation_cindex(model, heads, adapter, root, manifest, fold_data, val_ids,
                       device, seed, epoch, concordance):
    """One validation pass over the whole validation fold at a given epoch.

    This mirrors the original ``_summary`` used by the official loop: the
    validation fold is evaluated in full, the risk is the original
    ``-sum(cumprod(1 - sigmoid(logits)))`` and the concordance index is
    computed over the same events and times the final evaluation uses.  The
    fixed patch subset (``sample_seed=0``) and the evaluation seed schedule
    are the ones the repeated evaluation uses, so this pass is exactly
    repeat 0 of the final evaluation of the same weights.

    Only the forward pass is performed here; the research diagnostics
    (donor probes, static control, matched distances) are not needed to rank
    epochs and are computed once, on the retained best weights, afterwards.
    """
    model.eval()
    heads.eval()
    logits = []
    with torch.no_grad():
        for index, pid in enumerate(val_ids):
            batch = patient._batch(root, manifest, fold_data, [pid], device, torch,
                                   sample_seed=0)
            schedule_seed = case_seed(seed, 0, index, evaluation=True)
            patient._rng(schedule_seed, torch, np)
            adapter.set_noise_seed(schedule_seed + 27001)
            prediction, _ = model(**batch)
            _finite(prediction, "per-epoch validation prediction")
            logits.append(prediction.detach().cpu())
    combined = torch.cat(logits)
    risks = -(1 - combined.sigmoid()).cumprod(-1).sum(-1)
    events = np.asarray([manifest["patients"][pid]["censor"] < 1 for pid in val_ids], dtype=bool)
    times = np.asarray([manifest["patients"][pid]["time"] for pid in val_ids], dtype=float)
    try:
        score = float(concordance(events, times, risks.numpy())[0])
        if not math.isfinite(score):
            score = None
        status = "computed" if score is not None else "undefined_nonfinite"
    except ValueError as exc:
        score, status = None, "undefined: " + str(exc)
    return score, status


def run_arm_best(config, root, manifest, arm, initial_core, initial_probe, reference, *,
                 fold=0, output_root=None, _generated_test=False):
    """Train one arm under the original per-epoch best-checkpoint rule.

    The model computation, hyperparameters, initialization, batching, patch
    budget, loss terms and evaluation contract are identical to
    ``redundancy_engine.run_arm``.  The only behavioural difference is the
    checkpoint rule: the best validation epoch is retained, reloaded after the
    final epoch and re-evaluated with the full repeated-evaluation contract.
    """
    if arm not in ARMS or arm not in config["variants"]:
        raise ValueError("arm outside the declared eleven-arm queue")
    if config["study"] != "blca" or config["seeds"] != [3]:
        raise ValueError("this queue is locked to BLCA seed3")
    if fold not in config["folds"]:
        raise ValueError("requested fold is outside the declared queue")
    if config.get("feature_encoder") != "UNI2-h":
        raise ValueError("feature_encoder must be explicitly UNI2-h")
    if config.get("dias_distance", .05) != .05 or config.get("slotbert_temperature", .5) != .5:
        raise ValueError("declared DIAS distance/Slot-BERT temperature differ from fixed contracts")
    if config.get("content_target_dim", 32) != 32 or reference.target_dim != 32:
        raise ValueError("query-free reconstruction target dimension is fixed at 32")
    if _generated_test:
        if manifest.get("generated_test_only") is not True or config.get("device") != "cpu":
            raise ValueError("private shape override requires an explicitly generated CPU manifest")
    elif config.get("projection_dim", 256) != 256 or config.get("slot_iters", 10) != 10:
        raise ValueError("patient queue preserves native 256 dimensions and 10 iterations")
    if not reference.fitted:
        raise ValueError("a separately fitted train-only content reference is required")

    root = Path(root)
    out = Path(output_root) / "arms" / arm if output_root is not None else root / "arms" / arm
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refusing to overwrite or resume prior arm output")
    source_before = patient._source_hashes(config)
    if source_before != manifest["source_hashes"]:
        raise RuntimeError("source changed after preparation")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA is unavailable; no silent CPU fallback")
    torch.set_num_threads(int(config.get("torch_threads", 2)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    fold_data = manifest["folds"][str(fold)]
    train_ids, val_ids = fold_data["train"], fold_data["val"]
    if not train_ids or not val_ids or set(train_ids) & set(val_ids):
        raise ValueError("nonempty disjoint training/validation IDs required")
    if reference.train_case_count != len(train_ids):
        raise ValueError("reference fit count does not match training cohort")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seed = config["seeds"][0]
    initial_stream_seed = int(seed) * 1_000_003 + int(fold) * 7919 + 13
    protocol = {"arm": arm, "contract": ARM_CONTRACTS[arm], "fold": fold, "seed": seed,
                "initial_core_hash": state_hash(initial_core), "initial_probe_hash": state_hash(initial_probe),
                "source_hashes": source_before, "epochs": config["epochs"], "batch_size": config["batch_size"],
                "patches": config["patches"], "checkpoint_selection": CHECKPOINT_SELECTION,
                "native_forward_protocol": "B1_for_all_arms_accumulated_into_logical_batch",
                "probe_protocol": "one_detached_update_then_frozen_head_core_objective_per_logical_batch",
                "content_reference_fit": "training_inputs_only_no_outcomes",
                "initial_stream_seed": initial_stream_seed,
                "initial_stream_seed_scope": "first training logical batch patch draw only; arm-independent within a fold",
                "per_epoch_validation": "one full validation pass after each epoch; the global random streams are "
                                        "restored afterwards so the training trajectory is unchanged relative to the "
                                        "final-epoch queue",
                "checkpoint_rule_source": CHECKPOINT_RULE_SOURCE,
                "added_loss_slot_calls": {"wsi": 0, "omic": 0}}
    counters = {"root_forwards": 0, "slot_attention_calls": 0, "slot_attention_iterations": 0,
                "core_backward_calls": 0, "probe_backward_calls": 0,
                "core_optimizer_steps": 0, "probe_optimizer_steps": 0,
                "component_autograd_grad_calls": 0}
    try:
        model, heads = _construct(manifest, config["source_root"], config.get("projection_dim", 256),
                                  config.get("slot_iters", 10), device)
        model.load_state_dict(initial_core, strict=True)
        heads.load_state_dict(initial_probe, strict=True)
        if state_hash(model.state_dict()) != protocol["initial_core_hash"] or state_hash(heads.state_dict()) != protocol["initial_probe_hash"]:
            raise RuntimeError("paired initialization mismatch")
        core_optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        probe_optimizer = torch.optim.Adam(heads.parameters(), lr=config.get("probe_lr", config["lr"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(core_optimizer, T_max=config["epochs"], eta_min=1e-6)
        probe_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(probe_optimizer, T_max=config["epochs"], eta_min=1e-6)
        from sksurv.metrics import concordance_index_censored
        curve = []
        epoch_validation = []
        best = {"epoch": None, "cindex": None, "state": None}
        with make_native_adapter(model, ARM_CONTRACTS[arm]["adapter"], capture=True) as adapter, SlotCapture(model) as capture:
            patient._rng(initial_stream_seed, torch, np)
            for epoch in range(config["epochs"]):
                model.train()
                heads.train()
                order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(seed * 1000 + epoch)).tolist()
                for start in range(0, len(order), config["batch_size"]):
                    ids = [train_ids[index] for index in order[start:start + config["batch_size"]]]
                    batch = patient._batch(root, manifest, fold_data, ids, device, torch)
                    targets = [reference.targets(_item(batch, index)) for index in range(len(ids))]
                    seeds = [case_seed(seed, epoch, start + index) for index in range(len(ids))]
                    result = execute_logical_batch(model, heads, adapter, capture, batch, targets, arm,
                                                   core_optimizer, probe_optimizer, seeds,
                                                   measure_gradient_components=start == 0)
                    for key in counters:
                        counters[key] += result[key]
                    curve.append({"epoch": epoch + 1, "batch_start": start, "patient_ids": ids,
                                  "native_seeds": seeds, "learning_rate": core_optimizer.param_groups[0]["lr"], **result})
                    patient._dump(out / "curve.json", curve)
                scheduler.step()
                probe_scheduler.step()
                # Per-epoch validation, reproducing the original rule.  The
                # evaluation reseeds the global streams, so the guard restores
                # them before the next epoch draws its patch subsets.
                with _StreamGuard():
                    value, status = _validation_cindex(model, heads, adapter, root, manifest,
                                                        fold_data, val_ids, device, seed,
                                                        epoch + 1, concordance_index_censored)
                counters["root_forwards"] += len(val_ids)
                counters["slot_attention_calls"] += 2 * len(val_ids)
                counters["slot_attention_iterations"] += (model.slot_attention_wsi.iters + model.slot_attention_omic.iters) * len(val_ids)
                epoch_validation.append({"epoch": epoch + 1, "cindex": value,
                                          "cindex_status": status})
                patient._dump(out / "epoch_validation.json", epoch_validation)
                # The original rule keeps the epoch whose value is greater or
                # equal to the running maximum, so a later equal value wins.
                if value is not None and (best["cindex"] is None or value >= best["cindex"]):
                    best = {"epoch": epoch + 1, "cindex": value,
                            "state": {"state_dict": _cpu_state(model),
                                      "probe_state_dict": _cpu_state(heads)}}
                print(f"arm={arm} epoch={epoch+1}/{config['epochs']} "
                      f"val_cindex={value if value is not None else 'undefined'} "
                      f"best_epoch={best['epoch']} best_cindex={best['cindex']}", flush=True)
            # The capture context must still be active here: the repeated
            # evaluation reads the captured slot outputs of the prediction
            # path, exactly as the final-epoch queue does.
            torch.save({"state_dict": _cpu_state(model), "probe_state_dict": _cpu_state(heads),
                        "reference_state": reference.state_dict(), "optimizer_state": core_optimizer.state_dict(),
                        "probe_optimizer_state": probe_optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                        "probe_scheduler_state": probe_scheduler.state_dict(), "protocol": protocol,
                        "omic_sizes": manifest["omic_sizes"], "encoding_dim": manifest["encoding_dim"],
                        "counters": dict(counters)}, out / "last.pt")
            if best["state"] is None:
                raise RuntimeError("no epoch produced a defined validation concordance index")
            torch.save(best["state"], out / "best.pt")
            # Reload the retained best weights and evaluate them with the full
            # repeated-evaluation contract, exactly as the original does after
            # its training loop.
            model.load_state_dict(best["state"]["state_dict"], strict=True)
            heads.load_state_dict(best["state"]["probe_state_dict"], strict=True)
            evaluations, diagnostics, baseline_slots = [], [], None
            for repeat in range(config["eval_repeats"]):
                evaluation, rows, raw = _evaluate_repeat(model, heads, adapter, capture, reference, root, manifest,
                                    fold_data, val_ids, device, seed, repeat, concordance_index_censored,
                                    baseline_slots=baseline_slots)
                if repeat == 0:
                    baseline_slots = {item["patient_id"]: item["slots"] for item in raw}
                evaluations.append(evaluation)
                diagnostics.extend(rows)
                extra = len(val_ids) if repeat == 0 and len(val_ids) > 1 else 0
                counters["root_forwards"] += len(val_ids) + extra
                counters["slot_attention_calls"] += 2 * (len(val_ids) + extra)
                counters["slot_attention_iterations"] += (model.slot_attention_wsi.iters + model.slot_attention_omic.iters) * (len(val_ids) + extra)
                torch.save(raw, out / f"raw_validation_repeat_{repeat:02d}.pt")
                patient._dump(out / "eval_repeats.json", evaluations)
                patient._dump(out / "patient_diagnostics.json", diagnostics)
            torch.save({"patient_ids": val_ids, "diagnostics": diagnostics,
                        "raw_repeat_files": [f"raw_validation_repeat_{index:02d}.pt"
                                             for index in range(config["eval_repeats"])]}, out / "diagnostics.pt")
        patient._write_predictions(out / "validation_predictions.csv", manifest, fold_data, val_ids, evaluations)
        source_after = patient._source_hashes(config)
        if source_before != source_after:
            raise RuntimeError("original source changed during execution")
        summary = {"status": "completed", "arm": arm, "fold": fold, "seed": seed,
                   "epochs": config["epochs"], "training_cases": len(train_ids), "validation_cases": len(val_ids),
                   "protocol": protocol, "counters": counters, "evaluation_summary": patient._repeat_summary(evaluations, np),
                   "slot_diagnostics": _summarize_diagnostics(diagnostics), "source_unchanged": True,
                   "last_checkpoint_sha256": patient._sha(out / "last.pt"),
                   "best_checkpoint_sha256": patient._sha(out / "best.pt"),
                   "best_epoch": best["epoch"], "best_epoch_cindex_single_pass": best["cindex"],
                   "epoch_validation": epoch_validation,
                   "elapsed_seconds": time.perf_counter() - started,
                   "cuda_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   "claim_boundary": "single_fold_best_validation_checkpoint_exploratory_not_generalization_or_clinical_benefit"}
        patient._dump(out / "summary.json", summary)
        return summary
    except BaseException as exc:
        patient._dump(out / "failure.json", {"status": "failed", "type": type(exc).__name__, "message": str(exc),
                                             "counters_completed": counters, "elapsed_seconds": time.perf_counter() - started})
        raise
