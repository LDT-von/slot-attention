"""Anti-collapse variant of the best-checkpoint redundancy diagnostic.

Why this module exists
-----------------------
The read-only analysis of the completed 30/4096/20 five-fold queue found
that the slot collapse is an *optimizer step event*, not an epoch event:
the wsi effective rank falls from above 2 to below 1.5 inside a single
step, and in three of five folds the collapsing step follows a gradient
spike of 1e6-4.5e6 while the median gradient norm of the same run is
about 30-50.  The Gram objective delays the collapse relative to the
baseline but does not prevent it: R5 survives to epoch 2-17 depending on
the fold, and the baseline is already collapsed at its very first step in
two folds.

The only element of the eleven-arm catalogue that ever produced an
uncollapsed slot set was the Gram term, and it produced one transiently.
This module therefore tests whether the collapse can be *held* rather
than merely delayed, by adding two mechanisms that are absent from every
completed queue:

``gradient_clip_norm``
    Clip the core gradient norm after the backward pass and before the
    optimizer step.  This directly bounds the step that the analysis
    identified as the collapse trigger.

``gram_weight_decay``
    A multiplicative decay of the Gram weight over the training run.  The
    Gram term is what opens the uncollapsed window, and the analysis shows
    the window closes again; a decaying weight lets the objective act
    early and relaxes later, so the survival objective does not have to
    fight a permanently active anti-collapse penalty.

Scope and honesty
-----------------
This module does not modify ``slotspe_proof/redundancy_engine.py``,
``slotspe_proof/redundancy_best_engine.py`` or any file pinned by the
completed queues.  It is a separate protocol with its own output
directory, its own checkpoints and its own record.

A run under this module answers one question: can the slot geometry be
held uncollapsed for the whole run?  It does not by itself answer whether
spread slots carry patient-specific content; that question is answered by
re-running the deletion probe on the resulting checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from . import patient
from .redundancy_engine import (ARM_CONTRACTS, SlotCapture, _check_capture,
                                _construct, _cpu_state, _evaluate_repeat,
                                _finite, _item, _summarize_diagnostics,
                                case_seed, make_native_adapter, state_hash)
from .redundancy_best_engine import (CHECKPOINT_RULE_SOURCE, CHECKPOINT_SELECTION,
                                     _StreamGuard, _validation_cindex)

ANTI_COLLAPSE_ARMS = ("A0_clip", "A1_clip_decay")
ARM_CONTRACTS_ANTI = {
    "A0_clip": {"adapter": "S0", "gram_weight": 10.0, "slotbert_weight": 0.0,
                "content_weight": 5.0, "slotbert_temperature": 0.5,
                "clip": True, "decay": False},
    "A1_clip_decay": {"adapter": "S0", "gram_weight": 10.0, "slotbert_weight": 0.0,
                      "content_weight": 5.0, "slotbert_temperature": 0.5,
                      "clip": True, "decay": True},
}

# The Gram weight is scaled by this factor over the whole run when the decay
# is enabled.  The exponent is applied to the normalized progress so the
# weight reaches the floor exactly at the last epoch.
GRAM_DECAY_FLOOR = 0.1


def _grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().norm(2) ** 2)
    return math.sqrt(total)


def _clip_gradients(parameters, max_norm):
    """Clip the gradient norm in place and return the pre-clip norm."""
    norm = _grad_norm(parameters)
    if max_norm is None or max_norm <= 0 or not math.isfinite(norm):
        return norm, False
    if norm <= max_norm:
        return norm, False
    scale = max_norm / (norm + 1e-12)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.detach().mul_(scale)
    return norm, True


def _gram_weight(contract, epoch_index, epochs):
    """Return the Gram weight for one epoch.

    Without decay the weight is constant, which reproduces the completed
    R5 arm exactly.  With decay it is scaled by a power of the normalized
    epoch progress, so it is largest at the first epoch and reaches
    ``GRAM_DECAY_FLOOR`` at the last one.
    """
    weight = float(contract["gram_weight"])
    if not weight or not contract.get("decay"):
        return weight
    if epochs <= 1:
        return weight
    progress = min(max(epoch_index / (epochs - 1), 0.0), 1.0)
    scale = GRAM_DECAY_FLOOR ** progress
    return weight * scale


def run_arm_anti_collapse(config, root, manifest, arm, initial_core, initial_probe,
                          reference, *, fold=0, output_root=None,
                          _generated_test=False):
    """Train one anti-collapse arm under the original best-checkpoint rule.

    The model computation, initialization, batching, patch budget, loss
    terms, evaluation contract and checkpoint rule match
    ``redundancy_best_engine.run_arm_best``.  The only differences are the
    gradient clipping and the optional Gram weight decay, both of which
    are recorded in the arm contract and in the run protocol.
    """
    if arm not in ANTI_COLLAPSE_ARMS:
        raise ValueError("unknown anti-collapse arm")
    if arm not in config["variants"]:
        raise ValueError("arm outside the declared anti-collapse queue")
    if config["study"] != "blca" or config["seeds"] != [3]:
        raise ValueError("this queue is locked to BLCA seed3")
    if fold not in config["folds"]:
        raise ValueError("requested fold is outside the declared queue")
    if config.get("feature_encoder") != "UNI2-h":
        raise ValueError("feature_encoder must be explicitly UNI2-h")
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
    out.mkdir(parents=True, exist_ok=True)
    device = config["device"]
    train_ids = manifest["folds"][str(fold)]["train"]
    val_ids = manifest["folds"][str(fold)]["val"]
    contract = ARM_CONTRACTS_ANTI[arm]
    clip_norm = config.get("gradient_clip_norm")
    seed = config["seeds"][0]
    initial_stream_seed = int(seed) * 1_000_003 + int(fold) * 7919 + 13
    protocol = {"arm": arm, "contract": contract, "fold": fold,
                "seed": seed, "epochs": config["epochs"],
                "patches": config["patches"], "batch_size": config["batch_size"],
                "eval_repeats": config["eval_repeats"],
                "lr": config["lr"], "probe_lr": config.get("probe_lr", config["lr"]),
                "gradient_clip_norm": clip_norm,
                "gram_decay": contract["decay"],
                "gram_decay_floor": GRAM_DECAY_FLOOR if contract["decay"] else None,
                "checkpoint_selection": CHECKPOINT_SELECTION,
                "checkpoint_rule_source": CHECKPOINT_RULE_SOURCE,
                "initial_stream_seed": initial_stream_seed,
                "initial_core_hash": state_hash(initial_core),
                "initial_probe_hash": state_hash(initial_probe)}
    counters = {"root_forwards": 0, "slot_attention_calls": 0, "slot_attention_iterations": 0,
                "core_backward_calls": 0, "probe_backward_calls": 0,
                "core_optimizer_steps": 0, "probe_optimizer_steps": 0,
                "clipped_steps": 0, "clipped_norm_sum": 0.0}
    started = time.perf_counter()
    model, heads = _construct(manifest, config["source_root"], config.get("projection_dim", 256),
                              config.get("slot_iters", 10), device)
    model.load_state_dict(initial_core, strict=True)
    heads.load_state_dict(initial_probe, strict=True)
    if state_hash(model.state_dict()) != protocol["initial_core_hash"] or \
            state_hash(heads.state_dict()) != protocol["initial_probe_hash"]:
        raise RuntimeError("paired initialization mismatch")
    core_optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    probe_optimizer = torch.optim.Adam(heads.parameters(), lr=config.get("probe_lr", config["lr"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(core_optimizer, T_max=config["epochs"], eta_min=1e-6)
    probe_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(probe_optimizer, T_max=config["epochs"], eta_min=1e-6)

    from sksurv.metrics import concordance_index_censored
    curve = []
    epoch_validation = []
    best = {"epoch": None, "cindex": None, "state": None}
    with make_native_adapter(model, contract["adapter"], capture=True) as adapter, \
            SlotCapture(model) as capture:
        patient._rng(initial_stream_seed, torch, np)
        for epoch in range(config["epochs"]):
            model.train()
            heads.train()
            order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(
                seed * 1000 + epoch)).tolist()
            epoch_gram_weight = _gram_weight(contract, epoch, config["epochs"])
            for start in range(0, len(order), config["batch_size"]):
                ids = [train_ids[index] for index in order[start:start + config["batch_size"]]]
                batch = patient._batch(root, manifest, _fold_data(manifest, fold), ids, device, torch)
                targets = [reference.targets(_item(batch, index)) for index in range(len(ids))]
                seeds = [case_seed(seed, epoch, start + index) for index in range(len(ids))]
                result = _anti_logical_batch(model, heads, adapter, capture, batch, targets,
                                             contract, epoch_gram_weight, clip_norm,
                                             core_optimizer, probe_optimizer, seeds)
                for key in counters:
                    counters[key] += result[key]
                curve.append({"epoch": epoch + 1, "batch_start": start, "patient_ids": ids,
                              "native_seeds": seeds,
                              "learning_rate": core_optimizer.param_groups[0]["lr"],
                              "gram_weight": epoch_gram_weight, **result})
                patient._dump(out / "curve.json", curve)
            scheduler.step()
            probe_scheduler.step()
            with _StreamGuard():
                value, status = _validation_cindex(model, heads, adapter, root, manifest,
                                                    _fold_data(manifest, fold), val_ids, device,
                                                    seed, epoch + 1,
                                                    concordance_index_censored)
            counters["root_forwards"] += len(val_ids)
            counters["slot_attention_calls"] += 2 * len(val_ids)
            counters["slot_attention_iterations"] += (model.slot_attention_wsi.iters
                                                      + model.slot_attention_omic.iters) * len(val_ids)
            epoch_validation.append({"epoch": epoch + 1, "cindex": value,
                                      "cindex_status": status})
            patient._dump(out / "epoch_validation.json", epoch_validation)
            if value is not None and (best["cindex"] is None or value >= best["cindex"]):
                best = {"epoch": epoch + 1, "cindex": value,
                        "state": {"state_dict": _cpu_state(model),
                                  "probe_state_dict": _cpu_state(heads)}}
            print(f"arm={arm} epoch={epoch+1}/{config['epochs']} "
                  f"val_cindex={value if value is not None else 'undefined'} "
                  f"best_epoch={best['epoch']}", flush=True)
        torch.save({"state_dict": _cpu_state(model), "probe_state_dict": _cpu_state(heads),
                    "reference_state": reference.state_dict(),
                    "optimizer_state": core_optimizer.state_dict(),
                    "probe_optimizer_state": probe_optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "probe_scheduler_state": probe_scheduler.state_dict(),
                    "protocol": protocol, "omic_sizes": manifest["omic_sizes"],
                    "encoding_dim": manifest["encoding_dim"],
                    "counters": dict(counters)}, out / "last.pt")
        if best["state"] is None:
            raise RuntimeError("no epoch produced a defined validation concordance index")
        torch.save(best["state"], out / "best.pt")
        model.load_state_dict(best["state"]["state_dict"], strict=True)
        heads.load_state_dict(best["state"]["probe_state_dict"], strict=True)
        evaluations, diagnostics, baseline_slots = [], [], None
        for repeat in range(config["eval_repeats"]):
            evaluation, rows, raw = _evaluate_repeat(model, heads, adapter, capture, reference,
                                                      root, manifest, _fold_data(manifest, fold),
                                                      val_ids, device, seed, repeat,
                                                      concordance_index_censored,
                                                      baseline_slots=baseline_slots)
            if repeat == 0:
                baseline_slots = {item["patient_id"]: item["slots"] for item in raw}
            evaluations.append(evaluation)
            diagnostics.extend(rows)
            extra = len(val_ids) if repeat == 0 and len(val_ids) > 1 else 0
            counters["root_forwards"] += len(val_ids) + extra
            counters["slot_attention_calls"] += 2 * (len(val_ids) + extra)
            counters["slot_attention_iterations"] += (model.slot_attention_wsi.iters
                                                      + model.slot_attention_omic.iters) * (len(val_ids) + extra)
            torch.save(raw, out / f"raw_validation_repeat_{repeat:02d}.pt")
            patient._dump(out / "eval_repeats.json", evaluations)
            patient._dump(out / "patient_diagnostics.json", diagnostics)
        torch.save({"patient_ids": val_ids, "diagnostics": diagnostics,
                    "raw_repeat_files": [f"raw_validation_repeat_{index:02d}.pt"
                                         for index in range(config["eval_repeats"])]},
                   out / "diagnostics.pt")
    patient._write_predictions(out / "validation_predictions.csv", manifest,
                                _fold_data(manifest, fold), val_ids, evaluations)
    summary = {"status": "completed", "arm": arm, "fold": fold, "seed": seed,
               "epochs": config["epochs"], "patches": config["patches"],
               "training_cases": len(train_ids), "validation_cases": len(val_ids),
               "protocol": protocol, "counters": counters,
               "evaluation_summary": _summarize_diagnostics(diagnostics),
               "slot_diagnostics": _summarize_diagnostics(diagnostics),
               "best_epoch": best["epoch"],
               "best_epoch_cindex_single_pass": best["cindex"],
               "elapsed_seconds": time.perf_counter() - started,
               "claim_boundary": "single_fold_single_seed_exploratory_no_patient_efficacy_established"}
    patient._dump(out / "summary.json", summary)
    return summary


def _fold_data(manifest, fold):
    return manifest["folds"][str(fold)]


def _anti_logical_batch(model, heads, adapter, capture, batch, targets, contract,
                        gram_weight, clip_norm, core_optimizer, probe_optimizer, seeds):
    """One matched core and one detached-probe update, with clipping.

    This mirrors ``redundancy_engine.execute_logical_batch`` exactly except
    that the Gram weight can vary per epoch and the core gradient is
    clipped after the backward pass.  The probe is still detached from the
    native model and the check for a probe leak into the core is retained.
    """
    from .redundancy_engine import _component_gradient_norms, _frozen, _finite_module, _gradient_norm
    from .redundancy_content import coverage_set_mse
    from .slot_regularization import slot_gram_loss, slotbert_loss
    count = batch["x_wsi"].shape[0]
    if count < 1 or len(seeds) != count or len(targets) != count or len(set(seeds)) != count:
        raise ValueError("one unique matched seed and target map per example required")
    if not model.training:
        raise ValueError("training batch requires model.train()")
    if adapter.variant != contract["adapter"]:
        raise ValueError("adapter and arm disagree")
    core_optimizer.zero_grad(set_to_none=True)
    probe_optimizer.zero_grad(set_to_none=True)
    logits, auxiliary, slots, metrics = [], [], [], []
    for index, seed in enumerate(seeds):
        patient._rng(int(seed), torch, np)
        adapter.set_noise_seed(int(seed) + 27001)
        prediction, native_auxiliary = model(**_item(batch, index))
        _finite(prediction, "training prediction")
        current = _check_capture(capture, True)
        slots.append(current)
        logits.append(prediction)
        auxiliary.append(torch.as_tensor(native_auxiliary, device=prediction.device))
        metrics.append({name: _angular_metrics(value) for name, value in current.items()})
    logits = torch.cat(logits)
    main = model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None) / count
    auxiliary = torch.stack(auxiliary).mean()
    probe_loss = torch.stack([coverage_set_mse(heads[name](current[name].detach()), targets[index][name])
                              for index, current in enumerate(slots) for name in heads]).mean()
    _finite(probe_loss, "probe loss")
    probe_loss.backward()
    probe_gradient = _gradient_norm(heads)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("diagnostic probe leaked gradients into native core")
    probe_optimizer.step()
    _finite_module(heads, "updated probe")
    probe_optimizer.zero_grad(set_to_none=True)
    zero = main.new_zeros(())
    gram, paper, content = zero, zero, zero
    with _frozen(heads):
        if gram_weight:
            gram = torch.stack([slot_gram_loss(current[name])
                                for current in slots for name in heads]).mean()
        if contract["slotbert_weight"]:
            paper = torch.stack([slotbert_loss(current[name], contract["slotbert_temperature"])
                                 for current in slots for name in heads]).mean()
        if contract["content_weight"]:
            content = torch.stack([coverage_set_mse(heads[name](current[name]), targets[index][name])
                                   for index, current in enumerate(slots) for name in heads]).mean()
        added = gram_weight * gram + contract["slotbert_weight"] * paper \
            + contract["content_weight"] * content
        total = main + auxiliary + added
        _finite(total, "core objective")
        components = (_component_gradient_norms(model, {"main_nll": main,
                                                       "native_auxiliary": auxiliary,
                                                       "weighted_added_objective": added}))
        total.backward()
        gradient = _gradient_norm(model)
        if any(parameter.grad is not None for parameter in heads.parameters()):
            raise RuntimeError("core objective changed the frozen diagnostic heads")
        pre_clip_norm, clipped = _clip_gradients(list(model.parameters()),
                                                  clip_norm if contract.get("clip") else None)
        # The norm that actually reaches the optimizer is the post-clip one.
        # The pre-clip norm is kept separately so the effect of the clip is
        # auditable from the stored curve.
        applied_norm = _gradient_norm(model)
        core_optimizer.step()
        _finite_module(model, "updated core")
    return {"main_nll": float(main.detach()), "native_auxiliary": float(auxiliary.detach()),
            "gram_loss": float(gram.detach()), "slotbert_loss": float(paper.detach()),
            "content_loss": float(content.detach()), "added_loss": float(added.detach()),
            "total_loss": float(total.detach()), "probe_loss": float(probe_loss.detach()),
            "core_gradient_norm": applied_norm, "probe_gradient_norm": probe_gradient,
            "pre_clip_gradient_norm": pre_clip_norm,
            "gradient_was_clipped": bool(clipped),
            "case_count": count, "root_forwards": count,
            "slot_attention_calls": count * 3,
            "slot_attention_iterations": count * (model.slot_attention_wsi.iters
                                                 + 2 * model.slot_attention_omic.iters),
            "core_backward_calls": 1, "probe_backward_calls": 1,
            "core_optimizer_steps": 1, "probe_optimizer_steps": 1,
            "slot_metrics": metrics,
            "component_gradient_norms": components,
            "component_autograd_grad_calls": components["autograd_grad_calls"]
            if components else 0,
            "clipped_steps": 1 if clipped else 0,
            "clipped_norm_sum": pre_clip_norm if clipped else 0.0}


def _angular_metrics(value):
    from .redundancy_engine import angular_metrics
    return angular_metrics(value)
