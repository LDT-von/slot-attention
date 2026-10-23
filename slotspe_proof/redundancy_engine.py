"""Matched single-fold SlotSPE redundancy experiments, explicitly executed only.

Original source is never edited.  All arms use single-example native forwards,
identical logical batches and detached diagnostic-head updates.  Added losses
observe only prediction-path slot call zero, never reconstruction-path slots.
Patient-level outputs belong exclusively in the caller's ignored output root.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import patient
from .model import build_model, make_args
from .native_variants import make_native_adapter
from .slot_regularization import (
    QueryFreeSlotDecoder, SlotCapture, slot_gram_loss, slotbert_loss,
)
from .redundancy_content import coverage_set_mse


ARMS = ("S0", "S4", "R1", "R2", "R3", "R4", "R5", "S1", "S2", "S3", "S5")
ARM_CONTRACTS = {
    arm: {"adapter": arm if arm.startswith("S") else "S0",
          "gram_weight": 10.0 if arm in {"R1", "R5"} else 0.0,
          "slotbert_weight": 1.0 if arm in {"R2", "R3"} else 0.0,
          "content_weight": 5.0 if arm in {"R3", "R4", "R5"} else 0.0,
          "slotbert_temperature": 0.5}
    for arm in ARMS
}


def state_hash(state):
    """Hash ordered tensor contents rather than nondeterministic archive bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _cpu_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _construct(manifest, source_root, projection_dim, slot_iters, device):
    args = make_args(manifest["omic_sizes"], encoding_dim=manifest["encoding_dim"])
    args.wsi_projection_dim = int(projection_dim)
    args.slot_iters = int(slot_iters)
    model = build_model(source_root, args, "original").to(device)
    heads = nn.ModuleDict({name: QueryFreeSlotDecoder(int(projection_dim), 32)
                           for name in ("wsi", "omic")}).to(device)
    return model, heads


def make_initial_states(manifest, source_root, seed, device="cpu",
                        projection_dim=256, slot_iters=10):
    """One reproducible initialization, preserving the caller's random streams."""
    py_state, np_state = random.getstate(), np.random.get_state()
    # _rng/torch.manual_seed also seed every CUDA generator even for CPU models.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            patient._rng(int(seed), torch, np)
            model, heads = _construct(manifest, source_root, projection_dim, slot_iters, device)
            return _cpu_state(model), _cpu_state(heads)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def case_seed(seed, epoch, position, *, evaluation=False):
    """Unique case seeds shared across arms, irrespective of variable slot K."""
    if min(int(seed), int(epoch), int(position)) < 0:
        raise ValueError("seed indices must be nonnegative")
    return int(seed) * 10**12 + (10**11 if evaluation else 0) + int(epoch) * 10**6 + int(position)


def _item(batch, index):
    return {name: value[index:index + 1] if torch.is_tensor(value) else value
            for name, value in batch.items()}


def _finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("nonfinite " + label)


def _gradient_norm(module):
    gradients = [p.grad.detach() for p in module.parameters() if p.grad is not None]
    if not gradients:
        raise RuntimeError("no connected gradients")
    norms = torch._foreach_norm(gradients, 2) if hasattr(torch, "_foreach_norm") else [g.norm() for g in gradients]
    result = torch.stack(norms).norm()
    _finite(result, "gradient")
    return float(result)


def _finite_module(module, label):
    for name, value in module.state_dict().items():
        _finite(value, f"{label} parameter/buffer {name}")


def _component_gradient_norms(model, losses):
    """Diagnostic partial gradients only; autograd.grad never fills .grad."""
    named = [(name, getattr(model, "slot_attention_" + name).to_v.weight)
             for name in ("wsi", "omic")]
    result = {"parameters": [name + ".to_v.weight" for name, _ in named],
              "scope": "two_native_SA_value_projection_weights_not_full_model",
              "autograd_grad_calls": 0}
    parameters = [parameter for _, parameter in named]
    for label, loss in losses.items():
        gradients = (torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                     if loss.requires_grad else [None] * len(parameters))
        result["autograd_grad_calls"] += int(loss.requires_grad)
        values = {}
        for (name, _), gradient in zip(named, gradients):
            if gradient is None:
                values[name] = {"connected": False, "norm": None}
            else:
                _finite(gradient, "component gradient " + label)
                values[name] = {"connected": True, "norm": float(gradient.detach().norm())}
        result[label] = values
    return result


@contextlib.contextmanager
def _frozen(module):
    original = [p.requires_grad for p in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(module.parameters(), original):
            parameter.requires_grad_(flag)


def _check_capture(capture, training):
    expected = {"wsi": 1, "omic": 2 if training else 1}
    observed = {name: len(values) for name, values in capture.slots.items()}
    if observed != expected:
        raise RuntimeError(f"native slot call contract changed: {observed}, expected {expected}")
    return {name: values[0] for name, values in capture.slots.items()}


def angular_metrics(slots):
    """Pair metrics are undefined for a single retained slot, never zero."""
    value = slots.detach().double()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError("expected nonempty [K,D] slots")
    _finite(value, "slots")
    norms = value.norm(dim=-1)
    valid = bool((norms > 1e-8).all())
    count = len(value)
    result = {"slot_count": count, "nonzero_count": int((norms > 1e-8).sum()),
              "min_norm": float(norms.min()), "pair_metrics_defined": count >= 2 and valid}
    gram = F.normalize(value, dim=-1) @ F.normalize(value, dim=-1).T
    result["effective_rank"] = float(count**2 / gram.square().sum()) if valid else None
    names = ("cosine", "absolute_cosine", "rms_cosine", "max_cosine", "fraction_cosine_above_09")
    if result["pair_metrics_defined"]:
        pairs = gram[~torch.eye(count, dtype=torch.bool, device=value.device)]
        result.update(zip(names, (float(pairs.mean()), float(pairs.abs().mean()),
                     float(pairs.square().mean().sqrt()), float(pairs.max()),
                     float((pairs > .9).double().mean()))))
    else:
        result.update({name: None for name in names})
    return result


def reading_overlap(record):
    attention = record["attentions"][-1].double()
    # Head-averaged, row-normalized reads; report per-head representation apart.
    reads = attention.mean(1)[0]
    reads = reads / reads.sum(-1, keepdim=True).clamp_min(1e-12)
    if len(reads) < 2:
        return None
    values = [torch.minimum(reads[left], reads[right]).sum()
              for left in range(len(reads)) for right in range(left + 1, len(reads))]
    return float(torch.stack(values).mean())


def execute_logical_batch(model, heads, adapter, capture, batch, targets, arm,
                          core_optimizer, probe_optimizer, seeds, *, measure_gradient_components=False):
    """Perform one matched core and one detached-probe update on invented/real input.

    Caller owns explicit execution authorization, data preparation and contexts.
    ``targets`` is one detached modality mapping per case, allowing variable K.
    Graphs are retained only across this logical minibatch.
    """
    if arm not in ARM_CONTRACTS:
        raise ValueError("unknown arm")
    count = batch["x_wsi"].shape[0]
    if count < 1 or len(seeds) != count or len(targets) != count or len(set(seeds)) != count:
        raise ValueError("one unique matched seed and target map per example required")
    if not model.training:
        raise ValueError("training batch requires model.train()")
    settings = ARM_CONTRACTS[arm]
    if adapter.variant != settings["adapter"]:
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
        metrics.append({name: angular_metrics(value) for name, value in current.items()})
    logits = torch.cat(logits)
    main = model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None) / count
    auxiliary = torch.stack(auxiliary).mean()
    probe_loss = torch.stack([coverage_set_mse(heads[name](current[name].detach()), targets[index][name])
                              for index, current in enumerate(slots) for name in heads]).mean()
    _finite(probe_loss, "probe loss")
    probe_loss.backward()
    probe_gradient = _gradient_norm(heads)
    # Detached probe training must not back-propagate into the native model.
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("diagnostic probe leaked gradients into native core")
    probe_optimizer.step()
    _finite_module(heads, "updated probe")
    probe_optimizer.zero_grad(set_to_none=True)
    zero = main.new_zeros(())
    gram, paper, content = zero, zero, zero
    with _frozen(heads):
        if settings["gram_weight"]:
            gram = torch.stack([slot_gram_loss(current[name]) for current in slots for name in heads]).mean()
        if settings["slotbert_weight"]:
            paper = torch.stack([slotbert_loss(current[name], settings["slotbert_temperature"])
                                 for current in slots for name in heads]).mean()
        if settings["content_weight"]:
            content = torch.stack([coverage_set_mse(heads[name](current[name]), targets[index][name])
                                   for index, current in enumerate(slots) for name in heads]).mean()
        added = settings["gram_weight"] * gram + settings["slotbert_weight"] * paper + settings["content_weight"] * content
        total = main + auxiliary + added
        _finite(total, "core objective")
        components = (_component_gradient_norms(model, {"main_nll": main, "native_auxiliary": auxiliary,
                                                       "weighted_added_objective": added})
                      if measure_gradient_components else None)
        total.backward()
        gradient = _gradient_norm(model)
        if any(parameter.grad is not None for parameter in heads.parameters()):
            raise RuntimeError("core objective changed the frozen diagnostic heads")
        core_optimizer.step()
        _finite_module(model, "updated core")
    if hasattr(adapter, "commit_codebooks"):
        adapter.commit_codebooks()
    return {"main_nll": float(main.detach()), "native_auxiliary": float(auxiliary.detach()),
            "gram_loss": float(gram.detach()), "slotbert_loss": float(paper.detach()),
            "content_loss": float(content.detach()), "added_loss": float(added.detach()),
            "total_loss": float(total.detach()), "probe_loss": float(probe_loss.detach()),
            "core_gradient_norm": gradient, "probe_gradient_norm": probe_gradient,
            "case_count": count, "root_forwards": count, "slot_attention_calls": count * 3,
            "slot_attention_iterations": count * (model.slot_attention_wsi.iters + 2 * model.slot_attention_omic.iters),
            "core_backward_calls": 1, "probe_backward_calls": 1,
            "core_optimizer_steps": 1, "probe_optimizer_steps": 1, "slot_metrics": metrics,
            "component_gradient_norms": components,
            "component_autograd_grad_calls": components["autograd_grad_calls"] if components else 0}


def _mean_optional(values):
    finite = [float(value) for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def matched_angular_distance(slots, reference):
    """Coverage-aware rectangular matching of 1-cos; zero vectors are N/A.

    Scaling unit vectors by sqrt(D/2) makes mean squared feature distance
    exactly 1-cos. Unmatched slots in either set still contribute a cost.
    This distance remains defined for singleton sets; singleton pair cosine
    within a set remains undefined in ``angular_metrics``.
    """
    value, other = slots.detach().double(), reference.detach().double()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if other.ndim == 3 and other.shape[0] == 1:
        other = other[0]
    if value.ndim != 2 or other.ndim != 2 or value.shape[-1] != other.shape[-1]:
        raise ValueError("matching requires two [K,D] slot sets")
    _finite(value, "matching slots")
    _finite(other, "matching reference")
    if min(len(value), len(other)) < 1 or bool((value.norm(dim=-1) <= 1e-8).any()) or bool((other.norm(dim=-1) <= 1e-8).any()):
        return None
    scale = math.sqrt(value.shape[-1] / 2)
    return float(coverage_set_mse(F.normalize(value, dim=-1) * scale,
                                  F.normalize(other, dim=-1) * scale))


def _summarize_diagnostics(rows):
    result = {}
    for name in ("wsi", "omic"):
        metrics = [row["modalities"][name] for row in rows]
        result[name] = {key: _mean_optional([value.get(key) for value in metrics])
                        for key in ("slot_count", "nonzero_count", "min_norm", "cosine", "absolute_cosine",
                                    "rms_cosine", "effective_rank", "read_overlap", "own_mse", "donor_mse", "static_mse",
                                    "same_noise_changed_input_distance", "same_input_changed_noise_distance")}
        result[name]["case_draws"] = len(metrics)
        result[name]["valid_pair_draws"] = sum(bool(value["pair_metrics_defined"]) for value in metrics)
        result[name]["single_slot_draws"] = sum(value["slot_count"] == 1 for value in metrics)
        own, donor = result[name]["own_mse"], result[name]["donor_mse"]
        result[name]["donor_to_own_mse_ratio"] = donor / own if own is not None and own > 0 and donor is not None else None
    return result


def _evaluate_repeat(model, heads, adapter, capture, reference, root, manifest, fold_data,
                     ids, device, seed, repeat, concordance, baseline_slots=None):
    model.eval()
    heads.eval()
    logits, rows, targets, predictions, raw = [], [], [], [], []
    total_nll = 0.0
    with torch.no_grad():
        for index, pid in enumerate(ids):
            # Fixed patch subset: the repeated evaluations exist to measure
            # evaluation randomness, so their input must not change between
            # repeats. The original loader does not sample for validation.
            batch = patient._batch(root, manifest, fold_data, [pid], device, torch,
                                   sample_seed=0)
            target = reference.targets(batch)
            schedule_seed = case_seed(seed, repeat, index, evaluation=True)
            patient._rng(schedule_seed, torch, np)
            adapter.set_noise_seed(schedule_seed + 27001)
            prediction, _ = model(**batch)
            _finite(prediction, "evaluation prediction")
            current = _check_capture(capture, False)
            logits.append(prediction.detach().cpu())
            total_nll += float(model.loss_fn(prediction, y=batch["y"], c=batch["c"], t=None))
            record_map = {record["modality"]: record for record in adapter.records if record["call_index"] == 0}
            if set(record_map) != {"wsi", "omic"}:
                raise RuntimeError("missing prediction-path allocation records")
            row = {"patient_id": pid, "repeat": repeat, "native_seed": schedule_seed, "modalities": {}}
            predicted = {name: heads[name](current[name]).detach().cpu() for name in heads}
            detached_target = {name: value.detach().cpu() for name, value in target.items()}
            raw_item = {"patient_id": pid, "repeat": repeat,
                        "slots": {name: value.detach().cpu() for name, value in current.items()},
                        "probe_predictions": predicted}
            if repeat == 0:
                raw_item["final_attention"] = {name: record_map[name]["attentions"][-1] for name in heads}
                raw_item["targets"] = detached_target
            for name in heads:
                value = angular_metrics(current[name])
                value["read_overlap"] = reading_overlap(record_map[name])
                value["own_mse"] = float(coverage_set_mse(predicted[name], detached_target[name]))
                value["static_mse"] = float(coverage_set_mse(reference.static_targets[name].cpu(), detached_target[name]))
                value["same_noise_changed_input_distance"] = None
                value["same_input_changed_noise_distance"] = (
                    matched_angular_distance(current[name].cpu(), baseline_slots[pid][name])
                    if repeat > 0 and baseline_slots is not None else None)
                row["modalities"][name] = value
            # Re-run on a different case with exactly the original root and
            # allocation seeds. Unlike donor-head swapping, this isolates the
            # input change from native random initialization/Gumbel draws.
            if repeat == 0 and len(ids) > 1:
                donor_id = ids[(index + 1) % len(ids)]
                donor_batch = patient._batch(root, manifest, fold_data, [donor_id], device, torch,
                                             sample_seed=0)
                patient._rng(schedule_seed, torch, np)
                adapter.set_noise_seed(schedule_seed + 27001)
                donor_logits, _ = model(**donor_batch)
                _finite(donor_logits, "same-noise donor prediction")
                donor_slots = _check_capture(capture, False)
                row["same_noise_donor_patient_id"] = donor_id
                raw_item["same_noise_donor_slots"] = {name: value.detach().cpu() for name, value in donor_slots.items()}
                for name in heads:
                    row["modalities"][name]["same_noise_changed_input_distance"] = matched_angular_distance(current[name], donor_slots[name])
                    row["modalities"][name]["same_noise_donor_slot_count"] = donor_slots[name].shape[1]
            rows.append(row)
            predictions.append(predicted)
            targets.append(detached_target)
            raw.append(raw_item)
        for index, row in enumerate(rows):
            donor = (index + 1) % len(rows)
            row["donor_patient_id"] = ids[donor] if len(rows) > 1 else None
            for name in heads:
                row["modalities"][name]["donor_mse"] = (float(coverage_set_mse(predictions[donor][name], targets[index][name]))
                                                            if len(rows) > 1 else None)
    logits = torch.cat(logits)
    risks = -(1 - logits.sigmoid()).cumprod(-1).sum(-1)
    events = np.asarray([manifest["patients"][pid]["censor"] < 1 for pid in ids], dtype=bool)
    times = np.asarray([manifest["patients"][pid]["time"] for pid in ids], dtype=float)
    try:
        score = float(concordance(events, times, risks.numpy())[0])
        if not math.isfinite(score):
            score = None
        status = "computed" if score is not None else "undefined_nonfinite"
    except ValueError as exc:
        score, status = None, "undefined: " + str(exc)
    evaluation = {"repeat": repeat, "nll": total_nll / len(ids), "cindex": score,
                  "cindex_status": status, "risks": risks.tolist(), "logits": logits.tolist(),
                  "seed_schedule": "case_seed(seed,repeat,validation_position,evaluation=True)",
                  "slot_diagnostics": _summarize_diagnostics(rows)}
    return evaluation, rows, raw


def run_arm(config, root, manifest, arm, initial_core, initial_probe, reference, *, _generated_test=False):
    """Train one declared arm, retaining receipts even on failure; no resume."""
    if arm not in ARMS or arm not in config["variants"]:
        raise ValueError("arm outside the declared eleven-arm queue")
    if config["study"] != "blca" or config["folds"] != [0] or config["seeds"] != [3]:
        raise ValueError("this queue is locked to BLCA fold0 seed3")
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
    out = root / "arms" / arm
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
    fold_data = manifest["folds"]["0"]
    train_ids, val_ids = fold_data["train"], fold_data["val"]
    if not train_ids or not val_ids or set(train_ids) & set(val_ids):
        raise ValueError("nonempty disjoint training/validation IDs required")
    if reference.train_case_count != len(train_ids):
        raise ValueError("reference fit count does not match training cohort")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seed = config["seeds"][0]
    protocol = {"arm": arm, "contract": ARM_CONTRACTS[arm], "fold": 0, "seed": seed,
                "initial_core_hash": state_hash(initial_core), "initial_probe_hash": state_hash(initial_probe),
                "source_hashes": source_before, "epochs": config["epochs"], "batch_size": config["batch_size"],
                "patches": config["patches"], "checkpoint_selection": "fixed_final_epoch_no_selection",
                "native_forward_protocol": "B1_for_all_arms_accumulated_into_logical_batch",
                "probe_protocol": "one_detached_update_then_frozen_head_core_objective_per_logical_batch",
                "content_reference_fit": "training_inputs_only_no_outcomes",
                "added_loss_slot_calls": {"wsi": 0, "omic": 0},
                "raw_data_privacy": "all_outputs_in_ignored_local_evidence",
                "started_utc": patient._now()}
    patient._dump(out / "protocol.json", protocol)
    counters = {"root_forwards": 0, "slot_attention_calls": 0, "slot_attention_iterations": 0,
                "core_backward_calls": 0, "probe_backward_calls": 0, "core_optimizer_steps": 0,
                "probe_optimizer_steps": 0, "component_autograd_grad_calls": 0}
    if device.type == "cuda":
        # torch.cuda.reset_peak_memory_stats raises "Invalid device argument"
        # when no CUDA context exists yet, because it is called before the
        # model is moved onto the device. Initialise the context explicitly so
        # the peak-memory baseline is actually reset rather than skipped.
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
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
        curve = []
        with make_native_adapter(model, ARM_CONTRACTS[arm]["adapter"], capture=True) as adapter, SlotCapture(model) as capture:
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
                print(f"arm={arm} epoch={epoch+1}/{config['epochs']} core_steps={counters['core_optimizer_steps']} loss={curve[-1]['total_loss']:.5f}", flush=True)
            torch.save({"state_dict": _cpu_state(model), "probe_state_dict": _cpu_state(heads),
                        "reference_state": reference.state_dict(), "optimizer_state": core_optimizer.state_dict(),
                        "probe_optimizer_state": probe_optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                        "probe_scheduler_state": probe_scheduler.state_dict(), "protocol": protocol,
                        "omic_sizes": manifest["omic_sizes"], "encoding_dim": manifest["encoding_dim"],
                        "counters": dict(counters)}, out / "last.pt")
            from sksurv.metrics import concordance_index_censored
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
        summary = {"status": "completed", "arm": arm, "fold": 0, "seed": seed,
                   "epochs": config["epochs"], "training_cases": len(train_ids), "validation_cases": len(val_ids),
                   "protocol": protocol, "counters": counters, "evaluation_summary": patient._repeat_summary(evaluations, np),
                   "slot_diagnostics": _summarize_diagnostics(diagnostics), "source_unchanged": True,
                   "last_checkpoint_sha256": patient._sha(out / "last.pt"), "elapsed_seconds": time.perf_counter() - started,
                   "cuda_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   "claim_boundary": "single_fold_short_budget_exploratory_not_generalization_or_clinical_benefit"}
        patient._dump(out / "summary.json", summary)
        return summary
    except BaseException as exc:
        patient._dump(out / "failure.json", {"status": "failed", "type": type(exc).__name__, "message": str(exc),
                                             "counters_completed": counters, "elapsed_seconds": time.perf_counter() - started})
        raise
