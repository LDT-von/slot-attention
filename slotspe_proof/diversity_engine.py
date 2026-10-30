"""SlotSPE SCM/SRO loss arms using the unchanged best-checkpoint trainer.

The historical engine is reused through a function-local globals namespace;
no module globals or original files are patched.  Only the training objective
and graph-connected observation of the native S0 reads are added.  All input,
prediction, auxiliary-loss, diagnostic-probe and checkpoint rules stay with the
existing best-checkpoint engine.  Patient execution belongs to an explicit
caller/queue; importing this module never loads patient data.
"""
from __future__ import annotations

import math
from pathlib import Path
import threading
from types import FunctionType, SimpleNamespace

import numpy as np
import torch

from . import patient, redundancy_best_engine as best, redundancy_engine as base
from .assignment_router import AssignmentConfig, SlotAssignmentRouter, replay_with_assignment
from .native_allocation import NativeAllocation
from .redundancy_content import coverage_set_mse
from .slot_diversity_losses import (
    slot_cosine_margin_loss, slot_read_overlap_loss, warmup_factor,
)


ARMS = ("SCM", "SRO", "R4_SCM", "R4_SRO")
DEFAULT_DIVERSITY = {
    "cosine_margin": 0.90,
    "read_overlap_margin": 0.90,
    "weight": 0.10,
    "warmup_epochs": 5,
}
VALID_MASK_KEY = "_slot_diversity_wsi_valid_mask"
_RUN_LOCK = threading.Lock()


def validate_diversity_config(values=None):
    """Return a strict copy; unknown settings must not silently do nothing."""
    provided = {} if values is None else dict(values)
    unknown = set(provided) - set(DEFAULT_DIVERSITY)
    if unknown:
        raise ValueError(f"unknown diversity settings: {sorted(unknown)}")
    result = {**DEFAULT_DIVERSITY, **provided}
    for name in ("cosine_margin", "read_overlap_margin"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be finite in [0,1)")
        result[name] = float(value)
    value = result["weight"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("weight must be finite and nonnegative")
    result["weight"] = float(value)
    warmup = result["warmup_epochs"]
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup_epochs must be a nonnegative integer")
    return result


def arm_contract(arm, diversity=None):
    if arm not in ARMS:
        raise ValueError("arm must be SCM, SRO, R4_SCM or R4_SRO")
    return {
        "adapter": "S0", "gram_weight": 0.0, "slotbert_weight": 0.0,
        "slotbert_temperature": 0.5,
        "content_weight": 5.0 if arm.startswith("R4_") else 0.0,
        "diversity_kind": "SCM" if arm.endswith("SCM") else "SRO",
        "diversity": validate_diversity_config(diversity),
        "diversity_modalities": "equal_mean_wsi_omic",
        "diversity_slot_calls": {"wsi": 0, "omic": 0},
        "read_loss_scope": "last_iteration_head_mean_real_tokens_only",
        "no_hazard_variance_or_added_prediction_head": True,
    }


class GraphReadingAllocation(NativeAllocation):
    """S0 replay with the final prediction-path reads retained for autograd.

    Detached ``records`` retain the historical diagnostics interface.  The
    prediction tensor, RNG draws, model state and unmasked original attention
    computation are unchanged.  Masks are used only by the SRO loss.
    """

    def __init__(self, model, capture=True):
        super().__init__(model, "S0", capture=capture)
        self.graph_attentions = {}

    def _reset(self, model, args):
        super()._reset(model, args)
        self.graph_attentions.clear()

    def _forward(self, attention, inputs, modality, num_slots=None):
        index = self._calls[modality]
        self._calls[modality] += 1
        count = attention.num_slots if num_slots is None else num_slots
        mu = attention.slots_mu.expand(inputs.shape[0], count, -1)
        sigma = attention.slots_logsigma.exp().expand(inputs.shape[0], count, -1)
        initial = mu + sigma * torch.randn(mu.shape, device=inputs.device, dtype=inputs.dtype)
        router = SlotAssignmentRouter(AssignmentConfig(mode="original", eps=attention.eps))
        result = replay_with_assignment(attention, inputs, initial, router)
        if index == 0 and self.model.training and torch.is_grad_enabled():
            self.graph_attentions[modality] = result["attentions"][-1]
        if self.capture:
            self.records.append({
                "modality": modality, "call_index": index, "mode": "original",
                "source_input_count": inputs.shape[1],
                "initial": initial.detach().cpu(),
                "final": result["final"].detach().cpu(),
                "attentions": [value.detach().cpu() for value in result["attentions"]],
            })
        return result["final"]

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            self.graph_attentions.clear()


def add_real_token_mask(batch, manifest, ids):
    """Use recorded real-patch counts, never zero-valued feature heuristics.

    The published sampler puts all real rows first and appends zero padding.
    A subset of a long bag contains only real rows.  This metadata mask is not
    passed to any attention operator and therefore does not repair padding.
    """
    if batch["x_wsi"].ndim != 3 or len(ids) != len(batch["x_wsi"]):
        raise ValueError("batch/identity metadata disagree")
    width = batch["x_wsi"].shape[1]
    counts = []
    for pid in ids:
        count = manifest["patients"][pid].get("full_patch_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("full_patch_count metadata is required for every training case")
        counts.append(min(count, width))
    mask = torch.arange(width, device=batch["x_wsi"].device)[None, :] < torch.tensor(
        counts, device=batch["x_wsi"].device)[:, None]
    return {**batch, VALID_MASK_KEY: mask}


def _gradient_diagnostics(model, losses):
    """Partial Q/V gradients, not a claim about the entire model's gradient."""
    names, parameters = [], []
    for modality in ("wsi", "omic"):
        module = getattr(model, "slot_attention_" + modality)
        for part in ("to_q", "to_v"):
            names.append(f"{modality}.{part}.weight")
            parameters.append(getattr(module, part).weight)
    result = {"parameters": names, "scope": "native_SA_Q_and_V_weights_only", "autograd_grad_calls": 0}
    retained = {}
    for label, loss in losses.items():
        gradients = (torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                     if loss.requires_grad else [None] * len(parameters))
        result["autograd_grad_calls"] += int(loss.requires_grad)
        result[label] = {}
        retained[label] = []
        for name, parameter, gradient in zip(names, parameters, gradients):
            if gradient is None:
                result[label][name] = {"connected": False, "norm": None}
                retained[label].append(torch.zeros_like(parameter).reshape(-1))
            else:
                base._finite(gradient, "diversity diagnostic gradient")
                value = gradient.detach()
                result[label][name] = {"connected": True, "norm": float(value.norm())}
                retained[label].append(value.reshape(-1))
        retained[label] = torch.cat(retained[label])
    result["cosines"] = {}
    for label in ("main_nll", "weighted_content"):
        left, right = retained[label], retained["weighted_diversity"]
        denom = left.norm() * right.norm()
        result["cosines"][label + "_vs_diversity"] = float((left @ right) / denom) if float(denom) > 0 else None
    return result


def execute_diversity_batch(model, heads, adapter, capture, batch, targets, arm,
                            core_optimizer, probe_optimizer, seeds, *, epoch,
                            diversity=None, measure_gradient_components=False):
    """The original detached-probe/core update, plus one scoped loss term."""
    settings = arm_contract(arm, diversity)
    options = settings["diversity"]
    count = batch["x_wsi"].shape[0]
    if count < 1 or len(seeds) != count or len(targets) != count or len(set(seeds)) != count:
        raise ValueError("one unique matched seed and target map per example required")
    if not model.training or adapter.variant != "S0":
        raise ValueError("diversity loss arms require training with the S0 native adapter")
    factor = warmup_factor(epoch, options["warmup_epochs"])
    effective_weight = options["weight"] * factor
    if settings["diversity_kind"] == "SRO" and effective_weight and VALID_MASK_KEY not in batch:
        raise ValueError("SRO requires metadata-derived real-token validity")
    core_optimizer.zero_grad(set_to_none=True)
    probe_optimizer.zero_grad(set_to_none=True)
    logits, auxiliary, slots, metrics, diverse_terms, diverse_stats = [], [], [], [], [], []
    # Remove private metadata from the model kwargs and from content targets.
    model_batch = {key: value for key, value in batch.items() if key != VALID_MASK_KEY}
    for index, seed in enumerate(seeds):
        patient._rng(int(seed), torch, np)
        adapter.set_noise_seed(int(seed) + 27001)
        prediction, native_auxiliary = model(**base._item(model_batch, index))
        base._finite(prediction, "training prediction")
        current = base._check_capture(capture, True)
        slots.append(current)
        logits.append(prediction)
        auxiliary.append(torch.as_tensor(native_auxiliary, device=prediction.device))
        metrics.append({name: base.angular_metrics(value) for name, value in current.items()})
        if effective_weight:
            case_stats = {}
            for name in ("wsi", "omic"):
                if settings["diversity_kind"] == "SCM":
                    term, stats = slot_cosine_margin_loss(current[name], margin=options["cosine_margin"])
                else:
                    attention = adapter.graph_attentions.get(name)
                    if attention is None or not attention.requires_grad:
                        raise RuntimeError("SRO must observe graph-connected prediction-path reads")
                    valid = (batch[VALID_MASK_KEY][index:index + 1] if name == "wsi" else
                             torch.ones((1, attention.shape[-1]), dtype=torch.bool, device=attention.device))
                    term, stats = slot_read_overlap_loss(attention, valid, margin=options["read_overlap_margin"])
                diverse_terms.append(term)
                case_stats[name] = stats
            diverse_stats.append(case_stats)
    logits = torch.cat(logits)
    main = model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None) / count
    auxiliary = torch.stack(auxiliary).mean()
    probe_loss = torch.stack([
        coverage_set_mse(heads[name](current[name].detach()), targets[index][name])
        for index, current in enumerate(slots) for name in heads]).mean()
    base._finite(probe_loss, "probe loss")
    probe_loss.backward()
    probe_gradient = base._gradient_norm(heads)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("diagnostic probe leaked gradients into native core")
    probe_optimizer.step()
    base._finite_module(heads, "updated probe")
    probe_optimizer.zero_grad(set_to_none=True)
    zero = main.new_zeros(())
    content = zero
    diverse = torch.stack(diverse_terms).mean() if diverse_terms else zero
    with base._frozen(heads):
        if settings["content_weight"]:
            content = torch.stack([
                coverage_set_mse(heads[name](current[name]), targets[index][name])
                for index, current in enumerate(slots) for name in heads]).mean()
        weighted_content = settings["content_weight"] * content
        weighted_diverse = effective_weight * diverse
        # Exact historical addition order when diversity is disabled.
        added = 0.0 * zero + 0.0 * zero + weighted_content
        if effective_weight:
            added = added + weighted_diverse
        total = main + auxiliary + added
        base._finite(total, "core objective")
        components = _gradient_diagnostics(model, {
            "main_nll": main, "native_auxiliary": auxiliary,
            "weighted_content": weighted_content, "weighted_diversity": weighted_diverse,
        }) if measure_gradient_components else None
        total.backward()
        gradient = base._gradient_norm(model)
        if any(parameter.grad is not None for parameter in heads.parameters()):
            raise RuntimeError("core objective changed the frozen diagnostic heads")
        core_optimizer.step()
        base._finite_module(model, "updated core")
    return {
        "main_nll": float(main.detach()), "native_auxiliary": float(auxiliary.detach()),
        "gram_loss": 0.0, "slotbert_loss": 0.0, "content_loss": float(content.detach()),
        "diversity_kind": settings["diversity_kind"], "diversity_loss": float(diverse.detach()),
        "scm_loss": float(diverse.detach()) if settings["diversity_kind"] == "SCM" else 0.0,
        "sro_loss": float(diverse.detach()) if settings["diversity_kind"] == "SRO" else 0.0,
        "diversity_epoch": epoch, "diversity_warmup_factor": factor,
        "diversity_effective_weight": effective_weight,
        "weighted_diversity_loss": float(weighted_diverse.detach()),
        "weighted_content_loss": float(weighted_content.detach()),
        "diversity_metrics": diverse_stats,
        "added_loss": float(added.detach()), "total_loss": float(total.detach()),
        "probe_loss": float(probe_loss.detach()), "core_gradient_norm": gradient,
        "probe_gradient_norm": probe_gradient, "case_count": count,
        "root_forwards": count, "slot_attention_calls": count * 3,
        "slot_attention_iterations": count * (model.slot_attention_wsi.iters + 2 * model.slot_attention_omic.iters),
        "core_backward_calls": 1, "probe_backward_calls": 1,
        "core_optimizer_steps": 1, "probe_optimizer_steps": 1, "slot_metrics": metrics,
        "component_gradient_norms": components,
        "component_autograd_grad_calls": components["autograd_grad_calls"] if components else 0,
    }


def run_diversity_arm(config, root, manifest, arm, initial_core, initial_probe, reference, *,
                      fold=0, output_root=None, _generated_test=False):
    """Run a new arm while preserving the historical trainer's state machine.

    A caller must explicitly authorize patient execution before using this
    function. The queue is the CLI execution gate. Generated tests must use
    the historical engine's generated-only CPU override.
    """
    options = validate_diversity_config(config.get("diversity"))
    contract = arm_contract(arm, options)
    train_ids = manifest["folds"][str(fold)]["train"]
    for pid in train_ids:
        count = manifest["patients"][pid].get("full_patch_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("full_patch_count metadata is required before running a diversity arm")
    batch_size = config.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    batches_per_epoch = math.ceil(len(train_ids) / batch_size)
    if batches_per_epoch < 1:
        raise ValueError("nonempty training fold required")
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("diversity engine is serial within a process; use separate processes per arm")
    calls = 0
    try:
        def batch_with_validity(root_arg, manifest_arg, fold_data, ids, device, torch_arg,
                                np=None, budget=None, sample_seed=None):
            batch = patient._batch(root_arg, manifest_arg, fold_data, ids, device, torch_arg,
                                   np=np, budget=budget, sample_seed=sample_seed)
            return add_real_token_mask(batch, manifest_arg, ids)

        def executor(*args, **kwargs):
            nonlocal calls
            epoch = calls // batches_per_epoch + 1
            if epoch > config["epochs"]:
                raise RuntimeError("training calls exceeded the declared epoch budget")
            result = execute_diversity_batch(*args, epoch=epoch, diversity=options, **kwargs)
            calls += 1
            return result

        def adapter_factory(model, variant, capture=True):
            if variant != "S0":
                raise ValueError("diversity uses the original allocation only")
            return GraphReadingAllocation(model, capture=capture)

        namespace = dict(best.run_arm_best.__globals__)
        namespace.update({
            "ARMS": ARMS,
            "ARM_CONTRACTS": {name: arm_contract(name, options) for name in ARMS},
            "execute_logical_batch": executor,
            "make_native_adapter": adapter_factory,
            "patient": SimpleNamespace(**{**vars(patient), "_batch": batch_with_validity}),
        })
        runner = FunctionType(best.run_arm_best.__code__, namespace,
                              name="run_slot_diversity_best", argdefs=best.run_arm_best.__defaults__,
                              closure=best.run_arm_best.__closure__)
        runner.__kwdefaults__ = dict(best.run_arm_best.__kwdefaults__ or {})
        result = runner(config, root, manifest, arm, initial_core, initial_probe, reference,
                        fold=fold, output_root=output_root, _generated_test=_generated_test)
        if calls != config["epochs"] * batches_per_epoch:
            raise RuntimeError("training-call count differs from the declared epoch budget")
        result["loss_adapter"] = {
            "schema": "slotspe_diversity_loss_adapter_v1", "contract": contract,
            "historical_best_engine_reused_without_mutation": True,
            "original_padding_in_forward_unchanged": True,
            "real_token_mask_scope": "SRO_objective_only",
            "new_prediction_parameters": 0,
            "exact_collapse_escape_guaranteed": False,
            "no_claim_of_patient_benefit": True,
        }
        out = (Path(output_root) if output_root is not None else Path(root)) / "arms" / arm
        patient._dump(out / "summary.json", result)
        return result
    finally:
        _RUN_LOCK.release()
