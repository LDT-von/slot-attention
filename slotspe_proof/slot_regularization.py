"""Generated-data probes and losses on unmodified native SlotSPE outputs.

These experimental losses do not replace slot representations, add slot IDs,
or change the original forward function. They are not patient-ready methods.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _validate_slots(slots):
    if slots.ndim != 3 or min(slots.shape) < 1 or slots.shape[1] < 2:
        raise ValueError("expected nonempty [B,K,D] with at least two slots")
    if not slots.is_floating_point() or not bool(torch.isfinite(slots).all()):
        raise ValueError("slots must be finite floating tensors")
    if bool((slots.norm(dim=-1) <= 1e-8).any()):
        raise ValueError("zero slots do not establish angular diversity")


def _gram(slots):
    _validate_slots(slots)
    unit = F.normalize(slots, dim=-1)
    return unit @ unit.transpose(-1, -2)


def slot_gram_loss(slots):
    """Mean squared off-diagonal cosine; antipodal slots are penalized too."""
    gram = _gram(slots)
    mask = ~torch.eye(slots.shape[1], dtype=torch.bool, device=slots.device)
    return gram[:, mask].square().mean()


def slotbert_loss(slots, temperature=0.5):
    """Slot-BERT Eq.11 loss transplant, including its diagonal entries.

    The paper's temporal dimension is replaced by independent batch examples.
    This is not a reproduction of its video model or author implementation.
    The irreducible uniform-cross-entropy constant is log(K).
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    gram = _gram(slots)
    cost = gram - torch.eye(slots.shape[1], device=slots.device, dtype=slots.dtype)
    return -F.log_softmax(-cost / temperature, dim=-1).mean()


class SlotCapture:
    """Hold graph-connected outputs of each native slot-attention call.

    Call zero in each modality is used by the prediction path. The native
    training forward additionally reuses the omic module for reconstruction.
    Hooks only observe outputs; they never return replacement tensors.
    """

    def __init__(self, model):
        self.model = model
        self.slots = {"wsi": [], "omic": []}
        self._handles = []

    def __enter__(self):
        if self._handles:
            raise RuntimeError("capture context is already active")
        try:
            self._handles.append(self.model.register_forward_pre_hook(self._clear))
            for name in self.slots:
                module = getattr(self.model, "slot_attention_" + name)
                self._handles.append(module.register_forward_hook(self._hook(name)))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _clear(self, module, args):
        for values in self.slots.values():
            values.clear()

    def _hook(self, name):
        def observe(module, args, output):
            if not torch.is_tensor(output) or output.ndim != 3:
                raise ValueError("native slot attention must return [B,K,D]")
            self.slots[name].append(output)
        return observe

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for values in self.slots.values():
            values.clear()


class QueryFreeSlotDecoder(nn.Module):
    """One shared MLP per modality, with slots as its only input.

    No target tokens, target queries, slot identities, or positional embeddings
    are available to this head. Each slot produces one raw-feature prediction.
    """

    def __init__(self, slot_dim, output_dim):
        super().__init__()
        if min(slot_dim, output_dim) < 1:
            raise ValueError("feature dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(slot_dim, 2 * slot_dim), nn.ReLU(),
            nn.Linear(2 * slot_dim, output_dim),
        )

    def forward(self, slots):
        return self.network(slots)


def matched_set_mse(pred, target, reduction="mean"):
    """Hungarian set MSE, detached targets and differentiable selected costs."""
    from scipy.optimize import linear_sum_assignment

    if pred.ndim != 3 or pred.shape != target.shape or min(pred.shape) < 1:
        raise ValueError("prediction and target need identical nonempty [B,K,D]")
    if reduction not in ("mean", "none"):
        raise ValueError("reduction must be 'mean' or 'none'")
    if not bool(torch.isfinite(pred).all() and torch.isfinite(target).all()):
        raise ValueError("set inputs must be finite")
    costs = (pred[:, :, None] - target.detach()[:, None]).square().mean(-1)
    values = []
    for cost in costs:
        rows, columns = linear_sum_assignment(cost.detach().cpu().double().numpy())
        values.append(cost[rows, columns].mean())
    result = torch.stack(values)
    return result.mean() if reduction == "mean" else result


def slot_metrics(slots):
    """Raw-output angular checks; no centering, projection, or pair pruning."""
    _validate_slots(slots)
    slots = slots.detach().double()
    gram = _gram(slots)
    count = slots.shape[1]
    mask = ~torch.eye(count, dtype=torch.bool, device=slots.device)
    pairs = gram[:, mask]
    return {
        "cosine": float(pairs.mean()),
        "absolute_cosine": float(pairs.abs().mean()),
        "rms_cosine": float(pairs.square().mean().sqrt()),
        "max_cosine": float(pairs.max()),
        "fraction_cosine_above_09": float((pairs > 0.9).double().mean()),
        "effective_rank": float((count ** 2 / gram.square().sum((-1, -2))).mean()),
        "min_norm": float(slots.norm(dim=-1).min()),
        "slot_count": count,
        "case_count": slots.shape[0],
    }
