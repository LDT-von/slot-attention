"""Shared/private slot adapters and explicitly scoped dependence objectives.

These objectives constrain a set-content proxy and the shared/private *summary*.
They do not establish semantic disentanglement or independence of every slot.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_STD_FLOOR = 1e-10
_GRAM_FLOOR = 1e-12


def _tensor(name: str, value: Tensor, ndim: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided or value.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{name} must be a dense real floating-point tensor")
    if value.ndim != ndim or any(size == 0 for size in value.shape):
        raise ValueError(f"{name} must have {ndim} nonempty dimensions")
    if value.device.type == "meta":
        raise ValueError(f"{name} must contain materialized values")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _same_device(first_name: str, first: Tensor, second_name: str, second: Tensor) -> None:
    if first.device != second.device:
        raise ValueError(f"{first_name} and {second_name} must be on the same device")


class SharedPrivateSlots(nn.Module):
    """An equivariant adapter with a query-free, private-only content head.

    All affine layers start on CPU.  Their initialization restores the caller's
    CPU RNG state and never seeds or forks a CUDA generator.  Move/cast the
    resulting module explicitly before passing slots on another device/dtype.
    """

    def __init__(
        self,
        dimension: int = 256,
        shared_dim: int = 64,
        private_dim: int = 192,
        target_dim: int = 32,
        seed: int = 731,
    ) -> None:
        super().__init__()
        for name, value in (
            ("dimension", dimension), ("shared_dim", shared_dim),
            ("private_dim", private_dim), ("target_dim", target_dim),
        ):
            _positive_int(name, value)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        self.dimension = dimension
        self.shared_dim = shared_dim
        self.private_dim = private_dim
        self.target_dim = target_dim
        self.seed = seed
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.shared_encoder = nn.Linear(dimension, shared_dim, device="cpu")
            self.private_encoder = nn.Linear(dimension, private_dim, device="cpu")
            self.shared_projection = nn.Linear(shared_dim, dimension, device="cpu")
            self.private_projection = nn.Linear(private_dim, dimension, device="cpu")
            self.content_head = nn.Sequential(
                nn.Linear(private_dim, private_dim, device="cpu"),
                nn.GELU(),
                nn.Linear(private_dim, target_dim, device="cpu"),
            )

    def forward(self, slots: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        _tensor("slots", slots, 3)
        if slots.shape[-1] != self.dimension:
            raise ValueError(f"slots feature dimension must be {self.dimension}")
        parameter = self.shared_encoder.weight
        if slots.device != parameter.device or slots.dtype != parameter.dtype:
            raise ValueError("slots must match the adapter parameter device and dtype")
        shared = self.shared_encoder(slots.mean(dim=1))
        private = self.private_encoder(slots)
        latent = (
            self.shared_projection(shared).unsqueeze(1)
            + self.private_projection(private)
        )
        content = self.content_head(private)
        for name, value, ndim in (
            ("shared", shared, 2), ("private", private, 3),
            ("latent", latent, 3), ("content", content, 3),
        ):
            _tensor(name, value, ndim)
        return latent, {"shared": shared, "private": private, "content": content}


def content_set_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared Hungarian loss covering every target exactly once.

    The target count M may be smaller than the number of predictions K.  The
    assignment is a discrete, detached choice; the selected prediction costs
    remain differentiable.  Fixed reference targets never receive gradients.
    Patients are averaged equally, including those with fewer than K targets.
    """
    _tensor("pred", pred, 3)
    _tensor("target", target, 3)
    _same_device("pred", pred, "target", target)
    if pred.shape[0] != target.shape[0] or pred.shape[2] != target.shape[2]:
        raise ValueError("pred and target must have matching batch and feature dimensions")
    if target.shape[1] > pred.shape[1]:
        raise ValueError("target count M must not exceed prediction count K")
    # Imported only for matching; scipy sees detached, finite CPU costs.
    from scipy.optimize import linear_sum_assignment

    difference = pred.to(torch.float64).unsqueeze(2) - target.detach().to(torch.float64).unsqueeze(1)
    costs = difference.square().mean(dim=-1)
    _tensor("matching costs", costs, 3)
    patient_losses = []
    for cost in costs:
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        if len(columns) != target.shape[1]:
            raise RuntimeError("assignment failed to cover every content target")
        row_index = torch.as_tensor(rows, dtype=torch.long, device=cost.device)
        column_index = torch.as_tensor(columns, dtype=torch.long, device=cost.device)
        patient_losses.append(cost[row_index, column_index].mean())
    loss = torch.stack(patient_losses).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("content matching loss must be finite")
    return loss


def target_variance_floor(target: Tensor) -> Tensor:
    """Return the per-patient lower bound for an identical prediction set."""
    _tensor("target", target, 3)
    reference = target.detach().to(torch.float64)
    result = (reference - reference.mean(dim=1, keepdim=True)).square().mean(dim=(1, 2))
    _tensor("target variance floor", result, 1)
    return result


def private_summary(private: Tensor) -> Tensor:
    """Permutation-invariant first/second slot moments, computed in float64."""
    _tensor("private", private, 3)
    values = private.to(torch.float64)
    result = torch.cat((values.mean(dim=1), values.square().mean(dim=1)), dim=-1)
    _tensor("private summary", result, 2)
    return result


def _standardize_columns(values: Tensor, name: str) -> Tensor:
    centered = values.to(torch.float64) - values.to(torch.float64).mean(dim=0, keepdim=True)
    std = centered.square().mean(dim=0, keepdim=True).sqrt()
    if not bool(torch.isfinite(std).all()) or bool((std <= _STD_FLOOR).any()):
        raise ValueError(f"{name} scale degeneracy: column std must exceed {_STD_FLOOR}")
    # No epsilon and no detach: both value- and gradient-level scale invariance
    # depend on differentiating through the current logical batch's statistics.
    standardized = centered / (std * math.sqrt(values.shape[1]))
    _tensor(f"standardized {name}", standardized, 2)
    return standardized


def _centered_gaussian_gram(values: Tensor, name: str) -> Tensor:
    distances = (values.unsqueeze(1) - values.unsqueeze(0)).square().sum(dim=-1)
    gram = torch.exp(-0.5 * distances)
    count = values.shape[0]
    center = torch.eye(count, dtype=torch.float64, device=values.device) - torch.ones(
        (count, count), dtype=torch.float64, device=values.device
    ) / count
    centered = center @ gram @ center
    norm = torch.linalg.matrix_norm(centered, ord="fro")
    if not bool(torch.isfinite(norm)) or bool(norm <= _GRAM_FLOOR):
        raise ValueError(f"{name} kernel degeneracy: centered Gram norm must exceed {_GRAM_FLOOR}")
    return centered


def summary_hsic(shared: Tensor, summary: Tensor) -> Tensor:
    """Biased, standardized Gaussian HSIC on N distinct patient records.

    The value is tr(K H L H)/N**2, using bandwidth 1.  Batch standardization
    couples rows, so this training penalty is not an unbiased estimator or an
    independence test with a published significance threshold.  It addresses
    only shared versus the provided summary, not every private slot.
    """
    _tensor("shared", shared, 2)
    _tensor("summary", summary, 2)
    _same_device("shared", shared, "summary", summary)
    if shared.shape[0] != summary.shape[0]:
        raise ValueError("shared and summary must have the same patient count")
    if shared.shape[0] < 2:
        raise ValueError("HSIC requires at least two distinct patients")
    shared_gram = _centered_gaussian_gram(_standardize_columns(shared, "shared"), "shared")
    summary_gram = _centered_gaussian_gram(_standardize_columns(summary, "summary"), "summary")
    hsic = (shared_gram * summary_gram).sum() / shared.shape[0] ** 2
    if not bool(torch.isfinite(hsic)) or bool(hsic < -1e-12):
        raise ValueError("HSIC must be finite and nonnegative within floating-point tolerance")
    return hsic


def separation_terms(shared: Tensor, private: Tensor) -> Dict[str, Tensor]:
    """Return HSIC, the shared variance penalty, and their fixed-weight sum."""
    _tensor("shared", shared, 2)
    _tensor("private", private, 3)
    _same_device("shared", shared, "private", private)
    if shared.shape[0] != private.shape[0]:
        raise ValueError("shared and private must have the same patient count")
    hsic = summary_hsic(shared, private_summary(private))
    values = shared.to(torch.float64)
    variance = torch.relu(1.0 - torch.sqrt(values.var(dim=0, unbiased=False) + 1e-4)).mean()
    separation = 0.1 * hsic + 0.1 * variance
    if not bool(torch.isfinite(separation)):
        raise ValueError("separation loss must be finite")
    return {"hsic": hsic, "variance": variance, "separation": separation}
