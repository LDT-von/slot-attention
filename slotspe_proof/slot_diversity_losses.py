"""Bounded, scale-aware SlotSPE diversity objectives on native outputs.

SCM penalizes only slot-direction cosine above a declared margin. SRO applies
the same soft margin to Bhattacharyya overlap of native reading distributions.
Neither objective guarantees escape from an exactly symmetric collapse, nor
does a lower objective establish biological specialization or predictive gain.
The valid-token mask affects SRO's loss only; it never masks the native forward.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _margin(value):
    if isinstance(value, bool):
        raise ValueError("margin must be finite and in [0, 1)")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("margin must be finite and in [0, 1)") from exc
    if not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("margin must be finite and in [0, 1)")
    return value


def _work_tensor(value, name, dimensions):
    if (not torch.is_tensor(value) or value.ndim not in dimensions or
            min(value.shape) < 1 or not value.is_floating_point()):
        raise ValueError(f"{name} must be a nonempty floating tensor of rank {dimensions}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value if value.dtype == torch.float64 else value.float()


def _hinge_pairs(pairs, margin):
    # Normalize by the remaining interval so both new objectives lie in [0, 1].
    return (F.relu(pairs - margin) / (1.0 - margin)).square()


def slot_cosine_margin_loss(slots, margin=0.90):
    """Return SCM and scalar diagnostics for native slots ``[B, K, D]``.

    For each directed off-diagonal pair, SCM is
    ``relu((cosine - margin) / (1 - margin))**2``. Pairs are averaged within
    each patient and then across patients. Negative and sufficiently small
    positive cosine incur no penalty: this is not Gram orthogonalization.
    Positive slot rescaling leaves SCM unchanged. Zero vectors are rejected,
    since they do not define directions. A single slot yields graph-connected
    zero. Computation uses float32, or float64 when the input uses float64.
    """
    margin = _margin(margin)
    work = _work_tensor(slots, "slots", (3,))
    batch, count, _ = work.shape
    # Rescale before taking a norm to avoid overflow/underflow of squared norms.
    scale = work.abs().amax(dim=-1, keepdim=True)
    if bool((scale == 0).any()):
        raise ValueError("zero slots do not define angular diversity")
    scaled = work / scale
    unit = scaled / scaled.norm(dim=-1, keepdim=True)
    stats = {"margin": margin, "patient_count": batch, "slot_count": count,
             "pair_count_per_patient": count * (count - 1)}
    if count == 1:
        loss = work.sum() * 0.0
        stats.update(loss=0.0, mean_pair_cosine=0.0, max_pair_cosine=0.0,
                     violating_pair_fraction=0.0, pair_statistics_defined=False)
        return loss, stats
    gram = (unit @ unit.transpose(-1, -2)).clamp(-1.0, 1.0)
    off_diagonal = ~torch.eye(count, device=work.device, dtype=torch.bool)
    pairs = gram[:, off_diagonal]
    loss = _hinge_pairs(pairs, margin).mean(dim=1).mean()
    detached = pairs.detach()
    stats.update(loss=float(loss.detach()), mean_pair_cosine=float(detached.mean()),
                 max_pair_cosine=float(detached.max()),
                 violating_pair_fraction=float((detached > margin).float().mean()),
                 pair_statistics_defined=True)
    return loss, stats


def slot_read_overlap_loss(attentions, valid_mask, margin=0.90, eps=1e-8):
    """Return SRO and scalar diagnostics on graph-connected native readings.

    ``attentions`` is ``[B, H, K, N]`` or ``[B, K, N]``; heads are averaged
    first. ``valid_mask`` is boolean ``[B, N]`` and identifies real input
    tokens. For each patient, keep valid tokens only, add ``eps``, normalize
    each slot's row, and calculate ``BC(i,j)=sum_n sqrt(p_i[n]*p_j[n])``.
    The pair cost is ``relu((BC - margin) / (1 - margin))**2``.

    Every patient contributes equally, including graph-zero cases with only
    one valid token or one slot (reading diversity is unavailable there).
    Such cases are counted separately and are not evidence of successful
    diversity. Pair-overlap diagnostics exclude those undefined cases.
    Invalid tokens have exactly zero loss gradient. This helper never changes
    attention tensors or the model's forward behavior. All-zero valid reading
    rows are rejected instead of fabricating a distribution from smoothing.
    """
    margin = _margin(margin)
    if isinstance(eps, bool):
        raise ValueError("eps must be finite and positive")
    try:
        eps = float(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError("eps must be finite and positive") from exc
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    work = _work_tensor(attentions, "attentions", (3, 4))
    if bool((work < 0).any()):
        raise ValueError("attentions must be nonnegative")
    if work.ndim == 4:
        work = work.mean(dim=1)
    batch, count, tokens = work.shape
    if (not torch.is_tensor(valid_mask) or valid_mask.dtype != torch.bool or
            valid_mask.shape != (batch, tokens) or valid_mask.device != work.device):
        raise ValueError("valid_mask must be boolean [B, N] on the attention device")
    if not bool(valid_mask.any(dim=-1).all()):
        raise ValueError("every patient must have at least one real token")
    costs, pair_means, pair_maxima, violations = [], [], [], []
    skipped = 0
    off_diagonal = ~torch.eye(count, device=work.device, dtype=torch.bool)
    for index in range(batch):
        selected = work[index, :, valid_mask[index]]
        if bool((selected.sum(dim=-1) <= 0).any()):
            raise ValueError("each slot needs positive mass over real tokens")
        if count == 1 or selected.shape[-1] == 1:
            costs.append(selected.sum() * 0.0)
            skipped += 1
            continue
        # Native reading rows are normalized; smoothing makes sqrt safe even
        # for artificial exact-zero rows used by diagnostic counterexamples.
        probability = selected + eps
        probability = probability / probability.sum(dim=-1, keepdim=True)
        root = probability.sqrt()
        overlap = (root @ root.transpose(-1, -2)).clamp(0.0, 1.0)
        pairs = overlap[off_diagonal]
        costs.append(_hinge_pairs(pairs, margin).mean())
        detached = pairs.detach()
        pair_means.append(detached.mean())
        pair_maxima.append(detached.max())
        violations.append((detached > margin).float().mean())
    loss = torch.stack(costs).mean()
    stats = {"loss": float(loss.detach()), "margin": margin, "eps": eps,
             "patient_count": batch, "slot_count": count,
             "pair_count_per_patient": count * (count - 1),
             "eligible_patient_count": batch - skipped,
             "skipped_no_read_diversity_count": skipped,
             "min_valid_token_count": int(valid_mask.sum(-1).min()),
             "max_valid_token_count": int(valid_mask.sum(-1).max()),
             "pair_statistics_defined": bool(pair_means),
             "mean_pair_overlap": float(torch.stack(pair_means).mean()) if pair_means else 0.0,
             "max_pair_overlap": float(torch.stack(pair_maxima).max()) if pair_maxima else 0.0,
             "violating_pair_fraction": float(torch.stack(violations).mean()) if violations else 0.0}
    return loss, stats


def warmup_factor(epoch, warmup_epochs=5):
    """Linear multiplier for a one-based epoch; zero warmup means full weight."""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("epoch must be a positive one-based integer")
    if (isinstance(warmup_epochs, bool) or not isinstance(warmup_epochs, int) or
            warmup_epochs < 0):
        raise ValueError("warmup_epochs must be a nonnegative integer")
    return 1.0 if warmup_epochs == 0 else min(epoch / warmup_epochs, 1.0)
