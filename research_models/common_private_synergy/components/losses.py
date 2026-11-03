"""Unvalidated loss components. No model, data loader or import-time tensor work.

Tensor execution is deferred: this delivery was checked by AST only. Correlation
is computed across distinct patients, never over patches, slots or repeated
views. These objectives do not identify biological roles, MI or PID synergy.
"""
from dataclasses import dataclass
import math
from typing import Hashable, Literal, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def _positive_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


def _floating_matrix(name: str, x: Tensor) -> None:
    if not isinstance(x, Tensor) or x.ndim != 2 or min(x.shape) < 1:
        raise ValueError(f'{name} must be a nonempty [patients, dimensions] tensor')
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f'{name} requires float32/float64; explicitly upcast AMP inputs')
    if torch.is_autocast_enabled(x.device.type):
        raise ValueError('disable autocast around these correlation/survival components')
    if not torch.isfinite(x).all():
        raise ValueError(f'{name} contains nonfinite values')


def discrete_hazard_nll(
    logits: Tensor,
    completed_intervals: Tensor,
    event: Tensor,
    *,
    reduction: Literal['none', 'mean', 'sum'] = 'mean',
) -> Tensor:
    """NLL for right censoring with an explicit completed-interval convention.

    ``logits[b,k]`` gives logit P(event in interval k | survived before k).
    ``completed_intervals`` is int64; ``event`` is bool, both shape [B].
    For an event, n is the number of prior complete intervals, and the event
    occurs in interval n (0 <= n < K). For censoring, n is the count of intervals
    COMPLETELY observed event-free (0 <= n <= K). Censoring inside an interval
    contributes only its preceding complete intervals, not the containing one.

    This discretized likelihood discards partial-interval follow-up; it does
    not model interval-censored events or informative censoring. An observation
    censored before completing interval 0 has NLL 0; mean includes that row.
    Train-only bin construction and continuous-time-to-count mapping are the
    caller's responsibility. No fitting or bin inference occurs here.
    """
    _floating_matrix('logits', logits)
    if reduction not in ('none', 'mean', 'sum'):
        raise ValueError('invalid reduction')
    batch, bins = logits.shape
    if not isinstance(completed_intervals, Tensor) or not isinstance(event, Tensor):
        raise ValueError('labels must be tensors')
    if completed_intervals.shape != (batch,) or event.shape != (batch,):
        raise ValueError('one label pair per patient required')
    if completed_intervals.dtype != torch.int64 or event.dtype != torch.bool:
        raise ValueError('completed_intervals must be int64 and event must be bool')
    if completed_intervals.device != logits.device or event.device != logits.device:
        raise ValueError('labels and logits must share device')
    if ((completed_intervals < 0) | (completed_intervals > bins)).any():
        raise ValueError('completed_intervals outside [0, K]')
    if (event & (completed_intervals == bins)).any():
        raise ValueError('an event must occur within the modeled horizon')
    axis = torch.arange(bins, device=logits.device)[None, :]
    survived = axis < completed_intervals[:, None]
    nll = (F.softplus(logits) * survived).sum(dim=1)
    # Clamp only the already-validated censor-at-horizon sentinel. It never
    # changes an event index and its selected value contributes no event loss.
    selected = logits.gather(1, completed_intervals.clamp_max(bins - 1)[:, None]).squeeze(1)
    nll = nll + F.softplus(-selected) * event
    if not torch.isfinite(nll).all():
        raise FloatingPointError('NLL overflow; inspect logit magnitude and precision')
    return nll if reduction == 'none' else nll.mean() if reduction == 'mean' else nll.sum()


@dataclass(frozen=True)
class CorrelationPolicy:
    """Caller must choose the minimum patient batch and projection bound.

    Identity-target correlations with positive off-diagonal weight additionally
    require D <= B-1. When all four terms are enabled, dc+2*dp <= B-1 is also
    required for the idealized zero-epsilon targets to be jointly attainable.
    These are necessary sample-rank conditions, not statistical guarantees.
    ``max_projection_dim`` includes the concatenated common/private dimension
    used by each intra-modality loss. No rank guard can be disabled silently.
    """
    min_patients: int
    max_projection_dim: int
    eps: float = 1e-4
    offdiag_weight: float = 0.0051
    normalization: Literal['mean', 'sum'] = 'mean'

    def __post_init__(self) -> None:
        for name, value, lower in (('min_patients', self.min_patients, 2),
                                   ('max_projection_dim', self.max_projection_dim, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f'{name} must be an integer >= {lower}')
        _positive_finite('eps', self.eps)
        if isinstance(self.offdiag_weight, bool) or not isinstance(self.offdiag_weight, (int, float)) or not math.isfinite(self.offdiag_weight) or self.offdiag_weight < 0:
            raise ValueError('offdiag_weight must be finite and nonnegative')
        if self.normalization not in ('mean', 'sum'):
            raise ValueError('normalization must be mean or sum')


@dataclass(frozen=True)
class DeCURWeights:
    """All objectives are opt-in. Zero weights skip lookup and computation."""
    intra_wsi: float = 0.0
    intra_omic: float = 0.0
    cross_common: float = 0.0
    cross_private: float = 0.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


@dataclass
class LossBundle:
    terms: dict[str, Tensor]
    weighted_total: Tensor | None
    diagnostics: dict[str, dict[str, Tensor]]
    skipped: tuple[str, ...]


def _patient_keys(keys: Sequence[Hashable], batch: int) -> None:
    if isinstance(keys, (str, bytes)) or len(keys) != batch or len(set(keys)) != batch:
        raise ValueError('batch needs one distinct patient key per row; repeated views are not patients')


def _correlation(
    x: Tensor, y: Tensor, *, identity_target: bool,
    policy: CorrelationPolicy, patient_keys: Sequence[Hashable],
) -> tuple[Tensor, dict[str, Tensor]]:
    _floating_matrix('x', x)
    _floating_matrix('y', y)
    if x.shape != y.shape or x.device != y.device or x.dtype != y.dtype:
        raise ValueError('correlation inputs need matching shape, device and dtype')
    batch, dim = x.shape
    _patient_keys(patient_keys, batch)
    if batch < policy.min_patients:
        raise ValueError('insufficient distinct-patient batch for declared correlation policy')
    if dim > policy.max_projection_dim:
        raise ValueError('projection exceeds declared dimension budget')
    if identity_target and policy.offdiag_weight > 0 and dim > batch - 1:
        raise ValueError('identity correlation is rank-infeasible: require projection dimension <= B-1')
    xc, yc = x - x.mean(0), y - y.mean(0)
    vx, vy = xc.square().mean(0), yc.square().mean(0)
    correlation = (xc / torch.sqrt(vx + policy.eps)).T @ (yc / torch.sqrt(vy + policy.eps)) / batch
    if not torch.isfinite(correlation).all() or not torch.isfinite(vx).all() or not torch.isfinite(vy).all():
        raise FloatingPointError('correlation moments overflow')
    # Zero variance is a diagnostic, not an exception; a caller can report
    # collapse and continue. Epsilon alone does not repair zero gradients there.
    diagnostics = {'variance_x': vx.detach(), 'variance_y': vy.detach(),
                   'fraction_below_eps_x': (vx < policy.eps).to(x.dtype).mean().detach(),
                   'fraction_below_eps_y': (vy < policy.eps).to(y.dtype).mean().detach()}
    return correlation, diagnostics


def _correlation_penalty(c: Tensor, target: float, policy: CorrelationPolicy) -> Tensor:
    diagonal = (torch.diagonal(c) - target).square()
    mask = ~torch.eye(c.shape[0], device=c.device, dtype=torch.bool)
    offdiag = c[mask].square()
    if policy.normalization == 'sum':
        return diagonal.sum() + policy.offdiag_weight * offdiag.sum()
    return diagonal.mean() + policy.offdiag_weight * (offdiag.mean() if offdiag.numel() else c.new_zeros(()))


def decur_losses(
    view_a: Mapping[str, Tensor], view_b: Mapping[str, Tensor], *,
    patient_keys: Sequence[Hashable], weights: DeCURWeights, policy: CorrelationPolicy,
) -> LossBundle:
    """Group-summary adaptation, not a faithful reproduction of DeCUR training.

    Keys are common_wsi/private_wsi/common_omic/private_omic, each [B,D].
    Only enabled terms inspect their keys. Within a modality, common and private
    projections are concatenated; both augmented views must have matching
    dimensions. Cross-modal terms average same-view comparisons over a/b.
    Shared patient_keys declares the SAME row order for every tensor. Upstream
    code must enforce pairing; this function cannot detect falsely labeled rows.

    The optional mean normalization separately averages diagonal/off-diagonal
    entries and changes the meaning of their coefficient relative to DeCUR sums.
    Batch accumulation is not supported: accumulating independently computed
    microbatch correlations/gradients does not give full-batch correlations.
    If all weights are zero, weighted_total is None (no tensor is inspected).
    """
    terms: dict[str, Tensor] = {}
    diagnostics: dict[str, dict[str, Tensor]] = {}
    skipped: list[str] = []
    total: Tensor | None = None

    if all(value > 0 for value in vars(weights).values()) and policy.offdiag_weight > 0:
        # With exact idealized targets, both views of each coordinate coincide;
        # common coordinates are shared while the two private subspaces must be
        # mutually orthogonal AND orthogonal to common. Per-term rank checks
        # alone therefore miss the necessary joint bound dc + dp_w + dp_o.
        for view in (view_a, view_b):
            for key in ('common_wsi', 'common_omic', 'private_wsi', 'private_omic'):
                _floating_matrix(key, view[key])
            batch, dc = view['common_wsi'].shape
            dp_w, dp_o = view['private_wsi'].shape[1], view['private_omic'].shape[1]
            if dc + dp_w + dp_o > batch - 1:
                raise ValueError('joint identity/zero targets are rank-infeasible: require dc+dp_w+dp_o <= B-1')

    def evaluate(label: str, x: Tensor, y: Tensor, target: float) -> Tensor:
        c, diagnostic = _correlation(x, y, identity_target=(target == 1.0),
                                     policy=policy, patient_keys=patient_keys)
        diagnostics[label] = diagnostic
        return _correlation_penalty(c, target, policy)

    for name, weight in vars(weights).items():
        if weight == 0:
            skipped.append(name)
            continue
        if name.startswith('intra_'):
            modality = name.removeprefix('intra_')
            pairs = []
            for view in (view_a, view_b):
                common, private = view[f'common_{modality}'], view[f'private_{modality}']
                _floating_matrix('common', common)
                _floating_matrix('private', private)
                if common.shape[0] != private.shape[0] or common.device != private.device or common.dtype != private.dtype:
                    raise ValueError('within-modality projections must share batch, device and dtype')
                pairs.append(torch.cat((common, private), dim=1))
            loss = evaluate(name, pairs[0], pairs[1], 1.0)
        else:
            role = name.removeprefix('cross_')
            target = 1.0 if role == 'common' else 0.0
            loss = sum(evaluate(f'{name}_view_{i}', view[f'{role}_wsi'], view[f'{role}_omic'], target)
                       for i, view in enumerate((view_a, view_b))) / 2
        terms[name] = loss
        total = weight * loss if total is None else total + weight * loss
    return LossBundle(terms, total, diagnostics, tuple(skipped))


def variance_floor(
    features: Tensor, *, patient_keys: Sequence[Hashable],
    target_std: float, eps: float = 1e-4, min_patients: int = 2,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Optional across-patient variance hinge; not within-patient slot diversity.

    Separate from DeCUR weights and off by default in the planned objective.
    At exact constant features its derivative is zero despite a positive loss;
    it cannot guarantee escape from collapse or task/biological relevance.
    Do not substitute K slots for B patients. Measure on declared coordinates;
    arbitrary latent rescaling can satisfy this hinge without useful content.
    """
    _floating_matrix('features', features)
    _patient_keys(patient_keys, features.shape[0])
    if isinstance(min_patients, bool) or not isinstance(min_patients, int) or min_patients < 2 or features.shape[0] < min_patients:
        raise ValueError('invalid/insufficient minimum patient count')
    _positive_finite('target_std', target_std)
    _positive_finite('eps', eps)
    if target_std <= math.sqrt(eps):
        raise ValueError('target_std must exceed sqrt(eps)')
    variance = (features - features.mean(0)).square().mean(0)
    if not torch.isfinite(variance).all():
        raise FloatingPointError('variance overflow')
    std = torch.sqrt(variance + eps)
    return F.relu(target_std - std).mean(), {'variance': variance.detach(), 'std': std.detach()}
