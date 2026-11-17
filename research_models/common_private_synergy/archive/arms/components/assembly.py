"""Assembled common/private model container.

This module wires the reviewed parts together without adding a data loader,
training loop, feature adapter, or patient-data access.  A single encoder is
created per modality and reused for both views.  The low-dimensional
projectors are used only by the optional correlation objectives; prediction
always consumes the unprojected summaries.

The class is an implementation surface for later authorized tensor checks.
Importing this module does not construct a model or execute a forward pass.
The role names remain architectural roles, not identified biological factors
or a PID estimator.
"""
from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .contracts import ArchitectureConfig
from .losses import (
    CorrelationPolicy,
    DeCURWeights,
    LossBundle,
    decur_losses,
    discrete_hazard_nll,
    variance_floor,
)
from .readout import AdditiveInteractionReadout
from .slots import AllSlotPool, RoleSlotEncoder, SummaryProjector


def _nonnegative_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite nonnegative number')
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative number')


@dataclass(frozen=True)
class ObjectiveWeights:
    """Independent switches for the assembled objective.

    ``task`` is the main hazard NLL, ``common_task`` is the optional mean of
    the two modality-local common heads, and ``decur`` contains the four
    separately switchable representation terms.  All auxiliary terms default
    to zero so the first assembled baseline is a strict task-only model.
    """

    task: float = 1.0
    common_task: float = 0.0
    variance: float = 0.0
    variance_target_std: float = 1.0
    variance_eps: float = 1e-4
    decur: DeCURWeights = field(default_factory=DeCURWeights)

    def __post_init__(self) -> None:
        _nonnegative_finite('task', self.task)
        _nonnegative_finite('common_task', self.common_task)
        _nonnegative_finite('variance', self.variance)
        if isinstance(self.variance_target_std, bool) or not isinstance(self.variance_target_std, (int, float)):
            raise ValueError('variance_target_std must be finite and positive')
        if not math.isfinite(self.variance_target_std) or self.variance_target_std <= 0:
            raise ValueError('variance_target_std must be finite and positive')
        if isinstance(self.variance_eps, bool) or not isinstance(self.variance_eps, (int, float)):
            raise ValueError('variance_eps must be finite and positive')
        if not math.isfinite(self.variance_eps) or self.variance_eps <= 0:
            raise ValueError('variance_eps must be finite and positive')
        if not isinstance(self.decur, DeCURWeights):
            raise ValueError('decur must be DeCURWeights')


class CommonPrivateSynergyModel(nn.Module):
    """Container for the reviewed two-modality role architecture.

    Inputs to :meth:`forward` are already tokenized feature tensors:
    ``wsi=[B,Nw,Dw]`` and ``omic=[B,No,Do]``.  Explicit boolean masks are
    required for both modalities; padding is never silently treated as a real
    token.  A second view is optional for the task-only baseline and, when
    supplied, uses the exact same modality encoders, pooling modules and
    projectors as the first view.
    """

    _ROLE_NAMES = ('common_wsi', 'private_wsi', 'common_omic', 'private_omic')

    def __init__(self, config: ArchitectureConfig, *, ssl_projection_dim: int = 2):
        super().__init__()
        if type(ssl_projection_dim) is not int or ssl_projection_dim < 1:
            raise ValueError('ssl_projection_dim must be a positive integer')
        self.config = config
        self.ssl_projection_dim = ssl_projection_dim

        # One encoder per modality is deliberately shared by both views.
        self.wsi_encoder = RoleSlotEncoder(config.wsi_dim, config)
        self.omic_encoder = RoleSlotEncoder(config.omic_dim, config)

        self.pools = nn.ModuleDict({
            role: AllSlotPool(config.width) for role in self._ROLE_NAMES
        })
        self.projectors = nn.ModuleDict({
            role: SummaryProjector(config.width, ssl_projection_dim)
            for role in self._ROLE_NAMES
        })
        self.readout = AdditiveInteractionReadout(config)

    def _check_pair(self, wsi: Tensor, omic: Tensor, wsi_mask: Tensor, omic_mask: Tensor) -> None:
        if not isinstance(wsi, Tensor) or not isinstance(omic, Tensor):
            raise ValueError('wsi and omic must be tensors')
        if wsi.ndim != 3 or omic.ndim != 3 or wsi.shape[0] != omic.shape[0]:
            raise ValueError('paired inputs must be [patients,tokens,features] tensors')
        if wsi_mask.dtype != torch.bool or omic_mask.dtype != torch.bool:
            raise ValueError('explicit boolean masks are required')
        if wsi_mask.shape != wsi.shape[:2] or omic_mask.shape != omic.shape[:2]:
            raise ValueError('masks must match the first two input dimensions')
        if wsi_mask.device != wsi.device or omic_mask.device != omic.device:
            raise ValueError('masks must share the corresponding input device')

    def encode_view(
        self,
        wsi: Tensor,
        omic: Tensor,
        *,
        wsi_mask: Tensor,
        omic_mask: Tensor,
    ) -> dict[str, Any]:
        """Encode one paired view and return summaries, projections and logits."""
        self._check_pair(wsi, omic, wsi_mask, omic_mask)
        wsi_parts = self.wsi_encoder(wsi, wsi_mask)
        omic_parts = self.omic_encoder(omic, omic_mask)

        summaries = {
            'common_wsi': self.pools['common_wsi'](wsi_parts['common_slots']),
            'private_wsi': self.pools['private_wsi'](wsi_parts['private_slots']),
            'common_omic': self.pools['common_omic'](omic_parts['common_slots']),
            'private_omic': self.pools['private_omic'](omic_parts['private_slots']),
        }
        # Keep projections separate from prediction features.  Empty roles are
        # omitted so an 8:0/0:8 structural control cannot masquerade as a
        # valid SSL term through a projector bias.
        ssl = {
            role: self.projectors[role](summary)
            for role, summary in summaries.items()
            if (role.startswith('common_') and self.config.common_slots > 0)
            or (role.startswith('private_') and self.config.private_slots > 0)
        }
        readout = self.readout(
            summaries['common_wsi'], summaries['private_wsi'],
            summaries['common_omic'], summaries['private_omic'],
        )
        return {
            'wsi_parts': wsi_parts,
            'omic_parts': omic_parts,
            'summaries': summaries,
            'ssl': ssl,
            'readout': readout,
        }

    def forward(
        self,
        wsi: Tensor,
        omic: Tensor,
        wsi_mask: Tensor,
        omic_mask: Tensor,
        *,
        wsi_view_b: Tensor | None = None,
        omic_view_b: Tensor | None = None,
        wsi_mask_b: Tensor | None = None,
        omic_mask_b: Tensor | None = None,
    ) -> dict[str, Any]:
        """Run one view and optionally a second, parameter-shared view.

        The method has no optimizer, data access, split logic or patient-ID
        handling.  Passing any part of view B requires all four B tensors so
        that a caller cannot accidentally compare unpaired views.
        """
        first = self.encode_view(wsi, omic, wsi_mask=wsi_mask, omic_mask=omic_mask)
        second_args = (wsi_view_b, omic_view_b, wsi_mask_b, omic_mask_b)
        if any(value is not None for value in second_args):
            if any(value is None for value in second_args):
                raise ValueError('second view requires wsi, omic and both explicit masks')
            second = self.encode_view(
                wsi_view_b, omic_view_b,
                wsi_mask=wsi_mask_b, omic_mask=omic_mask_b,
            )
        else:
            second = None
        return {
            'view_a': first,
            'view_b': second,
            # Convenience aliases for a task-only caller.  The source of these
            # logits remains view A's strict readout.
            'logits': first['readout']['logits'],
            'additive_logits': first['readout']['additive_logits'],
            'interaction_logits': first['readout']['interaction_logits'],
        }

    def _check_objective_roles(self, weights: ObjectiveWeights) -> None:
        decur = weights.decur
        if decur.cross_common > 0 and self.config.common_slots == 0:
            raise ValueError('cross_common requires a nonempty common role')
        if decur.cross_private > 0 and self.config.private_slots == 0:
            raise ValueError('cross_private requires a nonempty private role')
        if (decur.intra_wsi > 0 or decur.intra_omic > 0) and (
            self.config.common_slots == 0 or self.config.private_slots == 0
        ):
            raise ValueError('intra-modality DeCUR terms require both role groups')

    def objective(
        self,
        output: Mapping[str, Any],
        completed_intervals: Tensor,
        event: Tensor,
        *,
        weights: ObjectiveWeights = ObjectiveWeights(),
        patient_keys: Sequence[Any] | None = None,
        correlation_policy: CorrelationPolicy | None = None,
    ) -> dict[str, Any]:
        """Compute enabled objective terms without selecting checkpoints.

        The caller supplies already mapped discrete survival labels and, when
        representation terms are enabled, one distinct key per independent
        patient row.  No term is calculated when its weight is zero.  This
        method is intentionally separate from ``forward`` so a future adapter
        can audit split-specific binning and view provenance before invoking it.
        """
        if not isinstance(output, Mapping) or 'view_a' not in output:
            raise ValueError('output must be the result of forward')
        if not isinstance(weights, ObjectiveWeights):
            raise ValueError('weights must be ObjectiveWeights')
        self._check_objective_roles(weights)
        first = output['view_a']
        readout = first['readout']
        terms: dict[str, Tensor] = {}
        diagnostics: dict[str, Any] = {}
        skipped: list[str] = []
        total: Tensor | None = None

        def add(name: str, value: Tensor, weight: float) -> None:
            nonlocal total
            terms[name] = value
            total = weight * value if total is None else total + weight * value

        if weights.task > 0:
            add('task', discrete_hazard_nll(readout['logits'], completed_intervals, event), weights.task)
        else:
            skipped.append('task')

        if weights.common_task > 0:
            if not readout['common_active']:
                raise ValueError('common_task requires a nonempty common role')
            common_losses = [
                discrete_hazard_nll(readout[key], completed_intervals, event)
                for key in ('common_wsi_logits', 'common_omic_logits')
            ]
            add('common_task', torch.stack(common_losses).mean(), weights.common_task)
        else:
            skipped.append('common_task')

        decur = weights.decur
        if any(value > 0 for value in vars(decur).values()):
            second = output.get('view_b')
            if second is None:
                raise ValueError('enabled DeCUR terms require a second paired view')
            if patient_keys is None:
                raise ValueError('enabled DeCUR terms require distinct patient_keys')
            policy = correlation_policy or CorrelationPolicy(
                min_patients=8, max_projection_dim=self.ssl_projection_dim * 2,
            )
            ssl_a, ssl_b = first['ssl'], second['ssl']
            decur_bundle: LossBundle = decur_losses(
                ssl_a, ssl_b, patient_keys=patient_keys, weights=decur, policy=policy,
            )
            terms.update({f'decur_{name}': value for name, value in decur_bundle.terms.items()})
            diagnostics['decur'] = decur_bundle.diagnostics
            for name, value in vars(decur).items():
                if value == 0:
                    skipped.append(f'decur_{name}')
                else:
                    total = value * decur_bundle.terms[name] if total is None else total + value * decur_bundle.terms[name]
        else:
            skipped.extend(f'decur_{name}' for name in vars(decur))

        if weights.variance > 0:
            if patient_keys is None:
                raise ValueError('variance term requires distinct patient_keys')
            features = torch.cat(tuple(first['summaries'][name] for name in self._ROLE_NAMES), dim=1)
            value, variance_diagnostics = variance_floor(
                features, patient_keys=patient_keys,
                target_std=weights.variance_target_std, eps=weights.variance_eps,
            )
            add('variance', value, weights.variance)
            diagnostics['variance'] = variance_diagnostics
        else:
            skipped.append('variance')

        return {
            'terms': terms,
            'weighted_total': total,
            'diagnostics': diagnostics,
            'skipped': tuple(skipped),
        }


def build_model(config: ArchitectureConfig, *, ssl_projection_dim: int = 2) -> CommonPrivateSynergyModel:
    """Explicit construction helper; no implicit defaults from data or files."""
    return CommonPrivateSynergyModel(config, ssl_projection_dim=ssl_projection_dim)
