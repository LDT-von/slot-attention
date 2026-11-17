"""Train-fitted adapter for the scaled 16-slot architecture.

What changes relative to the archived `ContentAdapter`
------------------------------------------------------
The archived adapter produced the omics ENCODER INPUT as a padded
gene-position vector plus a pathway one-hot, and reduced the omics TARGET by a
train-fold SVD.  The scaled architecture replaces the omics ENCODER INPUT with
329 per-pathway MLP tokens, so the adapter now has two distinct jobs:

  * TARGET construction (unchanged): the frozen train-fold SVD reduction that
    the measured probes used.  The role reference and the role supervision are
    built on this target, so it must not drift.
  * ENCODER INPUT construction (changed): per-pathway standardization fitted
    on TRAIN rows only, applied to the raw pathway expression before it enters
    the 329 independent MLPs.

Why per-pathway standardization
-------------------------------
The raw cached pathway expression is already a normalized expression matrix,
but the 329 pathways do not share a scale, and the raw range observed in the
prepared BLCA cache is wide (about -12 to +4).  An unscaled input to an
ELU + AlphaDropout pathway MLP is not a neutral choice, it changes what the
network sees.  Standardizing EACH pathway by its own train-fold moments makes
the 329 networks comparable while keeping the pathway identity, and it is
fitted on train rows only.

The moments are FROZEN after fitting and are stored as buffers, so they cannot
drift during role training.

The p-side encoder input normalization is unchanged from the archived arms:
the train-fold moments of the per-patient MEAN matrix.  The p-side ROLE
TARGET may still be the bag MEAN or the bag SPREAD, exactly as in the
archived arms.
"""
from __future__ import annotations

import torch

from ..patient_pilot_engine import ContentAdapter
from ..components.spread_target import bag_spread


class ScaledContentAdapter(ContentAdapter):
    """Train-fitted target reduction plus per-pathway input standardization."""

    SCHEMA = 'scaled_content_adapter_v1'

    def __init__(self, target_matrix, g, sizes, target_dim, *, input_mean, input_sd,
                 pathway_mean, pathway_sd, p_target_mode='spread'):
        super().__init__(target_matrix, g, sizes, target_dim)
        if p_target_mode not in ('mean', 'spread'):
            raise ValueError('p_target_mode must be mean or spread')
        if len(input_mean) != len(self.stats['p']['mean']) or len(input_sd) != len(input_mean):
            raise ValueError('input moment dimension mismatch')
        if not torch.isfinite(input_mean).all() or not torch.isfinite(input_sd).all():
            raise ValueError('nonfinite input moments')
        if float(input_sd.min()) <= 0:
            raise ValueError('positive input scale required')
        self.p_target_mode = p_target_mode
        self.stats['p']['input_mean'] = input_mean.detach().double().clone()
        self.stats['p']['input_sd'] = input_sd.detach().double().clone()
        # Per-pathway encoder input moments, one vector per pathway.
        if len(pathway_mean) != len(self.sizes) or len(pathway_sd) != len(self.sizes):
            raise ValueError('one input moment vector per pathway is required')
        for index, (mu, sd) in enumerate(zip(pathway_mean, pathway_sd)):
            if len(mu) != self.sizes[index] or len(sd) != self.sizes[index]:
                raise ValueError('pathway %d input moments do not match its gene count' % index)
            if not torch.isfinite(mu).all() or not torch.isfinite(sd).all() or float(sd.min()) <= 0:
                raise ValueError('finite pathway input moments with positive scale are required')
        self.pathway_mean = [mu.detach().double().clone() for mu in pathway_mean]
        self.pathway_sd = [sd.detach().double().clone() for sd in pathway_sd]

    # ------------------------------------------------------------- targets
    def p_target(self, patches):
        """Per-patient role target from the patient's REAL patches.

        `mean`  the archived per-patient mean over real patches.
        `spread` the per-feature standard deviation of the real patches.
        """
        if self.p_target_mode == 'mean':
            return self.target(patches.double().mean(0), 'p')
        return self.target(bag_spread(patches), 'p')

    # -------------------------------------------------------- encoder input
    def patches_input(self, patches):
        """Standardized patch matrix for the p encoder input."""
        return (patches - self.stats['p']['input_mean']) / self.stats['p']['input_sd']

    def omics_input(self, omics):
        """Per-pathway standardized gene vectors, one [genes_i] tensor each."""
        if [len(x) for x in omics] != self.sizes:
            raise ValueError('pathway/gene order schema changed')
        result = []
        for index, values in enumerate(omics):
            x = values.double()
            if len(x) != self.sizes[index]:
                raise ValueError('pathway %d gene count changed' % index)
            result.append((x - self.pathway_mean[index]) / self.pathway_sd[index])
        return result

    def state(self):
        return {'sizes': self.sizes, 'stats': self.stats,
                'p_target_mode': self.p_target_mode,
                'pathway_mean': self.pathway_mean, 'pathway_sd': self.pathway_sd,
                'schema': self.SCHEMA}


def fit_pathway_moments(store, ids, sizes):
    """Fit the per-pathway encoder input moments on TRAIN rows only.

    The moments are accumulated in float64 over the train patients and then
    frozen.  A pathway with a single gene has zero variance; its scale is
    clamped to 1 so that the standardization is defined and the raw value
    survives unchanged apart from the centering.
    """
    if not ids:
        raise ValueError('training patients are required')
    if len(sizes) != len(sizes):
        raise ValueError('invalid size list')
    sums = [torch.zeros(n, dtype=torch.float64) for n in sizes]
    squares = [torch.zeros(n, dtype=torch.float64) for n in sizes]
    for pid in ids:
        _, omics = store.load(pid)
        if len(omics) != len(sizes):
            raise ValueError('pathway count changed during the fit')
        for index, values in enumerate(omics):
            x = values.detach().to(dtype=torch.float64)
            if len(x) != sizes[index]:
                raise ValueError('pathway %d gene count changed during the fit' % index)
            sums[index] += x
            squares[index] += x * x
    means, scales = [], []
    for index, n in enumerate(sizes):
        mean = sums[index] / len(ids)
        variance = (squares[index] / len(ids) - mean * mean).clamp_min(0.0)
        sd = variance.sqrt()
        sd = torch.where(sd > 1e-6, sd, torch.ones_like(sd))
        means.append(mean)
        scales.append(sd)
    return means, scales
