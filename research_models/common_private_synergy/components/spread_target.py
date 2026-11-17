"""Spread-aware role target for the user's common/private role architecture.

Why this module exists
----------------------
The archived role target of `common_private_synergy` is ONE statistic per
patient: the mean over the patient's patch bag, reduced by a train-fold PCA
to `target_dim` dimensions (plus a pathway-expression PCA of the same
width). Under the architecture's own reference geometry - train-only PCA of
both modalities, then the linear COMMON / RESIDUAL split, then a survival
readout - that target has a measured fold-mean validation C-index of 0.6875
over six cohort/fold cells.

A bag MEAN discards the intra-bag distribution. A bag whose patches form one
tight cluster and a bag whose patches split into three distinct tissue
populations can share the same mean. The role model is a *set* model; its
supervision should describe the set, not only its centroid.

Two distribution-aware constructions were built and measured:

    * `DistributionRoleTarget` (see `distribution_target.py`): eight
      deterministic k-means centres over the patient's own projected patch
      cloud. Measured fold-mean 0.6856 - NOT better than the bag mean
      (P(better) = 0.88, 4 of 6 cells).
    * `SpreadRoleTarget` (this module): the per-feature standard deviation of
      the patient's patch bag. Measured fold-mean 0.6993 at the archived
      widths (target_dim 8, common_rank 2) and 0.7071 at target_dim 16,
      common_rank 6, against 0.6875 for the bag mean. At the archived widths
      only the target STATISTIC changes: +0.0117, better in 5 of 6 cells,
      paired bootstrap over cells 95% CI [+0.0020, +0.0218], P = 0.993.

The SPREAD is the measured winner and is therefore the construction this
module implements. The cluster-centre construction is retained in the
repository as the measured alternative and is not deleted.

What this module adds
---------------------
`SpreadRoleTarget` holds the FROZEN TRAIN-FOLD statistics of the spread
construction:

    * the per-patient statistic itself - the per-feature standard deviation of
      the patient's REAL patch bag - which needs no fitting, because it is a
      property of the bag;
    * the train-fold moments of that per-patient spread MATRIX, which place
      different bags on one comparable scale. These moments are fitted on
      TRAIN ROWS ONLY and are never refitted.

The standardization is deliberately the standardization of the SPREAD MATRIX,
not of the raw patch features. Standardizing a spread by the raw patch
feature moments would mix two different quantities and was an earlier, WRONG
version of this module; the measured probe standardized the spread matrix and
that is what is implemented here.

The PCA reduction that follows this standardization is the ARCHIVED one: the
archived `ContentAdapter` reduces a target matrix by a train-fold SVD, and
the spread arm reuses that adapter unchanged so that the reduction, the
score scale and the encoder input normalization stay identical to the
archived arm.

Scope and honesty
-----------------
This module changes what the role slots are asked to reconstruct. It does
NOT by itself raise any survival score: the 0.6993 / 0.7071 figures are
LINEAR PROBES of the target's information ceiling under the architecture's own
geometry, measured by the read-only probe
`local_evidence/diag_20261116/spread_target_width_isolation.py`. A trained
model still has to fit the target, and a trained model can still land BELOW
the ceiling, as the archived model did: it reported 0.6151 while its own role
outputs supported 0.6599 against a 0.6803 target ceiling.

The spread is a statistic of the bag, NOT an identified tissue type or a
certified biological factor.
"""
from __future__ import annotations

import torch
from torch import nn


def bag_spread(patches):
    """The per-feature SPREAD of one patient's real patch bag.

    This is the ONE definition of the spread statistic in the repository. The
    measured probe and the trained arm both call it, so the probe and the
    arm cannot drift apart.

    Convention: the n-1 (unbiased) standard deviation, which is the
    convention the measured probe used. A single real patch has no spread;
    the statistic is then zero, which is a measured fact about that bag
    rather than an imputed value.
    """
    if not isinstance(patches, torch.Tensor) or patches.ndim != 2:
        raise ValueError('a patch bag must be a [patches, features] tensor')
    if len(patches) < 2:
        return torch.zeros(patches.shape[1], dtype=torch.float64)
    return patches.detach().to(dtype=torch.float64).std(0)


class SpreadRoleTarget(nn.Module):
    """Frozen train-fold scaling plus the per-patient bag-spread target.

    Buffers only: there is no optimizer-visible parameter and no refit
    method, so the target cannot drift during role training.

    The p side returns the per-feature standard deviation of the patient's
    patch bag, standardized by the train-fold moments of the per-patient
    spread matrix. The g side returns the train-fold-scaled pathway
    expression, matching the archived adapter's omics behaviour, because the
    measured gain comes from the WSI side and the omics side must not change
    at the same time.
    """
    SCHEMA = 'common_private_spread_role_target_v2'

    def __init__(self, spread_mean, spread_sd, g_mean, g_sd, *, provenance: dict):
        super().__init__()
        if not isinstance(provenance, dict) or provenance.get('fit_split') != 'train':
            raise ValueError('validated train provenance is required')
        for name, value in [('spread_mean', spread_mean), ('spread_sd', spread_sd),
                            ('g_mean', g_mean), ('g_sd', g_sd)]:
            if not isinstance(value, torch.Tensor) or value.ndim != 1 or len(value) == 0:
                raise ValueError('nonempty 1-D frozen statistic required: ' + name)
            if not torch.isfinite(value).all():
                raise ValueError('nonfinite frozen statistic: ' + name)
        if float(spread_sd.min()) <= 0 or float(g_sd.min()) <= 0:
            raise ValueError('positive frozen scale required')
        self.register_buffer('spread_mean', spread_mean.detach().double().clone())
        self.register_buffer('spread_sd', spread_sd.detach().double().clone())
        self.register_buffer('g_mean', g_mean.detach().double().clone())
        self.register_buffer('g_std', g_sd.detach().double().clone())
        self.metadata = {**provenance, 'schema': self.SCHEMA,
                         'target_statistic': 'per_feature_standard_deviation_of_real_patches',
                         'standardization': 'train-fold moments of the per-patient spread matrix',
                         'semantic_certification': 'not_established_by_fitting'}

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, spread_rows, omics_vectors, *, provenance: dict):
        """Fit the frozen statistics on TRAIN rows only.

        `spread_rows` is a list of per-patient spread vectors, one per
        training patient; each vector holds the per-feature standard
        deviation of that patient's own real patches. `omics_vectors` is a
        list of 1-D tensors with one entry per training patient.

        The per-patient statistic needs no fitting, because it is a property
        of the bag. Only the global scaling is fitted, and it is fitted on
        train rows.
        """
        if not spread_rows or len(spread_rows) != len(omics_vectors):
            raise ValueError('paired training spread rows and omics vectors required')
        feature_dim = spread_rows[0].shape[0]
        for row in spread_rows:
            if row.ndim != 1 or len(row) != feature_dim:
                raise ValueError('every training spread row must share one feature dimension')
        stacked = torch.stack([row.detach().double() for row in spread_rows])
        spread_mean = stacked.mean(0)
        spread_sd = stacked.std(0, unbiased=False).clamp_min(1e-4)
        del stacked

        g_values = torch.cat([x.detach().double().reshape(-1) for x in omics_vectors])
        g_mean = g_values.mean()
        g_std = g_values.std().clamp_min(1e-4)
        del g_values
        return cls(spread_mean, spread_sd, g_mean, g_std, provenance=provenance)

    # -------------------------------------------------------------- targets
    bag_spread = staticmethod(bag_spread)

    @torch.no_grad()
    def p_target(self, patches: torch.Tensor) -> torch.Tensor:
        """One patient's patch bag -> [features] standardized spread target."""
        if patches.ndim != 2 or patches.shape[1] != len(self.spread_mean):
            raise ValueError('patch bag feature dimension does not match the frozen reference')
        spread = bag_spread(patches)
        return (spread - self.spread_mean) / self.spread_sd

    @torch.no_grad()
    def g_target(self, omics: torch.Tensor) -> torch.Tensor:
        """One patient's pathway expression -> [genes] scaled omics target."""
        values = omics.detach().to(dtype=torch.float64).reshape(-1)
        if len(values) != len(self.g_mean):
            raise ValueError('omics dimension does not match the frozen reference')
        return (values - self.g_mean) / self.g_std

    # ------------------------------------------------------------- contract
    def target_dim_per_patient(self) -> int:
        return int(len(self.spread_mean))

    def state(self):
        return {'spread_mean': self.spread_mean, 'spread_sd': self.spread_sd,
                'g_mean': self.g_mean, 'g_std': self.g_std}

    def get_extra_state(self):
        return dict(self.metadata)

    def set_extra_state(self, state):
        if state.get('schema') != self.SCHEMA or state.get('fit_split') != 'train':
            raise ValueError('invalid serialized spread target provenance')
        self.metadata = dict(state)
