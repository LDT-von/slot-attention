"""Distribution-aware role target for the user's common/private role architecture.

Why this module exists
----------------------
The archived role target of `common_private_synergy` is ONE statistic per
patient: the mean over the patient's patch bag, reduced by a train-fold PCA to
`target_dim` dimensions (plus a pathway-expression PCA of the same width). The
measured consequence, on the user's own cohort and under the user's own split,
is a survival-information ceiling of 0.6803 for ANY readout on those features
(ridge Cox 0.6803; Random Survival Forest 0.656; Extra Survival Trees 0.663;
Coxnet 0.676). The encoder is therefore asked to reconstruct a target that
provably cannot carry 0.70.

A bag mean also discards the intra-bag distribution: a bag whose patches form
one tight cluster and a bag whose patches split into three distinct tissue
populations can share the same mean. The role model is a *set* model; its
supervision should describe the set, not its centroid.

What this module adds
---------------------
`DistributionRoleTarget` computes a per-patient role target from the shape of
the patient's patch bag rather than from its centroid:

  * every patch is standardized with TRAIN-FOLD global moments and projected
    into a fixed `target_dim`-dimensional random space;
  * a deterministic k-means over the patient's own patch cloud yields
    `clusters` centres;
  * the centres are flattened into one target vector, so one patient carries
    `clusters * target_dim` numbers instead of `target_dim`.

This is the same KIND of construction the third-party original uses for its
content reference (slotspe_proof/redundancy_content.py). It is implemented
here as a new component of the USER's architecture; the original source is not
modified, imported or reused.

Honest scope
------------
This module changes what the role slots are asked to reconstruct. It does NOT
by itself raise any survival score. Whether a richer target raises the
information ceiling is measured separately, by the read-only probe
`local_evidence/diag_20261116/cluster_target_ceiling.py`. A richer target can
also make the role-fitting problem harder: with `clusters * target_dim`
coordinates to reconstruct, the same decoder width may no longer be enough.

The centres are a deterministic dictionary over the patient's own patch cloud.
They are NOT identified tissue types, and the cluster ORDER carries no
semantics: it is fixed by the deterministic farthest-point initialisation and
Lloyd steps, exactly as in the construction this module mirrors.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def deterministic_centres(points: torch.Tensor, clusters: int, steps: int = 5) -> torch.Tensor:
    """Deterministic farthest-point initialisation plus fixed-count Lloyd steps.

    `points` is [tokens, features]. The procedure is deterministic and does not
    use any random state: the first centre is the point furthest from the mean,
    each later centre is the point furthest from the already chosen centres,
    and the Lloyd refinement runs a fixed number of steps. An empty cluster
    keeps its previous centre instead of being dropped, so the output always
    has exactly `min(clusters, tokens)` rows.

    The row ORDER is a property of the dictionary, not a semantic order.
    """
    if points.ndim != 2 or min(points.shape) < 1:
        raise ValueError('nonempty [tokens, features] input required')
    if not points.is_floating_point() or not torch.isfinite(points).all():
        raise ValueError('finite floating points required')
    if clusters < 1 or steps < 0:
        raise ValueError('clusters must be >= 1 and steps >= 0')
    work = points.detach().to(dtype=torch.float64)
    count = min(clusters, work.shape[0])
    first = int((work - work.mean(0)).square().sum(-1).argmax())
    selected = [first]
    distance = (work - work[first]).square().sum(-1)
    for _ in range(1, count):
        candidate = distance.clone()
        candidate[selected] = -1
        index = int(candidate.argmax())
        selected.append(index)
        distance = torch.minimum(distance, (work - work[index]).square().sum(-1))
    centres = work[selected].clone()
    for _ in range(steps):
        labels = (work[:, None] - centres[None]).square().sum(-1).argmin(-1)
        updated = []
        for k in range(count):
            member = labels == k
            updated.append(work[member].mean(0) if bool(member.any()) else centres[k])
        centres = torch.stack(updated)
    return centres.to(dtype=points.dtype)


class DistributionRoleTarget(nn.Module):
    """Frozen train-fold statistics plus a per-patient distribution target.

    Buffers only: there is no fit method and no optimizer-visible parameter, so
    the target cannot drift during role training. `fit` is called explicitly on
    TRAIN rows; `target` never changes fitted state.

    The `p` side uses the patient's own patch cloud. The `g` side uses the
    patient's pathway blocks: every pathway is projected with its own random
    matrix and the projected blocks are concatenated, so the omics target
    describes the DISTRIBUTION over pathways rather than one concatenated
    vector's centroid.
    """

    SCHEMA = 'common_private_distribution_role_target_v1'

    def __init__(self, p_mean, p_std, p_projection, g_mean, g_std, g_projections,
                 *, clusters: int, km_steps: int, target_dim: int, provenance: dict):
        super().__init__()
        if clusters < 1 or km_steps < 0 or target_dim < 1:
            raise ValueError('explicit positive clusters/target_dim and nonnegative km_steps required')
        if not isinstance(provenance, dict) or not provenance.get('fit_split') == 'train':
            raise ValueError('validated train provenance is required')
        self.register_buffer('p_mean', p_mean.detach().double().clone())
        self.register_buffer('p_std', p_std.detach().double().clone())
        self.register_buffer('p_projection', p_projection.detach().double().clone())
        self.register_buffer('g_mean', g_mean.detach().double().clone())
        self.register_buffer('g_std', g_std.detach().double().clone())
        for index, matrix in enumerate(g_projections):
            self.register_buffer(f'g_projection_{index}', matrix.detach().double().clone())
        self.g_projection_count = len(g_projections)
        self.clusters = int(clusters)
        self.km_steps = int(km_steps)
        self.target_dim = int(target_dim)
        self.metadata = {**provenance, 'schema': self.SCHEMA,
                         'clusters': self.clusters, 'km_steps': self.km_steps,
                         'target_dim': self.target_dim,
                         'semantic_certification': 'not_established_by_fitting'}

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, patch_bags, omics_blocks, *, clusters: int, km_steps: int,
            target_dim: int, projection_seed: int, provenance: dict):
        """Fit the frozen statistics on TRAIN rows only.

        `patch_bags` is a list of [tokens, features] tensors, one per training
        patient. `omics_blocks` is a list of lists of 1-D tensors, one list of
        pathway blocks per training patient.
        """
        if not patch_bags or len(patch_bags) != len(omics_blocks):
            raise ValueError('paired training bags and pathway blocks required')
        feature_dim = patch_bags[0].shape[1]
        for bag in patch_bags:
            if bag.ndim != 2 or bag.shape[1] != feature_dim:
                raise ValueError('every training bag must share one feature dimension')
        # Global patch moments over the training bags.
        stacked = torch.cat([bag.detach().double() for bag in patch_bags], 0)
        p_mean = stacked.mean(0)
        p_std = stacked.std(0).clamp_min(1e-4)
        del stacked

        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(projection_seed))
        p_projection = (torch.randn(feature_dim, target_dim, generator=generator,
                                    dtype=torch.float64) / math.sqrt(feature_dim))

        block_sizes = [len(block) for block in omics_blocks[0]]
        for block in omics_blocks:
            if [len(x) for x in block] != block_sizes:
                raise ValueError('every training patient must expose the same pathway blocks')
        g_values = torch.cat([x.detach().double().reshape(-1)
                              for block in omics_blocks for x in block])
        g_mean = g_values.mean()
        g_std = g_values.std().clamp_min(1e-4)
        del g_values

        g_projections = []
        for index, size in enumerate(block_sizes):
            block_generator = torch.Generator(device='cpu')
            block_generator.manual_seed(int(projection_seed) + 104729 * (index + 1))
            g_projections.append(torch.randn(size, target_dim, generator=block_generator,
                                             dtype=torch.float64) / math.sqrt(max(size, 1)))
        return cls(p_mean, p_std, p_projection, g_mean, g_std, g_projections,
                   clusters=clusters, km_steps=km_steps, target_dim=target_dim,
                   provenance=provenance)

    # -------------------------------------------------------------- targets
    @torch.no_grad()
    def p_target(self, patches: torch.Tensor) -> torch.Tensor:
        """One patient's patch bag -> [clusters * target_dim] target vector."""
        if patches.ndim != 2 or patches.shape[1] != self.p_mean.shape[0]:
            raise ValueError('patch bag feature dimension does not match the frozen reference')
        projected = ((patches.detach().to(dtype=torch.float64) - self.p_mean)
                     / self.p_std) @ self.p_projection
        centres = deterministic_centres(projected, self.clusters, self.km_steps)
        return centres.reshape(-1).to(dtype=torch.float64)

    @torch.no_grad()
    def g_target(self, blocks) -> torch.Tensor:
        """One patient's pathway blocks -> [clusters * target_dim] target vector.

        Each pathway is projected with its own frozen matrix, and the
        resulting PROJECTED PATHWAY VECTORS form the point cloud that is
        clustered. The cloud is therefore a matrix with one row per pathway,
        which is the shape the original's content reference clusters; it is
        not one concatenated vector.
        """
        if len(blocks) != self.g_projection_count:
            raise ValueError('pathway block count does not match the frozen reference')
        rows = []
        for index, block in enumerate(blocks):
            values = block.detach().to(dtype=torch.float64).reshape(-1)
            matrix = getattr(self, f'g_projection_{index}')
            if len(values) != matrix.shape[0]:
                raise ValueError('pathway block size does not match the frozen projection')
            rows.append(((values - self.g_mean) / self.g_std) @ matrix)
        cloud = torch.stack(rows)
        centres = deterministic_centres(cloud, self.clusters, self.km_steps)
        return centres.reshape(-1).to(dtype=torch.float64)

    # ------------------------------------------------------------- contract
    def target_dim_per_patient(self) -> int:
        return self.clusters * self.target_dim

    def state(self):
        return {'metadata': dict(self.metadata), 'clusters': self.clusters,
                'km_steps': self.km_steps, 'target_dim': self.target_dim,
                'g_projection_count': self.g_projection_count}

    def get_extra_state(self):
        return dict(self.metadata)

    def set_extra_state(self, state):
        if state.get('schema') != self.SCHEMA or state.get('fit_split') != 'train':
            raise ValueError('invalid serialized distribution target provenance')
        self.metadata = dict(state)
