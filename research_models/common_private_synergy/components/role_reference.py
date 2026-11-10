"""Frozen, train-fitted linear role references; no loader or implicit fit.

Reference roles are relative to a declared target/function family, not certified
biological factors or PID atoms. Fit/selection/calibration are separate actions.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, asdict

import torch
from torch import nn


@dataclass(frozen=True)
class ReferenceProvenance:
    fit_split: str
    target_definition: str
    source: str

    def __post_init__(self):
        if self.fit_split != 'train':
            raise ValueError('reference fitting is restricted to declared train data')
        if not self.target_definition.strip() or not self.source.strip():
            raise ValueError('explicit target definition and provenance required')


def _matrix(x):
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or min(x.shape) < 1:
        raise ValueError('nonempty [samples, fixed_target_dimensions] required')
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('finite floating targets required')
    return x.detach().to(dtype=torch.float64, device='cpu').clone()


def _whiten(x, tolerance):
    mean = x.mean(0)
    centered = x - mean
    covariance = centered.T @ centered / len(x)
    values, vectors = torch.linalg.eigh(covariance)
    keep = values > tolerance * max(float(values.max()), 1e-12)
    if not keep.any():
        raise ValueError('no varying target coordinates')
    # Discard null directions instead of presenting regularized whitening as exact.
    transform = vectors[:, keep] / values[keep].sqrt()
    return mean, transform, centered @ transform


class FrozenRoleReference(nn.Module):
    """Only buffers; no fit method and no optimizer-visible parameters.

    `targets` detaches labels. `project` keeps decoder gradients. The reference
    is fitted once, and must be independently audited before semantic claims.
    """
    def __init__(self, means, transforms, projectors, *, provenance, mode, singular_values=()):
        super().__init__()
        self.metadata = {**asdict(provenance), 'mode': mode, 'schema': 'frozen_role_reference_v1',
                         'semantic_certification': 'not_established_by_fitting'}
        for modality in ('p', 'g'):
            mu, w, projection = (collection[modality].detach().double().clone()
                                  for collection in (means, transforms, projectors))
            if mu.ndim != 1 or w.ndim != 2 or w.shape[0] != len(mu):
                raise ValueError('mean/target transform shape mismatch')
            if projection.shape != (w.shape[1], w.shape[1]):
                raise ValueError('projector shape mismatch')
            if not all(torch.isfinite(t).all() for t in (mu, w, projection)):
                raise ValueError('nonfinite reference')
            if not torch.allclose(projection, projection.T, atol=1e-9, rtol=0) or not torch.allclose(projection @ projection, projection, atol=1e-9, rtol=0):
                raise ValueError('fixed orthogonal projector required')
            self.register_buffer(modality + '_mean', mu)
            self.register_buffer(modality + '_transform', w)
            self.register_buffer(modality + '_projector', projection)
        self.register_buffer('singular_values', torch.as_tensor(singular_values, dtype=torch.float64).clone())

    @classmethod
    def estimated(cls, p, g, *, common_rank, provenance, coordinate_system, tolerance=1e-10):
        if not isinstance(provenance, ReferenceProvenance):
            raise ValueError('validated train provenance required')
        if coordinate_system not in ('whitened', 'target'):
            raise ValueError('explicit whitened or target role coordinate system required')
        p, g = _matrix(p), _matrix(g)
        if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('positive finite covariance tolerance required')
        if len(p) != len(g) or len(p) < 3:
            raise ValueError('paired train rows required')
        pm, pw, pz = _whiten(p, tolerance)
        gm, gw, gz = _whiten(g, tolerance)
        if type(common_rank) is not int or not 0 <= common_rank <= min(pz.shape[1], gz.shape[1]):
            raise ValueError('explicit feasible common rank required; no selected default')
        u, s, vh = torch.linalg.svd(pz.T @ gz / len(p), full_matrices=True)
        transforms = {'p': pw, 'g': gw}
        if coordinate_system == 'whitened':
            projections = {'p': u[:, :common_rank] @ u[:, :common_rank].T,
                           'g': vh[:common_rank].T @ vh[:common_rank]}
        else:
            # Whitening is used for relation discovery only. Pull canonical
            # score directions back, then define a projection in the declared
            # fixed target metric. Sample-whitened orthogonality can otherwise
            # put an empirical G regression term into the true local residual.
            qp = torch.linalg.qr(pw @ u[:, :common_rank], mode='reduced').Q
            qg = torch.linalg.qr(gw @ vh[:common_rank].T, mode='reduced').Q
            projections = {'p': qp @ qp.T, 'g': qg @ qg.T}
            transforms = {'p': torch.eye(p.shape[1], dtype=p.dtype),
                          'g': torch.eye(g.shape[1], dtype=g.dtype)}
        result = cls({'p': pm, 'g': gm}, transforms, projections,
                   provenance=provenance, mode='estimated_linear', singular_values=s)
        fingerprint = hashlib.sha256()
        for value in (p, g):
            fingerprint.update(str(tuple(value.shape)).encode())
            fingerprint.update(value.numpy().tobytes())
        result.metadata.update(fit_samples=len(p), fit_targets_sha256=fingerprint.hexdigest(),
                               common_rank=common_rank, covariance_tolerance=tolerance,
                               coordinate_system=coordinate_system)
        return result

    @classmethod
    def oracle(cls, *, p_transform, g_transform, p_projector, g_projector, provenance):
        if not isinstance(provenance, ReferenceProvenance):
            raise ValueError('validated provenance required')
        return cls({'p': p_transform.new_zeros(p_transform.shape[0]),
                    'g': g_transform.new_zeros(g_transform.shape[0])},
                   {'p': p_transform, 'g': g_transform}, {'p': p_projector, 'g': g_projector},
                   provenance=provenance, mode='oracle_known_generator')

    def get_extra_state(self):
        return dict(self.metadata)

    def set_extra_state(self, state):
        if state.get('schema') != 'frozen_role_reference_v1' or state.get('fit_split') != 'train':
            raise ValueError('invalid serialized reference provenance')
        self.metadata = dict(state)

    def dimension(self, modality):
        return getattr(self, modality + '_transform').shape[1]

    def transform(self, x, modality):
        w, mean = getattr(self, modality + '_transform'), getattr(self, modality + '_mean')
        if x.ndim != 2 or len(x) == 0 or x.shape[1] != len(mean) or x.dtype != w.dtype or x.device != w.device or not torch.isfinite(x).all():
            raise ValueError('target dimensions, finite values, device and dtype must match frozen reference')
        return (x - mean) @ w

    def project(self, x, modality, role):
        p = getattr(self, modality + '_projector')
        if role == 'residual':
            p = torch.eye(len(p), dtype=p.dtype, device=p.device) - p
        elif role != 'common':
            raise ValueError('role must be common or residual')
        return x @ p

    @torch.no_grad()
    def targets(self, p, g):
        if len(p) != len(g):
            raise ValueError('paired target row counts required')
        result = {}
        for modality, x in [('p', p), ('g', g)]:
            z = self.transform(x.detach(), modality)
            for role in ('common', 'residual'):
                result[modality + '_' + role] = self.project(z, modality, role)
        return result

    def fingerprint(self):
        h = hashlib.sha256()
        import json
        h.update(json.dumps(self.metadata, sort_keys=True).encode())
        for name, value in self.named_buffers():
            h.update(name.encode())
            h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


def role_content_losses(outputs, targets):
    """Per-coordinate MSE; target count is unrelated to slot count."""
    if outputs.keys() != targets.keys() or not outputs:
        raise ValueError('same explicit role keys required')
    result = {}
    for key in outputs:
        if outputs[key].shape != targets[key].shape or outputs[key].ndim != 2:
            raise ValueError('role output/target shape mismatch')
        result[key] = (outputs[key] - targets[key].detach()).square().mean()
    return result
