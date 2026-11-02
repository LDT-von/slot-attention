"""Train-only observed prototype/PCA targets. No outcome or patient adapter.

Clustering is a frozen auxiliary teacher, not prediction-time slot selection.
Prespecified descriptor/content feature subspaces are an explicit assumption.
An unsupervised high-variance target need not be useful for survival.
"""
import numpy as np
import torch
from torch import nn


def _validate(x, valid, descriptor_columns, content_columns):
    if x.ndim != 3 or not x.is_floating_point():
        raise ValueError('floating [example, token, feature] input required')
    if valid.shape != x.shape[:2] or valid.dtype != torch.bool:
        raise ValueError('boolean token mask required')
    if not bool(valid.any(1).all()):
        raise ValueError('empty modality unsupported')
    columns = tuple(descriptor_columns) + tuple(content_columns)
    if not descriptor_columns or not content_columns or len(set(columns)) != len(columns):
        raise ValueError('nonempty disjoint feature subspaces required')
    if min(columns) < 0 or max(columns) >= x.shape[-1]:
        raise ValueError('feature column out of range')
    if not bool(torch.isfinite(x[valid]).all()):
        raise ValueError('nonfinite observed token')


def _kmeans(z, weights, clusters, seed, restarts=3, steps=50):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        centers = [z[rng.choice(len(z), p=weights/weights.sum())]]
        for _ in range(1, clusters):
            distance = ((z[:, None]-np.asarray(centers)[None])**2).sum(-1).min(1)
            mass = weights*distance
            if mass.sum() < 1e-12:
                raise ValueError('insufficient distinct observed descriptors')
            centers.append(z[rng.choice(len(z), p=mass/mass.sum())])
        centers = np.stack(centers)
        for _ in range(steps):
            distances = ((z[:, None]-centers[None])**2).sum(-1)
            ids = distances.argmin(1)
            updated = []
            for k in range(clusters):
                mask = ids == k
                if not mask.any():
                    updated.append(z[(weights*distances.min(1)).argmax()])
                else:
                    updated.append(np.average(z[mask], axis=0, weights=weights[mask]))
            updated = np.stack(updated)
            if np.allclose(updated, centers, atol=1e-9, rtol=0):
                centers = updated
                break
            centers = updated
        score = (((z[:, None]-centers[None])**2).sum(-1).min(1)*weights).sum()
        if best is None or score < best[0]:
            best = score, centers.copy()
    centers = best[1]
    # A deterministic dictionary order, not an identified semantic order.
    order = sorted(range(clusters), key=lambda k: tuple(centers[k]))
    return centers[order]


def _aggregate(x, valid, descriptor_columns, content_columns, mean, scale, centers):
    desc = (x[..., descriptor_columns]-mean)/scale
    assignment = ((desc[:, :, None]-centers[None, None])**2).sum(-1).argmin(-1)
    mask = (assignment[:, :, None] == np.arange(len(centers))[None, None]) & valid[..., None]
    count = mask.sum(1)
    means = np.einsum('bnk,bnc->bkc', mask, x[..., content_columns]) / np.maximum(count[..., None], 1)
    return assignment, count, means


class FrozenPrototypeTargets(nn.Module):
    """Buffers fitted from training inputs only; forward never updates them."""
    def __init__(self, descriptor_columns, content_columns, descriptor_mean, descriptor_scale,
                 centers, content_mean, axes, thresholds, fit_counts, fit_rows):
        super().__init__()
        self.descriptor_columns = tuple(descriptor_columns)
        self.content_columns = tuple(content_columns)
        self.fit_rows = int(fit_rows)
        for name, value in [('descriptor_mean', descriptor_mean), ('descriptor_scale', descriptor_scale),
                            ('centers', centers), ('content_mean', content_mean), ('axes', axes),
                            ('thresholds', thresholds), ('fit_counts', fit_counts)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64).clone())

    @classmethod
    def fit(cls, x, valid, *, clusters, descriptor_columns, content_columns, seed):
        _validate(x, valid, descriptor_columns, content_columns)
        if clusters < 1 or clusters > int(valid.sum()):
            raise ValueError('invalid prototype count')
        a = x.detach().cpu().double().numpy().copy()
        m = valid.detach().cpu().numpy()
        # Invalid/padded tokens never participate, even if they contain NaN.
        a[~m] = 0
        weights = np.broadcast_to(1/m.sum(1, keepdims=True), m.shape)[m]
        desc = a[..., descriptor_columns][m]
        center = np.average(desc, axis=0, weights=weights)
        scale = np.sqrt(np.average((desc-center)**2, axis=0, weights=weights))
        scale = np.where(scale < 1e-8, 1., scale)
        centers = _kmeans((desc-center)/scale, weights, clusters, seed)
        _, count, pooled = _aggregate(a, m, descriptor_columns, content_columns, center, scale, centers)
        means, axes, thresholds, fit_counts = [], [], [], []
        for k in range(clusters):
            values = pooled[count[:, k] > 0, k]
            if len(values) < 8:
                raise ValueError('prototype observed in fewer than eight training examples')
            average = values.mean(0)
            _, singular, vt = np.linalg.svd(values-average, full_matrices=False)
            if singular[0] < 1e-8:
                raise ValueError('constant prototype content has no PCA target')
            axis = vt[0]
            axis *= 1 if axis[np.abs(axis).argmax()] >= 0 else -1
            means.append(average); axes.append(axis)
            thresholds.append(np.median((values-average)@axis)); fit_counts.append(len(values))
        return cls(descriptor_columns, content_columns, center, scale, centers, np.stack(means),
                   np.stack(axes), np.array(thresholds), np.array(fit_counts), len(x))

    @torch.no_grad()
    def forward(self, x, valid=None):
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if valid is None else valid
        _validate(x, valid, self.descriptor_columns, self.content_columns)
        # Buffers stay immutable. Caller must move the module if changing device.
        values = x.detach().double().masked_fill(~valid[..., None], 0)
        desc = (values[..., self.descriptor_columns]-self.descriptor_mean)/self.descriptor_scale
        assignment = (desc[:, :, None]-self.centers[None, None]).square().sum(-1).argmin(-1)
        mask = (assignment[..., None] == torch.arange(len(self.centers), device=x.device)) & valid[..., None]
        count = mask.sum(1)
        pooled = torch.einsum('bnk,bnc->bkc', mask.double(), values[..., self.content_columns])/count.clamp_min(1)[..., None]
        scores = ((pooled-self.content_mean)*self.axes).sum(-1)-self.thresholds
        return {'labels': (scores > 0).to(x.dtype), 'observed': count > 0,
                'scores': scores, 'token_cluster': assignment, 'counts': count}
