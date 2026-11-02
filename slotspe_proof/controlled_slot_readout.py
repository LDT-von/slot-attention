"""Diagnostic factorial controls: role access, phi shape and pooling scale.

Fixed signed roles are a dense form of typed aggregation, not attention,
learned slot semantics or a claimed new mechanism. Every slot participates.
"""
import copy

import torch
from torch import nn


def hadamard(count, *, device=None, dtype=None):
    if count < 1 or count & (count-1):
        raise ValueError('positive power-of-two role count required')
    matrix = torch.ones(1, 1, device=device, dtype=dtype)
    while matrix.shape[0] < count:
        matrix = torch.cat([torch.cat([matrix, matrix], 1),
                            torch.cat([matrix, -matrix], 1)], 0)
    return matrix


def controlled_pool(encoded, ids, role_access, scale):
    if encoded.ndim != 3 or encoded.shape[2] % encoded.shape[1]:
        raise ValueError('encoded width must contain complete role blocks')
    count, width = encoded.shape[1:]
    if ids.shape != (count,) or ids.dtype != torch.long or not torch.equal(ids.sort().values, torch.arange(count, device=ids.device)):
        raise ValueError('complete paired role IDs required')
    if role_access not in ['blind', 'signed'] or scale not in ['mean', 'sum']:
        raise ValueError('unknown role access or scale')
    if role_access == 'signed':
        matrix = hadamard(count, device=encoded.device, dtype=encoded.dtype)
        blocks = torch.arange(width, device=encoded.device)//(width//count)
        encoded = encoded*matrix[ids][:, blocks][None]
    pooled = encoded.mean(1)
    return pooled if scale == 'mean' else count*pooled


class ControlledSlotReadout(nn.Module):
    def __init__(self, reference, *, phi_shape, role_access, scale, seed):
        super().__init__()
        if phi_shape not in ['full', 'block']:
            raise ValueError('unknown phi shape')
        self.count = reference.role_offsets[0].shape[0]
        self.width = reference.phi[0][-1].out_features
        self.phi_shape, self.role_access, self.scale = phi_shape, role_access, scale
        if self.width % self.count:
            raise ValueError('complete output blocks required')
        self.head = copy.deepcopy(reference.head)
        self.role_offsets = copy.deepcopy(reference.role_offsets)
        if phi_shape == 'full':
            self.phi = copy.deepcopy(reference.phi)
        else:
            dimension = reference.role_offsets[0].shape[1]
            block = self.width//self.count
            baseline = sum(p.numel() for p in reference.phi[0].parameters())
            hidden = max(1, round((baseline-block)/(dimension+1+block)))
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(seed)
                self.phi = nn.ModuleList([
                    nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, block))
                    for _ in range(2)])

    def forward(self, common, private, role_indices=None):
        features = []
        for j, (phi, value) in enumerate(zip(self.phi, private)):
            ids = torch.arange(self.count, device=value.device) if role_indices is None else role_indices[j]
            encoded = phi(value+self.role_offsets[j][ids][None])
            if self.phi_shape == 'block':
                encoded = encoded.repeat(1, 1, self.count)
            features.append(controlled_pool(encoded, ids, self.role_access, self.scale))
        return self.head(torch.cat([common, *features], -1))
