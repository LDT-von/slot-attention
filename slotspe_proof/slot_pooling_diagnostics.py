"""Privileged synthetic controls for rate allocation and all-slot pooling.

Role-bank pooling is a typed concatenation control, not a novel mechanism or
an accepted replacement for the research model. No patient inputs are used.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .xor_readout_diagnostics import oracle_inputs, mixture_survival_logits


CONDITIONS = ('clean_single', 'single_total1', 'repeated_total1',
              'repeated_total4', 'noise_only')


def gaussian_code(rate):
    if not math.isfinite(rate) or rate < 0:
        raise ValueError('finite nonnegative rate required')
    return math.sqrt(-math.expm1(-2*rate)), math.exp(-rate)


def code_inputs(shared, bits, noise, condition):
    if condition not in CONDITIONS:
        raise ValueError('unknown synthetic condition')
    mode = {'clean_single': 'before_phi_clean_single',
            'single_total1': 'before_phi_rate1_single',
            'repeated_total1': 'before_phi_rate1_repeated',
            'repeated_total4': 'before_phi_rate1_repeated',
            'noise_only': 'before_phi_noise_only'}[condition]
    g, private = oracle_inputs(shared, bits, noise, mode)
    if condition == 'repeated_total1':
        amplitude, sigma = gaussian_code(.25)
        for j, eps in enumerate(noise):
            private[j][:, :, 0] = amplitude*bits[:, j, None]+sigma*eps[:, :, 0]
    return g, private


def code_bayes_logits(common, private, condition):
    if condition == 'noise_only':
        probability = torch.full_like(common[:, 0], .5)
    elif condition == 'clean_single':
        p, q = [(v[:, 0, 0] > 0).to(v.dtype) for v in private]
        probability = p*(1-q)+(1-p)*q
    else:
        count = 1 if condition == 'single_total1' else 4
        rate = .25 if condition == 'repeated_total1' else 1.
        amplitude, sigma = gaussian_code(rate)
        ps = [torch.sigmoid(2*amplitude*v[:, :count, 0].sum(1)/(sigma*sigma))
              for v in private]
        p, q = ps
        probability = p*(1-q)+(1-p)*q
    return mixture_survival_logits(common[:, 0], probability)


class RoleBankReadout(nn.Module):
    """Sum fixed one-hot-role tensor features, retaining every role block.

    Joint permutation of values and role IDs is invariant. Value permutation
    without role IDs is not invariant. This requires stable learned slot roles;
    IDs are not semantic supervision. Fixed masks are not dynamic gates.
    """
    def __init__(self, reference, seed):
        super().__init__()
        self.count = reference.role_offsets[0].shape[0]
        width = reference.phi[0][-1].out_features
        dimension = reference.role_offsets[0].shape[1]
        if width % self.count:
            raise ValueError('readout width must divide into complete role blocks')
        self.block = width//self.count
        baseline = sum(p.numel() for p in reference.phi[0].parameters())
        hidden = max(1, round((baseline-self.block)/(dimension+1+self.block)))
        self.head = copy.deepcopy(reference.head)
        self.role_offsets = copy.deepcopy(reference.role_offsets)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.phi = nn.ModuleList([
                nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                              nn.Linear(hidden, self.block)) for _ in range(2)])

    def forward(self, common, private, role_indices=None):
        pooled = []
        for j, (phi, value) in enumerate(zip(self.phi, private)):
            ids = torch.arange(self.count, device=value.device) if role_indices is None else role_indices[j]
            if value.shape[1] != self.count or not torch.equal(ids.sort().values, torch.arange(self.count, device=value.device)):
                raise ValueError('all roles required with a complete paired permutation')
            encoded = phi(value+self.role_offsets[j][ids][None])
            roles = F.one_hot(ids, self.count).to(encoded.dtype)
            pooled.append(torch.einsum('kr,bkd->brd', roles, encoded).flatten(1))
        return self.head(torch.cat([common, *pooled], -1))
