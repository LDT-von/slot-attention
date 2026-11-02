"""Clarified finite diagnostic target; never part of neural training loss."""
import torch
from .slot_increment_probe import slot_increment_probe


def effects(factors, task):
    if task == 'factorwise': return factors
    if task != 'paired_xor': raise ValueError('known prognostic task required')
    p, q = factors[:, :2], factors[:, 2:]
    return -torch.stack([p[:, 0]*q[:, 0], p[:, 1]*q[:, 1], p[:, 0]*q[:, 1], p[:, 1]*q[:, 0]], -1)


def effect_probe(frozen, task):
    return {mode: slot_increment_probe(*[(r[mode]['common'], r[mode]['phi'], effects(r['factors'], task)) for r in frozen])
            for mode in ['sample', 'all_mean']}
