"""Aggregate diagnostics for raw slots, read distributions and role outputs.

This module does not make a causal or biological interpretation. Tensors are
never written by these functions. Callers must keep patient-level data private.
"""
import math
import torch
import torch.nn.functional as F
from .components.slots import READ_EPS
from .components.losses import discrete_hazard_nll


def read_mass_summary(mass):
    """Aggregate [patients,iterations,slots] pre-smoothing assignment mass.

    The threshold is the historical clamp level, not a dead-slot definition.
    Low mass is valid and reported; invalid numeric values still raise.
    """
    if mass.ndim != 3 or not mass.numel() or not torch.isfinite(mass).all() or (mass < 0).any():
        raise ValueError('finite nonnegative nonempty [patients,iterations,slots] mass required')
    mass = mass.detach().double()
    low = mass < READ_EPS
    return {'threshold': READ_EPS, 'threshold_scope': 'historical_clamp_level_not_dead_slot_definition',
            'observations': mass.numel(), 'minimum': float(mass.min()),
            'below_threshold_count': int(low.sum()), 'below_threshold_fraction': float(low.double().mean()),
            'below_threshold_count_by_iteration': low.sum(dim=(0, 2)).tolist(),
            'below_threshold_count_by_slot': low.sum(dim=(0, 1)).tolist(),
            'minimum_by_iteration': mass.amin(dim=(0, 2)).tolist()}


def read_diagnostics_summary(observations):
    """Aggregate detached [bag observations, iteration, slot] metrics.

    The two-dimensional arrays have iteration and slot axes only. They never
    retain a patient axis; counts are repeated observations, not independent
    patients or uncertainty estimates.
    """
    if not observations or 'raw_mass' not in observations:
        raise ValueError('nonempty read diagnostics including raw_mass required')
    shape = observations['raw_mass'].shape
    result = {'axes': ['iteration', 'slot'], 'bag_observations': shape[0],
              'observation_count': observations['raw_mass'].numel(), 'metrics': {}}
    for name, value in observations.items():
        if value.ndim != 3 or value.shape != shape or not value.numel() or not torch.isfinite(value).all():
            raise ValueError('aligned finite nonempty [bags,iterations,slots] diagnostics required')
        x = value.detach().double()
        result['metrics'][name] = {
            'mean': float(x.mean()), 'minimum': float(x.min()), 'maximum': float(x.max()),
            'mean_by_iteration_slot': x.mean(0).tolist(),
            'minimum_by_iteration_slot': x.amin(0).tolist(),
            'maximum_by_iteration_slot': x.amax(0).tolist(),
        }
    low = observations['raw_mass'] < READ_EPS
    result['raw_mass_threshold'] = READ_EPS
    result['below_threshold_count_by_iteration_slot'] = low.sum(0).tolist()
    result['below_threshold_fraction_by_iteration_slot'] = low.double().mean(0).tolist()
    result['threshold_scope'] = 'historical_clamp_level_not_dead_slot_definition'
    return result


def _effective_rank(x):
    """Entropy rank after centering, excluding numerical roundoff singular values."""
    x = x.double()
    tolerance = torch.finfo(x.dtype).eps * max(x.shape) * torch.linalg.vector_norm(x)
    x = x - x.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(x)
    eigen = singular[singular > tolerance].square()
    total = eigen.sum()
    if total == 0:
        return 0.0
    prob = eigen[eigen > 0] / total
    return float(torch.exp(-(prob * prob.log()).sum()))


def slot_summary(slots, attention):
    """Summarize [patients,slots,width] and normalized [patients,slots,tokens]."""
    if slots.ndim != 3 or attention.ndim != 3 or slots.shape[:2] != attention.shape[:2]:
        raise ValueError('expected aligned [patients,slots,width/tokens]')
    if not slots.numel() or not attention.numel() or not torch.isfinite(slots).all() or not torch.isfinite(attention).all():
        raise ValueError('finite nonempty slots and attention required')
    if (attention < 0).any() or not torch.allclose(attention.sum(-1), torch.ones_like(attention[..., 0]), atol=1e-5, rtol=1e-5):
        raise ValueError('attention must be nonnegative and normalized over tokens')
    x, a = slots.detach().double(), attention.detach().double()
    patients, count, width = x.shape
    norms = x.norm(dim=-1)
    # A zero vector has undefined cosine; never turn it into apparent diversity.
    pairs, overlaps, undefined = [], [], 0
    for i in range(count):
        for j in range(i + 1, count):
            valid = (norms[:, i] > 0) & (norms[:, j] > 0)
            undefined += int((~valid).sum())
            if valid.any():
                pairs.extend(F.cosine_similarity(x[valid, i], x[valid, j], dim=-1).tolist())
            overlaps.extend(torch.minimum(a[:, i], a[:, j]).sum(-1).tolist())
    return {
        'patients': patients, 'slots': count, 'width': width,
        'mean_norm': float(norms.mean()),
        'mean_norm_by_slot': norms.mean(dim=0).tolist(),
        'zero_slot_fraction': float((norms == 0).double().mean()),
        'zero_fraction_by_slot': (norms == 0).double().mean(dim=0).tolist(),
        'across_patient_std_rms': float(x.std(dim=0, unbiased=False).square().mean().sqrt()),
        'across_patient_std_rms_by_slot': x.std(dim=0, unbiased=False).square().mean(dim=-1).sqrt().tolist(),
        'centered_patient_effective_rank_by_slot': [_effective_rank(x[:, i, :]) for i in range(count)],
        'effective_rank_tolerance': 'eps64 * max(patients,width) * uncentered Frobenius norm',
        'patient_rank_upper_bound': min(patients - 1, width),
        'pair_cosine_mean': sum(pairs) / len(pairs) if pairs else None,
        'pair_cosine_undefined_count': undefined,
        'read_overlap_mean': sum(overlaps) / len(overlaps) if overlaps else None,
        'pair_status': 'not_applicable_one_slot' if count == 1 else ('undefined_zero_vectors' if not pairs else 'computed'),
    }


def role_summary(roles):
    """Across-patient variation and rank of actual decoded role outputs."""
    if roles.ndim != 2 or not roles.numel() or not torch.isfinite(roles).all():
        raise ValueError('finite nonempty [patients,features] required')
    r = roles.detach().double()
    return {'patients': len(r), 'mean_norm': float(r.norm(dim=-1).mean()),
            'across_patient_std_rms': float(r.std(dim=0, unbiased=False).square().mean().sqrt()),
            'centered_patient_effective_rank': _effective_rank(r),
            'effective_rank_tolerance': 'eps64 * max(patients,width) * uncentered Frobenius norm',
            'patient_rank_upper_bound': min(len(r) - 1, r.shape[1])}


@torch.no_grad()
def ablate_each_slot(model, output, completed=None, event=None):
    """Return mean absolute logit change for removal from each role pool.

    This is a deterministic model-input intervention, not conditional CPI.
    The empty-group output is the same fixed zero used by the model itself.
    Optional labels add deleted NLL and signed (deleted minus full) NLL;
    omitting labels preserves the legacy logit-only return dictionary.
    """
    if (completed is None) != (event is None):
        raise ValueError('both completed and event labels are required')
    with_nll = completed is not None
    base_nll = discrete_hazard_nll(output['prediction'], completed, event) if with_nll else None
    results, deleted_nll, delta_nll = {}, {}, {}
    for modality in ('p', 'g'):
        for role, part_key in (('common', 'common_slots'), ('residual', 'private_slots')):
            key = modality + '_' + role
            slots = output['parts'][modality][part_key]
            changes, nlls, deltas = [], [], []
            for index in range(slots.shape[1]):
                kept = torch.cat((slots[:, :index], slots[:, index + 1:]), dim=1)
                roles = dict(output['roles'])
                if kept.shape[1]:
                    decoded = model.decoders[key](model.pools[key](kept))
                    roles[key] = model.reference.project(decoded, modality, role)
                else:
                    roles[key] = torch.zeros_like(roles[key])
                alt = model.predict_roles(roles)['prediction']
                changes.append(float((alt - output['prediction']).abs().mean()))
                if with_nll:
                    nll = discrete_hazard_nll(alt, completed, event)
                    nlls.append(float(nll))
                    deltas.append(float(nll - base_nll))
            results[key] = changes
            if with_nll:
                deleted_nll[key], delta_nll[key] = nlls, deltas
    if with_nll:
        return {'mean_abs_logit_change': results, 'NLL': deleted_nll, 'delta_NLL': delta_nll}
    return results


def better_validation(candidate, best):
    """Exact C-index, then NLL, then earliest epoch; reject undefined metrics."""
    def check(row):
        c, n, e = row['C_index'], row['NLL'], row['epoch']
        if not isinstance(e, int) or e < 1 or c is None or n is None or not math.isfinite(c) or not math.isfinite(n):
            raise ValueError('finite validation metrics and positive integer epoch required')
        return (c, -n, -e)
    candidate_key = check(candidate)
    return best is None or candidate_key > check(best)
