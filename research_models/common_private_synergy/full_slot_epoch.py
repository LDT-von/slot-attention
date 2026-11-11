"""Fixed-sample epoch evaluation for the empirical role model.

Returns aggregates only. Selection and checkpointing are the caller's duty;
this module is not itself a runnable patient-training protocol.
"""
import math
import numpy as np
import torch
from sksurv.metrics import concordance_index_censored

from .components.losses import discrete_hazard_nll
from .components.role_reference import role_content_losses
from .full_slot_diagnostics import ablate_each_slot, role_summary, slot_summary
from .patient_pilot_engine import batch, model_forward


@torch.no_grad()
def evaluate_epoch(model, store, ids, adapter, boundaries, config, device, *, seed, survival):
    """Recreate the same ordered validation bags for every call with this seed.

    Return aggregate metrics, raw-slot/read/role diagnostics and ablations.
    The caller must validate the prepared split and identity independently.
    """
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('nonempty unique ordered validation ids required')
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    roles, slots, attention, ablations = {}, {}, {}, {}
    losses, logits = [], []
    role_sums = {}
    for start in range(0, len(ids), config['batch_size']):
        batch_ids = ids[start:start + config['batch_size']]
        b = batch(store, batch_ids, adapter, boundaries, config, device, generator=rng)
        out = model_forward(model, b)
        target = model.reference.targets(b['p_target'], b['g_target'])
        for key, value in role_content_losses(out['roles'], target).items():
            role_sums[key] = role_sums.get(key, 0.) + float(value) * len(batch_ids)
        if survival:
            losses.append(discrete_hazard_nll(out['prediction'], b['completed'], b['event'], reduction='none').cpu())
            logits.append(out['prediction'].cpu())
            for key, values in ablate_each_slot(model, out).items():
                if key not in ablations:
                    ablations[key] = [0.] * len(values)
                ablations[key] = [old + new * len(batch_ids) for old, new in zip(ablations[key], values)]
        for modality in ('p', 'g'):
            part = out['parts'][modality]
            cut = model.allocations[modality].common
            for role, part_key, section in (('common', 'common_slots', slice(None, cut)),
                                            ('residual', 'private_slots', slice(cut, None))):
                key = modality + '_' + role
                roles.setdefault(key, []).append(out['roles'][key].cpu())
                slots.setdefault(key, []).append(part[part_key].cpu())
                attention.setdefault(key, []).append(part['read_attention'][:, section].cpu())
    result = {
        'role_target_MSE': {k: v / len(ids) for k, v in role_sums.items()},
        'roles': {k: role_summary(torch.cat(v)) for k, v in roles.items()},
        'slots': {k: slot_summary(torch.cat(v), torch.cat(attention[k])) for k, v in slots.items()},
        'validation_sampling_seed': seed,
        'ablation_mean_abs_logit_change': {k: [v / len(ids) for v in values] for k, values in ablations.items()},
        'ablation_scope': 'slot removal in role pool; not conditional CPI',
    }
    if survival:
        logits = torch.cat(logits)
        risks = -(1 - logits.sigmoid()).cumprod(-1).sum(-1).numpy()
        events = np.array([store.manifest['patients'][i]['censor'] < 1 for i in ids], dtype=bool)
        times = np.array([store.manifest['patients'][i]['time'] for i in ids])
        try:
            cindex = float(concordance_index_censored(events, times, risks)[0])
            if not math.isfinite(cindex):
                raise ValueError('nonfinite C-index')
            status = 'computed'
        except ValueError as error:
            cindex, status = None, 'undefined: ' + str(error)
        result.update(NLL=float(torch.cat(losses).mean()), C_index=cindex, C_index_status=status)
    return result
