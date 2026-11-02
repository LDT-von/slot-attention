"""Equal-width whole-set versus validation-selected single-slot probe.

This measures predictive access using a finite nonlinear kernel family; it
does not estimate MI, establish independent factors, or certify redundancy.
With multiple modalities, 'single' means one slot PER modality. Averaging
occurs after the existing phi, matching the network's actual readout.
"""
from itertools import product
import torch
from .incremental_probe import incremental_probe


def _validate(parts):
    counts=None;dimensions=None
    for common,groups,target in parts:
        if not isinstance(groups,(list,tuple)) or not groups:
            raise ValueError('nonempty modality slot lists required')
        if common.ndim!=2 or target.ndim not in [1,2]:raise ValueError('common/target shape')
        for slots in groups:
            if slots.ndim!=3 or slots.shape[0]!=len(common) or min(slots.shape)<1:
                raise ValueError('expected aligned [examples, slots, width]')
            if not bool(torch.isfinite(slots).all()):raise ValueError('nonfinite slots')
        if len(target)!=len(common):raise ValueError('target rows do not align')
        shape=tuple(g.shape[1] for g in groups);dims=tuple(g.shape[2] for g in groups)
        if counts is None:counts,dimensions=shape,dims
        if shape!=counts or dims!=dimensions:raise ValueError('modality slot counts/widths differ across partitions')
    return counts


def slot_increment_probe(fit,validation,test):
    """Partitions: (common [N,C], list of phi-slots [N,K,D], target [N,...]).

    Independent sample partitions are the caller's responsibility. Candidates
    have identical pooled width and the same kernel grid/centres. Effective
    degrees of freedom and larger single-selection search are disclosed.
    """
    parts=[fit,validation,test];counts=_validate(parts)
    candidates=[]
    for indices in product(*[range(k) for k in counts]):
        inputs=[(g,torch.cat([s[:,k] for s,k in zip(groups,indices)],-1),y)
                for g,groups,y in parts]
        result=incremental_probe(*inputs)
        candidates.append({'indices':list(indices),'probe':result})
    selected=min(candidates,key=lambda x:x['probe']['joint']['validation_mse'])
    full=incremental_probe(*[(g,torch.cat([s.mean(1) for s in groups],-1),y) for g,groups,y in parts])
    target=test[2].double();mean=fit[2].double().mean(0)
    denominator=float((target-mean).square().mean())
    gain=selected['probe']['joint']['test_mse']-full['joint']['test_mse']
    return {'best_single_indices':selected['indices'],'single_candidate_count':len(candidates),
            'selected_single':selected['probe'],'all_slots':full,
            'all_vs_single_test_mse_gain':gain,'gain_relative_to_fit_mean_baseline':gain/max(denominator,1e-12),
            'pooled_feature_width_per_arm':sum(s.shape[-1] for s in fit[1]),
            'selection':'single slot(s) and kernel choices use validation only; test scored after selection',
            'interpretation':'finite-function-class gain; may reflect denoising/decoding, not uniquely complementary content'}
