"""One controlled below-capacity fee; no new encoder/readout or MI estimator."""
import math
import torch
from .crossmodal_shared_bottleneck import shared_private_objective


def private_rate_objective(out, time_bin, event, *, eta=0., private_weight=.3, capacity=1., shared_weight=.1):
    if not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError('eta must lie in[0,1]')
    terms = shared_private_objective(out, time_bin, event, shared_weight=shared_weight,
                                    private_weight=private_weight, capacity=capacity)
    means = [r.mean(0) for r in out['private_rate_per_slot']]
    if any(not bool(torch.isfinite(r).all()) for r in means):
        raise ValueError('finite rates required')
    bounded = torch.stack([r.clamp_max(capacity).mean() for r in means]).mean()
    fee = terms['private_budget_hinge']+eta*bounded
    # eta=0 preserves the original loss graph exactly, including operation order.
    loss = terms['loss'] if eta == 0 or private_weight == 0 else terms['loss']+private_weight*eta*bounded
    return {**terms, 'loss': loss, 'private_fee': fee,
            'private_rate_mean': torch.stack([r.mean() for r in means]).mean(),
            'active_slot_fraction': torch.cat([r > capacity for r in means]).float().mean()}
