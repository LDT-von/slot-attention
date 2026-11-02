"""Own-view Gaussian shared posteriors and cooperative all-slot private rate.

Synthetic research model only. Shared rates concern the actual sampled C,
not posterior means or unlimited Monte Carlo averaged predictions. The
cross-modal conditional-information identity requires each shared posterior
to depend only on its own modality. No slot-content oracle/teacher is used.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .common_private_survival import CommonPrivateSurvival, gaussian_kl, masked_mean, survival_nll
from .prior_normalized_slots import normalize_gaussian


def _noise_like(reference, supplied):
    if supplied is None:
        return torch.randn_like(reference)
    if supplied.shape != reference.shape or supplied.dtype != reference.dtype or supplied.device != reference.device:
        raise ValueError('noise shape, dtype and device must match posterior')
    if not bool(torch.isfinite(supplied).all()):
        raise ValueError('nonfinite noise')
    return supplied


class CrossmodalSharedBottleneck(CommonPrivateSurvival):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width, dimension = self.common[0].in_features, self.common[0].out_features
        replacements = []
        # Preserve core RNG and original mean initialization. Variance starts at 1.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(83491)
            for original in self.common:
                layer = nn.Linear(width, 2*dimension, device='cpu')
                with torch.no_grad():
                    layer.weight[:dimension].copy_(original.weight)
                    layer.bias[:dimension].copy_(original.bias)
                    layer.weight[dimension:].zero_(); layer.bias[dimension:].zero_()
                replacements.append(layer)
        self.common = nn.ModuleList(replacements)
        k, d = self.encoders[0].initial.shape[0], self.posterior[0].out_features//2
        self.role_offsets = nn.ParameterList([nn.Parameter(torch.zeros(k, d)) for _ in range(2)])

    def readout(self, common, private, role_indices=None):
        pooled = []
        for j, (phi, u) in enumerate(zip(self.phi, private)):
            count = self.role_offsets[j].shape[0]
            ids = torch.arange(count, device=u.device) if role_indices is None else role_indices[j]
            if u.shape[1] != count or not torch.equal(ids.sort().values, torch.arange(count, device=u.device)):
                raise ValueError('all slot roles required, with complete paired permutation')
            pooled.append(phi(u+self.role_offsets[j][ids][None]).mean(1))
        return self.head(torch.cat([common, *pooled], -1))

    def forward(self, wsi, omic, wsi_mask=None, omic_mask=None, *, sample=True,
                shared_noise=None, private_noise=None):
        if not sample and (shared_noise is not None or private_noise is not None):
            raise ValueError('mean diagnostic must not silently ignore noise')
        if shared_noise is not None and len(shared_noise) != 2:
            raise ValueError('two shared noise tensors required')
        if private_noise is not None and len(private_noise) != 2:
            raise ValueError('two private noise tensors required')
        xs = [wsi, omic]
        masks = [torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if m is None else m
                 for x, m in zip(xs, [wsi_mask, omic_mask])]
        slots = [encoder(x, mask) for encoder, x, mask in zip(self.encoders, xs, masks)]
        shared_means, shared_lvs, shared = [], [], []
        # No concatenated/cross-view input may enter these own-view posteriors.
        for j, (head, s) in enumerate(zip(self.common, slots)):
            mu, lv = head(s.mean(1)).chunk(2, -1); lv = lv.clamp(-8, 4)
            c = mu
            if sample:
                eps = _noise_like(mu, None if shared_noise is None else shared_noise[j])
                c = mu+(.5*lv).exp()*eps
            shared_means.append(mu); shared_lvs.append(lv); shared.append(c)
        common = (shared[0]+shared[1])/2
        common_mean = (shared_means[0]+shared_means[1])/2
        directions = [gaussian_kl(shared_means[j], shared_lvs[j], shared_means[1-j], shared_lvs[1-j]).sum(-1)
                      for j in range(2)]
        private, private_means, private_lvs, rates = [], [], [], []
        for j, s in enumerate(slots):
            g = common.detach()
            mu, lv = self.posterior[j](torch.cat([s, g[:, None].expand(-1, s.shape[1], -1)], -1)).chunk(2, -1)
            pm, plv = self.prior[j](g).unsqueeze(1).chunk(2, -1)
            lv, plv = lv.clamp(-8, 4), plv.clamp(-8, 4)
            vm, vlv = normalize_gaussian(mu, lv, pm, plv)
            v = vm
            if sample:
                eps = _noise_like(vm, None if private_noise is None else private_noise[j])
                v = vm+(.5*vlv).exp()*eps
            private.append(v); private_means.append(vm); private_lvs.append(vlv)
            rates.append(gaussian_kl(vm, vlv, torch.zeros_like(vm), torch.zeros_like(vlv)).sum(-1))
        target = [masked_mean(x, mask).detach() for x, mask in zip(xs, masks)]
        cross = sum(F.mse_loss(self.cross_target[j](shared[j]), target[1-j]) for j in range(2))/2
        return {'logits': self.readout(common, private), 'common': common, 'common_mean': common_mean,
            'shared_mean': shared_means, 'shared_logvar': shared_lvs, 'shared_sample': shared,
            'shared_rate_per_direction': directions, 'shared_rate': (directions[0]+directions[1])/2,
            'private': private, 'private_mean': private_means, 'private_logvar': private_lvs,
            'private_rate_per_slot': rates, 'slots': slots, 'cross': cross,
            'mean_alignment_diagnostic': F.mse_loss(shared_means[0], shared_means[1]),
            'sampling_mode': 'stochastic_single_draw' if sample else 'posterior_means_diagnostic_only'}


def shared_private_objective(out, time_bin, event, *, shared_weight=.1, private_weight=.3, capacity=1.):
    if out['sampling_mode'] != 'stochastic_single_draw':
        raise ValueError('objective requires actual sampled shared/private readout')
    if not all(math.isfinite(v) and v >= 0 for v in [shared_weight, private_weight, capacity]):
        raise ValueError('finite nonnegative rate weights/capacity required')
    nll = survival_nll(out['logits'], time_bin, event)
    common_rate = out['shared_rate'].mean()
    violation = torch.stack([torch.relu(rate.mean(0)-capacity).mean() for rate in out['private_rate_per_slot']]).mean()
    loss = nll+.1*out['cross']+shared_weight*common_rate+private_weight*violation
    return {'loss': loss, 'nll': nll, 'shared_rate': common_rate, 'private_budget_hinge': violation}
