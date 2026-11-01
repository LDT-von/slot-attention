"""Research prototype: conditional slot bottleneck and selection-free survival.

Inputs are frozen, training-fold-standardized token features, NOT raw slides/RNA.
No patient loader or queue integration. Common/private are operational roles,
not identified biological factors. Sampling is explicit even in eval mode.
"""
import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(x, mask):
    if mask.shape != x.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('boolean mask must match token axes')
    if not mask.any(1).all():
        raise ValueError('empty modality is unsupported')
    return x.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)


class SlotEncoder(nn.Module):
    def __init__(self, input_dim, width, slots, iterations=3):
        super().__init__()
        self.project = nn.Linear(input_dim, width)
        self.initial = nn.Parameter(torch.randn(slots, width) * .1)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.query = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.update = nn.GRUCell(width, width)
        self.iterations = iterations

    def forward(self, x, mask):
        masked_mean(x, mask)  # enforce valid bags
        x = x.masked_fill(~mask[..., None], 0)
        features = self.project(x)
        k, v = self.key(features), self.value(features)
        s = self.initial[None].expand(x.shape[0], -1, -1)
        for _ in range(self.iterations):
            logits = torch.einsum('bkd,bnd->bkn', self.query(self.norm(s)), k) / k.shape[-1] ** .5
            # Competition during slot construction; no selection at readout.
            a = logits.softmax(1) * mask[:, None, :]
            a = a / a.sum(-1, keepdim=True).clamp_min(1e-8)
            messages = a @ v
            s = self.update(messages.reshape(-1, s.shape[-1]), s.reshape(-1, s.shape[-1])).reshape_as(s)
        return s


def gaussian_kl(mu, lv, prior_mu, prior_lv):
    return .5 * (prior_lv - lv + (lv.exp() + (mu-prior_mu).square()) * (-prior_lv).exp() - 1)


class CommonPrivateSurvival(nn.Module):
    def __init__(self, wsi_dim, omic_dim, width=128, slots=8, common_dim=32, private_dim=32, bins=4):
        super().__init__()
        self.encoders = nn.ModuleList([SlotEncoder(d, width, slots) for d in (wsi_dim, omic_dim)])
        self.common = nn.ModuleList([nn.Linear(width, common_dim) for _ in range(2)])
        self.posterior = nn.ModuleList([nn.Linear(width+common_dim, 2*private_dim) for _ in range(2)])
        # Same conditional prior for every slot: no slot-ID coding/ordering assumption.
        self.prior = nn.ModuleList([nn.Linear(common_dim, 2*private_dim) for _ in range(2)])
        self.phi = nn.ModuleList([nn.Sequential(nn.Linear(private_dim, width), nn.GELU(), nn.Linear(width, width)) for _ in range(2)])
        self.cross_target = nn.ModuleList([nn.Linear(common_dim, omic_dim), nn.Linear(common_dim, wsi_dim)])
        self.head = nn.Sequential(nn.Linear(common_dim+2*width, width), nn.GELU(), nn.Linear(width, bins))

    def readout(self, common, private):
        return self.head(torch.cat([common] + [f(u).mean(1) for f, u in zip(self.phi, private)], -1))

    def forward(self, wsi, omic, wsi_mask=None, omic_mask=None, *, sample=True, noise=None):
        xs = [wsi, omic]
        masks = [torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if m is None else m
                 for x, m in zip(xs, [wsi_mask, omic_mask])]
        slots = [e(x, m) for e, x, m in zip(self.encoders, xs, masks)]
        cs = [f(s.mean(1)) for f, s in zip(self.common, slots)]
        common = (cs[0]+cs[1]) / 2
        private, kls = [], []
        for j, s in enumerate(slots):
            # KL cannot directly push the conditioning summary to hide information.
            # Survival/alignment/cross prediction still train that summary.
            g = common.detach()
            mu, lv = self.posterior[j](torch.cat([s, g[:, None].expand(-1, s.shape[1], -1)], -1)).chunk(2, -1)
            pm, plv = self.prior[j](g).unsqueeze(1).chunk(2, -1)
            lv, plv = lv.clamp(-8, 4), plv.clamp(-8, 4)
            if sample:
                eps = torch.randn_like(mu) if noise is None else noise[j]
                u = mu + (.5*lv).exp()*eps
            else:
                u = mu
            private.append(u)
            kls.append(gaussian_kl(mu, lv, pm, plv).sum((1, 2)))
        target = [masked_mean(x, m).detach() for x, m in zip(xs, masks)]
        cross = sum(F.mse_loss(self.cross_target[j](cs[j]), target[1-j]) for j in range(2)) / 2
        alignment = F.mse_loss(cs[0], cs[1])
        return {'logits': self.readout(common, private), 'common': common, 'private': private,
                'slots': slots, 'kl_joint': torch.stack(kls).sum(0), 'cross': cross, 'alignment': alignment,
                'latent_dimensions': sum(u.shape[1]*u.shape[2] for u in private)}


def survival_nll(logits, time_bin, event):
    """Event in bin t; censoring means known survival THROUGH bin t.

    Clinical discretization must respect this convention; bins fit on train only.
    """
    if time_bin.shape != logits.shape[:1] or event.shape != time_bin.shape:
        raise ValueError('one time and event indicator per example required')
    if not ((time_bin >= 0) & (time_bin < logits.shape[1])).all():
        raise ValueError('time bins out of range')
    if not ((event == 0) | (event == 1)).all():
        raise ValueError('event must be binary')
    axis = torch.arange(logits.shape[1], device=logits.device)[None]
    survived = axis < time_bin[:, None]
    survived = survived | ((axis == time_bin[:, None]) & ~event.bool()[:, None])
    loss = (F.softplus(logits)*survived).sum(1)
    loss = loss + F.softplus(-logits.gather(1, time_bin[:, None]).squeeze(1))*event
    return loss.mean()


def objective(out, time_bin, event, beta=.001):
    nll = survival_nll(out['logits'], time_bin, event)
    kl = out['kl_joint'].mean()/out['latent_dimensions']
    return nll + .1*out['cross'] + .1*out['alignment'] + beta*kl
