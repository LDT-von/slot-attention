"""Privileged generated bags with two independent factors per modality.

No patient adapter. Role channels, measured factors and population hazards
are declared synthetic assumptions; true factors are diagnostics, not losses.
"""
import torch
from torch.nn import functional as F

TASKS = ('factorwise', 'paired_xor')


def generate(seed, n, task):
    if task not in TASKS or n < 1:
        raise ValueError('known task and positive sample count required')
    rng = torch.Generator().manual_seed(seed)
    shared = 2*torch.randint(2, (n,), generator=rng).float()-1
    factors = 2*torch.randint(2, (n, 4), generator=rng).float()-1
    nuisance = torch.randn(n, 2, generator=rng)
    ids = torch.arange(2).repeat_interleave(8)
    xs = []
    for modality in range(2):
        x = torch.zeros(n, 16, 8)
        x[:, :, :2] = F.one_hot(ids, 2).float()[None]
        x[:, :, 2] = shared[:, None]+.05*torch.randn(n, 16, generator=rng)
        x[:, :, 3] = factors[:, 2*modality+ids]+.05*torch.randn(n, 16, generator=rng)
        x[:, :, 4] = nuisance[:, modality, None]+.05*torch.randn(n, 16, generator=rng)
        x[:, :, 5:] = .1*torch.randn(n, 16, 3, generator=rng)
        permutation = torch.rand(n, 16, generator=rng).argsort(1)
        xs.append(x.gather(1, permutation[:, :, None].expand(-1, -1, 8)))
    if task == 'factorwise': effects = factors
    else:
        p, q = factors[:, :2], factors[:, 2:]
        effects = -torch.stack([p[:, 0]*q[:, 0], p[:, 1]*q[:, 1],
                                p[:, 0]*q[:, 1], p[:, 1]*q[:, 0]], -1)
    logits = .3*shared[:, None]+1.2*effects+torch.tensor([-1.5, -.9, -.3, .3])
    hazards = torch.sigmoid(logits)
    hits = torch.rand(n, 4, generator=rng) < hazards
    latent_event_bin = torch.where(hits.any(1), hits.int().argmax(1), 4)
    censor = torch.searchsorted(torch.tensor([.1, .2, .3, 1.]), torch.rand(n, generator=rng))
    return {'xs': xs, 'shared': shared, 'factors': factors, 'nuisance': nuisance,
            'true_logits': logits, 'time': torch.minimum(latent_event_bin, censor),
            'event': (latent_event_bin <= censor).float()}


def noise(rng, n):
    return {'shared_noise': [torch.randn(n, 4, generator=rng) for _ in range(2)],
            'private_noise': [torch.randn(n, 2, 4, generator=rng) for _ in range(2)]}


def standardize(train, others):
    """Fit continuous token scales on training rows; keep role metadata exact."""
    states = []
    for x in train['xs']:
        values = x[:, :, 2:]
        states.append((values.mean((0, 1)), values.std((0, 1), unbiased=False).clamp_min(1e-6)))
    result = []
    for pack in [train, *others]:
        xs = []
        for x, (mean, std) in zip(pack['xs'], states):
            new = x.clone(); new[:, :, 2:] = (new[:, :, 2:]-mean)/std; xs.append(new)
        result.append({**pack, 'xs': xs})
    return result, [{'mean': mean.tolist(), 'std': std.tolist()} for mean, std in states]


def observed_bin_concordance(logits, time, event):
    """Generated-bin score only; censoring is known survival THROUGH its bin."""
    risk = -(1-torch.sigmoid(logits)).cumprod(-1).sum(-1)
    earlier = time[:, None] < time[None, :]
    event_before_same_bin_censor = (time[:, None] == time[None, :]) & ~event.bool()[None]
    pairs = event.bool()[:, None] & (earlier | event_before_same_bin_censor)
    count = int(pairs.sum())
    if not count: return None
    credit = (risk[:, None] > risk[None]).float()+.5*(risk[:, None] == risk[None]).float()
    return float(credit[pairs].sum()/count)
