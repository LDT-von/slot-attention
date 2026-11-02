"""Patient-level variational CLUB critic for common/private readout summaries.

Independent adaptation of Cheng et al., ICML 2020, Eq.14/15. The learned
Gaussian conditional need not be correct: values are NOT certified MI bounds.
Whole patients are rows; slots are symmetrically pooled before this module.
"""
import math
from contextlib import contextmanager
import torch
from torch import nn


def standardized_rows(x):
    if x.ndim != 2 or x.shape[0] < 2 or not x.is_floating_point():
        raise ValueError('expected [distinct patients, features], at least two patients')
    if not bool(torch.isfinite(x).all()):
        raise ValueError('nonfinite representation')
    return (x-x.mean(0,keepdim=True))/x.std(0,unbiased=False,keepdim=True).clamp_min(1e-4)


def log_ratio_from_gaussian(mu, logvar, y):
    """Off-diagonal negative pairs; sum over features, mean over patients.

    Excluding diagonal removes the (B-1)/B shrinkage of all-pair V-statistic.
    For iid patient rows this estimates the same population log-ratio as CLUB.
    Finite samples can yield negative values; never clip them into 'success'.
    """
    if mu.shape != logvar.shape or y.shape != mu.shape or y.ndim != 2 or len(y)<2:
        raise ValueError('matching [B,D] tensors with B>=2 required')
    pair = -.5*((y[None,:,:]-mu[:,None,:]).square()*(-logvar[:,None,:]).exp()
                  +logvar[:,None,:]+math.log(2*math.pi)).sum(-1)
    positive = pair.diag().mean()
    negative = (pair.sum()-pair.diag().sum())/(len(y)*(len(y)-1))
    return positive-negative


@contextmanager
def frozen_critic(critic):
    states=[p.requires_grad for p in critic.parameters()]
    try:
        for p in critic.parameters(): p.requires_grad_(False)
        yield
    finally:
        for p,state in zip(critic.parameters(),states): p.requires_grad_(state)


class GaussianCLUB(nn.Module):
    def __init__(self, common_dim, private_dim, hidden=64, seed=771):
        super().__init__()
        # Construction cannot perturb the native model/data RNG schedule.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.predictor=nn.Sequential(nn.Linear(common_dim,hidden,device='cpu'),nn.SiLU(),
                                         nn.Linear(hidden,hidden,device='cpu'),nn.SiLU(),
                                         nn.Linear(hidden,2*private_dim,device='cpu'))

    def parameters_of_conditional(self, common):
        mean,logvar=self.predictor(common).chunk(2,-1)
        return mean,logvar.clamp(-6.,3.)

    def fitting_loss(self, common, private):
        mean,lv=self.parameters_of_conditional(common.detach())
        y=private.detach()
        return .5*((y-mean).square()*(-lv).exp()+lv+math.log(2*math.pi)).mean()

    def forward(self, common, private):
        mean,lv=self.parameters_of_conditional(common)
        return log_ratio_from_gaussian(mean,lv,private)


def branch_features(net,out):
    return [f(u).mean(1) for f,u in zip(net.phi,out['private'])]


def separation_penalty(critics, common, private_features):
    """Per-dimension proxy; gradients reach private path, common is detached."""
    if not private_features or len(critics)!=len(private_features):
        raise ValueError('one critic for each nonempty private modality list required')
    x=standardized_rows(common.detach())
    return torch.stack([critic(x,standardized_rows(y))/y.shape[1]
                        for critic,y in zip(critics,private_features)]).mean()
