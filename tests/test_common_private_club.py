import math
import torch
import pytest
from slotspe_proof.common_private_club import GaussianCLUB,log_ratio_from_gaussian,frozen_critic,standardized_rows,separation_penalty


def test_log_ratio_matches_explicit_offdiagonal_pairs():
    x=torch.tensor([[0.],[1.],[2.]])
    y=torch.tensor([[2.],[1.],[0.]])
    value=log_ratio_from_gaussian(x,torch.zeros_like(x),y)
    positive=sum(-.5*(float(y[i]-x[i]))**2 for i in range(3))/3
    negative=sum(-.5*(float(y[j]-x[i]))**2 for i in range(3) for j in range(3) if i!=j)/6
    assert abs(float(value)-(positive-negative))<1e-6


def test_critic_construction_preserves_cpu_rng():
    before=torch.random.get_rng_state().clone();GaussianCLUB(2,3)
    assert torch.equal(before,torch.random.get_rng_state())


def test_fitting_detaches_representations():
    c=GaussianCLUB(2,3);x=torch.randn(8,2,requires_grad=True);y=torch.randn(8,3,requires_grad=True)
    c.fitting_loss(x,y).backward()
    assert x.grad is None and y.grad is None
    assert all(p.grad is not None for p in c.parameters())


def test_frozen_critic_preserves_private_gradient_and_restores_flags():
    c=GaussianCLUB(2,3);x=torch.randn(8,2);y=torch.randn(8,3,requires_grad=True)
    with frozen_critic(c): c(x,y).backward()
    assert all(p.grad is None and p.requires_grad for p in c.parameters())
    assert y.grad is not None and torch.isfinite(y.grad).all()


def test_standardization_scale_and_shift():
    x=torch.randn(16,3)
    assert torch.allclose(standardized_rows(x),standardized_rows(7*x+2),atol=1e-5)


def test_critic_wrong_direction_can_be_negative():
    x=torch.tensor([[-1.],[1.]])
    assert log_ratio_from_gaussian(-x,torch.zeros_like(x),x)<0


def test_cpu_constructor_never_seeds_cuda(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('CPU critic constructor must not seed CUDA')
    monkeypatch.setattr(torch.cuda,'manual_seed_all',forbidden)
    GaussianCLUB(2,3)


def test_missing_modality_critic_is_not_silently_ignored():
    with pytest.raises(ValueError):separation_penalty([GaussianCLUB(2,3)],torch.randn(8,2),[torch.randn(8,3),torch.randn(8,3)])
