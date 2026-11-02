import torch
import pytest
from slotspe_proof.prior_normalized_slots import PriorNormalizedSlotSurvival,normalize_gaussian
from slotspe_proof.common_private_survival import CommonPrivateSurvival,gaussian_kl,objective


@pytest.fixture(autouse=True)
def isolated_cpu_runtime():
    old=torch.get_num_threads();torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):yield
    torch.set_num_threads(old)


def test_gaussian_change_of_coordinates_preserves_kl():
    generator=torch.Generator().manual_seed(5)
    mu,lv,pm,plv=[torch.randn(3,4,5,generator=generator,dtype=torch.float64) for _ in range(4)]
    vm,vlv=normalize_gaussian(mu,lv,pm,plv)
    assert torch.allclose(gaussian_kl(mu,lv,pm,plv),gaussian_kl(vm,vlv,torch.zeros_like(vm),torch.zeros_like(vlv)),atol=1e-10)


def test_common_location_copy_is_removed_from_actual_private_code():
    g=torch.linspace(-2,2,100)[:,None];p=torch.sin(g*3)
    zero=torch.zeros_like(g)
    codes=[normalize_gaussian(a*g+p,zero,a*g,zero)[0] for a in [0.,1.,100.]]
    assert all(torch.allclose(codes[0],c,atol=1e-5) for c in codes)


def test_host_has_identical_parameters_and_same_rate_before_any_update():
    torch.default_generator.manual_seed(3)
    raw=CommonPrivateSurvival(6,6,width=16,slots=4,common_dim=4,private_dim=4)
    new=PriorNormalizedSlotSurvival(6,6,width=16,slots=4,common_dim=4,private_dim=4)
    new.load_state_dict(raw.state_dict())
    x=torch.randn(8,12,6);y=torch.randn_like(x)
    noises=[torch.randn(8,4,4) for _ in range(2)]
    a=raw(x,y,noise=noises);b=new(x,y,noise=noises)
    assert torch.equal(a['kl_joint'],b['kl_joint'])
    assert torch.allclose(b['kl_joint'],b['normalized_kl_joint'],atol=1e-5)
    assert sum(p.numel() for p in raw.parameters())==sum(p.numel() for p in new.parameters())
    assert len(new.prior[0]._forward_hooks)==0 and len(new.posterior[0]._forward_hooks)==0
    objective(b,torch.arange(8)%4,torch.ones(8),beta=.3).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())


def test_standard_normal_prior_recovers_original_readout_exactly():
    net=PriorNormalizedSlotSurvival(6,6,width=16,slots=4,common_dim=4,private_dim=4)
    with torch.no_grad():
        for prior in net.prior:prior.weight.zero_();prior.bias.zero_()
    out=net(torch.randn(8,12,6),torch.randn(8,12,6),sample=False)
    assert all(torch.equal(a,b) for a,b in zip(out['private'],out['raw_private']))
