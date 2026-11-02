import pytest
import torch
from slotspe_proof.crossmodal_shared_bottleneck import CrossmodalSharedBottleneck, shared_private_objective
from slotspe_proof.common_private_survival import gaussian_kl, survival_nll


@pytest.fixture(autouse=True)
def cpu():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]): yield
    torch.set_num_threads(old)


def net():
    return CrossmodalSharedBottleneck(8, 8, width=16, slots=4, common_dim=4, private_dim=4)


def noise(n=12):
    return dict(shared_noise=[torch.randn(n, 4) for _ in range(2)],
                private_noise=[torch.randn(n, 4, 4) for _ in range(2)])


def test_own_modality_posterior_is_invariant_to_other_modality_change():
    model = net(); a, b = torch.randn(12, 12, 8), torch.randn(12, 12, 8)
    eps = noise(); first = model(a, b, **eps); second = model(a, b+3, **eps)
    assert torch.equal(first['shared_mean'][0], second['shared_mean'][0])
    assert torch.equal(first['shared_logvar'][0], second['shared_logvar'][0])
    assert not torch.equal(first['shared_mean'][1], second['shared_mean'][1])


def test_readout_uses_actual_shared_samples_with_no_mean_bypass():
    model = net(); a, b = torch.randn(12, 12, 8), torch.randn(12, 12, 8)
    eps = noise(); out = model(a, b, **eps)
    assert torch.equal(out['common'], (out['shared_sample'][0]+out['shared_sample'][1])/2)
    assert torch.equal(out['logits'], model.readout(out['common'], out['private']))
    assert not torch.allclose(out['common'], out['common_mean'])
    changed = model(a, b, shared_noise=[x+2 for x in eps['shared_noise']], private_noise=eps['private_noise'])
    assert not torch.allclose(out['logits'], changed['logits'])


def test_full_noise_controls_are_reproducible_without_rng_consumption():
    model = net(); a, b = torch.randn(12, 12, 8), torch.randn(12, 12, 8); eps = noise()
    before = torch.random.get_rng_state().clone()
    first, second = model(a, b, **eps), model(a, b, **eps)
    assert torch.equal(first['logits'], second['logits'])
    assert torch.equal(before, torch.random.get_rng_state())


def test_private_rate_cannot_directly_update_shared_heads():
    model = net(); out = model(torch.randn(12, 12, 8), torch.randn(12, 12, 8), **noise())
    sum(r.mean() for r in out['private_rate_per_slot']).backward()
    assert all(p.grad is None for p in model.common.parameters())
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.encoders.parameters())


def test_survival_gradient_reaches_shared_variance_and_private_heads():
    model = net(); out = model(torch.randn(12, 12, 8), torch.randn(12, 12, 8), **noise())
    survival_nll(out['logits'], torch.arange(12)%4, torch.ones(12)).backward()
    for head in model.common:
        assert head.weight.grad[4:].abs().sum()>0
    assert all(p.grad is not None for p in model.posterior.parameters())


def test_shared_kl_has_correct_orientation_and_two_sided_gradient():
    model = net(); out = model(torch.randn(12, 12, 8), torch.randn(12, 12, 8), **noise())
    expected = gaussian_kl(out['shared_mean'][0], out['shared_logvar'][0], out['shared_mean'][1], out['shared_logvar'][1]).sum(-1)
    assert torch.equal(expected, out['shared_rate_per_direction'][0])
    out['shared_rate'].mean().backward()
    assert all(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.common.parameters())


def test_normalized_private_rate_matches_raw_conditional_kl():
    model = net(); out = model(torch.randn(12, 12, 8), torch.randn(12, 12, 8), **noise())
    for j, s in enumerate(out['slots']):
        g = out['common'].detach()
        mu, lv = model.posterior[j](torch.cat([s, g[:, None].expand(-1, 4, -1)], -1)).chunk(2, -1)
        pm, plv = model.prior[j](g).unsqueeze(1).chunk(2, -1)
        raw = gaussian_kl(mu, lv.clamp(-8, 4), pm, plv.clamp(-8, 4)).sum(-1)
        assert torch.allclose(raw, out['private_rate_per_slot'][j], atol=2e-6, rtol=1e-5)


def test_mean_diagnostic_cannot_silently_train_rate_objective():
    model = net(); a, b = torch.randn(12, 12, 8), torch.randn(12, 12, 8)
    out = model(a, b, sample=False)
    with pytest.raises(ValueError, match='sampled'): shared_private_objective(out, torch.arange(12)%4, torch.ones(12))
    with pytest.raises(ValueError, match='ignore'): model(a, b, sample=False, **noise())


def test_pair_permutation_invariance_and_no_slot_dropping():
    model = net(); g = torch.randn(12, 4); vs = [torch.randn(12, 4, 4) for _ in range(2)]
    with torch.no_grad():
        for e in model.role_offsets: e.normal_()
    perm = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(model.readout(g, vs), model.readout(g, [v[:, perm] for v in vs], [perm, perm]), atol=1e-6)
    with pytest.raises(ValueError): model.readout(g, [v[:, :2] for v in vs])


def test_eval_remains_stochastic_and_noise_shapes_rejected():
    model = net().eval(); a, b = torch.randn(12, 12, 8), torch.randn(12, 12, 8)
    assert not torch.equal(model(a, b)['logits'], model(a, b)['logits'])
    with pytest.raises(ValueError, match='noise shape'): model(a, b, shared_noise=[torch.zeros(12, 3)]*2)
