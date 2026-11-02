import pytest
import torch
from slotspe_proof.private_rate_fee import private_rate_objective
from slotspe_proof.crossmodal_shared_bottleneck import CrossmodalSharedBottleneck, shared_private_objective
from slotspe_proof.multifactor_survival_synthetic import generate, noise, standardize, observed_bin_concordance


def fake(rate):
    return {'sampling_mode': 'stochastic_single_draw', 'logits': torch.zeros(2, 4),
            'cross': torch.tensor(0.), 'shared_rate': torch.zeros(2),
            'private_rate_per_slot': [rate]}


@pytest.mark.parametrize('eta', [0., .1, 1.])
def test_piecewise_fee_has_prescribed_below_and_above_gradients(eta):
    rate = torch.tensor([[.5, 2.], [.5, 2.]], requires_grad=True)
    result = private_rate_objective(fake(rate), torch.zeros(2, dtype=torch.long), torch.ones(2), eta=eta)
    result['private_fee'].backward()
    assert torch.allclose(result['private_fee'], torch.tensor(eta*1.25+(1-eta)*.5))
    assert torch.allclose(rate.grad, torch.tensor([[eta/4, .25], [eta/4, .25]]))


def test_eta_zero_preserves_loss_and_all_model_gradients_exactly():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(653)
        model = CrossmodalSharedBottleneck(8, 8, width=16, slots=2, common_dim=4, private_dim=4)
    pack = generate(810653, 8, 'factorwise'); eps = noise(torch.Generator().manual_seed(7), 8)
    out = model(*pack['xs'], **eps)
    a = shared_private_objective(out, pack['time'], pack['event'])['loss']
    b = private_rate_objective(out, pack['time'], pack['event'], eta=0)['loss']
    assert a is b or torch.equal(a, b)
    ga = torch.autograd.grad(a, tuple(model.parameters()), retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(b, tuple(model.parameters()), allow_unused=True)
    assert all((x is None and y is None) or (x is not None and y is not None and torch.equal(x, y)) for x, y in zip(ga, gb))


def test_fee_gradient_routes_and_argument_guards():
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(1)
        model = CrossmodalSharedBottleneck(8, 8, width=16, slots=2, common_dim=4, private_dim=4)
    pack = generate(1, 8, 'factorwise')
    out = model(*pack['xs'], **noise(torch.Generator().manual_seed(1), 8))
    terms = private_rate_objective(out, pack['time'], pack['event'], eta=.1)
    terms['private_fee'].backward()
    assert all(p.grad is None for p in model.common.parameters())
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.encoders.parameters())
    for eta in [-.1, 1.1, float('nan')]:
        with pytest.raises(ValueError): private_rate_objective(out, pack['time'], pack['event'], eta=eta)
    mean = model(*pack['xs'], sample=False)
    with pytest.raises(ValueError): private_rate_objective(mean, pack['time'], pack['event'])


def test_generated_bags_are_observable_and_tasks_share_inputs_without_rng_effect():
    before = torch.get_rng_state().clone()
    a, b = generate(81, 32, 'factorwise'), generate(81, 32, 'paired_xor')
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(x, y) for x, y in zip(a['xs'], b['xs']))
    assert not torch.equal(a['true_logits'], b['true_logits'])
    for j, x in enumerate(a['xs']):
        assert torch.equal(x[:, :, :2].sum(1), torch.full((32, 2), 8.))
        role = x[:, :, :2].argmax(-1)
        measured = a['factors'][:, 2*j:2*j+2].gather(1, role)
        assert bool((x[:, :, 3]-measured).abs().max() < .3)
    assert bool(((a['time'] >= 0) & (a['time'] < 4)).all())


def test_normalization_fits_training_only_and_keeps_roles():
    train, held = generate(1, 32, 'factorwise'), generate(2, 12, 'factorwise')
    shifted = {**held, 'xs': [x+torch.tensor([0., 0., 100., 100., 100., 100., 100., 100.]) for x in held['xs']]}
    a, stats_a = standardize(train, [held]); b, stats_b = standardize(train, [shifted])
    assert stats_a == stats_b
    assert all(torch.equal(x, y) for x, y in zip(a[0]['xs'], b[0]['xs']))
    for raw, x in zip(train['xs'], a[0]['xs']):
        assert torch.equal(raw[:, :, :2], x[:, :, :2])
        assert torch.allclose(x[:, :, 2:].mean((0, 1)), torch.zeros(6), atol=1e-6)


def test_generated_bin_concordance_order_ties_and_survival_through_censor_bin():
    time = torch.tensor([0, 1, 2]); event = torch.ones(3)
    logits = torch.tensor([[2.]*4, [0.]*4, [-2.]*4])
    assert observed_bin_concordance(logits, time, event) == 1.
    assert observed_bin_concordance(-logits, time, event) == 0.
    assert observed_bin_concordance(torch.zeros_like(logits), time, event) == .5
    assert observed_bin_concordance(logits[:2], torch.zeros(2), torch.tensor([1., 0.])) == 1.
    assert observed_bin_concordance(logits, time, torch.zeros(3)) is None
