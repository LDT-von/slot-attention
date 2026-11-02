import math
import pytest
import torch
from slotspe_proof.xor_readout_diagnostics import oracle_inputs, oracle_bayes_logits, mixture_survival_logits, configure_readout_training
from slotspe_proof.crossmodal_shared_bottleneck import CrossmodalSharedBottleneck
from slotspe_proof.common_private_survival import gaussian_kl, survival_nll


def test_gaussian_oracle_rate_is_one_only_in_active_slots():
    a = math.sqrt(1-math.exp(-2)); mu = torch.zeros(8, 4, 4); lv = torch.zeros_like(mu)
    mu[:, 0, 0] = a; lv[:, 0, 0] = -2
    r = gaussian_kl(mu, lv, torch.zeros_like(mu), torch.zeros_like(lv)).sum(-1)
    assert torch.allclose(r[:, 0], torch.ones(8)) and torch.equal(r[:, 1:], torch.zeros(8, 3))


def test_clean_oracle_bayes_logits_equal_exact_xor_risk():
    bits = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    shared = torch.tensor([-.5, 0., .5, 1.])
    g, v = oracle_inputs(shared, bits, [torch.zeros(4, 4, 4)]*2, 'before_phi_clean_single')
    logits = oracle_bayes_logits(g, v, 'before_phi_clean_single')
    expected = (3*(bits[:, 0] != bits[:, 1]).float()-1.5+.3*shared)[:, None]+torch.tensor([-2., -1., 0., 1.])
    assert torch.allclose(logits, expected.double(), atol=2e-7)


def test_mixture_censored_likelihood_matches_direct_distribution_mixture():
    shared = torch.tensor([.7], dtype=torch.double); p = .37
    mixture = mixture_survival_logits(shared, torch.tensor([p]))
    components = [.3*shared[:, None]+s+shared.new_tensor([-2., -1., 0., 1.])[None] for s in [-1.5, 1.5]]
    for time in range(4):
        for event in [0., 1.]:
            t = torch.tensor([time]); e = torch.tensor([event])
            observed = torch.exp(-survival_nll(mixture, t, e))
            reference = (1-p)*torch.exp(-survival_nll(components[0], t, e))+p*torch.exp(-survival_nll(components[1], t, e))
            assert torch.allclose(observed, reference, atol=1e-8)
    naive = .37*torch.sigmoid(components[1])+.63*torch.sigmoid(components[0])
    assert not torch.allclose(torch.sigmoid(mixture)[:, 1:], naive[:, 1:])


def test_readout_configuration_cannot_update_encoders_or_posterior():
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(1)
        net = CrossmodalSharedBottleneck(8, 8, width=16, slots=4, common_dim=4, private_dim=4)
    assert sum(p.numel() for p in configure_readout_training(net)) == 1396
    assert all(not p.requires_grad for name, p in net.named_parameters() if name.startswith(('encoders.', 'common.', 'posterior.', 'prior.')))
    assert sum(p.numel() for p in configure_readout_training(net, head_only=True)) == 660


def test_noisy_oracle_reproducibility_and_layout_guards():
    shared = torch.zeros(8); bits = torch.ones(8, 2); noise = [torch.randn(8, 4, 4) for _ in range(2)]
    a = oracle_inputs(shared, bits, noise, 'before_phi_rate1_repeated')
    b = oracle_inputs(shared, bits, noise, 'before_phi_rate1_repeated')
    assert torch.equal(a[1][0], b[1][0])
    with pytest.raises(ValueError): oracle_inputs(shared, bits*0, noise, 'before_phi_rate1_repeated')
