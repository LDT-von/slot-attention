import math
import pytest
import torch

from slotspe_proof.common_private_survival import gaussian_kl
from slotspe_proof.crossmodal_shared_bottleneck import CrossmodalSharedBottleneck
from slotspe_proof.slot_pooling_diagnostics import gaussian_code, code_inputs, code_bayes_logits, RoleBankReadout
from slotspe_proof.xor_readout_diagnostics import oracle_bayes_logits


def test_equal_total_rate_is_not_equal_per_slot_rate():
    for count in [1, 4]:
        rate = 1/count
        a, sigma = gaussian_code(rate)
        kl = gaussian_kl(torch.tensor(a).double(), torch.tensor(2*math.log(sigma)).double(), torch.tensor(0.), torch.tensor(0.))
        assert float(kl)*count == pytest.approx(1., abs=2e-7)
    assert 4*math.expm1(.5) < math.expm1(2.)


def test_equal_rate_bayes_uses_all_four_noisy_observations():
    bits = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    s = torch.zeros(4); eps = [torch.zeros(4, 4, 4)]*2
    g, v = code_inputs(s, bits, eps, 'single_total1')
    assert torch.allclose(code_bayes_logits(g, v, 'single_total1'), oracle_bayes_logits(g, v, 'before_phi_rate1_single'))
    g, v = code_inputs(s, bits, eps, 'repeated_total1')
    correct = code_bayes_logits(g, v, 'repeated_total1')
    changed = [x.clone() for x in v]; changed[0][:, 1:, 0] *= -1
    assert not torch.allclose(correct, code_bayes_logits(g, changed, 'repeated_total1'))


def test_role_bank_preserves_roles_and_capacity_without_global_rng_effect():
    torch.manual_seed(1)
    reference = CrossmodalSharedBottleneck(8, 8, width=16, slots=4, common_dim=4, private_dim=4)
    state = torch.get_rng_state().clone()
    bank = RoleBankReadout(reference, 19)
    assert torch.equal(state, torch.get_rng_state())
    assert sum(p.numel() for p in bank.parameters()) == 1402
    x = [torch.randn(6, 4, 4), torch.randn(6, 4, 4)]; g = torch.randn(6, 4)
    permutation = torch.tensor([2, 0, 3, 1])
    own = bank(g, x)
    assert torch.allclose(own, bank(g, [v[:, permutation] for v in x], [permutation]*2), atol=1e-7)
    assert not torch.allclose(own, bank(g, [v[:, permutation] for v in x]))
    with pytest.raises(ValueError): bank(g, x, [torch.zeros(4, dtype=torch.long)]*2)


def test_every_role_block_has_a_value_gradient():
    reference = CrossmodalSharedBottleneck(8, 8, width=16, slots=4, common_dim=4, private_dim=4)
    bank = RoleBankReadout(reference, 19)
    x = [torch.randn(5, 4, 4, requires_grad=True) for _ in range(2)]
    bank(torch.zeros(5, 4), x).square().sum().backward()
    assert all(bool((v.grad.abs().sum((0, 2)) > 0).all()) for v in x)
