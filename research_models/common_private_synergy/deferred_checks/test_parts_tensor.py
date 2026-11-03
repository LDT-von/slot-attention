"""Prepared only, NOT EXECUTED. Requires explicit future tensor-test authorization.

Do not discover this directory from checks/run_static.py's unittest runner.
"""
import pytest
import torch
from research_models.common_private_synergy.components.contracts import ArchitectureConfig
from research_models.common_private_synergy.components.slots import RoleSlotEncoder, AllSlotPool
from research_models.common_private_synergy.components.readout import AdditiveInteractionReadout


def test_real_mask_and_attention_contract():
    torch.manual_seed(27)
    config = ArchitectureConfig(3, 3, width=4)
    encoder = RoleSlotEncoder(3, config)
    x = torch.randn(2, 5, 3)
    mask = torch.tensor([[True, True, False, False, False], [True]*5])
    out = encoder(x, mask)
    contaminated = x.clone()
    contaminated[~mask] = float('nan')
    replay = encoder(contaminated, mask)
    assert torch.equal(out['slots'], replay['slots'])
    assert out['common_slots'].shape == (2, 7, 4)
    assert out['private_slots'].shape == (2, 1, 4)
    assert torch.all(out['read_attention'][0, :, 2:] == 0)
    assert torch.allclose(out['read_attention'].sum(-1), torch.ones(2, 8), atol=1e-6)
    with pytest.raises(ValueError):
        encoder(x, torch.zeros_like(mask))


@pytest.mark.parametrize('count', [0, 8])
def test_empty_role_is_inactive(count):
    torch.manual_seed(28)
    config = ArchitectureConfig(3, 3, width=4, common_slots=count, use_interaction=True)
    readout = AdditiveInteractionReadout(config)
    x = [torch.randn(3, 4) for _ in range(4)]
    out = readout(*x)
    modified = [a.clone() for a in x]
    inactive = (0, 2) if count == 0 else (1, 3)
    for index in inactive:
        modified[index] += 19
    replay = readout(*modified)
    assert torch.equal(out['logits'], replay['logits'])
    if count == 0:
        assert out['common_wsi_logits'] is None
        assert not out['common_active']
        assert torch.all(out['g_diagnostic_only'] == 0)


def test_pool_permutation_and_all_slot_gradient_paths():
    torch.manual_seed(29)
    pool = AllSlotPool(4)
    x = torch.randn(2, 7, 4, requires_grad=True)
    out = pool(x)
    assert torch.allclose(out, pool(x[:, [3, 0, 6, 2, 1, 5, 4]]), atol=1e-6)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.all(x.grad.abs().sum(-1) > 0)
    assert torch.all(pool(torch.empty(2, 0, 4)) == 0)


def test_additive_mixed_difference_and_disabled_interaction():
    torch.manual_seed(30)
    config = ArchitectureConfig(3, 3, width=4)
    model = AdditiveInteractionReadout(config)
    w0, w1, o0, o1 = [[torch.randn(3, 4), torch.randn(3, 4)] for _ in range(4)]
    def f(w, o):
        out = model(w[0], w[1], o[0], o[1])
        assert torch.all(out['interaction_logits'] == 0)
        return out['logits']
    diff = f(w1, o1) - f(w1, o0) - f(w0, o1) + f(w0, o0)
    assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-6)
    f(w1, o1).sum().backward()
    assert model.left.weight.grad is None and model.cross_out.weight.grad is None


def test_state_dict_replay():
    torch.manual_seed(31)
    config = ArchitectureConfig(3, 3, width=4, use_interaction=True)
    a, b = AdditiveInteractionReadout(config), AdditiveInteractionReadout(config)
    b.load_state_dict(a.state_dict())
    x = [torch.randn(3, 4) for _ in range(4)]
    assert torch.equal(a(*x)['logits'], b(*x)['logits'])
