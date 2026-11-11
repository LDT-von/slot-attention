import pytest
import torch
from torch.nn import functional as F

from research_models.common_private_synergy.components.slots import READ_EPS, masked_slot_read
from research_models.common_private_synergy.full_slot_diagnostics import read_mass_summary, slot_summary


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('gap', [38., 1000.])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_starved_read_is_normalized_masked_differentiable_and_visible(dtype, gap, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; server CUDA environment must exercise this case')
    logits = torch.zeros(2, 4, 7, dtype=dtype, device=device)
    logits[:, -1] = -gap
    logits.requires_grad_(True)
    mask = torch.tensor([[True, True, True, False, False, False, False], [True] * 7], device=device)
    attention, mass = masked_slot_read(logits, mask)
    assert (mass[:, -1] < READ_EPS).all()
    old_assignment = logits.softmax(1) * mask[:, None, :]
    old_read = old_assignment / old_assignment.sum(-1, keepdim=True).clamp_min(READ_EPS)
    assert (old_read[:, -1].sum(-1) < 1e-5).all()
    with pytest.raises(ValueError, match='normalized'):
        slot_summary(torch.ones(2, 4, 3, dtype=dtype, device=device), old_read)
    assert torch.allclose(attention.sum(-1), torch.ones(2, 4, dtype=dtype, device=device))
    assert torch.count_nonzero(attention.masked_select(~mask[:, None, :].expand_as(attention))) == 0
    slot_summary(torch.ones(2, 4, 3, dtype=dtype, device=device), attention)
    reading = read_mass_summary(mass[:, None, :])
    assert reading['below_threshold_count'] == 2 and reading['observations'] == 8
    assert reading['below_threshold_count_by_slot'] == [0, 0, 0, 2]
    assert reading['below_threshold_count_by_iteration'] == [2]
    loss = (attention * torch.arange(7, dtype=dtype, device=device)).sum()
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad.masked_select(~mask[:, None, :].expand_as(logits))) == 0


def test_all_real_matches_original_add_epsilon_l1_formula():
    logits = torch.randn(2, 4, 11, generator=torch.Generator().manual_seed(42), dtype=torch.float64)
    attention, _ = masked_slot_read(logits, torch.ones(2, 11, dtype=torch.bool))
    expected = F.normalize(logits.softmax(1) + READ_EPS, p=1, dim=-1)
    # norm(p=1) and sum may differ by reduction roundoff; compare at float64
    # precision rather than requiring identical reduction implementations.
    torch.testing.assert_close(attention, expected, atol=1e-15, rtol=1e-15)


def test_padding_has_no_epsilon_mass_and_empty_or_nonfinite_bags_rejected():
    logits = torch.randn(1, 4, 6, dtype=torch.float64)
    mask = torch.tensor([[True, True, False, False, False, False]])
    first, _ = masked_slot_read(logits, mask)
    changed = logits.clone(); changed[:, :, 2:] = 1e100
    second, _ = masked_slot_read(changed, mask)
    assert torch.equal(first, second)
    with pytest.raises(ValueError):
        masked_slot_read(logits, torch.zeros_like(mask))
    logits[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        masked_slot_read(logits, mask)
