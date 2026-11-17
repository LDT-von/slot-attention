import pytest
import torch

from research_models.common_private_synergy.model import RoleSlotSurvival, decur_style_losses, _zero_loss


def test_shapes_all_slots_and_gradient():
    torch.manual_seed(13)
    model = RoleSlotSurvival(5, 3, width=8, bins=4)
    a, b = torch.randn(5, 6, 5), torch.randn(5, 4, 3)
    out = model(a, b)
    assert out['logits'].shape == (5, 4)
    assert [tuple(s.shape) for s in out['slots']] == [(5, 8, 8), (5, 8, 8)]
    out['logits'].sum().backward()
    for encoder in model.encoders:
        assert torch.all(encoder.initial.grad.abs().sum(dim=1) > 0)


def test_views_losses_and_backward():
    torch.manual_seed(14)
    model = RoleSlotSurvival(5, 3, width=6)
    inputs = (torch.randn(6, 7, 5), torch.randn(6, 4, 3))
    view1 = model(*inputs)
    view2 = model(*(x + torch.randn_like(x) * .05 for x in inputs))
    losses = decur_style_losses(view1, view2)
    assert set(losses) == {'intra_wsi', 'intra_omic', 'cross_common', 'cross_private'}
    assert all(torch.isfinite(v) and v >= 0 for v in losses.values())
    sum(losses.values()).backward()
    assert all(m.initial.grad is not None and torch.isfinite(m.initial.grad).all() for m in model.encoders)


def test_correlation_rejects_single_or_constant_patient_batch():
    def view(n):
        return {'common': [torch.ones(n, 3), torch.ones(n, 3)],
                'private': [torch.ones(n, 3), torch.ones(n, 3)]}
    with pytest.raises(ValueError, match='at least 2'):
        decur_style_losses(view(1), view(1))
    with pytest.raises(ValueError, match='variance'):
        decur_style_losses(view(3), view(3))


def test_masked_padding_and_invalid_pair():
    torch.manual_seed(15)
    model = RoleSlotSurvival(5, 3, width=4)
    a, b = torch.randn(2, 3, 5), torch.randn(2, 2, 3)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    modified = a.clone()
    modified[~mask] = 10000
    assert torch.equal(model(a, b, mask)['logits'], model(modified, b, mask)['logits'])
    with pytest.raises(ValueError, match='paired'):
        model(a, b[:1])


def test_cross_private_diagonal_and_offdiagonal_weights():
    c = torch.tensor([[1., 2.], [3., 4.]])
    assert torch.isclose(_zero_loss(c, 0.0), torch.tensor(8.5))
    assert torch.isclose(_zero_loss(c, 0.5), torch.tensor(11.75))
