"""Generated tensor checks for the new losses; no patient data or models."""
import math

import pytest
import torch

from slotspe_proof.slot_diversity_losses import (
    slot_cosine_margin_loss, slot_read_overlap_loss, warmup_factor,
)


def test_scm_known_cosine_margin_and_no_antipodal_penalty():
    cosine = 0.95
    slots = torch.tensor([[[1., 0.], [cosine, math.sqrt(1 - cosine ** 2)]]],
                         dtype=torch.float64, requires_grad=True)
    loss, stats = slot_cosine_margin_loss(slots)
    assert loss.item() == pytest.approx(0.25)
    assert stats["mean_pair_cosine"] == pytest.approx(cosine)
    loss.backward()
    assert torch.isfinite(slots.grad).all() and slots.grad.abs().sum() > 0
    antipodal = torch.tensor([[[1., 0.], [-1., 0.]]])
    assert slot_cosine_margin_loss(antipodal)[0].item() == 0


def test_scm_scale_permutation_and_batch_duplication_invariance():
    slots = torch.tensor([[[1., .1], [1., .2], [.7, .3]]], dtype=torch.float64)
    value, _ = slot_cosine_margin_loss(slots)
    scaled = slots * torch.tensor([1e-15, 4., 1e15], dtype=torch.float64)[None, :, None]
    assert slot_cosine_margin_loss(scaled)[0].item() == pytest.approx(value.item())
    assert slot_cosine_margin_loss(slots[:, [2, 0, 1]])[0].item() == pytest.approx(value.item())
    assert slot_cosine_margin_loss(slots.repeat(4, 1, 1))[0].item() == pytest.approx(value.item())


def test_scm_one_outlier_does_not_hide_duplicate_pairs():
    slots = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]])
    loss, _ = slot_cosine_margin_loss(slots)
    assert loss.item() == pytest.approx(1 / 3)


def test_scm_exact_collapse_has_positive_loss_but_zero_gradient():
    slots = torch.tensor([[[1., 0.], [1., 0.], [1., 0.]]], requires_grad=True)
    loss, _ = slot_cosine_margin_loss(slots)
    assert loss.item() == pytest.approx(1.)
    loss.backward()
    assert torch.equal(slots.grad, torch.zeros_like(slots))


def test_scm_single_slot_graph_zero_and_low_precision_promotion():
    slots = torch.ones(2, 1, 3, dtype=torch.float16, requires_grad=True)
    loss, stats = slot_cosine_margin_loss(slots)
    assert loss.dtype == torch.float32 and loss.item() == 0
    assert stats["pair_statistics_defined"] is False
    loss.backward()
    assert torch.equal(slots.grad, torch.zeros_like(slots))


def test_sro_known_bhattacharyya_and_asymmetric_gradient():
    attention = torch.tensor([[[.8, .2], [.7, .3]]], dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss, stats = slot_read_overlap_loss(attention, mask, eps=1e-12)
    expected = math.sqrt(.8 * .7) + math.sqrt(.2 * .3)
    assert stats["mean_pair_overlap"] == pytest.approx(expected)
    assert loss.item() == pytest.approx(((expected - .9) / .1) ** 2)
    loss.backward()
    assert torch.isfinite(attention.grad).all() and attention.grad.abs().sum() > 0


def test_sro_padding_is_excluded_and_has_zero_gradient():
    attention = torch.tensor([[[.8, 99., .2], [.7, 3., .3]]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False, True]])
    before = attention.detach().clone()
    loss, stats = slot_read_overlap_loss(attention, mask)
    expected = slot_read_overlap_loss(attention[:, :, [0, 2]], torch.ones(1, 2, dtype=torch.bool))[0]
    torch.testing.assert_close(loss, expected)
    assert stats["min_valid_token_count"] == 2
    loss.backward()
    assert torch.equal(attention.grad[:, :, 1], torch.zeros_like(attention.grad[:, :, 1]))
    assert attention.grad[:, :, [0, 2]].abs().sum() > 0
    assert torch.equal(before, attention.detach())


def test_sro_heads_are_averaged_before_pair_comparison():
    attention = torch.tensor([[[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]]])
    loss, stats = slot_read_overlap_loss(attention, torch.ones(1, 2, dtype=torch.bool))
    assert loss.item() == pytest.approx(1., abs=3e-6)
    assert stats["mean_pair_overlap"] == pytest.approx(1., abs=3e-7)


def test_sro_exact_uniform_collapse_has_zero_gradient():
    attention = torch.full((1, 2, 4, 8), 1 / 8, dtype=torch.float64, requires_grad=True)
    loss, _ = slot_read_overlap_loss(attention, torch.ones(1, 8, dtype=torch.bool))
    assert loss.item() == pytest.approx(1.)
    loss.backward()
    assert torch.allclose(attention.grad, torch.zeros_like(attention.grad), atol=1e-12)


def test_sro_disjoint_readings_and_unavoidable_single_token_case():
    attention = torch.tensor([[[1., 0.], [0., 1.]]], requires_grad=True)
    loss, _ = slot_read_overlap_loss(attention, torch.ones(1, 2, dtype=torch.bool))
    assert loss.item() == 0
    mixed = torch.tensor([[[.6, .4], [.5, .5]], [[.3, .7], [.9, .1]]], requires_grad=True)
    mixed_mask = torch.tensor([[True, True], [True, False]])
    full_loss, stats = slot_read_overlap_loss(mixed, mixed_mask)
    first_loss, _ = slot_read_overlap_loss(mixed[:1], mixed_mask[:1])
    assert full_loss.item() == pytest.approx(first_loss.item() / 2)
    assert stats["eligible_patient_count"] == 1 and stats["skipped_no_read_diversity_count"] == 1
    full_loss.backward()
    assert torch.equal(mixed.grad[1], torch.zeros_like(mixed.grad[1]))


def test_sro_single_slot_is_graph_connected_zero():
    attention = torch.full((1, 1, 3), 1 / 3, requires_grad=True)
    loss, stats = slot_read_overlap_loss(attention, torch.ones(1, 3, dtype=torch.bool))
    assert loss.item() == 0 and stats["pair_statistics_defined"] is False
    loss.backward()
    assert torch.equal(attention.grad, torch.zeros_like(attention.grad))


def test_sro_permutation_batch_invariance_and_no_tensor_diagnostics():
    attention = torch.tensor([[[.4, .4, .2], [.3, .5, .2], [.3, .4, .3]]], dtype=torch.float64)
    mask = torch.ones(1, 3, dtype=torch.bool)
    loss, stats = slot_read_overlap_loss(attention, mask)
    permuted = attention[:, [2, 0, 1]][:, :, [1, 2, 0]]
    assert slot_read_overlap_loss(permuted, mask)[0].item() == pytest.approx(loss.item())
    assert slot_read_overlap_loss(attention.repeat(3, 1, 1), mask.repeat(3, 1))[0].item() == pytest.approx(loss.item())
    assert all(not torch.is_tensor(value) for value in stats.values())


@pytest.mark.parametrize("margin", [-.1, 1., float("nan"), float("inf"), True])
def test_both_losses_reject_invalid_margins(margin):
    with pytest.raises(ValueError, match="margin"):
        slot_cosine_margin_loss(torch.ones(1, 2, 3), margin)
    with pytest.raises(ValueError, match="margin"):
        slot_read_overlap_loss(torch.ones(1, 2, 3), torch.ones(1, 3, dtype=torch.bool), margin)


@pytest.mark.parametrize("slots", [torch.zeros(1, 2, 3), torch.ones(0, 2, 3),
                                  torch.ones(2, 3), torch.ones(1, 2, 3, dtype=torch.int64),
                                  torch.full((1, 2, 3), float("nan"))])
def test_scm_rejects_undefined_directions_and_invalid_shapes(slots):
    with pytest.raises(ValueError):
        slot_cosine_margin_loss(slots)


@pytest.mark.parametrize("attention, mask", [
    (torch.ones(1, 2, 3), torch.zeros(1, 3, dtype=torch.bool)),
    (torch.zeros(1, 2, 3), torch.ones(1, 3, dtype=torch.bool)),
    (torch.full((1, 2, 3), -1.), torch.ones(1, 3, dtype=torch.bool)),
    (torch.full((1, 2, 3), float("inf")), torch.ones(1, 3, dtype=torch.bool)),
    (torch.ones(1, 2, 3), torch.ones(1, 3)),
    (torch.ones(1, 2, 3), torch.ones(2, 3, dtype=torch.bool)),
])
def test_sro_rejects_invalid_readings_or_masks(attention, mask):
    with pytest.raises(ValueError):
        slot_read_overlap_loss(attention, mask)


@pytest.mark.parametrize("eps", [0, -1, float("nan"), True])
def test_sro_rejects_invalid_smoothing(eps):
    with pytest.raises(ValueError, match="eps"):
        slot_read_overlap_loss(torch.ones(1, 2, 3), torch.ones(1, 3, dtype=torch.bool), eps=eps)


def test_warmup_is_one_based_and_only_changes_new_loss_weight():
    assert [warmup_factor(epoch) for epoch in range(1, 8)] == pytest.approx([.2, .4, .6, .8, 1., 1., 1.])
    assert warmup_factor(1, 0) == 1
    for epoch, length in [(0, 5), (True, 5), (1., 5), (1, -1), (1, True)]:
        with pytest.raises(ValueError):
            warmup_factor(epoch, length)
