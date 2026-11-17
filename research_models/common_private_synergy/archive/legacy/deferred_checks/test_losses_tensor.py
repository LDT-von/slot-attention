"""Deferred tensor acceptance cases; NOT executed in the preparation stage.

This file executes no neural network but does use PyTorch tensors/autograd.
Run only in a later explicitly authorized tensor-validation stage. The current
delivery has AST checks only; these tests must not be counted as passed.
"""
import math

import pytest
import torch

from research_models.common_private_synergy.components.losses import (
    CorrelationPolicy, DeCURWeights, decur_losses, discrete_hazard_nll, variance_floor,
)


def test_nll_completed_interval_convention():
    # All hazard probabilities .5: event in bin 1 -> 2 log 2; censor after
    # two complete bins -> 2 log 2; before first full bin -> 0; horizon -> 3 log2.
    logits = torch.zeros(4, 3, dtype=torch.float64)
    n = torch.tensor([1, 2, 0, 3], dtype=torch.int64)
    event = torch.tensor([True, False, False, False])
    actual = discrete_hazard_nll(logits, n, event, reduction='none')
    expected = torch.tensor([2, 2, 0, 3], dtype=torch.float64) * math.log(2)
    torch.testing.assert_close(actual, expected)


def test_nll_labels_are_not_silently_coerced():
    logits = torch.zeros(1, 2)
    with pytest.raises(ValueError, match='int64'):
        discrete_hazard_nll(logits, torch.tensor([0.0]), torch.tensor([True]))
    with pytest.raises(ValueError, match='horizon'):
        discrete_hazard_nll(logits, torch.tensor([2]), torch.tensor([True]))


def test_disabled_terms_inspect_nothing():
    bundle = decur_losses({}, {}, patient_keys=(), weights=DeCURWeights(),
                          policy=CorrelationPolicy(8, 4))
    assert bundle.weighted_total is None
    assert bundle.terms == {} and len(bundle.skipped) == 4


def test_constant_features_produce_diagnostic_not_exception():
    view = {'common_wsi': torch.ones(8, 2), 'common_omic': torch.ones(8, 2)}
    bundle = decur_losses(view, view, patient_keys=tuple(range(8)),
                          weights=DeCURWeights(cross_common=1), policy=CorrelationPolicy(8, 4))
    torch.testing.assert_close(bundle.terms['cross_common'], torch.tensor(1.0))
    assert bundle.diagnostics['cross_common_view_0']['fraction_below_eps_x'] == 1


def test_joint_rank_guard_is_stricter_than_each_matrix():
    # Each intra dimension is 6 <= B-1=7, but dc+2dp=9 > 7.
    view = {name: torch.ones(8, 3) for name in
            ('common_wsi', 'private_wsi', 'common_omic', 'private_omic')}
    with pytest.raises(ValueError, match='joint'):
        decur_losses(view, view, patient_keys=tuple(range(8)),
                     weights=DeCURWeights(1, 1, 1, 1), policy=CorrelationPolicy(8, 6))


def test_patient_keys_cannot_duplicate_views_as_independent_patients():
    view = {'common_wsi': torch.ones(8, 2), 'common_omic': torch.ones(8, 2)}
    with pytest.raises(ValueError, match='distinct patient'):
        decur_losses(view, view, patient_keys=(0,) * 8,
                     weights=DeCURWeights(cross_common=1), policy=CorrelationPolicy(8, 4))


def test_orthogonal_known_answer_and_normalization():
    # Three independent +/-1 factors on all eight states. All variances one,
    # off diagonal zero; finite epsilon shrinks each self-correlation to 1/(1+eps).
    common = torch.tensor([[a] for a in (-1.0, 1.0) for b in (-1.0, 1.0) for c in (-1.0, 1.0)])
    pw = torch.tensor([[b] for a in (-1.0, 1.0) for b in (-1.0, 1.0) for c in (-1.0, 1.0)])
    po = torch.tensor([[c] for a in (-1.0, 1.0) for b in (-1.0, 1.0) for c in (-1.0, 1.0)])
    view = {'common_wsi': common, 'common_omic': common, 'private_wsi': pw, 'private_omic': po}
    eps = .01
    mean = decur_losses(view, view, patient_keys=tuple(range(8)), weights=DeCURWeights(1, 1, 1, 1),
                        policy=CorrelationPolicy(8, 2, eps=eps, normalization='mean'))
    summed = decur_losses(view, view, patient_keys=tuple(range(8)), weights=DeCURWeights(1, 1, 1, 1),
                          policy=CorrelationPolicy(8, 2, eps=eps, normalization='sum'))
    expected = torch.tensor((eps / (1 + eps)) ** 2)
    torch.testing.assert_close(mean.terms['cross_common'], expected, atol=1e-8, rtol=1e-4)
    torch.testing.assert_close(mean.terms['cross_private'], torch.tensor(0.0))
    torch.testing.assert_close(summed.terms['intra_wsi'], 2 * mean.terms['intra_wsi'])


def test_variance_floor_exact_collapse_has_zero_gradient():
    # A deliberate negative control, not proof that anti-collapse succeeds.
    features = torch.ones(8, 2, requires_grad=True)
    loss, diagnostics = variance_floor(features, patient_keys=tuple(range(8)), target_std=1.0)
    grad, = torch.autograd.grad(loss, features)
    assert loss > 0
    assert torch.equal(grad, torch.zeros_like(features))
    assert torch.equal(diagnostics['variance'], torch.zeros(2))
