"""Generated feature checks; these tests never load patient artifacts."""
from __future__ import annotations

import copy

import pytest
import torch

from survot_rank.research.slotspe_proof.redundancy_content import (
    ContentReference,
    coverage_set_mse,
)
from survot_rank.research.slotspe_proof.slot_regularization import matched_set_mse


def _cases(count=5, offset=0.0):
    generator = torch.Generator().manual_seed(83)
    return [{
        "x_wsi": torch.randn(1, 9 + index, 6, generator=generator) + offset,
        "x_omic1": torch.randn(1, 3, generator=generator) + offset,
        "x_omic2": torch.randn(1, 5, generator=generator) + offset,
        "x_omic3": torch.randn(1, 7, generator=generator) + offset,
        # Invalid outcome values must not enter any feature reference operation.
        "y": torch.tensor([float("nan")]), "c": object(),
    } for index in range(count)]


def _reference(**kwargs):
    return ContentReference([3, 5, 7], 6, target_dim=4, clusters=8,
                            kmeans_steps=3, **kwargs)


def _same(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _same(first, second)
    else:
        assert left == right


def test_streaming_training_moments_use_each_heterogeneous_raw_feature():
    cases = _cases()
    reference = _reference().fit(iter(cases))
    raw = [torch.cat([case["x_wsi"][0] for case in cases])]
    raw += [torch.cat([case[f"x_omic{index}"] for case in cases]) for index in (1, 2, 3)]
    assert reference.train_case_count == 5
    assert reference.feature_counts == (55, 5, 5, 5)
    for value, mean, std in zip(raw, reference.means, reference.stds):
        assert torch.allclose(mean, value.double().mean(0), atol=1e-12)
        assert torch.allclose(std, value.double().std(0, correction=0), atol=1e-12)
    assert [tuple(x.shape) for x in reference.projections] == [(6, 4), (3, 4), (5, 4), (7, 4)]
    assert all(1 <= len(x) <= 8 for x in reference.static_targets.values())


def test_targets_are_heterogeneous_input_sets_and_never_refit_on_heldout():
    reference = _reference().fit(_cases())
    before = reference.state_dict()
    case = _cases(1, offset=11.0)[0]
    saved = {key: value.clone() for key, value in case.items() if torch.is_tensor(value)}
    targets = reference.targets(case)
    assert targets["wsi"].shape == (8, 4)
    assert targets["omic"].shape == (3, 4)
    assert all(value.dtype == case["x_wsi"].dtype for value in targets.values())
    assert all(not value.requires_grad for value in targets.values())
    # Three pathways fit into three target tokens; each uses its own raw size.
    projected = torch.cat([
        ((case[f"x_omic{i}"][0].double() - reference.means[i]) / reference.stds[i])
        @ reference.projections[i] for i in (1, 2, 3)
    ]).reshape(3, 4)
    distance = (targets["omic"].double()[:, None] - projected[None]).square().sum(-1)
    assert torch.all(distance.min(-1).values < 1e-10)
    _same(reference.state_dict(), before)
    for key, value in saved.items():
        torch.testing.assert_close(case[key], value, equal_nan=True)
    with pytest.raises(RuntimeError, match="frozen"):
        reference.fit(_cases(1, offset=999.0))


def test_seeded_fit_and_targets_preserve_global_rng_and_inputs():
    cases = _cases()
    before = {key: value.clone() for key, value in cases[0].items() if torch.is_tensor(value)}
    rng = torch.random.get_rng_state().clone()
    first = _reference(seed=731).fit(cases)
    first_target = first.targets(cases[0])
    second = _reference(seed=731).fit(cases)
    second_target = second.targets(cases[0])
    assert torch.equal(torch.random.get_rng_state(), rng)
    _same(first.state_dict(), second.state_dict())
    _same(first_target, second_target)
    for key, value in before.items():
        torch.testing.assert_close(cases[0][key], value, equal_nan=True)
    other_seed = _reference(seed=732).fit(cases)
    assert not torch.equal(first.projections[0], other_seed.projections[0])
    # Fixed projections do not change with the supplied training-fold values.
    changed_fold = _reference(seed=731).fit(_cases(offset=99.0))
    _same(first.projections, changed_fold.projections)
    assert not torch.equal(first.means[0], changed_fold.means[0])


def test_checkpoint_roundtrip_is_weights_only_and_contains_no_raw_features(tmp_path):
    original = _reference().fit(_cases())
    path = tmp_path / "reference.pt"
    torch.save(original.state_dict(), path)
    loaded = torch.load(path, weights_only=True)
    restored = _reference().load_state_dict(loaded)
    _same(original.state_dict(), restored.state_dict())
    _same(original.targets(_cases(1)[0]), restored.targets(_cases(1)[0]))
    assert set(vars(restored)) == {
        "omic_sizes", "encoding_dim", "target_dim", "clusters", "kmeans_steps", "seed",
        "_fitted", "train_case_count", "feature_counts", "means", "stds", "projections",
        "static_targets",
    }
    snapshot = restored.state_dict()
    loaded["means"][0].zero_()
    loaded["static_targets"]["wsi"].zero_()
    _same(restored.state_dict(), snapshot)


@pytest.mark.parametrize("mutation", [
    lambda state: state.update({"extra": 0}),
    lambda state: state.update({"seed": 18}),
    lambda state: state.update({"feature_counts": [1, 1, 1, 1]}),
    lambda state: state["means"].__setitem__(0, torch.zeros(7)),
    lambda state: state["stds"].__setitem__(1, torch.zeros(3)),
    lambda state: state["projections"][0].fill_(float("nan")),
    lambda state: state["static_targets"].update({"omic": torch.zeros(0, 4)}),
])
def test_checkpoint_strict_validation_is_atomic(mutation):
    reference = _reference().fit(_cases())
    before = reference.state_dict()
    bad = copy.deepcopy(before)
    mutation(bad)
    with pytest.raises(ValueError):
        reference.load_state_dict(bad)
    _same(before, reference.state_dict())


def test_equal_count_loss_matches_existing_hungarian_and_target_has_no_gradient():
    generator = torch.Generator().manual_seed(114)
    pred = torch.randn(1, 8, 4, generator=generator, requires_grad=True)
    target = torch.randn(1, 8, 4, generator=generator, requires_grad=True)
    loss = coverage_set_mse(pred, target)
    expected = matched_set_mse(pred, target)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    loss.backward()
    assert pred.grad is not None and bool(torch.isfinite(pred.grad).all())
    assert bool((pred.grad.abs().sum(-1) > 0).all())
    assert target.grad is None


def test_rectangular_loss_covers_missing_target_and_unused_prediction():
    one_slot = torch.tensor([[0.0]], requires_grad=True)
    two_targets = torch.tensor([[0.0], [10.0]], requires_grad=True)
    loss, details = coverage_set_mse(one_slot, two_targets, return_details=True)
    assert float(loss.detach()) == 50.0
    assert details == {
        "prediction_count": 1, "target_count": 2, "matched_count": 1,
        "unmatched_prediction_count": 0, "unmatched_target_count": 1,
    }
    loss.backward()
    assert float(one_slot.grad) == -10.0 and two_targets.grad is None
    extra_slot = torch.tensor([[0.0], [10.0]], requires_grad=True)
    reverse, details = coverage_set_mse(extra_slot, torch.tensor([[0.0]]), True)
    assert float(reverse.detach()) == 50.0
    assert details["unmatched_prediction_count"] == 1
    reverse.backward()
    assert float(extra_slot.grad[1]) == 10.0


def test_set_loss_is_permutation_invariant_and_single_slot_not_automatically_zero():
    pred = torch.tensor([[1.0, 2.0], [9.0, 5.0], [-4.0, 8.0]])
    target = torch.tensor([[2.0, 1.0], [5.0, 9.0]])
    expected = coverage_set_mse(pred, target)
    torch.testing.assert_close(coverage_set_mse(pred[[2, 0, 1]], target.flip(0)), expected)
    assert float(coverage_set_mse(pred[:1], target)) > 0


def test_empty_constant_short_and_batch_inputs_have_explicit_contracts():
    reference = _reference()
    with pytest.raises(ValueError, match="empty training fold"):
        reference.fit(iter(()))
    with pytest.raises(RuntimeError, match="fit"):
        reference.targets(_cases(1)[0])
    batch = {
        "x_wsi": torch.ones(2, 3, 6), "x_omic1": torch.ones(2, 3),
        "x_omic2": torch.ones(2, 5), "x_omic3": torch.ones(2, 7),
    }
    reference.fit([batch])
    assert reference.train_case_count == 2
    assert all(torch.equal(std, torch.full_like(std, 1e-4)) for std in reference.stds)
    with pytest.raises(ValueError, match="batch size one"):
        reference.targets(batch)
    single = {key: value[:1] for key, value in batch.items()}
    assert reference.targets(single)["wsi"].shape == (3, 4)
    assert all(bool(torch.isfinite(value).all()) for value in reference.targets(single).values())


@pytest.mark.parametrize("bad_input", [
    lambda batch: batch.update({"x_wsi": torch.zeros(1, 0, 6)}),
    lambda batch: batch.update({"x_omic2": torch.zeros(1, 4)}),
    lambda batch: batch["x_wsi"].fill_(float("nan")),
    lambda batch: batch.update({"x_omic1": torch.zeros(1, 3, dtype=torch.long)}),
])
def test_invalid_raw_inputs_fail_closed(bad_input):
    batch = _cases(1)[0]
    bad_input(batch)
    with pytest.raises(ValueError):
        _reference().fit([batch])


@pytest.mark.parametrize("pred,target", [
    (torch.zeros(0, 4), torch.zeros(2, 4)),
    (torch.zeros(2, 4), torch.zeros(2, 3)),
    (torch.zeros(2, 2, 4), torch.zeros(2, 4)),
    (torch.tensor([[float("inf")]]), torch.zeros(1, 1)),
])
def test_invalid_set_inputs_are_rejected(pred, target):
    with pytest.raises(ValueError):
        coverage_set_mse(pred, target)
