"""Tensor-free contracts for the opt-in capacity training protocol."""

import copy
import math

import pytest

from survot_rank.research.slotspe_proof.problem_capacity import (
    capacity_plan,
    validate_matched_state_contract,
    _epoch_orders,
    _step_seed,
)


def test_capacity_factors_keep_selection_and_omic_budget_fixed():
    plan = capacity_plan({})
    assert plan["capacity_counts"] == [4, 8, 16]
    assert plan["selection"] == {"wsi_k": 2, "omic_k": 2, "omic_num_slots": 8}
    assert plan["initialization"] == "same_new_initial_state"
    assert plan["checkpoint_rule"] == "final_epoch_only_no_best_selection"


@pytest.mark.parametrize("counts", [[1, 8], [4, 4], [8], [True, 8], [4.0, 8], "4,8"])
def test_capacity_plan_rejects_ambiguous_or_impossible_factors(counts):
    with pytest.raises(ValueError):
        capacity_plan({"capacity_counts": counts})


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, True, "0.001"])
def test_capacity_plan_rejects_invalid_learning_rates(value):
    with pytest.raises(ValueError):
        capacity_plan({"capacity_lr": value})


def test_capacity_plan_rejects_unsupported_modality():
    with pytest.raises(ValueError):
        capacity_plan({"capacity_modality": "both"})


def _contract():
    return {
        "parameter_numel": 6,
        "state": {"weight": {"shape": [2, 3], "dtype": "torch.float32", "sha256": "same-bytes"}},
    }


def test_matched_state_accepts_identical_contracts():
    reference = _contract()
    validate_matched_state_contract(reference, copy.deepcopy(reference))


@pytest.mark.parametrize("field,value", [("shape", [3, 2]), ("dtype", "torch.float64"), ("sha256", "other-bytes")])
def test_matched_state_rejects_shape_dtype_or_initial_value_changes(field, value):
    reference = _contract()
    changed = copy.deepcopy(reference)
    changed["state"]["weight"][field] = value
    with pytest.raises(ValueError):
        validate_matched_state_contract(reference, changed)


def test_matched_state_rejects_parameter_count_or_key_changes():
    reference = _contract()
    changed = copy.deepcopy(reference)
    changed["parameter_numel"] += 1
    with pytest.raises(ValueError):
        validate_matched_state_contract(reference, changed)
    changed = copy.deepcopy(reference)
    changed["state"]["unexpected"] = changed["state"].pop("weight")
    with pytest.raises(ValueError):
        validate_matched_state_contract(reference, changed)


def test_epoch_orders_are_shared_reproducible_permutations():
    ids = ["a", "b", "c", "d", "e"]
    first = _epoch_orders(ids, 3, 3)
    assert first == _epoch_orders(ids, 3, 3)
    assert all(sorted(order) == ids for order in first)
    assert ids == ["a", "b", "c", "d", "e"]


@pytest.mark.parametrize("ids", [[], ["a", "a"]])
def test_epoch_orders_reject_empty_or_duplicate_sample_lists(ids):
    with pytest.raises(ValueError):
        _epoch_orders(ids, 2, 3)


def test_step_seeds_are_arm_independent_and_separate_training_validation():
    assert _step_seed(3, "train", 0, 0) == _step_seed(3, "train", 0, 0)
    assert _step_seed(3, "train", 0, 0) != _step_seed(3, "train", 0, 1)
    assert _step_seed(3, "train", 0, 0) != _step_seed(3, "val", 0, 0)
    assert 0 <= _step_seed(3, "train", 0, 0) <= 2**32 - 1
