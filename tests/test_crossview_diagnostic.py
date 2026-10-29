"""Generated finite examples only; no patient input or neural model execution."""
import math

import pytest

from slotspe_proof.crossview_diagnostic import (
    empirical_categorical_oracle,
    empirical_mse_oracle,
)


def test_duplicated_slots_remain_perfect_after_one_deletion():
    rows = [("a", (0, 0)), ("b", (1, 1))]
    result = empirical_categorical_oracle(rows)
    assert result["full_oracle_loss"] == 0
    assert result["drop_oracle_loss"] == 0
    assert result["empirical_oracle_gap"] == 0
    assert result["coverage"]["records"] == 2
    assert result["coverage"]["deletion_occurrences"] == 4
    assert result["coverage"]["drop_singleton_buckets"] == 2


def test_three_multiset_categorical_example_has_entropy_h_one_third():
    rows = [("00", (0, 0)), ("01", (0, 1)), ("11", (1, 1))]
    result = empirical_categorical_oracle(rows)
    h = -(1 / 3) * math.log(1 / 3) - (2 / 3) * math.log(2 / 3)
    assert result["full_oracle_loss"] == 0
    assert result["drop_oracle_loss"] == pytest.approx(h)
    assert result["empirical_oracle_gap"] == pytest.approx(h)
    assert result["coverage"]["full_buckets"] == 3
    assert result["coverage"]["drop_buckets"] == 2
    assert result["coverage"]["full_singleton_record_fraction"] == 1
    assert result["coverage"]["drop_any_singleton_record_fraction"] == 0


def test_three_multiset_continuous_example_has_mse_one_eighteenth():
    rows = [((0.0,), (0, 0)), ((0.5,), (0, 1)), ((1.0,), (1, 1))]
    result = empirical_mse_oracle(rows)
    assert result["target_dimension"] == 1
    assert result["full_oracle_loss"] == 0
    assert result["drop_oracle_loss"] == pytest.approx(1 / 18)
    assert result["empirical_oracle_gap"] == pytest.approx(1 / 18)


def test_permutation_invariance_and_json_compatible_summary():
    rows = [((0.0, 0.25), (2, 1, 0)), ((1.0, 0.75), (0, 2, 1)),
            ((0.5, 0.5), (0, 0, 1))]
    permuted = [(target, tuple(reversed(codes))) for target, codes in rows]
    result = empirical_mse_oracle(rows)
    assert result == empirical_mse_oracle(permuted)
    assert result["empirical_oracle_gap"] >= 0
    import json
    json.dumps(result, allow_nan=False)


def test_deletion_bucket_support_counts_independent_records_not_occurrences():
    rows = [("a", (0, 0)), ("b", (0, 1)), ("c", (3, 3))]
    result = empirical_categorical_oracle(rows)
    coverage = result["coverage"]
    assert coverage["records"] == 3
    assert coverage["independent_records_assumed"] == 3
    assert coverage["deletion_occurrences"] == 6
    assert coverage["drop_buckets"] == 3
    assert coverage["drop_singleton_buckets"] == 2
    assert coverage["drop_any_singleton_record_fraction"] == pytest.approx(2 / 3)
    assert coverage["drop_mean_singleton_deletion_probability"] == pytest.approx(0.5)


def test_one_slot_empty_remaining_multiset_and_nonzero_conditional_variance():
    rows = [((0.0,), (0,)), ((1.0,), (1,))]
    result = empirical_mse_oracle(rows)
    assert result["coverage"]["drop_buckets"] == 1
    assert result["drop_oracle_loss"] == pytest.approx(0.25)
    assert result["empirical_oracle_gap"] == pytest.approx(0.25)


@pytest.mark.parametrize("rows", [
    [],
    [(1, ())],
    [(1, (0,)), (2, (0, 1))],
    [(1, (-1,))],
    [(1, (1.0,))],
    [(1, (True,))],
    [(1, "12")],
    [(1, (0,)), (2,)],
    [([1], (0,))],
    [(float("nan"), (0,))],
    [(("nested", float("inf")), (0,))],
])
def test_rejects_invalid_categorical_rows(rows):
    with pytest.raises(ValueError):
        empirical_categorical_oracle(rows)


@pytest.mark.parametrize("rows", [
    [],
    [((), (0,))],
    [((1.0,), (0,)), ((1.0, 2.0), (1,))],
    [((float("nan"),), (0,))],
    [((float("inf"),), (0,))],
    [((True,), (0,))],
    [(("1",), (0,))],
    [((1.0,), (-1,))],
    [((1.0,), (0,)), ((2.0,), (0, 1))],
    [((1e308,), (0,)), ((-1e308,), (0,))],
])
def test_rejects_invalid_mse_rows_and_numerical_overflow(rows):
    with pytest.raises(ValueError):
        empirical_mse_oracle(rows)


def test_aggregation_is_explicitly_not_a_population_certificate():
    result = empirical_mse_oracle([((1.0,), (0, 0))])
    assert "Empirical oracle" in result["interpretation"]
    assert "not population mutual information" in result["interpretation"]
    assert "not verified" in result["interpretation"]
