"""Generated examples only; no patient input, no checkpoint, no model execution."""
import math

import numpy as np
import pytest

from slotspe_proof.crossview_deletion import (
    assign_codes,
    build_rows,
    fit_codebook,
    geometry_summary,
    multiset_structure,
    patient_codes,
    permuted_codes_control,
    run_deletion_oracle,
)


def test_fit_codebook_is_deterministic_and_recovers_separated_clusters():
    rng = np.random.default_rng(0)
    data = np.concatenate([
        rng.normal(0.0, 0.05, size=(20, 3)),
        rng.normal(5.0, 0.05, size=(20, 3)),
    ])
    first = fit_codebook(data, 2)
    second = fit_codebook(data, 2)
    assert np.array_equal(first, second)
    codes = assign_codes(data, first)
    # The two well-separated clusters must land in different codes.
    assert codes[:20].tolist() == [codes[0]] * 20
    assert codes[20:].tolist() == [codes[20]] * 20
    assert codes[0] != codes[20]


def test_fit_codebook_requires_enough_vectors():
    with pytest.raises(ValueError):
        fit_codebook(np.zeros((3, 2)), 4)


def test_fit_codebook_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        fit_codebook(np.zeros((4, 2)), 0)
    with pytest.raises(ValueError):
        fit_codebook(np.zeros((4, 2)), 2, steps=-1)
    with pytest.raises(ValueError):
        fit_codebook(np.array([[np.nan, 0.0]]), 1)


def test_assign_codes_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        assign_codes(np.zeros((2, 3)), np.zeros((2, 4)))


def test_patient_codes_returns_slot_length_tuple():
    matrix = np.array([[0.0, 0.0], [5.0, 5.0], [0.1, 0.1]])
    codes = patient_codes(matrix, fit_codebook(matrix, 2))
    assert len(codes) == 3
    assert all(isinstance(value, int) for value in codes)


def test_permutation_invariant_codes_do_not_change_the_oracle():
    rows = [((0.0,), (0, 1)), ((1.0,), (1, 0)), ((2.0,), (0, 0))]
    codes = {"a": (0, 1), "b": (1, 0), "c": (0, 0)}
    targets = {"a": (0.0,), "b": (1.0,), "c": (2.0,)}
    forward = run_deletion_oracle(codes, targets, ["a", "b", "c"], target_kind="mse")
    swapped = {"a": (1, 0), "b": (0, 1), "c": (0, 0)}
    backward = run_deletion_oracle(swapped, targets, ["a", "b", "c"], target_kind="mse")
    assert forward["empirical_oracle_gap"] == pytest.approx(backward["empirical_oracle_gap"])


def test_deletion_gap_is_nonnegative_and_zero_when_a_slot_is_redundant():
    # Both slots carry the same code, so deleting either changes nothing.
    codes = {"a": (0, 0), "b": (1, 1)}
    targets = {"a": 0, "b": 1}
    result = run_deletion_oracle(codes, targets, ["a", "b"], target_kind="category")
    assert result["empirical_oracle_gap"] == pytest.approx(0.0, abs=1e-12)
    assert result["drop_oracle_loss"] >= result["full_oracle_loss"]


def test_deletion_gap_is_positive_when_the_deleted_slot_is_the_only_evidence():
    # Patient a has a code that patient b lacks. The full multiset separates
    # the two patients, but deleting that code merges them into one bucket.
    codes = {"a": (0, 1), "b": (1, 1)}
    targets = {"a": 0, "b": 1}
    result = run_deletion_oracle(codes, targets, ["a", "b"], target_kind="category")
    assert result["full_oracle_loss"] == pytest.approx(0.0, abs=1e-12)
    assert result["empirical_oracle_gap"] > 0.0


def test_permutation_null_is_nonnegative_and_repeats_are_reproducible():
    codes = {"a": (0, 1), "b": (0, 2), "c": (1, 2)}
    targets = {"a": 0, "b": 1, "c": 2}
    first = permuted_codes_control(codes, targets, ["a", "b", "c"],
                                   target_kind="category", repeats=4)
    second = permuted_codes_control(codes, targets, ["a", "b", "c"],
                                    target_kind="category", repeats=4)
    assert first["null_gaps"] == second["null_gaps"]
    assert first["null_gap_mean"] >= 0.0
    assert first["null_gap_min"] >= 0.0


def test_multiset_structure_reports_unique_collapse():
    codes = {"a": (0, 1), "b": (1, 2), "c": (2, 3)}
    result = multiset_structure(codes, ["a", "b", "c"])
    assert result["distinct_multisets"] == 3
    assert result["patients_in_unique_multiset"] == 3
    assert result["patients_in_unique_multiset_fraction"] == pytest.approx(1.0)

    shared = {"a": (0, 0), "b": (0, 0), "c": (0, 0)}
    result = multiset_structure(shared, ["a", "b", "c"])
    assert result["distinct_multisets"] == 1
    assert result["largest_multiset_multiplicity"] == 3


def test_build_rows_validates_patient_presence_and_targets():
    with pytest.raises(ValueError):
        build_rows({"a": (0, 1)}, {"b": 0}, ["a", "b"])
    with pytest.raises(ValueError):
        build_rows({"a": (0, 1)}, {"a": (float("nan"),)}, ["a"])
    with pytest.raises(ValueError):
        build_rows({"a": (0, 1)}, {"a": (1, 2)}, ["a", "b"])


def test_run_deletion_oracle_rejects_unknown_target_kind():
    with pytest.raises(ValueError):
        run_deletion_oracle({"a": (0, 1)}, {"a": 0}, ["a"], target_kind="other")


def test_geometry_summary_reports_descriptive_metrics():
    matrix = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    result = geometry_summary([matrix])
    assert result["patients"] == 1
    assert result["mean_cosine"] is not None
    assert 0.0 <= result["mean_cosine"] <= 1.0
    assert result["mean_effective_rank"] is not None
    assert math.isfinite(result["mean_effective_rank"])
