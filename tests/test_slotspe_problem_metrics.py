"""Synthetic identities for mechanism measurements; no models or patient data."""

import json

import numpy as np
import pytest

from survot_rank.research.slotspe_proof.problem_metrics import matched_stability, slot_metrics, summary


def test_duplicate_and_distinct_slots_are_not_confused():
    duplicate = slot_metrics([[1, 2], [1, 2]], [[1, 0], [1, 0]])
    distinct = slot_metrics([[1, 0], [0, 1]], [[1, 0], [0, 1]])
    assert duplicate["feature"]["offdiag_cosine"][0]["value"] == pytest.approx(1)
    assert duplicate["feature"]["normalized_l2"][0]["value"] == 0
    assert duplicate["feature"]["centered_effective_rank"]["value"] is None
    assert distinct["feature"]["offdiag_cosine"][0]["value"] == 0
    assert distinct["feature"]["normalized_l2"][0]["value"] == pytest.approx(1)
    assert distinct["attention"]["offdiag_total_variation"][0]["value"] == 1
    assert distinct["attention"]["offdiag_jensen_shannon_bits"][0]["value"] == pytest.approx(1)
    assert duplicate["attention"]["offdiag_total_variation"][0]["value"] == 0


def test_cosine_does_not_hide_norm_differences_or_fabricate_zero_vectors():
    result = slot_metrics([[1, 0], [10, 0], [0, 0]], np.eye(3))
    assert result["feature"]["offdiag_cosine"][0]["value"] == 1
    assert result["feature"]["normalized_l2"][0]["value"] == pytest.approx(9 / np.sqrt(101))
    assert result["feature"]["norms"] == [1, 10, 0]
    assert result["feature"]["undefined_cosine_pairs"] == 2
    assert result["feature"]["offdiag_cosine"][1]["value"] is None
    assert result["feature"]["offdiag_cosine_summary"]["n"] == 1
    json.dumps(result, allow_nan=False)


def test_centering_removes_common_offset_and_reports_known_rank():
    features = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    reference = slot_metrics(features, np.eye(4))["feature"]["centered_effective_rank"]
    shifted = slot_metrics(features + 1000, np.eye(4))["feature"]["centered_effective_rank"]
    assert reference["value"] == pytest.approx(2)
    assert shifted["value"] == pytest.approx(2)
    assert reference["algebraic_rank"] == 2


def test_finite_small_vectors_keep_defined_cosines_despite_other_large_slots():
    result = slot_metrics([[1e-200, 0], [0, 1e200], [1e-200, 0]], np.eye(3))
    assert result["feature"]["undefined_cosine_pairs"] == 0
    assert result["feature"]["offdiag_cosine"][1]["value"] == pytest.approx(1)
    assert result["feature"]["norms"][0] == pytest.approx(1e-200, abs=0)
    json.dumps(result, allow_nan=False)


def test_attention_support_and_assignment_entropy_have_known_extremes():
    hard = slot_metrics(np.eye(2), [[5, 0, 0], [0, 7, 0]])["attention"]
    shared = slot_metrics(np.eye(2), np.ones((2, 3)))["attention"]
    assert hard["effective_token_support"] == [1, 1]
    assert hard["token_assignment_entropy_nats"] == [0, 0, None]
    assert hard["unclaimed_token_count"] == 1
    assert shared["effective_token_support"] == pytest.approx([3, 3])
    assert shared["mass_weighted_assignment_entropy_nats"] == pytest.approx(np.log(2))
    assert shared["token_assignment_normalized_entropy"] == pytest.approx([1, 1, 1])


def test_external_groups_separate_dedicated_and_fragmented_attention():
    groups = ["A", "A", "B", "B"]
    dedicated = slot_metrics(np.eye(2), [[1, 1, 0, 0], [0, 0, 1, 1]], groups)["groups"]
    fragmented = slot_metrics(np.eye(2), np.ones((2, 4)), groups)["groups"]
    assert dedicated["slot_purity"] == [1, 1]
    assert [x["normalized_entropy"] for x in dedicated["group_fragmentation"]] == [0, 0]
    assert fragmented["slot_purity"] == [0.5, 0.5]
    assert [x["normalized_entropy"] for x in fragmented["group_fragmentation"]] == pytest.approx([1, 1])
    assert [x["effective_slots"] for x in fragmented["group_fragmentation"]] == pytest.approx([2, 2])


def test_hungarian_recovers_permutation_without_claiming_index_stability():
    a = np.eye(3)
    permutation = [2, 0, 1]
    result = matched_stability(a, a[permutation], n_permutations=64)
    assert result["matched_mean_distance"] == 0
    assert result["unmatched_mean_distance"] == 1
    assert [x["slot_b"] for x in result["matches"]] == [1, 2, 0]
    assert result["assignment_ambiguous"] is False
    assert result["alternative_assignment_mean_distance_gap"] == pytest.approx(2 / 3)


def test_duplicate_attention_makes_matching_ambiguous():
    result = matched_stability(np.ones((3, 4)), np.ones((3, 4)))
    assert result["matched_mean_distance"] == 0
    assert result["assignment_ambiguous"] is True
    assert result["alternative_assignment_mean_distance_gap"] == 0


def test_local_chance_rng_is_reproducible_and_preserves_global_numpy_rng():
    np.random.seed(813)
    state = np.random.get_state()
    first = matched_stability(np.eye(4), np.eye(4), seed=42)
    after = np.random.random()
    np.random.set_state(state)
    expected = np.random.random()
    assert after == expected
    assert first == matched_stability(np.eye(4), np.eye(4), seed=42)


def test_one_slot_and_zero_mass_group_are_explicitly_undefined():
    result = slot_metrics([[0, 0]], [[1, 0]], ["present", "unclaimed"])
    assert result["feature"]["centered_effective_rank"]["value"] is None
    assert result["feature"]["offdiag_cosine_summary"]["n"] == 0
    assert result["groups"]["group_fragmentation"][1]["reason"] == "group_has_zero_attention_mass"
    matched = matched_stability([[1]], [[1]], n_permutations=0)
    assert matched["assignment_ambiguous"] is None
    assert matched["random_permutation_baseline"]["summary"]["n"] == 0
    json.dumps([result, matched], allow_nan=False)


@pytest.mark.parametrize("slots,attention", [
    ([[1]], [[0]]), ([[1]], [[-1]]), ([[float("nan")]], [[1]]),
    ([[1]], [[float("inf")]]), ([[1], [2]], [[1]]), ([], [[1]]),
])
def test_invalid_inputs_fail_loudly(slots, attention):
    with pytest.raises(ValueError):
        slot_metrics(slots, attention)


def test_summary_has_no_silent_nonfinite_filtering():
    assert summary([])["median"] is None
    assert summary([1, 2, 3, 4]) == {"n": 4, "median": 2.5, "q25": 1.75, "q75": 3.25, "min": 1, "max": 4}
    with pytest.raises(ValueError):
        summary([1, float("nan")])


def test_matching_refuses_different_token_shapes():
    with pytest.raises(ValueError):
        matched_stability(np.ones((2, 3)), np.ones((2, 4)))
