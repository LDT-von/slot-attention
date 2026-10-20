"""Synthetic assignment controls; no real patients or checkpoint loads."""
from itertools import permutations
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from survot_rank.research.slotspe_proof.p4_diagnostics import assignment, costs, patient_analysis, top_reading

ROOT = Path(__file__).resolve().parents[1]


def test_plan_does_not_load_torch_or_patient_tensors(tmp_path):
    code = "import sys; from survot_rank.research.slotspe_proof.p4_diagnostics import plan; assert not plan(sys.argv[1])['ready']; assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], cwd=ROOT, check=True, capture_output=True)


def test_known_permutation_is_recovered_without_calling_it_semantic_drift():
    attention = np.repeat(np.eye(8)[None], 2, axis=0)
    vectors = np.pad(np.eye(8), ((0, 0), (0, 248)))
    permutation = np.array([2, 4, 0, 7, 1, 6, 5, 3])
    tv, angular = costs(attention, attention[:, permutation], vectors, vectors[permutation])
    for value in (tv, angular):
        result = assignment(value)
        assert result["columns_for_reference_rows"] == np.argsort(permutation).tolist()
        assert result["matched_mean_cost"] == 0
        assert result["diagonal_mean_cost"] > 0
        assert not result["near_tie"]


def test_flat_reading_is_ambiguous_even_with_perfect_repeat_overlap():
    attention = np.full((20, 2, 8, 20), .05)
    vectors = np.broadcast_to(np.pad(np.eye(8), ((0, 0), (0, 248))), (20, 8, 256))
    report, raw = patient_analysis(attention, vectors)
    assert len(report["repeats_against_reference"]) == 19
    assert (raw["attention_costs"] == 0).all()
    for row in report["repeats_against_reference"]:
        assert row["attention_match"]["near_tie"]
        assert row["attention_match"]["second_best_mean_cost_gap"] == 0
        assert row["reading"]["top5_overlap_fraction_mean"] == 1
        assert abs(row["reading"]["current_top5_mass_above_uniform_mean"]) < 1e-15


def test_second_best_gap_matches_exhaustive_assignment():
    matrix = np.array([[.2, .1, .7], [.3, .4, .8], [.9, .4, .1]])
    totals = sorted(sum(matrix[i, p[i]] for i in range(3)) for p in permutations(range(3)))
    result = assignment(matrix)
    assert result["matched_mean_cost"] == pytest.approx(totals[0]/3)
    assert result["second_best_mean_cost_gap"] == pytest.approx((totals[1]-totals[0])/3)


def test_head_averaging_cannot_hide_distribution_change():
    a = np.array([[[.9, .1], [.9, .1]], [[.1, .9], [.1, .9]]])
    b = a[::-1].copy()
    assert np.array_equal(a.mean(0), b.mean(0))
    tv, _ = costs(a, b, np.eye(2), np.eye(2))
    np.testing.assert_allclose(tv, .8)


def test_near_uniform_top5_changes_can_have_tiny_weight_change():
    a = np.full((1, 8, 20), .05)
    b = a.copy()
    a[:, :, :5] += 1e-5
    a[:, :, 5:10] -= 1e-5
    b[:, :, :5] -= 1e-5
    b[:, :, 5:10] += 1e-5
    tv, _ = costs(a, b, np.eye(8), np.eye(8))
    assert tv.max() < .00011
    result = top_reading(a, b, np.arange(8))
    assert result["top5_overlap_fraction_mean"] == 0
    assert 0 < result["current_top5_mass_above_uniform_mean"] < .00006


@pytest.mark.parametrize("problem", ["normalization", "zero_vector", "nonfinite_cost"])
def test_invalid_raw_inputs_are_rejected(problem):
    a = np.full((2, 8, 11), 1/11)
    vectors = np.eye(8)
    if problem == "nonfinite_cost":
        with pytest.raises(ValueError):
            assignment(np.full((8, 8), float("nan")))
    else:
        if problem == "normalization":
            a[0, 0, 0] = 3
        else:
            vectors[0] = 0
        with pytest.raises(ValueError):
            costs(a, a, vectors, vectors)
