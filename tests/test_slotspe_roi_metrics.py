"""Invented input only: reading metrics do not imply patient performance."""

import json

import numpy as np
import pytest

from slotspe_proof.roi_metrics import analyze_attention


def test_uniform_reads_all_tokens_padding_and_annotated_area():
    attention = np.full((2, 8, 7), 1 / 7)
    real = np.array([True] * 5 + [False] * 2)
    tumor = np.array([True, False, True, False, False, False, False])
    out = analyze_attention(attention, real, tumor, top_k=3)

    assert out["n_real"] == 5 and out["n_padding"] == 2
    assert out["n_annotated_tumor"] == 2 and out["n_unknown_real"] == 3
    assert len(out["by_head"]) == 2
    for group in [out["head_mean"], *out["by_head"]]:
        assert group["status"] == "ok"
        np.testing.assert_allclose(group["padding_mass"], 2 / 7)
        np.testing.assert_allclose(group["real_mass"], 5 / 7)
        np.testing.assert_allclose(group["real_attention"], 1 / 5)
        np.testing.assert_allclose(group["overlap"], 1)
        np.testing.assert_allclose(group["uniform_tv"], 0)
        assert len(group["pair_overlaps"]) == 28
        assert group["tumor"]["status"] == "not_evaluable_insufficient_annotated_tumor"
        assert group["tumor"]["q"] == pytest.approx(0.4)
        np.testing.assert_allclose(group["tumor"]["mass"], 0.4)
        np.testing.assert_allclose(group["tumor"]["enrichment"], 1)
        np.testing.assert_allclose(group["tumor"]["mass_minus_q"], 0)
        np.testing.assert_allclose(group["tumor"]["unconditioned_overlap"], 0.4)
        np.testing.assert_allclose(group["tumor"]["conditioned_overlap"], 1)
        assert group["top"]["slots"][0]["input_row_indices"] == [0, 1, 2]
        assert group["top"]["slots"][0]["cutoff"]["exact_tie"]
        assert group["top"]["slots"][0]["cutoff"]["near_tie"]
        assert group["top"]["slots"][0]["mass"] == pytest.approx(3 / 5)
        assert group["top"]["slots"][0]["uniform_mass"] == pytest.approx(3 / 5)
    json.dumps(out, allow_nan=False)


def test_head_average_can_hide_opposing_per_head_reading():
    attention = np.array([
        [[0.9, 0.1], [0.1, 0.9]],
        [[0.1, 0.9], [0.9, 0.1]],
    ])
    out = analyze_attention(attention, [True, True], [True, True], min_tumor_patches=1)
    assert out["head_mean"]["overlap"][0][1] == pytest.approx(1.0)
    for head in out["by_head"]:
        assert head["overlap"][0][1] == pytest.approx(0.2)
        assert head["uniform_tv"] == pytest.approx([0.4, 0.4])


def test_same_cancer_class_different_patches_is_not_cancer_overlap():
    attention = np.array([[[0.8, 0, 0.1, 0.1], [0, 0.8, 0.1, 0.1]]])
    out = analyze_attention(attention, [True] * 4, [True, True, False, False],
                            min_tumor_patches=2)
    group = out["head_mean"]
    assert group["tumor"]["status"] == "eligible"
    assert group["tumor"]["mass"] == pytest.approx([0.8, 0.8])
    assert group["tumor"]["enrichment"] == pytest.approx([1.6, 1.6])
    assert group["tumor"]["conditioned_overlap"][0][1] == 0
    assert group["tumor"]["unconditioned_overlap"][0][1] == 0
    assert group["overlap"][0][1] == pytest.approx(0.2)


def test_padding_is_reported_and_real_conditional_distribution_is_separate():
    attention = np.array([[[0.2, 0.4, 0.4], [0.1, 0.2, 0.7]]])
    out = analyze_attention(attention, [True, True, False], [True, False, False],
                            min_tumor_patches=1, top_k=20)
    group = out["head_mean"]
    assert group["padding_mass"] == pytest.approx([0.4, 0.7])
    assert group["real_mass"] == pytest.approx([0.6, 0.3])
    np.testing.assert_allclose(group["real_attention"], [[1 / 3, 2 / 3]] * 2)
    assert group["overlap"][0][1] == pytest.approx(1)
    assert group["top"]["k"] == 2
    assert group["top"]["slots"][0]["cutoff"]["gap"] is None


def test_head_average_before_conditioning_preserves_native_padding_weight():
    # Different heads give different padding mass. Conditioning each head first
    # and then averaging changes the native head-average distribution.
    attention = np.array([
        [[0.80, 0.10, 0.10], [0.80, 0.10, 0.10]],
        [[0.02, 0.18, 0.80], [0.02, 0.18, 0.80]],
    ])
    out = analyze_attention(attention, [True, True, False], [True, False, False],
                            min_tumor_patches=1)
    np.testing.assert_allclose(out["head_mean"]["padding_mass"], [0.45, 0.45])
    np.testing.assert_allclose(out["head_mean"]["real_attention"],
                               [[0.41 / 0.55, 0.14 / 0.55]] * 2)
    post_conditioning_average = np.mean(
        [np.array(head["real_attention"]) for head in out["by_head"]], axis=0)
    assert not np.allclose(post_conditioning_average, out["head_mean"]["real_attention"])


def test_overlap_equals_one_minus_total_variation_on_generated_data():
    rng = np.random.default_rng(2089)
    attention = rng.dirichlet(np.ones(47), size=(4, 8))
    real = np.array([True] * 35 + [False] * 12)
    tumor = np.array([True] * 7 + [False] * 40)
    out = analyze_attention(attention, real, tumor, min_tumor_patches=1)
    for group in [out["head_mean"], *out["by_head"]]:
        conditional = np.asarray(group["real_attention"])
        tv = 0.5 * np.abs(conditional[:, None] - conditional[None, :]).sum(axis=-1)
        np.testing.assert_allclose(group["overlap"], 1 - tv, atol=1e-12)
        np.testing.assert_allclose(group["tumor"]["mass"], conditional[:, :7].sum(axis=-1))


def test_unknown_region_and_zero_tumor_attention_are_not_normal_or_nan():
    attention = np.array([[[0.0, 0.5, 0.5], [0.5, 0.25, 0.25]]])
    out = analyze_attention(attention, [True] * 3, [True, False, False],
                            min_tumor_patches=1)
    tumor = out["head_mean"]["tumor"]
    assert out["n_unknown_real"] == 2
    assert tumor["zero_tumor_mass_slots"] == [0]
    assert tumor["conditioned_overlap"][0][1] is None
    assert tumor["conditioned_overlap"][1][1] == pytest.approx(1)
    empty = analyze_attention(attention, [True] * 3, [False] * 3)
    assert empty["head_mean"]["tumor"]["status"] == "no_annotated_tumor"
    assert empty["head_mean"]["tumor"]["enrichment"] is None
    json.dumps(out, allow_nan=False)
    json.dumps(empty, allow_nan=False)


def test_no_real_or_zero_real_mass_explicitly_not_evaluable():
    attention = np.array([[[0.0, 1.0], [0.5, 0.5]]])
    no_real = analyze_attention(attention, [False, False], [False, False])
    assert no_real["head_mean"]["status"] == "not_evaluable_no_real_patches"
    partial = analyze_attention(attention, [True, False], [False, False])
    assert partial["head_mean"]["status"] == "not_evaluable_zero_real_attention_mass"
    assert partial["head_mean"]["real_attention"] is None
    json.dumps(partial, allow_nan=False)


def test_cutoff_near_ties_are_flagged_without_changing_ranking():
    attention = np.array([[[0.4, 0.3 + 1e-10, 0.3 - 1e-10], [0.4, 0.3, 0.3]]])
    out = analyze_attention(attention, [True] * 3, [False] * 3, top_k=2,
                            tie_atol=1e-9, tie_rtol=0)
    slot0, slot1 = out["head_mean"]["top"]["slots"]
    assert slot0["input_row_indices"] == [0, 1]
    assert slot0["cutoff"]["near_tie"] and not slot0["cutoff"]["exact_tie"]
    assert slot1["cutoff"]["exact_tie"] and slot1["cutoff"]["near_tie"]
    assert slot0["cutoff"]["gap"] == pytest.approx(2e-10)


@pytest.mark.parametrize("attention, real, tumor", [
    (np.ones((2, 3)) / 3, [True] * 3, [False] * 3),
    (np.ones((1, 1, 3)) / 3, [True] * 3, [False] * 3),
    (np.array([[[0.2, 0.8], [0.5, np.nan]]]), [True] * 2, [False] * 2),
    (np.array([[[0.2, 0.8], [-0.1, 1.1]]]), [True] * 2, [False] * 2),
    (np.ones((1, 2, 2)), [True] * 2, [False] * 2),
    (np.ones((1, 2, 2)) / 2, [1, 0], [False] * 2),
    (np.ones((1, 2, 2)) / 2, [True, False], [False, True]),
    (np.ones((1, 2, 2)) / 2, [True], [False] * 2),
])
def test_invalid_inputs_fail_closed(attention, real, tumor):
    with pytest.raises(ValueError):
        analyze_attention(attention, real, tumor)


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"top_k": True},
                                   {"min_tumor_patches": 0}, {"tie_atol": -1},
                                   {"tie_rtol": np.inf}])
def test_invalid_parameters_fail_closed(kwargs):
    with pytest.raises(ValueError):
        analyze_attention(np.ones((1, 2, 2)) / 2, [True, True], [False, False], **kwargs)
