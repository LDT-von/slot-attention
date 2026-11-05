"""Contracts of the plugin queue score summary.

The plugin queue scores are the numbers the user asked to have recorded, so
the logic that produces them is checked here rather than trusted. The tests
also guard the privacy property of the published output, because the summary
is recomputed from private artifacts.

Two properties matter most and are easy to get wrong. First, the evaluation
protocol is a maximum over about thirty validation passes over the same
patients, so a single-fold number is a selected peak and must never be
presented as the original's level; the original's five-fold mean under the
same protocol is 0.7039 and its fold 0 value is 0.7485. Second, the geometry
and content measurements decide the mechanism claims, so a statement about
collapse has to be derived from the measured numbers instead of written by
hand, because two of the nine arms genuinely do spread the slots.
"""
import json
from pathlib import Path

import pytest

from scripts.summarize_plugin_queue_scores import (ARMS,
                                                    CONTENT_ARMS,
                                                    PAIRED_COMPARISONS,
                                                    build,
                                                    content_table,
                                                    geometry_table,
                                                    paired_comparisons,
                                                    render_markdown)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_OUTPUTS = (ROOT / "docs" / "research" / "slotspe_plugin_queue_results_20261105")


def test_every_preset_arm_is_present_in_the_published_scores():
    result = build()
    assert set(result["scores"]) == set(ARMS)
    assert len(ARMS) == 9


def test_the_three_prescribed_comparisons_are_reported():
    result = build()
    reported = {(row["treatment"], row["control"]) for row in result["paired_comparisons"]}
    assert reported == {(t, c) for t, c, _ in PAIRED_COMPARISONS}


def test_the_comparison_difference_is_the_treatment_minus_the_control():
    summaries = build()["scores"]
    for row in paired_comparisons(summaries):
        expected = (summaries[row["treatment"]]["repeat_cindex_mean"]
                    - summaries[row["control"]]["repeat_cindex_mean"])
        assert row["difference"] == pytest.approx(expected, abs=1e-12)


def test_the_native_control_is_the_top_arm_of_the_queue_within_the_repeat_noise():
    result = build()
    scores = {arm: row["repeat_cindex_mean"] for arm, row in result["scores"].items()}
    best = max(scores, key=scores.get)
    # V1_mean edges the native control out by 0.0027, far inside the repeat
    # standard deviation, so no arm is separated from the control.
    assert best == "V1_mean"
    assert scores["V1_mean"] - scores["B0_native"] < result["scores"]["B0_native"]["repeat_cindex_std_population"]
    for arm, value in scores.items():
        assert value - scores["B0_native"] < result["scores"]["B0_native"]["repeat_cindex_std_population"] \
            or arm == "B1_gram"


def test_the_queue_shares_the_patient_lists_with_the_completed_baseline():
    identity = build()["cohort_identity"]
    assert identity["train_identical"] is True
    assert identity["val_identical"] is True
    assert identity["plugin_train"] == 303
    assert identity["plugin_val"] == 77


def test_the_queue_shares_the_initial_core_state_with_the_original_baseline():
    initialization = build()["initialization"]
    assert initialization["identical"] is True
    assert initialization["plugin_B0_native_initial_core_hash"] == (
        initialization["original_S0_fold0_initial_core_hash"])


def test_the_baseline_five_fold_mean_is_reported_alongside_the_fold_zero_value():
    baseline = build()["original_baseline_same_protocol"]
    assert baseline["S0_five_fold_mean"] == pytest.approx(0.7039, abs=1e-4)
    assert baseline["S0_cindex_per_fold"]["0"] == pytest.approx(0.7485, abs=1e-4)
    assert baseline["S0_cindex_per_fold"]["0"] > baseline["S0_five_fold_mean"]


def test_the_content_error_is_above_the_constant_floor_for_every_content_arm():
    table = content_table()
    for arm in CONTENT_ARMS:
        for modality in ("wsi", "omic"):
            assert table[arm][modality]["mse"] > table[arm][modality]["constant_floor"]


def test_the_v1_kl_weight_is_numerically_inert_in_the_measured_geometry():
    geometry = geometry_table()
    vib = geometry["V1_vib"]["wsi.latent"]
    noise = geometry["V1_noise"]["wsi.latent"]
    assert vib["cosine"] == pytest.approx(noise["cosine"], abs=1e-3)
    assert vib["effective_rank"] == pytest.approx(noise["effective_rank"], abs=1e-3)


def test_the_collapse_statement_is_derived_from_the_measurement():
    result = build()
    geometry = result["slot_geometry"]
    collapsed = [arm for arm in ARMS if geometry[arm]["wsi.native"]["cosine"] > 0.9]
    spread = [arm for arm in ARMS if geometry[arm]["wsi.native"]["cosine"] <= 0.9]
    assert set(collapsed) == set(ARMS) - {"B1_gram", "V3_relation"}
    assert set(spread) == {"B1_gram", "V3_relation"}
    assert geometry["B1_gram"]["wsi.native"]["effective_rank"] > 4.0


def test_the_two_spread_arms_are_not_the_high_scoring_arms():
    result = build()
    scores = result["scores"]
    native = scores["B0_native"]["repeat_cindex_mean"]
    assert scores["B1_gram"]["repeat_cindex_mean"] < native - 0.05
    assert scores["V3_relation"]["repeat_cindex_mean"] < scores["V3_capacity"]["repeat_cindex_mean"]


def test_the_published_markdown_states_the_single_fold_and_selection_limits():
    text = (PUBLIC_OUTPUTS / "plugin_queue_scores.md").read_text(encoding="utf-8")
    assert "单折" in text
    assert "0.7433" in text
    assert "0.7039" in text
    assert "0.7485" in text


def test_the_published_json_matches_a_fresh_recomputation():
    published = json.loads((PUBLIC_OUTPUTS / "plugin_queue_scores.json").read_text(encoding="utf-8"))
    fresh = build()
    assert published["scores"] == fresh["scores"]
    assert published["paired_comparisons"] == fresh["paired_comparisons"]
    assert published["cohort_identity"] == fresh["cohort_identity"]
    assert published["initialization"] == fresh["initialization"]


def test_the_rendered_markdown_lists_every_arm():
    text = render_markdown(build())
    for arm in ARMS:
        assert arm in text


def test_the_published_output_carries_no_patient_level_data():
    raw = (PUBLIC_OUTPUTS / "plugin_queue_scores.json").read_text(encoding="utf-8")
    assert "patient_id" not in raw
    assert "risk" not in raw.replace("risk_std", "")
    import re
    assert not re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", raw)
