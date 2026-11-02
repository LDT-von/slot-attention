"""Contracts of the public queue-result summary.

The repository rule is that patient features, weights, predictions,
identifiers and raw per-patient arrays stay in the ignored ``local_evidence``
directory and never enter the public tree. The queue-result summary is a
public document that is recomputed from private artifacts, so it is exactly
the place where that rule could be broken by accident. These tests therefore
check the summary logic itself and the privacy property of its output.
"""
import json
import statistics
from pathlib import Path

import pytest

from scripts.summarize_queue_results import (ANTICOLLAPSE_ARMS,
                                              DIVERSITY_NEW_ARMS,
                                              DIVERSITY_REUSED_ARMS,
                                              diversity_summary,
                                              fold_arm_score,
                                              geometry_of,
                                              render_markdown)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_OUTPUTS = (ROOT / "docs" / "research" / "slotspe_queue_results_20261102")


def test_repeat_cindices_reads_only_the_scalar_cindex_of_each_repeat(tmp_path):
    # fold_arm_score looks under <root>/folds/<fold>/arms/<arm>, so the test
    # builds that layout rather than a bare arm directory.
    arm_dir = tmp_path / "folds" / "0" / "arms" / "arm"
    arm_dir.mkdir(parents=True)
    records = [{"repeat": index, "cindex": 0.5 + index / 100.0,
                "patient_ids": ["TCGA-XX-YYYY"], "risks": [1.0, 2.0]}
               for index in range(20)]
    (arm_dir / "eval_repeats.json").write_text(json.dumps(records), encoding="utf-8")
    score = fold_arm_score(tmp_path, 0, "arm")
    assert score["n_repeats"] == 20
    assert score["cindex_mean_over_repeats"] == pytest.approx(
        statistics.fmean(r["cindex"] for r in records))
    assert score["cindex_repeat_std_population"] == pytest.approx(
        statistics.pstdev([r["cindex"] for r in records]))


def test_geometry_computes_the_donor_to_own_ratio_when_it_is_not_stored(tmp_path):
    arm_dir = tmp_path / "arm"
    arm_dir.mkdir()
    records = [{"patient_id": "TCGA-XX-YYYY",
                "modalities": {"wsi": {"effective_rank": 3.0, "cosine": 0.5,
                                       "read_overlap": 0.9, "own_mse": 0.4,
                                       "donor_mse": 0.6, "static_mse": 1.0},
                               "omic": {"effective_rank": 2.0, "cosine": 0.3,
                                        "read_overlap": 0.8, "own_mse": 2.0,
                                        "donor_mse": 2.0, "static_mse": 4.0}}}]
    (arm_dir / "patient_diagnostics.json").write_text(json.dumps(records), encoding="utf-8")
    wsi = geometry_of(arm_dir, "wsi")
    omic = geometry_of(arm_dir, "omic")
    assert wsi["donor_to_own_mse_ratio"] == pytest.approx(1.5)
    assert wsi["effective_rank"] == pytest.approx(3.0)
    assert omic["donor_to_own_mse_ratio"] == pytest.approx(1.0)


def test_geometry_returns_none_when_the_artifact_is_absent(tmp_path):
    assert geometry_of(tmp_path / "missing", "wsi") is None
    assert fold_arm_score(tmp_path, 0, "missing") is None


def test_geometry_returns_none_for_a_file_that_is_not_a_record_list(tmp_path):
    arm_dir = tmp_path / "arm"
    arm_dir.mkdir()
    (arm_dir / "patient_diagnostics.json").write_text('{"not": "a list"}', encoding="utf-8")
    assert geometry_of(arm_dir, "wsi") is None


@pytest.mark.skipif(not (PUBLIC_OUTPUTS / "results.md").is_file(),
                    reason="the public summary has not been generated in this checkout")
def test_the_public_summary_contains_no_patient_identifier():
    text = (PUBLIC_OUTPUTS / "results.md").read_text(encoding="utf-8")
    assert "TCGA-" not in text
    payload = json.loads((PUBLIC_OUTPUTS / "results.json").read_text(encoding="utf-8"))
    assert payload["patient_identifiers_included"] is False
    assert payload["per_patient_arrays_included"] is False
    assert payload["patient_data_included"] is False


@pytest.mark.skipif(not (PUBLIC_OUTPUTS / "results.json").is_file(),
                    reason="the public summary has not been generated in this checkout")
def test_the_public_summary_reports_both_queues_with_their_manifest_state():
    payload = json.loads((PUBLIC_OUTPUTS / "results.json").read_text(encoding="utf-8"))
    assert payload["diversity_queue"]["manifest_status"] == "completed"
    assert payload["anti_collapse_queue"]["manifest_status"] == "failed"
    arms = {row["arm"] for row in payload["diversity_queue"]["aggregate"]}
    assert set(DIVERSITY_NEW_ARMS).issubset(arms)
    assert set(DIVERSITY_REUSED_ARMS).issubset(arms)


@pytest.mark.skipif(not (PUBLIC_OUTPUTS / "results.json").is_file(),
                    reason="the public summary has not been generated in this checkout")
def test_the_public_summary_marks_the_untrained_anti_collapse_folds_as_untrained():
    payload = json.loads((PUBLIC_OUTPUTS / "results.json").read_text(encoding="utf-8"))
    untrained = [row for row in payload["anti_collapse_queue"]["folds"] if not row["trained"]]
    assert len(untrained) == 4
    assert {row["fold"] for row in untrained} == {3, 4}
    assert {row["arm"] for row in untrained} == set(ANTICOLLAPSE_ARMS)


@pytest.mark.skipif(not (PUBLIC_OUTPUTS / "results.md").is_file(),
                    reason="the public summary has not been generated in this checkout")
def test_the_rendered_markdown_states_the_claim_boundary():
    text = (PUBLIC_OUTPUTS / "results.md").read_text(encoding="utf-8")
    assert "患者获益" in text
    assert "患者损害" in text
    assert "不构成临床效用" in text
