"""Contracts of the five-fold score statistics.

The five-fold statistics are the numbers the user asked for directly, so the
logic that produces them is checked here rather than trusted. The tests also
guard the privacy property of the published output, because the summary is
recomputed from private artifacts.

The central subtlety the tests pin down is that the two evaluation protocols
are not interchangeable. The per-epoch protocol takes a maximum over about
thirty validation passes, so it reads higher than the twenty-repeat protocol
on the same model. Mixing the two would produce a number that looks
comparable to the recorded baseline results and is not.
"""
import json
import statistics
from pathlib import Path

import pytest

from scripts.summarize_five_fold_scores import (ANTI_ARMS,
                                                 BASE_ARMS,
                                                 epoch_validation,
                                                 per_epoch_arm,
                                                 repeat_arm,
                                                 repeat_cindex)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_OUTPUTS = (ROOT / "docs" / "research" / "slotspe_queue_results_20261102")


def _write_arm(root, fold, arm, epochs):
    """Write one arm-fold of per-epoch validation values."""
    folder = root / "folds" / str(fold) / "arms" / arm
    folder.mkdir(parents=True, exist_ok=True)
    payload = [{"epoch": epoch, "cindex": 0.5 + epoch / 100.0}
               for epoch in range(1, epochs + 1)]
    (folder / "epoch_validation.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_epoch_validation_is_keyed_by_epoch(tmp_path):
    _write_arm(tmp_path, 0, "arm", 5)
    values = epoch_validation(tmp_path, 0, "arm")
    assert values == {1: 0.51, 2: 0.52, 3: 0.53, 4: 0.54, 5: 0.55}


def test_epoch_validation_returns_none_when_the_artifact_is_absent(tmp_path):
    assert epoch_validation(tmp_path, 0, "missing") is None


def test_epoch_validation_skips_records_without_a_cindex(tmp_path):
    folder = tmp_path / "folds" / "0" / "arms" / "arm"
    folder.mkdir(parents=True)
    (folder / "epoch_validation.json").write_text(
        json.dumps([{"epoch": 1, "cindex": 0.7}, {"epoch": 2, "cindex": None},
                    {"epoch": None, "cindex": 0.8}]), encoding="utf-8")
    assert epoch_validation(tmp_path, 0, "arm") == {1: 0.7}


def test_the_epoch_wise_mean_is_taken_only_over_epochs_every_fold_has(tmp_path):
    # Fold 0 reaches epoch 4, fold 1 reaches epoch 2 only. The common range
    # must be 1..2, so the epoch-wise mean must not use epochs 3 and 4.
    _write_arm(tmp_path, 0, "arm", 4)
    _write_arm(tmp_path, 1, "arm", 2)
    result = per_epoch_arm(tmp_path, "arm")
    assert result["common_epoch_range"] == [1, 2]
    assert [row["epoch"] for row in result["epoch_means"]] == [1, 2]
    assert result["best_epoch_mean"]["epoch"] == 2


def test_the_best_epoch_mean_and_the_mean_of_fold_bests_are_different_quantities(tmp_path):
    # Fold 0 peaks early, fold 1 peaks late, and the two together are
    # flat. The epoch-wise optimum and the average of the per-fold optima
    # must not be confused with each other.
    folder0 = tmp_path / "folds" / "0" / "arms" / "arm"
    folder0.mkdir(parents=True)
    (folder0 / "epoch_validation.json").write_text(
        json.dumps([{"epoch": 1, "cindex": 0.9}, {"epoch": 2, "cindex": 0.5}]),
        encoding="utf-8")
    folder1 = tmp_path / "folds" / "1" / "arms" / "arm"
    folder1.mkdir(parents=True)
    (folder1 / "epoch_validation.json").write_text(
        json.dumps([{"epoch": 1, "cindex": 0.5}, {"epoch": 2, "cindex": 0.9}]),
        encoding="utf-8")
    result = per_epoch_arm(tmp_path, "arm")
    assert result["best_epoch_mean"]["mean"] == pytest.approx(0.7)
    assert result["mean_of_fold_bests"] == pytest.approx(0.9)
    assert result["best_epoch_mean"]["mean"] != pytest.approx(result["mean_of_fold_bests"])


def test_repeat_cindex_averages_the_repeats_and_counts_them(tmp_path):
    folder = tmp_path / "folds" / "0" / "arms" / "arm"
    folder.mkdir(parents=True)
    records = [{"repeat": index, "cindex": 0.6 + index / 100.0,
                "patient_ids": ["TCGA-XX-YYYY"]} for index in range(20)]
    (folder / "eval_repeats.json").write_text(json.dumps(records), encoding="utf-8")
    value = repeat_cindex(tmp_path, 0, "arm")
    assert value["n"] == 20
    assert value["mean"] == pytest.approx(statistics.fmean(r["cindex"] for r in records))


def test_repeat_cindex_returns_none_when_the_artifact_is_absent(tmp_path):
    assert repeat_cindex(tmp_path, 0, "missing") is None


def test_repeat_arm_reports_only_the_folds_that_have_a_retained_evaluation(tmp_path):
    for fold in (0, 1, 2):
        folder = tmp_path / "folds" / str(fold) / "arms" / "arm"
        folder.mkdir(parents=True)
        (folder / "eval_repeats.json").write_text(
            json.dumps([{"cindex": 0.6 + fold / 100.0}] * 20), encoding="utf-8")
    result = repeat_arm(tmp_path, "arm")
    assert result["folds_with_data"] == [0, 1, 2]
    assert result["mean"] == pytest.approx(0.61)


@pytest.mark.skipif(not (PUBLIC_OUTPUTS / "five_fold_scores.md").is_file(),
                    reason="the five-fold summary has not been generated in this checkout")
def test_the_published_five_fold_summary_contains_no_patient_identifier():
    text = (PUBLIC_OUTPUTS / "five_fold_scores.md").read_text(encoding="utf-8")
    assert "TCGA-" not in text
    payload = json.loads((PUBLIC_OUTPUTS / "five_fold_scores.json").read_text(encoding="utf-8"))
    assert payload["patient_identifiers_included"] is False
    assert payload["per_patient_arrays_included"] is False
    assert payload["patient_data_included"] is False


@pytest.mark.skipif(not (PUBLIC_OUTPUTS / "five_fold_scores.json").is_file(),
                    reason="the five-fold summary has not been generated in this checkout")
def test_the_published_summary_states_both_protocols_and_their_coverage():
    payload = json.loads((PUBLIC_OUTPUTS / "five_fold_scores.json").read_text(encoding="utf-8"))
    protocols = payload["protocols"]
    assert "per_epoch_validation" in protocols
    assert "twenty_repeat_evaluation" in protocols
    for name in ("per_epoch_validation", "twenty_repeat_evaluation"):
        protocol = protocols[name]
        assert protocol["coverage"]
        assert protocol["caveat"]
    assert protocols["why_both"]
    # The twenty-repeat protocol cannot cover the anti-collapse folds whose
    # evaluation never completed.
    anti = payload["anti_collapse_repeats"]
    for arm in ANTI_ARMS:
        assert len(anti[arm]["folds_with_data"]) == 3
    for arm in BASE_ARMS:
        assert len(payload["baseline_repeats"][arm]["folds_with_data"]) == 5


@pytest.mark.skipif(not (PUBLIC_OUTPUTS / "five_fold_scores.md").is_file(),
                    reason="the five-fold summary has not been generated in this checkout")
def test_the_published_summary_states_the_claim_boundary():
    text = (PUBLIC_OUTPUTS / "five_fold_scores.md").read_text(encoding="utf-8")
    assert "患者获益" in text
    assert "患者损害" in text
    assert "不构成临床效用" in text
