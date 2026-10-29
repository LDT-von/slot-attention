"""Cohort aggregation checks on invented case summaries only."""

from copy import deepcopy
import json

import numpy as np
import pytest

from slotspe_proof.roi_cohort import EXPECTED_SEEDS, summarize_cohort
from slotspe_proof.roi_metrics import analyze_attention


def make_summary(kind="uniform", positives=20):
    real = [True] * 27 + [False] * 3
    tumor = [True] * positives + [False] * (30 - positives)
    if kind == "uniform":
        raw = np.full((2, 8, 30), 1 / 30)
    else:
        raw = np.zeros((2, 8, 30))
        raw[:, :4, 0] = 1
        raw[:, 4:, 1] = 1
    metrics = analyze_attention(raw, real, tumor)
    return {
        "schema": "slotspe_roi_case_analysis_v1",
        "row_manifest_sha256": "1" * 64,
        "pathway_names": ["invented-a", "invented-b"],
        "alignment": {"real_mask": real, "tumor_mask": tumor,
                      "fractions": [1.0 if v else None for v in tumor],
                      "sensitivity": {key: tumor.copy() for key in ("0.25", "0.5", "0.75")}},
        "wsi": metrics,
        "roi_threshold_sensitivity_head_mean": {key: deepcopy(metrics["head_mean"]["tumor"])
                                                 for key in ("0.25", "0.5", "0.75")},
        "provenance": {
            "study": "invented-blca", "fold": 0, "training_seed": 3,
            "signature_sha256": "a" * 64, "split_sha256": "b" * 64,
            "feature_encoder": "invented-encoder", "checkpoint_sha256": "c" * 64,
            "source_sha256": "d" * 64, "wsi_input_sha256": "e" * 64,
            "omic_input_sha256": "f" * 64, "captured_model_state_sha256": "3" * 64,
            "rng_entry_sha256": "4" * 64, "input_protocol": "synthetic_fixture",
            "checkpoint_selection": "invented", "split_role": "validation",
            "evaluation_seed": EXPECTED_SEEDS[0],
        },
    }


def records(case_key, summary, seeds=EXPECTED_SEEDS):
    result = []
    for seed in seeds:
        entry = deepcopy(summary)
        entry["provenance"]["evaluation_seed"] = seed
        entry["provenance"]["rng_entry_sha256"] = f"{seed:064x}"
        result.append({"case_key": case_key, "summary": entry})
    return result


def test_patient_is_unit_not_repeated_initializations():
    uniform = make_summary()
    distinct = make_summary("different")
    distinct["row_manifest_sha256"] = "2" * 64
    distinct["provenance"]["wsi_input_sha256"] = "9" * 64
    inputs = records("invented-case-1", uniform) + records("invented-case-2", distinct)
    inputs += records("invented-incomplete", uniform, EXPECTED_SEEDS[:3])
    result = summarize_cohort(inputs)
    assert result["patient_count_input"] == 3
    assert result["patient_count_complete"] == 2
    assert result["patient_count_incomplete"] == 1
    assert result["patient_count_roi_eligible"] == 2
    assert result["observation_count_input"] == 43
    assert result["distributions"]["overlap_median"]["n"] == 2
    assert result["distributions"]["overlap_median"]["median"] == pytest.approx(.5)
    assert result["distributions"]["near_uniform_repeat_fraction"]["mean"] == pytest.approx(.5)
    assert result["bootstrap_unit"] == "patient"
    assert len(result["overlap_by_head"]) == 2
    assert result["roi_area_threshold_sensitivity"]["0.25"]["patient_count_eligible"] == 2
    assert "invented-case" not in json.dumps(result, allow_nan=False)
    assert result == summarize_cohort(inputs)


def test_incomplete_and_insufficient_annotated_regions_are_explicit():
    patient = make_summary(positives=4)
    result = summarize_cohort(records("invented-case", patient))
    assert result["patient_count_complete"] == 1
    assert result["patient_count_roi_eligible"] == 0
    assert result["distributions"]["roi_mass_median"]["n"] == 0
    assert result["distributions"]["roi_mass_median"]["median"] is None
    assert result["distributions"]["overlap_median"]["bootstrap_median_95ci"] is None
    pending = summarize_cohort(records("invented-case", patient, EXPECTED_SEEDS[:1]))
    assert pending["status"] == "no_complete_patients"
    assert pending["patient_count_complete"] == 0
    assert pending["patient_count_incomplete"] == 1


def test_duplicate_and_unexpected_repeats_fail():
    sample = records("invented", make_summary(), EXPECTED_SEEDS[:1])[0]
    with pytest.raises(ValueError, match="duplicate"):
        summarize_cohort([sample, deepcopy(sample)])
    bad = deepcopy(sample)
    bad["summary"]["provenance"]["evaluation_seed"] = 88
    with pytest.raises(ValueError, match="predeclared"):
        summarize_cohort([bad])


@pytest.mark.parametrize("field", ["study", "fold", "training_seed", "signature_sha256", "split_sha256",
                                   "feature_encoder", "checkpoint_sha256", "source_sha256", "input_protocol",
                                   "checkpoint_selection", "captured_model_state_sha256"])
def test_incompatible_cohort_provenance_is_rejected(field):
    first = records("invented-a", make_summary(), EXPECTED_SEEDS[:1])[0]
    other = deepcopy(first)
    other["case_key"] = "invented-b"
    value = other["summary"]["provenance"][field]
    other["summary"]["provenance"][field] = (value + 1 if isinstance(value, int) else
        "9" * 64 if field.endswith("sha256") else value + "-different")
    with pytest.raises(ValueError, match="provenance|unsupported input protocol"):
        summarize_cohort([first, other])


@pytest.mark.parametrize("field", ["row_manifest_sha256", "wsi_input_sha256", "omic_input_sha256", "tumor_mask", "fractions"])
def test_same_patient_changed_inputs_or_roi_are_rejected(field):
    first, other = records("invented", make_summary(), EXPECTED_SEEDS[:2])
    summary = other["summary"]
    if field == "row_manifest_sha256":
        summary[field] = "9" * 64
    elif field in ("wsi_input_sha256", "omic_input_sha256"):
        summary["provenance"][field] = "9" * 64
    elif field == "tumor_mask":
        summary["alignment"][field][0] = False
    else:
        summary["alignment"][field][0] = .8
    with pytest.raises(ValueError, match="between repeats"):
        summarize_cohort([first, other])


def test_slot_pair_and_unknown_roi_validation():
    sample = records("invented", make_summary(), EXPECTED_SEEDS[:1])[0]
    bad = deepcopy(sample)
    bad["summary"]["wsi"]["head_mean"]["pair_overlaps"][0]["overlap"] = .1
    with pytest.raises(ValueError, match="disagree"):
        summarize_cohort([bad])
    sample["summary"]["wsi"]["head_mean"]["tumor"]["status"] = "no_annotated_tumor"
    with pytest.raises(ValueError, match="eligible"):
        summarize_cohort([sample])


def test_reused_rng_state_and_record_order():
    inputs = records("invented-a", make_summary()) + records("invented-b", make_summary("different"))
    assert summarize_cohort(inputs) == summarize_cohort(list(reversed(inputs)))
    invalid = deepcopy(inputs)
    invalid[1]["summary"]["provenance"]["rng_entry_sha256"] = invalid[0]["summary"]["provenance"]["rng_entry_sha256"]
    with pytest.raises(ValueError, match="duplicate initialization RNG"):
        summarize_cohort(invalid)


def test_threshold_masks_are_not_implicitly_interchanged():
    sample = records("invented", make_summary(), EXPECTED_SEEDS[:1])[0]
    sample["summary"]["alignment"]["sensitivity"]["0.25"][0] = False
    with pytest.raises(ValueError, match="sensitivity mask"):
        summarize_cohort([sample])
