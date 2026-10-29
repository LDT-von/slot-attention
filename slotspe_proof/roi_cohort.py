"""Patient-unit summaries of matched ROI analyses; no patient/model loading.

The caller owns a private, in-memory list of case summaries. A distinct case
key identifies each patient, and must not be shared with public outputs. The
20 slot initialization seeds are repeated measurements, not extra patients.
The code enforces matching declared provenance and fixed-input digests; it does
not independently establish real-world patient, slide or feature provenance.
"""

from collections import defaultdict
from itertools import combinations

import numpy as np


EXPECTED_SEEDS = tuple(range(2026102700, 2026102720))
MATCHED_FIELDS = ("study", "fold", "training_seed", "signature_sha256", "split_sha256",
                  "feature_encoder", "checkpoint_sha256", "source_sha256", "input_protocol",
                  "checkpoint_selection", "split_role", "captured_model_state_sha256")
PATIENT_INPUT_FIELDS = ("wsi_input_sha256", "omic_input_sha256")
HASH_FIELDS = {"signature_sha256", "split_sha256", "checkpoint_sha256", "source_sha256",
               "wsi_input_sha256", "omic_input_sha256", "captured_model_state_sha256", "rng_entry_sha256"}


def _digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _bounded(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} must be a real number")
    value = float(value)
    if not np.isfinite(value) or value < -1e-7 or value > 1 + 1e-7:
        raise ValueError(f"{label} must be finite and within [0, 1]")
    return min(1.0, max(0.0, value))


def _nonnegative(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} must be a real number")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return value


def _vector(values, name):
    if not isinstance(values, list) or len(values) != 8:
        raise ValueError(f"{name} must contain exactly 8 slots")
    return [_bounded(value, name) for value in values]


def _pair_values(group):
    if not isinstance(group, dict) or group.get("status") != "ok":
        raise ValueError("all included reading groups must be evaluable")
    pairs = group.get("pair_overlaps")
    if not isinstance(pairs, list) or len(pairs) != 28:
        raise ValueError("exactly 28 slot pairs are required")
    seen = {}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("invalid slot-pair record")
        a, b = pair.get("slot_a"), pair.get("slot_b")
        if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int)
                or not isinstance(b, int) or not 0 <= a < b < 8 or (a, b) in seen):
            raise ValueError("slot-pair indices must enumerate unique unordered pairs")
        seen[(a, b)] = _bounded(pair.get("overlap"), "overlap")
    if set(seen) != set(combinations(range(8), 2)):
        raise ValueError("slot-pair coverage is incomplete")
    matrix = group.get("overlap")
    if not isinstance(matrix, list) or len(matrix) != 8 or any(not isinstance(row, list) or len(row) != 8 for row in matrix):
        raise ValueError("overlap matrix must be 8 by 8")
    matrix = np.asarray([[_bounded(v, "overlap matrix") for v in row] for row in matrix])
    if not np.allclose(matrix, matrix.T, atol=1e-7, rtol=0) or not np.allclose(np.diag(matrix), 1.0, atol=1e-7, rtol=0):
        raise ValueError("overlap matrix must be symmetric with unit diagonal")
    if any(not np.isclose(value, matrix[a, b], rtol=1e-7, atol=1e-7) for (a, b), value in seen.items()):
        raise ValueError("pair values and overlap matrix disagree")
    return list(seen.values())


def _region_measure(region, tumor, real, pairs, tv):
    """Describe one predeclared ROI threshold with an explicit missing state."""
    result = {"roi_evaluable": False, "roi_mass_median": None, "roi_enrichment_median": None,
              "roi_conditional_overlap_median": None, "near_uniform": None, "common_roi_reading": None}
    if not isinstance(region, dict) or region.get("annotated_count") != sum(tumor):
        raise ValueError("ROI count is inconsistent")
    if sum(tumor) < 20:
        if region.get("status") not in ("no_annotated_tumor", "not_evaluable_insufficient_annotated_tumor"):
            raise ValueError("insufficient annotated area cannot be marked eligible")
        return result
    if region.get("status") != "eligible":
        raise ValueError("at least 20 annotated patches require eligible ROI metrics")
    mass = _vector(region.get("mass"), "ROI mass")
    values = region.get("enrichment")
    if not isinstance(values, list) or len(values) != 8:
        raise ValueError("ROI enrichment must contain exactly 8 slots")
    enrichment = [_nonnegative(value, "ROI enrichment") for value in values]
    q = sum(tumor) / sum(real)
    if not np.isclose(_bounded(region.get("q"), "ROI fraction"), q, atol=1e-7):
        raise ValueError("ROI fraction disagrees with the mask")
    if not np.allclose(enrichment, np.asarray(mass) / q, atol=1e-7, rtol=1e-7):
        raise ValueError("ROI enrichment disagrees with mass and fraction")
    conditional = region.get("conditioned_overlap")
    if not isinstance(conditional, list) or len(conditional) != 8 or any(not isinstance(r, list) or len(r) != 8 for r in conditional):
        raise ValueError("ROI conditional-overlap matrix must be 8 by 8")
    roi_pairs = []
    for a, b in combinations(range(8), 2):
        left, right = conditional[a][b], conditional[b][a]
        if left is None or right is None:
            if left is not None or right is not None or (mass[a] > 0 and mass[b] > 0):
                raise ValueError("missing ROI overlap must correspond to zero ROI mass")
            continue
        if mass[a] <= 0 or mass[b] <= 0:
            raise ValueError("zero ROI mass cannot have conditional overlap")
        left, right = _bounded(left, "ROI overlap"), _bounded(right, "ROI overlap")
        if not np.isclose(left, right, atol=1e-7):
            raise ValueError("ROI overlap must be symmetric")
        roi_pairs.append(left)
    high = sum(value >= 0.9 for value in pairs) >= 15
    # Undefined conditional pairs are never counted as evidence of high overlap.
    uniform = high and sum(t < .05 and .8 <= e <= 1.2 for t, e in zip(tv, enrichment)) >= 5
    focused = high and sum(e >= 1.5 and m >= .10 for e, m in zip(enrichment, mass)) >= 5 and sum(x >= .9 for x in roi_pairs) >= 15
    result.update({"roi_evaluable": True, "roi_mass_median": float(np.median(mass)),
                   "roi_enrichment_median": float(np.median(enrichment)),
                   "roi_conditional_overlap_median": float(np.median(roi_pairs)) if roi_pairs else None,
                   "near_uniform": uniform, "common_roi_reading": focused})
    return result


def _measure(summary):
    wsi = summary.get("wsi")
    if not isinstance(wsi, dict) or wsi.get("schema") != "slotspe_roi_attention_metrics_v1":
        raise ValueError("case must contain validated WSI metrics")
    shape = wsi.get("input_shape")
    if (not isinstance(shape, list) or len(shape) != 3 or shape[1] != 8
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in shape)):
        raise ValueError("expected [heads, 8 slots, tokens] input shape")
    aligned = summary.get("alignment")
    if not isinstance(aligned, dict):
        raise ValueError("case alignment is required")
    real, tumor = aligned.get("real_mask"), aligned.get("tumor_mask")
    fractions = aligned.get("fractions")
    if (not isinstance(real, list) or not isinstance(tumor, list) or len(real) != shape[2]
            or len(tumor) != shape[2] or any(type(v) is not bool for v in real + tumor)
            or any(t and not r for t, r in zip(tumor, real))):
        raise ValueError("invalid real/tumor alignment masks")
    if not isinstance(fractions, list) or len(fractions) != shape[2]:
        raise ValueError("one annotation area fraction is required for each token")
    for is_real, is_tumor, fraction in zip(real, tumor, fractions):
        if fraction is None:
            if is_tumor:
                raise ValueError("tumor label requires a measured annotation fraction")
        else:
            fraction = _bounded(fraction, "annotation area fraction")
            if not is_real or is_tumor != (fraction >= .5):
                raise ValueError("annotation fractions disagree with real/tumor masks")
    if (wsi.get("n_real") != sum(real) or wsi.get("n_padding") != len(real) - sum(real)
            or wsi.get("n_annotated_tumor") != sum(tumor)):
        raise ValueError("alignment counts disagree with metrics")
    head_mean = wsi.get("head_mean")
    pairs = _pair_values(head_mean)
    tv = _vector(head_mean.get("uniform_tv"), "uniform_tv")
    padding = _vector(head_mean.get("padding_mass"), "padding_mass")
    heads = wsi.get("by_head")
    if not isinstance(heads, list) or len(heads) != shape[0]:
        raise ValueError("one group per attention head is required")
    head_pair_medians = [float(np.median(_pair_values(head))) for head in heads]
    high = sum(value >= 0.9 for value in pairs) >= 15
    result = {"overlap_median": float(np.median(pairs)), "tv_median": float(np.median(tv)),
              "padding_mass_median": float(np.median(padding)), "high_overlap": high,
              "head_overlap_medians": head_pair_medians}
    region = head_mean.get("tumor")
    result.update(_region_measure(region, tumor, real, pairs, tv))
    sensitivity = summary.get("roi_threshold_sensitivity_head_mean")
    masks = aligned.get("sensitivity")
    if not isinstance(sensitivity, dict) or not isinstance(masks, dict):
        raise ValueError("predeclared ROI sensitivity metrics and masks are required")
    result["roi_sensitivity"] = {}
    for label, threshold in (("0.25", .25), ("0.5", .5), ("0.75", .75)):
        mask = masks.get(label)
        expected_mask = [bool(r and f is not None and f >= threshold) for r, f in zip(real, fractions)]
        if mask != expected_mask:
            raise ValueError("ROI sensitivity mask disagrees with area fractions")
        if label == "0.5" and sensitivity.get(label) != region:
            raise ValueError("main ROI and 0.5 sensitivity metrics disagree")
        result["roi_sensitivity"][label] = _region_measure(sensitivity.get(label), mask, real, pairs, tv)
    return result


def _distribution(values, rng, bootstrap_samples):
    if not values:
        return {"n": 0, "mean": None, "median": None, "q1": None, "q3": None,
                "bootstrap_median_95ci": None}
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite patient-level value")
    interval = None
    if len(x) >= 2:
        estimates = np.median(x[rng.integers(0, len(x), size=(bootstrap_samples, len(x)))], axis=1)
        interval = [float(v) for v in np.quantile(estimates, [.025, .975])]
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "q1": float(np.quantile(x, .25)), "q3": float(np.quantile(x, .75)),
            "bootstrap_median_95ci": interval}


def summarize_cohort(records, *, expected_seeds=EXPECTED_SEEDS, bootstrap_samples=2000, bootstrap_seed=20261027):
    """Aggregate complete repeats within patient before patient-level statistics.

    ``records`` contains {case_key, summary}. Keys remain internal and are not
    returned. Missing seeds mark a patient incomplete, never an independent row.
    Protocol thresholds are descriptive conventions, not clinical significance.
    """
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("at least one case observation is required")
    if (not isinstance(expected_seeds, (list, tuple)) or not expected_seeds
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in expected_seeds)
            or len(set(expected_seeds)) != len(expected_seeds)):
        raise ValueError("expected_seeds must be a nonempty list of unique nonnegative integers")
    if (isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1
            or isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0):
        raise ValueError("bootstrap configuration must use nonnegative integer seed and positive sample count")
    expected = set(expected_seeds)
    patients = defaultdict(dict)
    patient_rng_states = defaultdict(set)
    cohort_signature = None
    patient_signature = {}
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("case_key"), str) or not row["case_key"].strip():
            raise ValueError("each record requires a private, nonempty case_key")
        key = row["case_key"]
        summary = row.get("summary")
        if not isinstance(summary, dict) or summary.get("schema") != "slotspe_roi_case_analysis_v1":
            raise ValueError("each record must contain a case-analysis summary")
        prov = summary.get("provenance")
        if not isinstance(prov, dict) or any(field not in prov for field in (*MATCHED_FIELDS, *PATIENT_INPUT_FIELDS, "rng_entry_sha256")):
            raise ValueError("required cohort and fixed-input provenance is missing")
        for field in HASH_FIELDS:
            _digest(prov[field], field)
        for field in ("fold", "training_seed"):
            if isinstance(prov[field], bool) or not isinstance(prov[field], int) or prov[field] < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        for field in ("study", "feature_encoder", "input_protocol", "checkpoint_selection", "split_role"):
            if not isinstance(prov[field], str) or not prov[field].strip():
                raise ValueError(f"{field} must be a nonempty string")
        if prov["split_role"] != "validation":
            raise ValueError("cohort must use validation cases")
        if prov["input_protocol"] not in ("original_full_validation", "fixed_4096_diagnostic", "synthetic_fixture"):
            raise ValueError("unsupported input protocol")
        seed = prov.get("evaluation_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in expected:
            raise ValueError("evaluation seed is not in the predeclared schedule")
        pathways = summary.get("pathway_names")
        if not isinstance(pathways, list) or not pathways or any(not isinstance(x, str) or not x.strip() for x in pathways) or len(set(pathways)) != len(pathways):
            raise ValueError("pathway order must contain unique nonempty names")
        signature = (tuple(prov[field] for field in MATCHED_FIELDS), tuple(pathways), summary.get("wsi", {}).get("input_shape", [None])[0])
        if cohort_signature is None:
            cohort_signature = signature
        if signature != cohort_signature:
            raise ValueError("cohort provenance, pathway order or head count differs")
        mapping = _digest(summary.get("row_manifest_sha256"), "row_manifest_sha256")
        alignment = summary.get("alignment")
        if not isinstance(alignment, dict):
            raise ValueError("case alignment is required")
        fixed = (mapping, *(prov[field] for field in PATIENT_INPUT_FIELDS), alignment.get("real_mask"),
                 alignment.get("tumor_mask"), alignment.get("fractions"))
        if key in patient_signature and fixed != patient_signature[key]:
            raise ValueError("patient inputs, row mapping or annotation differ between repeats")
        patient_signature[key] = fixed
        if seed in patients[key]:
            raise ValueError("duplicate patient and initialization seed")
        if prov["rng_entry_sha256"] in patient_rng_states[key]:
            raise ValueError("duplicate initialization RNG state within a patient")
        patient_rng_states[key].add(prov["rng_entry_sha256"])
        patients[key][seed] = _measure(summary)

    complete = [list(patients[key].values()) for key in sorted(patients) if set(patients[key]) == expected]
    patient_rows = []
    for repeats in complete:
        if len({r["roi_evaluable"] for r in repeats}) != 1:
            raise ValueError("ROI eligibility changes between repeats")
        row = {name: float(np.median([r[name] for r in repeats]))
               for name in ("overlap_median", "tv_median", "padding_mass_median")}
        row["high_overlap_repeat_fraction"] = float(np.mean([r["high_overlap"] for r in repeats]))
        row["head_overlap_medians"] = np.median(np.asarray([r["head_overlap_medians"] for r in repeats]), axis=0).tolist()
        row["roi_evaluable"] = repeats[0]["roi_evaluable"]
        if row["roi_evaluable"]:
            for name in ("roi_mass_median", "roi_enrichment_median"):
                row[name] = float(np.median([r[name] for r in repeats]))
            conditional = [r["roi_conditional_overlap_median"] for r in repeats]
            row["roi_conditional_overlap_median"] = (float(np.median(conditional))
                if all(v is not None for v in conditional) else None)
            row["near_uniform_repeat_fraction"] = float(np.mean([r["near_uniform"] for r in repeats]))
            row["common_roi_reading_repeat_fraction"] = float(np.mean([r["common_roi_reading"] for r in repeats]))
        row["roi_sensitivity"] = {}
        for label in ("0.25", "0.5", "0.75"):
            region_repeats = [r["roi_sensitivity"][label] for r in repeats]
            if len({region["roi_evaluable"] for region in region_repeats}) != 1:
                raise ValueError("ROI sensitivity eligibility changes between repeats")
            if region_repeats[0]["roi_evaluable"]:
                sensitivity_row = {name: float(np.median([r[name] for r in region_repeats]))
                                   for name in ("roi_mass_median", "roi_enrichment_median")}
                conditional = [r["roi_conditional_overlap_median"] for r in region_repeats]
                sensitivity_row["roi_conditional_overlap_median"] = (float(np.median(conditional))
                    if all(v is not None for v in conditional) else None)
                sensitivity_row["near_uniform_repeat_fraction"] = float(np.mean([r["near_uniform"] for r in region_repeats]))
                sensitivity_row["common_roi_reading_repeat_fraction"] = float(np.mean([r["common_roi_reading"] for r in region_repeats]))
                row["roi_sensitivity"][label] = sensitivity_row
            else:
                row["roi_sensitivity"][label] = None
        patient_rows.append(row)
    roi_rows = [r for r in patient_rows if r["roi_evaluable"]]
    rng = np.random.default_rng(bootstrap_seed)
    distributions = {}
    for field in ("overlap_median", "tv_median", "padding_mass_median", "high_overlap_repeat_fraction"):
        distributions[field] = _distribution([r[field] for r in patient_rows], rng, bootstrap_samples)
    for field in ("roi_mass_median", "roi_enrichment_median", "roi_conditional_overlap_median",
                  "near_uniform_repeat_fraction", "common_roi_reading_repeat_fraction"):
        distributions[field] = _distribution([r[field] for r in roi_rows if r.get(field) is not None], rng, bootstrap_samples)
    by_head = [_distribution([r["head_overlap_medians"][i] for r in patient_rows], rng, bootstrap_samples)
               for i in range(cohort_signature[2])]
    roi_sensitivity = {}
    for label in ("0.25", "0.5", "0.75"):
        rows = [r["roi_sensitivity"][label] for r in patient_rows if r["roi_sensitivity"][label] is not None]
        fields = ("roi_mass_median", "roi_enrichment_median", "roi_conditional_overlap_median",
                  "near_uniform_repeat_fraction", "common_roi_reading_repeat_fraction")
        roi_sensitivity[label] = {"patient_count_eligible": len(rows), "distributions": {
            field: _distribution([r[field] for r in rows if r[field] is not None], rng, bootstrap_samples)
            for field in fields}}
    return {
        "schema": "slotspe_roi_cohort_summary_v1",
        "status": "ok" if complete else "no_complete_patients",
        "patient_count_input": len(patients), "patient_count_complete": len(complete),
        "patient_count_incomplete": len(patients) - len(complete), "patient_count_roi_eligible": len(roi_rows),
        "observation_count_input": len(records), "expected_repeats_per_patient": len(expected),
        "expected_seeds": sorted(expected), "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed, "bootstrap_unit": "patient",
        "matched_provenance": dict(zip(MATCHED_FIELDS, cohort_signature[0])),
        "pathway_count": len(cohort_signature[1]), "head_count": cohort_signature[2],
        "distributions": distributions, "overlap_by_head": by_head,
        "roi_area_threshold_sensitivity": roi_sensitivity,
        "interpretation": "Within-patient initialization repeats are not independent patients; thresholds are descriptive, and ROI-unmarked is unknown.",
        "private_case_keys_included": False,
    }
