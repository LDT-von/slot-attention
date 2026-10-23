"""Audit public-summary arithmetic and generated-matrix counterexamples only.

No model, checkpoint, patient cache, or private array is opened or executed.
Run from any directory with Python, NumPy and SciPy installed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from survot_rank.research.slotspe_proof.problem_metrics import slot_metrics


def digest(data):
    return hashlib.sha256(data).hexdigest()


def entropy(p):
    return float(-(p * np.log(p)).sum())


def main():
    # Native competition probabilities can have unequal slot masses while all
    # row-normalized pooling reads, and thus the reported derived entropy, agree.
    q = np.array([0.3] + [0.1] * 7, dtype=np.float64)
    competition = np.repeat(q[:, None], 256, axis=1)
    pooling = (competition + 1e-8) / (competition + 1e-8).sum(axis=1, keepdims=True)
    metric = slot_metrics(np.eye(8), pooling)
    derived = metric["attention"]["token_assignment_normalized_entropy"]
    assert np.allclose(competition.sum(axis=0), 1)
    assert not np.allclose(competition, 1 / 8)
    assert np.allclose(derived, 1)

    # Complementary heads can disappear in their mean. This is a mathematical
    # possibility, not a statement that the patient's individual heads do this.
    head_a = np.array([[1., 0.], [0., 1.]])
    head_b = head_a[::-1]
    head_mean = (head_a + head_b) / 2
    assert np.array_equal(head_mean[0], head_mean[1])

    centered = slot_metrics(np.eye(8), np.ones((8, 16)))["feature"]["centered_effective_rank"]
    assert centered["algebraic_rank"] == 7
    assert math.isclose(centered["value"], 7, abs_tol=1e-12)

    ratios = {}
    for name, primary, auxiliary, reconstruction in (
        ("wsi", 1.227, 1.387, 0.000411),
        ("omic", 1.555, 1.585, 0.002509),
    ):
        ratios[name] = {
            "scope": "ratio_of_reported_rounded_medians_not_median_of_patient_ratios",
            "weighted_reconstruction_over_main": reconstruction / primary,
            "main_over_weighted_reconstruction": primary / reconstruction,
            "log10_main_over_weighted_reconstruction": math.log10(primary / reconstruction),
            "aux_survival_over_weighted_reconstruction": auxiliary / reconstruction,
        }

    sources = ["models/SlotSPE.py", "models/slot_attention.py", "models/transformer.py",
               "models/omics_encoder.py", "utils/loss_func.py", "utils/core_utils.py"]
    roots = {
        "vendored": ROOT / "third_party/SlotSPE",
        "desktop_original": ROOT / "SlotSPE-main",
        "local_dct_reference": Path("E:/DCT-Reg/third_party/SlotSPE"),
    }
    source_hashes = {}
    for label, root in roots.items():
        source_hashes[label] = {}
        for name in sources:
            path = root / name
            if not path.is_file():
                source_hashes[label][name] = {"present": False}
                continue
            data = path.read_bytes()
            source_hashes[label][name] = {
                "present": True, "sha256": digest(data),
                "lf_normalized_sha256": digest(data.replace(b"\r\n", b"\n")),
            }
    comparisons = {}
    for label in ("desktop_original", "local_dct_reference"):
        comparisons[label] = {}
        for name in sources:
            a = source_hashes["vendored"][name]
            b = source_hashes[label][name]
            comparisons[label][name] = {
                "byte_equal": a.get("sha256") == b.get("sha256") if a.get("present") and b.get("present") else None,
                "lf_normalized_equal": a.get("lf_normalized_sha256") == b.get("lf_normalized_sha256") if a.get("present") and b.get("present") else None,
            }

    runner = ROOT / "survot_rank/research/slotspe_proof/problem_experiments.py"
    runner_bytes = runner.read_bytes()
    reported_runner_sha = "33f572ff3ee53620e16a4eb0b02cd3983d3076ec2f5bcf1c9b0f4c2b918f1502"
    output = {
        "schema": "slotspe_public_observe_summary_review_v1",
        "basis": "public_report_source_inspection_and_generated_arrays_only",
        "reported_observe_commit": "2a806c27e9c71bee0fe21d48d639ebc67b8b92ee",
        "server_raw_artifacts_received_and_audited": False,
        "patient_model_execution_this_review": False,
        "torch_imported": "torch" in sys.modules,
        "entropy_counterexample": {
            "native_competition_slot_probabilities": q.tolist(),
            "native_competition_normalized_entropy": entropy(q) / math.log(8),
            "native_max_abs_deviation_from_one_eighth": float(np.abs(q - 1 / 8).max()),
            "row_normalized_pooling_max_pairwise_abs_difference": float(np.abs(pooling - pooling[0]).max()),
            "derived_report_metric_entropy_min": min(derived),
            "derived_report_metric_entropy_max": max(derived),
            "conclusion": "derived_pooling_entropy_one_does_not_identify_native_competition_uniformity",
        },
        "head_averaging_counterexample": {
            "per_head_rows_differ": bool(np.any(head_a[0] != head_a[1])),
            "mean_head_rows_equal": bool(np.array_equal(head_mean[0], head_mean[1])),
        },
        "centered_rank_ceiling": {"4_slots": 3, "8_slots": 7, "16_slots": 15},
        "generated_centered_eye8_rank": centered,
        "reported_gradient_norm_ratios": ratios,
        "negative_gradient_cosine_patient_counts_of_77": {
            "main_vs_aux_survival": {"wsi": 9, "omic": 4},
            "main_vs_weighted_reconstruction": {"wsi": 44, "omic": 54},
            "aux_survival_vs_weighted_reconstruction": {"wsi": 55, "omic": 55},
        },
        "own_beats_mismatched_donor_counts_of_77": [40, 31, 72],
        "runner_identity": {
            "reported_sha256": reported_runner_sha,
            "local_sha256": digest(runner_bytes),
            "local_matches_report": digest(runner_bytes) == reported_runner_sha,
            "lf_normalized_matches_report": digest(runner_bytes.replace(b"\r\n", b"\n")) == reported_runner_sha,
            "automatic_source_gate_files": sources[:5],
            "reported_manual_eleven_file_check_independently_verified": False,
        },
        "local_source_hashes": source_hashes,
        "local_source_comparisons_to_vendored": comparisons,
        "server_source_identity_scope": "local_paths_are_not_the_server_root; remote_eleven_hash_values_not_received",
    }
    assert not output["torch_imported"]
    destination = Path(__file__).with_name("summary_verification.json")
    destination.write_bytes((json.dumps(output, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"output": str(destination), "entropy_counterexample": output["entropy_counterexample"],
                      "gradient_ratios": ratios, "runner_identity": output["runner_identity"],
                      "local_source_comparisons": comparisons}, ensure_ascii=False))


if __name__ == "__main__":
    main()
