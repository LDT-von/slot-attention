"""Regenerate the public 2026-10-22 GPU record from existing local artifacts.

Read-only derivation. This script starts no training, no inference, no patient
intervention and no server task. It only re-reads artifacts that already exist:

* the two completed local original-arm queue summaries under results/
* the aggregate summaries of already-completed read-only / frozen-checkpoint
  checks under the ignored local_evidence directory

Patient identifiers, weights, predictions and raw per-patient arrays never leave
local_evidence. This generator publishes queue-level aggregates only, and it
refuses to emit any record that still carries a patient identifier or a local
absolute path.

Usage:
    python -B scripts/record_slotspe_gpu_trisurv_20261022.py --plan
    python -B scripts/record_slotspe_gpu_trisurv_20261022.py --run
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics as st
import sys

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True

OUTPUT_DIR = REPO / "docs/research/slotspe_gpu_trisurv_20261022"
FIVE_FOLD = "results/slotspe_proof/blca_5fold_gpu_trisurv_a512169"
FIVE_FOLD_3SEED = "results/slotspe_proof/blca_5fold_3seed_gpu_trisurv_a512169"
EVIDENCE = "local_evidence"

PATIENT_ID = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}")
LOCAL_PATH = re.compile(r"/(?:data1|home|mnt|media|Users|Users)/[^\"\s,]*")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def guard(value, where):
    """Refuse to publish a record that still carries patient rows or host paths."""
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    identifiers = set(PATIENT_ID.findall(text))
    paths = sorted(set(LOCAL_PATH.findall(text)))
    if identifiers:
        raise RuntimeError(f"{where} would publish {len(identifiers)} patient identifiers")
    if paths:
        raise RuntimeError(f"{where} would publish host paths, e.g. {paths[:3]}")
    return text


def queue_run(name):
    root = REPO / name
    summary = read(root / "proof_summary.json")
    patients = summary["patients"]
    arms = [arm for row in patients["results"] for arm in row["arms"]]
    values = [arm["val_cindex_mean_exploratory"] for arm in arms]
    by_fold, by_seed = {}, {}
    for arm in arms:
        by_fold.setdefault(arm["fold"], []).append(arm["val_cindex_mean_exploratory"])
        by_seed.setdefault(arm["seed"], []).append(arm["val_cindex_mean_exploratory"])
    fold_means = [st.mean(by_fold[key]) for key in sorted(by_fold)]
    return {
        "output_root": name,
        "proof_summary_sha256": sha(root / "proof_summary.json"),
        "status": summary["status"],
        "transport_added": summary["transport_added"],
        "training_arms": summary["training_arms"],
        "repaired_models_constructed": summary["repaired_models_constructed"],
        "protocol_dependence": summary["protocol_dependence"],
        "scope": summary["scope"],
        "claims_not_supported": summary["claims_not_supported"],
        "pairs_expected": patients["expected_pairs"],
        "pairs_completed": patients["completed_pairs"],
        "missing_pairs": patients["missing_pairs"],
        "generated_utc": patients["generated_utc"],
        "cohort_counts": patients["counts"],
        "per_pair": [
            {
                "fold": arm["fold"],
                "seed": arm["seed"],
                "arm": arm["arm"],
                "epochs": arm["epochs"],
                "optimizer_steps": arm["optimizer_steps"],
                "parameter_count": arm["parameter_count"],
                "repeats": arm["repeats"],
                "defined_cindex_repeats": arm["defined_cindex_repeats"],
                "val_cindex_mean_exploratory": arm["val_cindex_mean_exploratory"],
                "val_cindex_std_population": arm["val_cindex_std_population"],
                "val_cindex_min_exploratory": arm["val_cindex_min_exploratory"],
                "val_cindex_max_exploratory": arm["val_cindex_max_exploratory"],
                "patient_risk_std_mean": arm["patient_risk_std_mean"],
                "patient_risk_std_max": arm["patient_risk_std_max"],
                "patient_risk_range_max": arm["patient_risk_range_max"],
                "evaluation_randomness_observed": arm["evaluation_randomness_observed"],
                "selector_parameter_count": arm["selector_parameter_count"],
                "main_gradient_none_parameter_count": arm["main_gradient_none_parameter_count"],
                "main_gradient_connected_parameter_count": arm["main_gradient_connected_parameter_count"],
                "disconnected_selector_status": arm["disconnected_selector_status"],
                "same_initial_state_sha256": arm["same_initial_state_sha256"],
                "checkpoint_sha256": arm["checkpoint_sha256"],
                "peak_allocated_cuda_bytes": arm["peak_allocated_cuda_bytes"],
                "elapsed_seconds": arm["elapsed_seconds"],
            }
            for arm in arms
        ],
        "descriptive_aggregate": {
            "pairs": len(values),
            "grand_mean": st.mean(values),
            "min": min(values),
            "max": max(values),
            "cross_run_sample_std": st.stdev(values),
            "per_fold_mean_over_seeds": {str(key): st.mean(by_fold[key]) for key in sorted(by_fold)},
            "per_seed_mean_over_folds": {str(key): st.mean(by_seed[key]) for key in sorted(by_seed)},
            "between_fold_std_of_fold_means": st.stdev(fold_means),
            "per_repeat_std_population_min": min(arm["val_cindex_std_population"] for arm in arms),
            "per_repeat_std_population_max": max(arm["val_cindex_std_population"] for arm in arms),
            "unique_initial_state_sha256": len({arm["same_initial_state_sha256"] for arm in arms}),
            "unique_checkpoint_sha256": len({arm["checkpoint_sha256"] for arm in arms}),
            "elapsed_seconds_total": sum(arm["elapsed_seconds"] for arm in arms),
            "peak_allocated_cuda_bytes_max": max(arm["peak_allocated_cuda_bytes"] for arm in arms),
            "source_warning": patients["validation_metrics_by_arm"]["original"]["warning"],
        },
        "structural_checks": patients["checks"],
        "reported_validation_metrics_by_arm": patients["validation_metrics_by_arm"],
    }


def redundancy_record():
    root = REPO / EVIDENCE / "retained_redundancy_gpu_trisurv_20261022"
    data = read(root / "aggregate_summary.json")
    modalities = {}
    for name, value in data["modalities"].items():
        distributions = value["distributions"]
        modalities[name] = {
            "candidate_counts": value["candidate_counts"],
            "selected_counts": value["selected_counts"],
            "patients_all_unique_pairs_above_0_95": value["patients_all_unique_pairs_above_0_95"],
            "patients_selected_pair_above_0_95": value["patients_selected_pair_above_0_95"],
            "patients_zero_gate_weight_sum": value["patients_zero_gate_weight_sum"],
            "distribution_means": {key: entry["mean"] for key, entry in distributions.items()},
        }
    return {
        "route": "retained_native_redundancy",
        "aggregate_summary_sha256": sha(root / "aggregate_summary.json"),
        "recorded_utc": data["recorded_utc"],
        "input_scope": data["scope"],
        "server_new_results": data["server_new_results"],
        "patient_count": data["patient_count"],
        "fold": data["fold"],
        "seed": data["seed"],
        "historical_epochs": data["historical_epochs"],
        "historical_optimizer_steps": data["historical_optimizer_steps"],
        "checkpoint_sha256_recorded": data["checkpoint_sha256_recorded"],
        "verified_historical_output_files": data["verified_historical_output_files"],
        "historical_native_source_matches_current": data["historical_native_source_matches_current"],
        "new_model_forward_passes": data["new_model_forward_passes"],
        "new_training_steps": data["new_training_steps"],
        "metric_definition": data["metric"],
        "cosine_0_95_is_descriptive_not_clinical_threshold": data["cosine_0_95_is_descriptive_not_clinical_threshold"],
        "modalities": modalities,
        "conclusion_boundary": data["conclusion_boundary"],
        "private_artifacts_not_published": ["patient_metrics.json"],
    }


def sparsity_off_record():
    root = REPO / EVIDENCE / "sparsity_off_gpu_trisurv_20261022"
    data = read(root / "aggregate_summary.json")
    receipt = read(root / "execution_receipt.json")
    arms = {}
    for name, value in data["arms"].items():
        arms[name] = {
            "heads": {head: entry["cindex"] for head, entry in value["heads"].items()},
            "native_nll_per_patient": {head: entry.get("native_nll_per_patient") for head, entry in value["heads"].items()},
            "differences_vs_native": value["differences_vs_native"],
        }
    return {
        "route": "sparsity_off_frozen_checkpoint_inference",
        "aggregate_summary_sha256": sha(root / "aggregate_summary.json"),
        "execution_receipt_sha256": sha(root / "execution_receipt.json"),
        "protocol": data["protocol"],
        "patients": data["patients"],
        "fold": data["fold"],
        "seed": data["seed"],
        "checkpoint_sha256": data["checkpoint_sha256"],
        "new_training_steps": data["new_training_steps"],
        "new_slot_generation": data["new_slot_generation"],
        "new_full_model_forward_calls": data["new_full_model_forward_calls"],
        "new_native_downstream_readout_sweeps": data["new_native_downstream_readout_sweeps"],
        "controls_max_abs_errors": data["controls_max_abs_errors"],
        "slot_vectors_before_gate_are_identical_by_design": data["slot_vectors_before_gate_are_identical_by_design"],
        "arms": arms,
        "conclusion_boundary": data["conclusion_boundary"],
        "runner_sha256": receipt["runner_sha256"],
        "source_hashes": receipt["source_hashes"],
        "model_state_unchanged": receipt["model_state_unchanged"],
        "original_inputs_unchanged": receipt["original_inputs_unchanged"],
        "external_rng_unchanged": receipt["external_rng_unchanged"],
        "private_artifacts_not_published": ["predictions.npz", "patient_order.json"],
    }


def p1_observations_record():
    root = REPO / EVIDENCE / "p1_observations_gpu_trisurv_20261022"
    receipt = read(root / "receipt.json")
    return {
        "route": "p1_native_slot_observation_export",
        "receipt_sha256": sha(root / "receipt.json"),
        "schema": receipt["schema"],
        "output_file_count": len(receipt["outputs"]),
        "observations_sha256": receipt["outputs"]["observations.json"],
        "patient_provenance_sha256": receipt["outputs"]["patient_provenance.json"],
        "observer_source_sha256": receipt["observer_source_sha256"],
        "exporter_source_sha256": receipt["exporter_source_sha256"],
        "original_artifacts_unchanged": receipt["original_artifacts_unchanged"],
        "new_optimizer_steps": 0,
        "input_scope": "local_trisurv_gpu_current_training_run_real_BLCA_cached_validation",
        "note": "逐患者记录与逐患者原始数组含患者标识，仅保留在忽略的local_evidence，不进入公开树；公开仅登记收据与文件哈希。",
        "private_artifacts_not_published": ["observations.json", "patient_provenance.json", "raw/*.npz"],
    }


def observe_record():
    root = REPO / EVIDENCE / "original_questions_gpu_trisurv_observe_01"
    receipt = read(root / "execution_receipt.json")
    return {
        "route": "original_questions_observe_phase",
        "execution_receipt_sha256": sha(root / "execution_receipt.json"),
        "status": receipt["status"],
        "phase": receipt["phase"],
        "started_utc": receipt["started_utc"],
        "finished_utc": receipt["finished_utc"],
        "environment": receipt["environment"],
        "source_hashes": receipt["source_hashes"],
        "helper_sha256": receipt["helper_sha256"],
        "runner_sha256": receipt["runner_sha256"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "source_unchanged": receipt["source_unchanged"],
        "patient_outputs_private": receipt["patient_outputs_private"],
        "result": receipt["result"],
        "observation_settings": {
            "fold": receipt["config"]["fold"],
            "seed": receipt["config"]["seed"],
            "repeats": receipt["config"]["repeats"],
            "reference_iterations": receipt["config"]["reference_iterations"],
            "capacity_counts": receipt["config"]["capacity_counts"],
            "capacity_modality": receipt["config"]["capacity_modality"],
            "dictionary_train_patients": receipt["config"]["dictionary_train_patients"],
            "replay_atol": receipt["config"]["replay_atol"],
        },
        "private_artifacts_not_published": ["patient_records.json", "objective_gradients.json", "patient_*_native.npz"],
    }


def build():
    five = queue_run(FIVE_FOLD)
    five3 = queue_run(FIVE_FOLD_3SEED)
    value = {
        "schema": "slotspe_gpu_trisurv_20261022_record_v1",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "SPE-09",
        "scope": "Record-only registration of already-completed local GPU runs and already-completed read-only or frozen-checkpoint checks. No new execution.",
        "environment": {
            "torch": "2.10.0+cu128",
            "cuda_runtime": "12.8",
            "gpu": "NVIDIA GeForce RTX 5090",
        },
        "shared_queue_protocol": {
            "experiment": "original_audit",
            "arms": ["original"],
            "epochs": 3,
            "patches": 256,
            "batch_size": 8,
            "eval_repeats": 20,
            "lr": 0.0005,
            "torch_threads": 2,
            "parameter_count": 33093390,
            "optimizer_steps_per_pair": 114,
            "transport_added": False,
            "repaired_models_constructed": False,
            "feature_set": "uni2-h",
        },
        "queue_runs": {"blca_5fold_seed3": five, "blca_5fold_3seed": five3},
        "read_only_and_frozen_checks": {
            "retained_native_redundancy": redundancy_record(),
            "sparsity_off_frozen_checkpoint": sparsity_off_record(),
            "p1_native_slot_observations": p1_observations_record(),
            "original_questions_observe": observe_record(),
        },
        "structural_observations": {
            "pairs_with_disconnected_selector": sum(
                run["structural_checks"]["original_disconnection_observed_pairs"] for run in (five, five3)
            ),
            "pairs_total": five["pairs_completed"] + five3["pairs_completed"],
            "selector_parameter_count": 8,
            "main_gradient_connected_parameter_count": 0,
            "interpretation": "选槽支路对融合主损失无梯度在全部20个已训练对中一致重现；这是P7的局部结构事实，不等于整个Slot Attention缺少生存监督，也不构成患者预测损害。",
        },
        "descriptive_measurement_spread": {
            "five_fold_seed3_mean": five["descriptive_aggregate"]["grand_mean"],
            "five_fold_seed3_cross_run_std": five["descriptive_aggregate"]["cross_run_sample_std"],
            "five_fold_3seed_mean": five3["descriptive_aggregate"]["grand_mean"],
            "five_fold_3seed_cross_run_std": five3["descriptive_aggregate"]["cross_run_sample_std"],
            "five_fold_3seed_between_fold_std": five3["descriptive_aggregate"]["between_fold_std_of_fold_means"],
            "five_fold_3seed_per_seed_means": five3["descriptive_aggregate"]["per_seed_mean_over_folds"],
            "note": "折间差异远大于种子间差异。此为测量离散度的描述，不是性能结论，也不构成对任何已登记对照的推翻。",
        },
        "checks": {
            "new_training_runs_this_stage": 0,
            "new_model_forward_passes_this_stage": 0,
            "new_patient_interventions_this_stage": 0,
            "server_execution_this_stage": False,
            "original_source_modified": False,
            "default_queue_config_modified": False,
            "results_directory_git_ignored": True,
            "local_evidence_git_ignored": True,
            "patient_identifiers_staged_to_git": False,
            "host_absolute_paths_staged_to_git": False,
        },
        "patient_predictive_harm_established": False,
        "correction_efficacy_established": False,
    }
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true")
    group.add_argument("--plan", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    sources = [
        REPO / FIVE_FOLD / "proof_summary.json",
        REPO / FIVE_FOLD_3SEED / "proof_summary.json",
        REPO / EVIDENCE / "retained_redundancy_gpu_trisurv_20261022/aggregate_summary.json",
        REPO / EVIDENCE / "sparsity_off_gpu_trisurv_20261022/aggregate_summary.json",
        REPO / EVIDENCE / "sparsity_off_gpu_trisurv_20261022/execution_receipt.json",
        REPO / EVIDENCE / "p1_observations_gpu_trisurv_20261022/receipt.json",
        REPO / EVIDENCE / "original_questions_gpu_trisurv_observe_01/execution_receipt.json",
    ]
    missing = [str(path.relative_to(REPO)) for path in sources if not path.is_file()]
    plan = {
        "schema": "slotspe_gpu_trisurv_record_plan_v1",
        "mode": "run" if args.run else "plan",
        "output_dir": str(args.output_dir),
        "ready": not missing,
        "missing_sources": missing,
        "source_sha256": {str(path.relative_to(REPO)): sha(path) for path in sources if path.is_file()},
        "will_publish": ["aggregate queue metrics", "aggregate redundancy statistics",
                         "aggregate sparsity-off readout metrics", "receipt hashes only"],
        "will_not_publish": ["patient identifiers", "per-patient arrays", "checkpoints", "host absolute paths"],
        "new_training_runs": 0,
        "new_model_forward_passes": 0,
        "server_execution": False,
    }
    if not args.run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if missing:
        parser.error("Missing source artifacts: " + ", ".join(missing))

    value = build()
    target = args.output_dir / "five_fold_and_mechanism_records.json"
    guard(value, target.name)
    write(target, value)
    print(json.dumps({"status": "completed", "output": str(target.relative_to(REPO)),
                      "sha256": sha(target), "bytes": target.stat().st_size,
                      "new_training_runs": 0, "new_model_forward_passes": 0,
                      "server_execution": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())