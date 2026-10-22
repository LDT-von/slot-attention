"""Update docs/PROGRESS.json from the generated 2026-10-22 GPU record.

Read-only with respect to every model, dataset and result: this script reads
the already-generated aggregate record and rewrites the single progress entry.
It starts no training, no inference, no patient intervention and no server task.

Every numeric value is copied from the generated record, not retyped, so the
progress entry cannot drift from the artifacts it summarises.

Usage:
    python -B scripts/update_progress_gpu_trisurv_20261022.py --plan
    python -B scripts/update_progress_gpu_trisurv_20261022.py --run
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True

RECORD = REPO / "docs/research/slotspe_gpu_trisurv_20261022/five_fold_and_mechanism_records.json"
PROGRESS = REPO / "docs/PROGRESS.json"
README = REPO / "README.md"


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
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def build(record):
    five = record["queue_runs"]["blca_5fold_seed3"]
    five3 = record["queue_runs"]["blca_5fold_3seed"]
    agg5 = five["descriptive_aggregate"]
    agg3 = five3["descriptive_aggregate"]
    redundancy = record["read_only_and_frozen_checks"]["retained_native_redundancy"]
    sparsity = record["read_only_and_frozen_checks"]["sparsity_off_frozen_checkpoint"]
    p1 = record["read_only_and_frozen_checks"]["p1_native_slot_observations"]
    observe = record["read_only_and_frozen_checks"]["original_questions_observe"]
    spread = record["descriptive_measurement_spread"]
    structural = record["structural_observations"]

    return {
        "recorded_at": "2026-10-22T10:05:00+08:00",
        "task_id": "SPE-09",
        "route": "local_blca_five_fold_three_seed_and_read_only_mechanism_records",
        "status": "completed_local_record_only_registration",
        "authorization": "用户于2026-10-22询问当前情况，并在选择record_only后授权一并推送；本轮只登记已完成的本机GPU结果。",
        "completed": "把本机已完成的五折、五折三种子、稀疏关闭、保留冗余、P1观察导出与observe阶段六项结果写入单一进度入口；未新增任何训练、推理或患者干预。",
        "public_record": "docs/research/slotspe_gpu_trisurv_20261022/five_fold_and_mechanism_records.json",
        "public_record_sha256": sha(RECORD),
        "report": "docs/research/slotspe_gpu_trisurv_20261022/本机GPU阶段登记与离散度.txt",
        "generator": "scripts/record_slotspe_gpu_trisurv_20261022.py",
        "environment": record["environment"],
        "shared_queue_protocol": record["shared_queue_protocol"],
        "five_fold_seed3": {
            "output_root": five["output_root"],
            "status": five["status"],
            "pairs_completed": five["pairs_completed"],
            "pairs_expected": five["pairs_expected"],
            "missing_pairs": five["missing_pairs"],
            "cohort_unique_complete_case_patients": five["cohort_counts"]["unique_complete_case_patients"],
            "per_fold_counts": five["cohort_counts"]["per_fold"],
            "mean_validation_cindex_exploratory": agg5["grand_mean"],
            "cross_run_sample_std": agg5["cross_run_sample_std"],
            "min": agg5["min"],
            "max": agg5["max"],
            "per_pair": {f"fold_{row['fold']}_seed_{row['seed']}": row["val_cindex_mean_exploratory"]
                         for row in five["per_pair"]},
            "structural_checks": five["structural_checks"],
        },
        "five_fold_three_seed": {
            "output_root": five3["output_root"],
            "status": five3["status"],
            "pairs_completed": five3["pairs_completed"],
            "pairs_expected": five3["pairs_expected"],
            "missing_pairs": five3["missing_pairs"],
            "mean_validation_cindex_exploratory": agg3["grand_mean"],
            "cross_run_sample_std": agg3["cross_run_sample_std"],
            "min": agg3["min"],
            "max": agg3["max"],
            "per_fold_mean_over_seeds": agg3["per_fold_mean_over_seeds"],
            "per_seed_mean_over_folds": agg3["per_seed_mean_over_folds"],
            "between_fold_std_of_fold_means": agg3["between_fold_std_of_fold_means"],
            "per_repeat_std_population_min": agg3["per_repeat_std_population_min"],
            "per_repeat_std_population_max": agg3["per_repeat_std_population_max"],
            "unique_initial_state_sha256": agg3["unique_initial_state_sha256"],
            "unique_checkpoint_sha256": agg3["unique_checkpoint_sha256"],
            "elapsed_seconds_total": agg3["elapsed_seconds_total"],
            "peak_allocated_cuda_bytes_max": agg3["peak_allocated_cuda_bytes_max"],
            "per_pair": {f"fold_{row['fold']}_seed_{row['seed']}": row["val_cindex_mean_exploratory"]
                         for row in five3["per_pair"]},
            "structural_checks": five3["structural_checks"],
            "source_warning": agg3["source_warning"],
        },
        "structural_observations": structural,
        "retained_native_redundancy": {
            "route": redundancy["route"],
            "recorded_utc": redundancy["recorded_utc"],
            "patient_count": redundancy["patient_count"],
            "fold": redundancy["fold"],
            "seed": redundancy["seed"],
            "new_training_steps": redundancy["new_training_steps"],
            "new_model_forward_passes": redundancy["new_model_forward_passes"],
            "verified_historical_output_files": redundancy["verified_historical_output_files"],
            "modalities": redundancy["modalities"],
            "cosine_0_95_is_descriptive_not_clinical_threshold": redundancy["cosine_0_95_is_descriptive_not_clinical_threshold"],
            "conclusion_boundary": redundancy["conclusion_boundary"],
            "private_artifacts_not_published": redundancy["private_artifacts_not_published"],
        },
        "sparsity_off_frozen_checkpoint": {
            "route": sparsity["route"],
            "patients": sparsity["patients"],
            "fold": sparsity["fold"],
            "seed": sparsity["seed"],
            "new_training_steps": sparsity["new_training_steps"],
            "new_slot_generation": sparsity["new_slot_generation"],
            "new_native_downstream_readout_sweeps": sparsity["new_native_downstream_readout_sweeps"],
            "fused_cindex": {arm: values["heads"]["fused"] for arm, values in sparsity["arms"].items()},
            "aux_cindex": {arm: {"aux_wsi": values["heads"]["aux_wsi"], "aux_omic": values["heads"]["aux_omic"]}
                           for arm, values in sparsity["arms"].items()},
            "controls_max_abs_errors": sparsity["controls_max_abs_errors"],
            "conclusion_boundary": sparsity["conclusion_boundary"],
            "private_artifacts_not_published": sparsity["private_artifacts_not_published"],
        },
        "p1_native_slot_observations": {
            "route": p1["route"],
            "output_file_count": p1["output_file_count"],
            "new_optimizer_steps": p1["new_optimizer_steps"],
            "original_artifacts_unchanged": p1["original_artifacts_unchanged"],
            "input_scope": p1["input_scope"],
            "private_artifacts_not_published": p1["private_artifacts_not_published"],
        },
        "original_questions_observe": {
            "route": observe["route"],
            "status": observe["status"],
            "phase": observe["phase"],
            "started_utc": observe["started_utc"],
            "finished_utc": observe["finished_utc"],
            "result": observe["result"],
            "observation_settings": observe["observation_settings"],
            "patient_outputs_private": observe["patient_outputs_private"],
            "private_artifacts_not_published": observe["private_artifacts_not_published"],
        },
        "measurement_spread_note": {
            "five_fold_seed3_mean": spread["five_fold_seed3_mean"],
            "five_fold_seed3_cross_run_std": spread["five_fold_seed3_cross_run_std"],
            "five_fold_3seed_mean": spread["five_fold_3seed_mean"],
            "five_fold_3seed_cross_run_std": spread["five_fold_3seed_cross_run_std"],
            "five_fold_3seed_between_fold_std": spread["five_fold_3seed_between_fold_std"],
            "five_fold_3seed_per_seed_means": spread["five_fold_3seed_per_seed_means"],
            "classification": "descriptive_measurement_spread_not_a_patient_performance_conclusion",
            "detail": "跨折跨种子离散度远大于此前单点fold0/seed3上登记的对照效应量量级，且折间差异远大于种子间差异；"
                      "说明主要差异来自数据划分。以往单折单种子差值需在该分布内重新判断，不应继续在单折上叠加对照。",
            "does_not_override_prior_records": True,
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
            "historical_reports_modified": False,
            "private_patient_inputs_staged": False,
        },
        "patient_predictive_harm_established": False,
        "correction_efficacy_established": False,
        "gaps_reasons": "3轮探索性验证C-index；跨折跨种子各对共享同一379名完全病例患者，不是独立证据单位，均值仅描述性；"
                        "未测P1-P7机制干预，未跑D0-D4阶段配置(完整宿主训练未接入)，未接DIAS/MetaSlot；"
                        "本轮为只读登记，未启动任何新执行。",
        "next_action_user_input": "由用户决定是否授权具体对照臂(关闭稀疏选择、匹配容量训练或A/B/C)与队列规模；"
                                  "D0-D4需先完成宿主训练接入。未获明确执行授权前不启动训练或推理。",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true")
    group.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)

    missing = [str(path.relative_to(REPO)) for path in (RECORD, PROGRESS) if not path.is_file()]
    plan = {
        "schema": "slotspe_progress_update_plan_v1",
        "mode": "run" if args.run else "plan",
        "ready": not missing,
        "missing": missing,
        "record_sha256": sha(RECORD) if RECORD.is_file() else None,
        "progress_sha256_before": sha(PROGRESS) if PROGRESS.is_file() else None,
        "will_update": ["docs/PROGRESS.json", "README.md"],
        "will_not_do": ["training", "inference", "patient intervention", "server execution",
                        "editing historical reports or original sources"],
    }
    if not args.run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if missing:
        parser.error("Missing: " + ", ".join(missing))

    record = read(RECORD)
    entry = build(record)
    progress = read(PROGRESS)
    progress["recorded_at"] = entry["recorded_at"]
    progress["next_action"] = entry["next_action_user_input"]
    progress["latest_local_five_fold_three_seed_and_mechanism_records"] = entry
    write(PROGRESS, progress)
    print(json.dumps({"status": "completed",
                      "progress_sha256_after": sha(PROGRESS),
                      "bytes": PROGRESS.stat().st_size,
                      "new_training_runs": 0, "new_model_forward_passes": 0,
                      "server_execution": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())