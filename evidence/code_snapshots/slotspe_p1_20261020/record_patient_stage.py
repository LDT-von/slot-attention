"""Record this completed read-only check without changing another research line."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import copy
import hashlib
import json
import shutil
import sys

ROOT = Path(r"E:\DCT-Reg")
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from survot_rank.research.slotspe_proof.p1_diagnostics import audit_export

OUT = ROOT / "docs/research/slotspe_p1_20261020"
RAW = ROOT / ".review/slotspe_p1_20261020/seed3_native_v2"
def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

audited = audit_export(RAW)
report = read(RAW / "observations.json")
for path, digest in report["source_inputs_sha256"].items():
    assert sha(Path(path)) == digest, path
summary = read(OUT / "descriptive_summary.json")
assert summary["raw_export_root"] == str(RAW)
assert summary["raw_observations_sha256"] == sha(RAW / "observations.json")
assert report["patient_count"] == 71
assert report["candidate_counts"] == {"wsi": [8], "omic": [8]}
assert report["selected_counts"] == {"wsi": [2], "omic": [2]}
assert report["replay"]["native_final_rng_sha256"] == report["replay"]["observed_final_rng_sha256"]
snapshot = ROOT / ".review/slotspe_p1_20261020/observer_source_v2.py"
shutil.copy2(ROOT / "survot_rank/research/slotspe_proof/p1_diagnostics.py", snapshot)
measurement = {
    "task_id": "SPE-08", "date": "2026-10-20", "scope": "historical_local_native_checkpoint_readonly_real_validation",
    "run_root": str(ROOT / ".review/slotspe_proof_20261018/patient"), "raw_export_root": str(RAW),
    "fold": 0, "seed": 3, "epochs_previously_trained": 3, "old_optimizer_steps": 111,
    "new_training_runs": 0, "new_optimizer_steps": 0, "patient_count": 71, "patches": 256,
    "measurement_versions_run": ["v1", "v2"], "native_validation_passes_including_v1_and_v2": 4,
    "measurement_versions_are_independent_patients": False, "final_export": "v2",
    "historical_prediction_replay_max_error": report["historical_replay_max_logit_difference"],
    "observer_prediction_max_error": report["observer_max_logit_difference"], "native_attention_update_reconstruction_max_error": report["reconstructed_native_update_max_error"],
    "all_rng_states_match": True, "checkpoint_tensors_unchanged": report["checkpoint_tensors_unchanged"],
    "original_artifacts_and_source_bytes_unchanged": True, "offline_raw_audit": audited,
    "P1_fixed_count_observed": True, "P1_harm_proved": False,
    "P2_native_attention_overlap_observed": True, "P2_functional_redundancy_or_harm_proved": False,
    "server_SPE09_run": "reported_completed_not_independently_verified", "server_raw_results_received": False,
    "new_tests": {"final_passed": 8, "scope": "invented inputs/serialization only", "counts_not_additive_across_reruns": True,
        "first_attempt": "2 passed and 1 temporary-directory setup error; measurement_tests.xml retained", "final_xml": "measurement_tests_complete.xml"},
    "observer_source_snapshot": str(snapshot), "observer_source_sha256": sha(snapshot),
    "observation_receipt_sha256": sha(RAW / "receipt.json"), "human_report": "P1_真实患者只读检查.txt",
    "next_scope": "P2：先查高读取重叠是否仍具有独立内容和不同计算贡献；不自动训练新模型或扩大患者网格。"
}
write(OUT / "measurement_receipt.json", measurement)

progress_path = ROOT / "docs/THREE_LINE_PROGRESS.json"
human_path = ROOT / "docs/三条研究路线_工作总览_20261019.txt"
before = read(progress_path)
progress = copy.deepcopy(before)
now = datetime.now(timezone(timedelta(hours=8)))
backup = ROOT / "docs/history" / ("slotspe_p1_patient_" + now.strftime("%Y%m%d_%H%M%S"))
backup.mkdir(parents=True, exist_ok=False)
for path in (progress_path, human_path):
    shutil.copy2(path, backup / path.name)
eid = "EV-SPE-P1-PATIENT-20261020"
paths = ["docs/research/slotspe_p1_20261020/" + name for name in ("measurement_receipt.json", "artifact_inventory.json", "descriptive_summary.json", "P1_真实患者只读检查.txt", "measurement_tests_complete.xml")]
progress["evidence"].append({"id": eid, "type": "historical_native_trained_checkpoint_readonly_observation",
    "details": "原版历史seed3末轮71人只读重放，预测/更新差0，全RNG终态一致；8/2固定实测，高attention重叠仅为描述。未训练，非SPE09服务器回传。",
    "paths": paths, "local_file_sha256": {p: sha(ROOT / p) for p in paths}, "raw_export": str(RAW)})
task = next(t for t in progress["tasks"] if t["id"] == "SPE-08")
task["last_checked"] = "2026-10-20"
task["status"] = "partially_completed_local"
task["next_action"] = measurement["next_scope"]
task["evidence_refs"].append(eid)
task["last_measurement"] = {"type": measurement["scope"], "patients": 71, "checkpoint_seed": 3, "P1_fixed_count_observed": True,
    "P1_predictive_harm": "not_established", "P2_attention_overlap": "observed_on_one_historical_checkpoint_one_rng_forward",
    "P2_functional_redundancy_or_harm": "not_established", "report": paths[0], "final_synthetic_measurement_checks": 8, "checks_not_additive": True}
task["evidence_level"] = "literature_source_plus_historical_trained_real_patient_readonly_observations"
task["patient_effect"] = "fixed_count_and_attention_overlap_observed_predictive_harm_not_proved"
line = next(l for l in progress["lines"] if l["id"] == "L2")
line["latest_native_readonly_check"] = task["last_measurement"]
line["evidence_refs"].append(eid)
line["bottleneck"] = "已找到并核验历史本机真实checkpoint；P1固定数和P2注意力读取高重叠已观察。是否功能冗余/损害仍待受控核查，SPE09服务器原始产物仍待验收。"
progress["current_priority_note"] = "SlotSPE逐项检查：P1历史71人8/2固定实测，P2读取重叠待功能核查；SPE09原始回传单独验收。"
progress["change_log"].append({"date": "2026-10-20", "scope": "SlotSPE_P1_historical_real_patient_readonly", "details": "只读原版seed3历史checkpoint与71人固定缓存；原预测/observer差0、全RNG一致，导出raw并离线重算验收；最终8项测量检查通过，非服务器SPE09回传。", "backup": str(backup.relative_to(ROOT)).replace("\\", "/")})
assert [t for t in before["tasks"] if not t["id"].startswith("SPE-")] == [t for t in progress["tasks"] if not t["id"].startswith("SPE-")]
assert [l for l in before["lines"] if l["id"] != "L2"] == [l for l in progress["lines"] if l["id"] != "L2"]
write(progress_path, progress)
human = human_path.read_text(encoding="utf-8-sig")
anchor = "2．SlotSPE：先干净原版检查问题，A/B/C用于后续归因。"
new = "\n2026年10月20日第二阶段：已复用10月18日本机历史原版seed3/3轮checkpoint，对71名验证患者只读重放，预测最大差0、observer差0、全RNG终态一致，原权重/缓存/源码不变。\nP1两模态实际均8候选/选2；P2末轮WSI槽对平均头attention overlap中位0.999428，逐头中位0.998662，说明读取分布高度重合，不能据此宣布功能冗余、坍缩或预测损害。\n报告：docs/research/slotspe_p1_20261020/P1_真实患者只读检查.txt；raw目录为.review/slotspe_p1_20261020/seed3_native_v2。最终8项合成测量/防篡改检查通过，不与旧检查或患者数相加。\n这次有真实历史缓存只读前向、没有新增训练；它不是SPE09服务器302/77数据回传。下一步查P2的独立内容和计算贡献，其他线不变。\n"
assert anchor in human
human_path.write_text(human.replace(anchor, anchor + new, 1), encoding="utf-8")
write(OUT / "progress_update_validation.json", {"non_slotspe_tasks_unchanged": True, "non_slotspe_lines_unchanged": True,
    "backup": str(backup), "raw_audit": audited, "original_sources_rehashed_unchanged": True, "new_training_runs": 0,
    "new_readonly_patient_forward_work": True})
print(json.dumps({"patient_count": 71, "raw_audit": audited["status"], "all_rng_match": True,
    "P1_fixed_count": True, "P1_harm_proved": False, "P2_attention_overlap": True, "server_raw_received": False,
    "report": str(OUT / "P1_真实患者只读检查.txt"), "backup": str(backup)}, ensure_ascii=False))
