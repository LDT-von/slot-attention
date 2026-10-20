"""Audit/record the completed P2 stage; no patient inference or training."""
import ast
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(r"E:\DCT-Reg")
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from survot_rank.research.slotspe_proof.p2_diagnostics import audit, _read, _sha, _write

RAW = ROOT / ".review/slotspe_p2_20261020/seed3_native_v1"
OUT = ROOT / "docs/research/slotspe_p2_20261020"
probe = ROOT / ".review/slotspe_p2_20261020/probe_source_v1.py"
current = ROOT / "survot_rank/research/slotspe_proof/p2_diagnostics.py"
audited = audit(RAW)
report = _read(RAW / "observations.json")
protocol = _read(RAW / "protocol.json")
for path, expected in report["source_inputs_sha256"].items():
    # The first probe's complete source was frozen before measurement. Only
    # offline metadata validation was strengthened afterward; do not re-run or
    # relabel the measured patient probe as a new independent experiment.
    actual = probe if Path(path) == current else Path(path)
    assert _sha(actual) == expected, path
assert _sha(probe) == _read(RAW / "receipt.json")["probe_source_sha256"]
def functions(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
old, new = functions(probe), functions(current)
unchanged_functions = ["protocol", "pair_choices", "transform", "native_readout", "perturbation_statistics", "risks", "summarize", "run"]
assert all(old[k] == new[k] for k in unchanged_functions)
auditor_snapshot = ROOT / ".review/slotspe_p2_20261020/auditor_source_v2.py"
assert not auditor_snapshot.exists()
shutil.copy2(current, auditor_snapshot)
tests = ET.parse(OUT / "measurement_tests_hardened.xml").getroot()
assert sum(int(t.attrib.get("tests", 0)) for t in tests.iter("testsuite")) == 10
assert sum(int(t.attrib.get(k, 0)) for t in tests.iter("testsuite") for k in ("failures", "errors", "skipped")) == 0

_write(OUT / "descriptive_summary.json", {"raw_export": str(RAW), "observation_sha256": _sha(RAW / "observations.json"),
    "protocol": protocol, "summaries": report["summaries"],
    "closest_pair_selected_members": {m: {str(i): sum(r["pair_choices"][m]["closest_pair_selected_members"] == i for r in report["records"]) for i in range(3)} for m in ("wsi", "omic")},
    "patients": report["patients"], "independent_patient_count": 71, "patient_pair_count_is_not_independent_sample_count": True,
    "outcomes_not_used": True, "C_index_not_evaluated": True})
measurement = {"task": "SPE-08/P2", "date": "2026-10-20", "scope": report["scope"], "raw_export": str(RAW),
    "fold": 0, "seed": 3, "previous_training_epochs": 3, "previous_optimizer_steps": 111,
    "patients": 71, "patches_in_original_cache": 256, "new_training_runs": 0, "new_optimizer_steps": 0,
    "new_full_patient_encoder_passes": 0, "retained_final_slot_conditions": len(protocol["arms"]),
    "controls": 4, "interventions": 7, "conditions_are_not_independent_patients_or_training_runs": True,
    "native_downstream_replay_max_error": report["native_downstream_replay_max_error"],
    "permutation_max_logit_error": report["permutation_max_logit_error"],
    "max_modal_mean_error": max(report["max_modal_mean_error_by_arm"].values()),
    "rng_unchanged": report["rng_before"] == report["rng_after"], "checkpoint_tensors_unchanged": True,
    "original_model_cache_P1_outputs_unchanged": True, "offline_raw_audit": audited,
    "source_snapshots": {"measurement_v1": {"path": str(probe), "sha256": _sha(probe)},
        "offline_auditor_v2": {"path": str(auditor_snapshot), "sha256": _sha(auditor_snapshot)}},
    "audit_hardening_after_measurement": {"changed": "metadata identity/RNG schema/original-input flag validation only",
        "measurement_functions_ast_unchanged": unchanged_functions, "patient_probes_repeated": False},
    "synthetic_checks": {"final_passed": 10, "counts_not_additive": True,
        "first_attempt": "4 passed, 1 temporary-directory setup error; retained measurement_tests.xml",
        "intermediate": "5 passed; retained measurement_tests_complete.xml", "final_xml": "measurement_tests_hardened.xml"},
    "P2_reading_overlap": "observed_on_historical_real_patient_native_attention",
    "P2_final_slots_identical": False, "P2_conditional_readout_sensitivity": "observed_with_fixed_keep_and_mean_preserving_latent_interventions",
    "P2_independent_biological_or_prognostic_information": "not_established", "P2_harmful_redundancy": "not_established",
    "P2_fragmentation": "not_established", "P2_correction_efficacy": "not_tested",
    "server_SPE09_status": "user_reported_completed_not_independently_verified", "server_raw_results_received": False,
    "raw_receipt_sha256": _sha(RAW / "receipt.json"),
    "next": "P2本地条件干预阶段完成；等待SPE09原始产物再按相同协议复核。后续逐项到P3，不新增训练预算或模块。"}
_write(OUT / "measurement_receipt.json", measurement)

lines = ["P2：槽冗余/碎片化核查——第三阶段（2026-10-20）", "",
    "1. 路线和任务号", "SlotSPE问题存在性，SPE-08/P2。只查原版，未加运输或新模块。", "",
    "2. 本轮完成与结论", "P2已发现真实患者的槽读取高度重合；本轮再查它们是否可以完全互相替代。",
    "复用10月18日本机原版fold0/seed3、3轮/111步末轮checkpoint，以及P1保留的71名验证患者末轮槽；没有新训练，也没有重新编码患者patch。",
    "固定原先keep，使用原生cross/self/预测头；11个计算条件=4个对照+7个干预。71名仍是同一批患者，不能按条件或2485个配对扩充样本数。",
    "P1病理槽注意力重叠系数中位0.999428，逐头中位0.998662，读取高度重合已有实测；槽向量并不完全相同。",
    "本次干预均保留每模态8槽和槽均值。最近对按逐头overlap均值选择，与患者结局无关。实际选中二槽单独合并，避免仅靠最近对选择作结论。", "",
    "操作                         风险绝对变化中位数       最大值             风险排序翻转",]
for arm, label in (("wsi_closest_pair", "合并最相似病理槽对"), ("omic_closest_pair", "合并最相似组学槽对"),
        ("wsi_selected_pair", "合并原来选中的病理二槽"), ("omic_selected_pair", "合并原来选中的组学二槽"),
        ("wsi_all_mean", "病理8槽全换成其均值"), ("omic_all_mean", "组学8槽全换成其均值"), ("both_all_mean", "两模态各自8槽全均值")):
    s = report["summaries"][arm]
    lines.append(f"{label:<22} {s['absolute_risk_difference']['median']:.9f}             {s['absolute_risk_difference']['max']:.9f}       {s['risk_order_reversals']}")
lines += ["", "原风险分数跨患者范围0.255700515；上述数值是预测敏感性，未计算C-index，不能解读为提分、掉分或临床损害。",
    "最相似病理槽对合并的最大logit变化中位0.000095129，71/71超过预先设定的1e-5数值阈值；阈值不是临床阈值。",
    "最近病理槽对：35人两槽均未选、35人一选一未选、1人两槽均选。原cross仍使用全部槽，不能把这些结果归因成选槽器单独的作用。",
    "白话结论：读得很像确实存在，但相似槽还不是完全可互换的复制品；原模型确实使用了槽间差异。差异可能含初始化残留，尚不能认定其代表不同组织或独立预后信息。",
    "本轮预测变化总体较小；这支持数值敏感性，不能直接支持P2是严重预测故障。槽间均值干预可能偏离训练分布，不能作为修正方法效果。", "",
    "3. 对应文件", str(OUT / "measurement_receipt.json"), str(OUT / "descriptive_summary.json"),
    str(RAW / "observations.json"), str(RAW / "protocol.json"), str(RAW / "receipt.json"), str(RAW / "raw"),
    str(current), str(ROOT / "scripts/check_slotspe_p2.py"), "本轮未提交、未push。", "",
    "4. 检查与服务器", "代码/合成测量检查最终10项通过，不与旧版5项或首次4项相加。首次1项临时目录设置错误已修正，失败XML保留。",
    "患者条件干预：本机实际执行，原生下游重放logit最大差0；仅置换槽顺序最大误差3.576278687e-7，风险最大误差1.909138123e-7。",
    "槽均值最大偏差2.384185791e-7；原权重、缓存、P1产物及桌面原版源码未变；Torch CPU/CUDA随机状态未变。",
    "逐患者原始/干预槽、keep、logits已保存并核对哈希，离线重算选择规则、干预、风险差、排序和全部摘要通过。",
    "验收后仅加固元数据核查，未重跑患者计算；测量源v1与验收源v2均单独封存，测量函数AST一致。",
    "服务器SPE09：只有用户/服务器AI文字完成说明，原始产物尚未回传验收。这71人是本机历史289/71数据，不能冒充服务器302/77结果。", "",
    "5. 未完成及原因", "有害功能冗余、组织/事件碎片化、独立预后信息及修正有效性均未证实。只有一个短程checkpoint的一次固定槽实现；无真实组织/事件标注，也无相应修正对照。",
    "不把上述干预当成可部署修正版，不依据观察结果改阈值，不强迫出现故障。", "",
    "6. 已安排下一步与所需材料", measurement["next"],
    "需要服务器AI回传实际配置/命令/日志、逐患者预测、槽干预与权重原始包；不需要SSH。本阶段不再追加患者训练。", ""]
human_report = OUT / "P2_真实患者槽干预核查.txt"
assert not human_report.exists()
human_report.write_text("\n".join(lines), encoding="utf-8")

progress_path = ROOT / "docs/THREE_LINE_PROGRESS.json"
human_path = ROOT / "docs/三条研究路线_工作总览_20261019.txt"
before = _read(progress_path)
progress = copy.deepcopy(before)
human_before = human_path.read_text(encoding="utf-8-sig")
backup = ROOT / "docs/history" / ("slotspe_p2_patient_" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S"))
backup.mkdir(parents=True, exist_ok=False)
for path in (progress_path, human_path):
    shutil.copy2(path, backup / path.name)
eid = "EV-SPE-P2-PATIENT-20261020"
assert not any(e["id"] == eid for e in progress["evidence"])
paths = [str((OUT / n).relative_to(ROOT)).replace("\\", "/") for n in ("measurement_receipt.json", "descriptive_summary.json", human_report.name, "measurement_tests_hardened.xml")]
progress["evidence"].append({"id": eid, "type": "historical_native_trained_final_slot_conditional_intervention",
    "details": "71名历史真实患者，固定keep、8槽数和槽均值，原生下游重放差0；均值合并会产生较小预测差，尚不证明有害冗余/碎片化/修正效果。最终10项工具检查通过，非服务器回传。",
    "paths": paths, "local_file_sha256": {p: _sha(ROOT / p) for p in paths}, "raw_export": str(RAW)})
task = next(t for t in progress["tasks"] if t["id"] == "SPE-08")
task.setdefault("historical_measurements", []).append(copy.deepcopy(task["last_measurement"]))
task["last_measurement"] = {"type": measurement["scope"], "patients": 71, "checkpoint_seed": 3,
    "P2_reading_overlap": "observed", "P2_conditional_prediction_sensitivity": "observed_small_differences_with_frozen_keep",
    "P2_harmful_redundancy_fragmentation_or_correction_efficacy": "not_established", "report": paths[0],
    "final_synthetic_measurement_checks": 10, "checks_not_additive": True, "new_optimizer_steps": 0, "server_raw_received": False}
task["next_action"] = measurement["next"]
task["evidence_refs"].append(eid)
task["patient_effect"] = "reading_overlap_and_conditional_prediction_sensitivity_observed_harm_not_proved"
task["expected_outputs"].append(str(human_report.relative_to(ROOT)).replace("\\", "/"))
task.setdefault("historical_local_stages", []).append(copy.deepcopy(task["local_stage"]))
task["local_stage"] = {"date": "2026-10-20", "completed": "P2历史71人末轮槽条件干预、原始验收与报告",
    "new_training_runs": 0, "patient_failure_proof_completed": False, "originals_unchanged": True}
line = next(l for l in progress["lines"] if l["id"] == "L2")
line["latest_native_readonly_check"] = copy.deepcopy(task["last_measurement"])
line["evidence_refs"].append(eid)
line["bottleneck"] = "P2已观察读取高重叠和较小的槽差异预测作用；有害冗余/碎片化/修正效果仍未证。SPE09服务器原始产物待验收，随后逐项到P3。"
progress["current_priority_note"] = "SlotSPE逐项检查：P1/P2本地只读阶段已记录；P2高读取重叠存在，有害冗余未证；下一项P3，SPE09原始回传单独验收。"
progress["change_log"].append({"date": "2026-10-20", "scope": "SlotSPE_P2_final_slot_conditional_interventions",
    "details": "71名历史原版患者末轮槽，4对照7干预，重放差0；差异影响预测但未证明损害。10项合成工具检查及原始验收通过，未训练/未push/非服务器回传。",
    "backup": str(backup.relative_to(ROOT)).replace("\\", "/")})
assert [t for t in before["tasks"] if t["id"] != "SPE-08"] == [t for t in progress["tasks"] if t["id"] != "SPE-08"]
assert [l for l in before["lines"] if l["id"] != "L2"] == [l for l in progress["lines"] if l["id"] != "L2"]
progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
anchor = "2．SlotSPE：先干净原版检查问题，A/B/C用于后续归因。"
addition = "\n2026年10月20日第三阶段P2：已对同一历史71人的末轮槽做4对照+7干预，原生下游重放差0、置换误差3.58e-7；原keep、8槽和槽均值保持。\n相似病理槽对合并后risk变化中位0.000019；病理8槽全均值后中位0.000987、最大0.002781。读取高重叠已见，但槽并非完全可互换；尚不能认定有害冗余、碎片化或修正有效。\n本机真实历史槽条件干预已完成，最终10项合成工具检查与原始验收通过，未新增训练/未push，不是SPE09服务器回传。报告：docs/research/slotspe_p2_20261020/P2_真实患者槽干预核查.txt。\n下一项P3；SPE09原始包仍待回传。下方为P1及此前阶段记录，患者数/条件/检查次数不相加。\n"
assert anchor in human_before
human = human_before.replace(anchor, anchor+addition, 1).replace("当前最高优先级：SlotSPE八类问题逐项核查，先P1；SPE-09原始产物验收并行。", "当前最高优先级：SlotSPE八类问题逐项核查，P1/P2本地阶段已登记，下一项P3；SPE-09原始产物验收并行。", 1)
assert human.split("3．运输插拔：", 1)[1] == human_before.split("3．运输插拔：", 1)[1]
assert human.split("1．V2：", 1)[1].split(anchor, 1)[0] == human_before.split("1．V2：", 1)[1].split(anchor, 1)[0]
human_path.write_text(human, encoding="utf-8")
_write(OUT / "progress_update_validation.json", {"all_other_tasks_unchanged": True, "non_slotspe_lines_unchanged": True,
    "V2_and_transport_human_sections_unchanged": True, "backup": str(backup), "raw_audit": audited,
    "original_input_hashes_rechecked": True, "frozen_probe_hash_rechecked": True, "new_training_runs": 0})
print(json.dumps({"stage": "P2", "raw_audit": audited, "checks": 10, "report": str(human_report), "backup": str(backup)}, ensure_ascii=False))
