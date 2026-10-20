"""Record audited P4 observations without running patient inference."""
import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(r"E:\DCT-Reg")
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from survot_rank.research.slotspe_proof.p4_diagnostics import audit, _read, _sha, _write
from survot_rank.research.slotspe_proof.p2_diagnostics import audit as audit_p2

RAW = ROOT / ".review/slotspe_p4_20261020/seed3_native_v1"
OUT = ROOT / "docs/research/slotspe_p4_20261020"
audited = audit(RAW)
report = _read(RAW / "observations.json")
negative = _read(OUT / "raw_negative_audit.json")
assert len(negative["checks"]) == 2 and all(c["rejected"] is True for c in negative["checks"])
tests = ET.parse(OUT / "measurement_tests.xml").getroot()
assert sum(int(s.attrib.get("tests", 0)) for s in tests.iter("testsuite")) == 9
assert sum(int(s.attrib.get(k, 0)) for s in tests.iter("testsuite") for k in ("failures", "errors", "skipped")) == 0
snapshot = RAW.parent / "probe_source_v1.py"
assert _sha(snapshot) == _sha(ROOT / "survot_rank/research/slotspe_proof/p4_diagnostics.py") == _read(RAW / "receipt.json")["probe_source_sha256"]
p2_root = ROOT / ".review/slotspe_p2_20261020/seed3_native_v1"
p2_audit = audit_p2(p2_root)
p2 = _read(p2_root / "observations.json")
manifest_path, = [Path(p) for p in report["source_inputs_sha256"] if Path(p).name == "manifest.json"]
manifest = _read(manifest_path)
annotation = {"scope": "only annotations carried by retained experiment evidence; not an exhaustive external-data search",
    "patient_record_fields": list(manifest["patients"][0]), "patient_count": 71,
    "patch_identity_available": True, "patch_features_available": True,
    "tissue_type_per_patch_labels_in_evidence": False, "fixed_slot_biological_type_labels_in_evidence": False,
    "survival_labels_are_tissue_semantics": False,
    "cross_patient_patch_indices_are_shared_semantic_coordinates": False,
    "P4_biological_semantic_drift_assessable_from_this_evidence": False}
_write(OUT / "annotation_inventory.json", annotation)
source = Path(manifest["source"])
excerpts = []
for name, needles in (("models/slot_attention.py", ("self.slots_mu =", "self.slots_logsigma =", "slots = mu + sigma", "self.to_q =", "self.gru =")),
        ("models/SlotSPE.py", ("x_inter = self.cross_attention", "wsi_intra =", "omic_intra =", "x_inter.mean(dim=1)"))):
    path = source / name
    lines = path.read_text(encoding="utf-8").splitlines()
    excerpts.append({"path": str(path), "sha256": _sha(path), "locations": [{"line": i+1, "text": line.strip()}
        for i, line in enumerate(lines) if any(needle in line for needle in needles)]})
_write(OUT / "source_and_permutation_evidence.json", {"excerpts": excerpts,
    "finding": "Shared slot initialization distribution and shared updates; no tissue label attached to array index. This is an anonymous set structure, not itself a survival failure.",
    "prior_real_patient_downstream_permutation_control": {"raw_root": str(p2_root), "audit": p2_audit,
        "receipt_sha256": _sha(p2_root / "receipt.json"), "simultaneous_slot_keep_permutation_max_logit_error": p2["permutation_max_logit_error"],
        "interpretation": "Conditional downstream permutation check, not a claim that permuting only slots while keeping masks fixed is invariant."}})

measurement = {"task": "SPE-08/P4", "date": "2026-10-20", "scope": report["scope"], "raw_export": str(RAW),
    "fold": 0, "seed": 3, "previous_training_epochs": 3, "previous_optimizer_steps": 111,
    "patients": 71, "repeats": 20, "reference_repeat": 0, "paired_repeats_per_patient": 19,
    "paired_repeats_per_modality": 1349, "paired_repeats_are_independent_patients": False,
    "new_training_runs": 0, "new_optimizer_steps": 0, "new_module_replays": 20, "new_full_patient_encoder_passes": 0,
    "new_performance_or_correction_comparisons": 0,
    "replay_max_logit_error": report["replay_max_logit_error"], "replay_max_final_slot_error": report["replay_max_final_slot_error"],
    "native_attention_update_max_error": report["native_attention_update_max_error"], "reference_vs_P1_attention_max_error": report["reference_vs_P1_attention_max_error"],
    "original_weights_sources_and_previous_artifacts_unchanged": True, "external_rng_unchanged": True, "hooks_removed": True,
    "probe_source_snapshot": {"path": str(snapshot), "sha256": _sha(snapshot)},
    "checks": {"synthetic_passed": 9, "raw_metadata_rejections_separate": 2, "counts_not_additive": True, "raw_audit": audited},
    "P4_no_fixed_index_semantics_guarantee": "confirmed_structural_property",
    "P4_same_patient_index_correspondence_changes_under_matching": "observed",
    "P4_residual_reading_distribution_difference_after_matching": "observed_small_TV_not_biological_semantics",
    "P4_real_tissue_semantic_drift": "not_established", "P4_cross_patient_semantics": "not_evaluated_no_labels",
    "P4_predictive_harm_or_correction_efficacy": "not_tested",
    "server_SPE09_status": "reported_completed_not_independently_verified", "server_raw_results_received": False,
    "summaries": report["summaries"], "raw_receipt_sha256": _sha(RAW / "receipt.json"),
    "next": "按顺序到P5：先核对真实输入中的补零/无mask影响，再区分人为零patch干预与真实无关组织占槽；不新增模块或训练。P4真实跨患者语义结论保留待组织/槽语义标注。"}
_write(OUT / "measurement_receipt.json", measurement)

lines = ["P4：槽编号与语义身份核查——第五阶段（2026-10-20）", "",
    "1. 路线和任务号", "SlotSPE原版问题存在性，SPE-08/P4。", "",
    "2. 本轮已经完成什么", "结构事实：原版槽共享[1,1,D]高斯参数和同一套更新权重，没有把槽编号绑定到组织类型。因而同编号没有固定生物学语义保证。匿名集合结构本身不等于预测故障。",
    "患者核查复用原版fold0/seed3、3轮/111步checkpoint，以及此前71名验证患者的20次随机状态和编码输入；新做20次原生模块只读重放，导出末轮逐头注意力。无新增训练、无新模型模块。",
    "对每名患者，以预先指定的第0次为参考；其余19次分别做8槽的注意力TV最佳匹配和槽向量余弦最佳匹配。病例结局、keep和编号不参与匹配。每模态1349组比较来自同一批71人，不是1349名独立患者。",
    "注意力TV是逐头分布差的平均：0代表权重分布相同，1代表完全分离；不是病理语义差异或临床阈值。", "",
    "模态       同编号TV中位数      最佳匹配TV中位数      匹配对应到不同编号的比例"]
for m, label in (("wsi", "病理"), ("omic", "组学")):
    s = report["summaries"][m]
    lines.append(f"{label}       {s['attention_diagonal_mean_TV']['median']:.9f}         {s['attention_matched_mean_TV']['median']:.9f}          {100*s['attention_reassigned_slots_fraction']:.3f}%")
lines += ["", "比例按重复对中的槽对应关系计算；不能写成该比例患者发生语义漂移。最佳匹配只给出低成本对应，不是槽的真实组织身份。",
    "病理最佳匹配后的Top-5 token交集比例中位0.25，组学0.10；但Top-5总权重比均匀分布多出的质量中位仅约0.00004815/0.00009293。接近均匀时名单容易变化，不能据Top-5名单变化宣布组织内容剧烈改变。",
    "已计算第二优匹配差距，病理/组学中位6.40e-6/1.47e-5；1e-7数值近并列阈值下分别有12/7组。没有把所有匹配都宣称唯一，也没有把该阈值当生物学判断标准。",
    "读取匹配与向量匹配的对应一致率中位均0.75，两者也不是完全相同的身份依据。注意力之外仍有GRU/残差信息，不能把注意力匹配视为槽全部内容的语义匹配。",
    "已知反向排列对照：71人两模态的最佳读取匹配成本均为0。此前P2真实患者槽与keep同步置换的最大logit差仅3.576e-7，说明单纯编号排列可以改变而预测基本不变。",
    "白话结论：编号不能直接当稳定语义标签已有结构和真实患者对应证据；编号变化不能直接算作语义丢失。本轮未证明真实组织语义漂移、跨患者语义混乱或预测损害。", "",
    "3. 对应文件路径、报告或提交号", str(OUT / "measurement_receipt.json"), str(OUT / "annotation_inventory.json"),
    str(OUT / "source_and_permutation_evidence.json"), str(RAW / "observations.json"), str(RAW / "receipt.json"),
    "raw/中每患者NPZ保存20次逐头注意力，以及19组注意力/向量8×8成本矩阵；observations保留每组匹配、次优差、Top-5统计和已知排列对照。",
    str(ROOT / "survot_rank/research/slotspe_proof/p4_diagnostics.py"), str(ROOT / "scripts/check_slotspe_p4.py"), "本轮未提交、未push。", "",
    "4. 检查是否通过；服务器是否实际运行；原始结果是否回传", "9项合成检查通过：已知排列恢复、均匀读取多解、逐头抵消、第二优匹配穷举对照、近均匀Top-5变化、无效输入拒绝及安全计划。",
    "另记2项真实结果副本防篡改检查：拒绝伪造语义已证明状态和伪造匹配成本；不与9项合成检查相加。",
    "本机真实患者模块重放已执行；logits/末轮槽/原注意力更新/P1第0次注意力最大误差均0。原权重、缓存、源码及此前产物保持，外部Torch RNG保持，hook移除。原始哈希、逐组成本、最佳/次佳匹配与摘要重算验收通过。",
    "服务器SPE09仍只有文字完成说明，原始包未回传验收。这是本机历史71人检查，不能代替服务器77人结果。", "",
    "5. 还有什么没完成，原因是什么", "已有证据包没有逐patch组织类型、槽对应组织标签或跨患者的语义真值。生存标签不是组织语义标签；不同患者patch编号也不是共同语义坐标。",
    "所以真实跨患者语义漂移、语义不稳定是否损害预测，以及固定语义机制是否有益均待证。单个短程checkpoint、注意力相似度和向量匹配不能补出这些标签。", "",
    "6. 已安排的下一步，需要提供什么", measurement["next"],
    "P5先复用现有材料；若要继续证明P4的真实组织语义问题，需要带患者/patch对应关系的组织标注或明确可验证的语义目标。服务器部分仍等待AI回传原始包，不需要SSH。", ""]
human_report = OUT / "P4_真实患者槽编号与语义核查.txt"
assert not human_report.exists()
human_report.write_text("\n".join(lines), encoding="utf-8")

progress_path, human_path = ROOT / "docs/THREE_LINE_PROGRESS.json", ROOT / "docs/三条研究路线_工作总览_20261019.txt"
before, human_before = _read(progress_path), human_path.read_text(encoding="utf-8-sig")
progress = copy.deepcopy(before)
backup = ROOT / "docs/history" / ("slotspe_p4_patient_" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S"))
backup.mkdir(parents=True, exist_ok=False)
for path in (progress_path, human_path):
    shutil.copy2(path, backup / path.name)
eid = "EV-SPE-P4-PATIENT-20261020"
assert not any(e["id"] == eid for e in progress["evidence"])
paths = [str((OUT / n).relative_to(ROOT)).replace("\\", "/") for n in ("measurement_receipt.json", "annotation_inventory.json", "source_and_permutation_evidence.json", human_report.name, "measurement_tests.xml", "raw_negative_audit.json")]
progress["evidence"].append({"id": eid, "type": "historical_native_within_patient_attention_and_vector_slot_matching",
    "details": "71人20次槽读取重放差0；最佳对应大多改变编号，匹配后读取TV较小。已知排列成本0，未证明真实组织语义漂移或损害。9项合成及另记2项原始副本检查通过，非服务器回传。",
    "paths": paths, "local_file_sha256": {p: _sha(ROOT / p) for p in paths}, "raw_export": str(RAW)})
task = next(t for t in progress["tasks"] if t["id"] == "SPE-08")
task.setdefault("historical_measurements", []).append(copy.deepcopy(task["last_measurement"]))
task["last_measurement"] = {"type": report["scope"], "patients": 71, "checkpoint_seed": 3,
    "P4_no_fixed_index_semantic_guarantee": "structural_property_confirmed", "P4_within_patient_matching": "observed_index_reassignment_and_small_residual_reading_TV",
    "P4_real_tissue_semantic_drift_or_predictive_harm": "not_established", "P4_cross_patient_semantic_labels": "unavailable_in_current_evidence",
    "report": paths[0], "synthetic_checks": 9, "raw_metadata_rejections_separate": 2, "checks_not_additive": True,
    "new_optimizer_steps": 0, "server_raw_received": False}
task["next_action"] = measurement["next"]
task["evidence_refs"].append(eid)
task["patient_effect"] = "anonymous_index_and_repeat_matching_observed_real_semantic_harm_not_proved"
task["expected_outputs"].append(str(human_report.relative_to(ROOT)).replace("\\", "/"))
task.setdefault("historical_local_stages", []).append(copy.deepcopy(task["local_stage"]))
task["local_stage"] = {"date": "2026-10-20", "completed": "P4历史71人逐头读取重放、最佳/次佳匹配、排列对照和语义证据边界核查",
    "new_training_runs": 0, "new_module_replays": 20, "originals_unchanged": True, "real_biological_semantic_drift_proved": False}
line = next(l for l in progress["lines"] if l["id"] == "L2")
line["latest_native_readonly_check"] = copy.deepcopy(task["last_measurement"])
line["evidence_refs"].append(eid)
line["bottleneck"] = "P4编号未绑定语义及真实患者匹配换号已观察；匹配后读取TV较小，真实组织/跨患者语义漂移缺标注，损害未证。下一项P5，SPE09原始包继续等待。"
progress["current_priority_note"] = "SlotSPE逐项检查：P1～P4本地阶段已登记；P4不能把编号置换等同语义漂移。下一项P5，服务器SPE09原始回传单独验收。"
progress["change_log"].append({"date": "2026-10-20", "scope": "SlotSPE_P4_slot_index_and_semantic_identity",
    "details": "71人20次读取与原版/P1/P3重放差0；Hungarian匹配及次优差、已知排列对照验收通过。9项合成+另记2项原始副本检查，真实语义漂移/损害未证。未训练/未push。",
    "backup": str(backup.relative_to(ROOT)).replace("\\", "/")})
assert [t for t in before["tasks"] if t["id"] != "SPE-08"] == [t for t in progress["tasks"] if t["id"] != "SPE-08"]
assert [l for l in before["lines"] if l["id"] != "L2"] == [l for l in progress["lines"] if l["id"] != "L2"]
progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
anchor = "2．SlotSPE：先干净原版检查问题，A/B/C用于后续归因。"
addition = "\n2026年10月20日第五阶段P4：同一历史71人20次逐头读取重放差0。病理最佳匹配约87.7%槽对应不同编号，匹配后读取TV中位0.000962；编号不可直接当语义标签。\n已知排列匹配成本0；近均匀权重下Top-5列表变化不等于语义漂移。本次无真实组织语义标注，尚未证明跨患者组织语义漂移或预测损害。\n9项合成检查及另记2项原始副本篡改拒绝通过；未训练/未push，非服务器回传。报告：docs/research/slotspe_p4_20261020/P4_真实患者槽编号与语义核查.txt。\n下一项P5；P4生物学语义证据保留待标注。以下为P1～P3等历史阶段，患者/重复/检查次数不相加。\n"
assert anchor in human_before
human = human_before.replace(anchor, anchor+addition, 1)
human = human.replace("当前最高优先级：SlotSPE八类问题逐项核查，P1/P2/P3本地阶段已登记，下一项P4；SPE-09原始产物验收并行。", "当前最高优先级：SlotSPE八类问题逐项核查，P1～P4本地阶段已登记，下一项P5；SPE-09原始产物验收并行。", 1)
assert human.split("3．运输插拔：", 1)[1] == human_before.split("3．运输插拔：", 1)[1]
assert human.split("1．V2：", 1)[1].split(anchor, 1)[0] == human_before.split("1．V2：", 1)[1].split(anchor, 1)[0]
human_path.write_text(human, encoding="utf-8")
_write(OUT / "progress_update_validation.json", {"all_other_tasks_unchanged": True, "non_slotspe_lines_unchanged": True,
    "V2_and_transport_human_sections_unchanged": True, "backup": str(backup), "raw_audit": audited,
    "original_hashes_rechecked": True, "new_training_runs": 0})
print(json.dumps({"stage": "P4", "raw_audit": audited, "synthetic_checks": 9, "separate_raw_metadata_checks": 2,
    "report": str(human_report), "backup": str(backup)}, ensure_ascii=False))
