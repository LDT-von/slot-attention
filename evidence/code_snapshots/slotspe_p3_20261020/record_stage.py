"""Record completed P3 without running patients or changing other lines."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(r"E:\DCT-Reg")
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from survot_rank.research.slotspe_proof.p3_diagnostics import audit, _read, _sha, _write

RAW = ROOT / ".review/slotspe_p3_20261020/seed3_native_v1"
OUT = ROOT / "docs/research/slotspe_p3_20261020"
probe = RAW.parent / "probe_source_v1.py"
audited = audit(RAW, probe)
report = _read(RAW / "observations.json")
negative = _read(OUT / "raw_negative_audit.json")
assert len(negative["checks"]) == 3 and all(c["rejected"] is True for c in negative["checks"])
tests = ET.parse(OUT / "measurement_tests_complete.xml").getroot()
assert sum(int(s.attrib.get("tests", 0)) for s in tests.iter("testsuite")) == 7
assert sum(int(s.attrib.get(k, 0)) for s in tests.iter("testsuite") for k in ("failures", "errors", "skipped")) == 0
auditor_source = RAW.parent / "auditor_source_v2.py"
assert not auditor_source.exists()
shutil.copy2(ROOT / "survot_rank/research/slotspe_proof/p3_diagnostics.py", auditor_source)

measurement = {"task": "SPE-08/P3", "date": "2026-10-20", "scope": report["scope"], "raw_export": str(RAW),
    "fold": 0, "seed": 3, "previous_training_epochs": 3, "previous_optimizer_steps": 111,
    "patients": 71, "patches": 256, "repeats_per_condition": 20, "conditions": 4,
    "new_training_runs": 0, "new_optimizer_steps": 0, "new_full_native_eval_passes": 20,
    "new_module_eval_passes": 80, "module_passes_include_20_native_alignment_replays": True,
    "independent_patient_count": 71, "conditions_and_repeats_are_not_independent_patients": True,
    "historical_20_repeat_replay_max_logit_error": report["historical_replay_max_logit_error"],
    "native_vs_module_replay_max_logit_error": report["modular_native_replay_max_logit_error"],
    "native_encoded_inputs_equal_across_all_repeats": True,
    "original_weights_sources_and_cache_unchanged": True, "controlled_replays_preserve_external_rng": True,
    "model_forward_or_parameters_replaced": False, "hooks_removed": True,
    "source_snapshots": {"measurement_v1": {"path": str(probe), "sha256": _sha(probe)},
        "offline_auditor_v2": {"path": str(auditor_source), "sha256": _sha(auditor_source)}},
    "audit_failure_preserved": {"error": "ValueError: Raw metrics differ", "cause": "Auditor combined 20 repeats in one float32 CPU sigmoid call; original evaluation calculated each 71-patient pass separately",
        "observed_max_rounding_difference": 4.76837158203125e-7,
        "fix": "Recompute each repeat with historical tensor shape; require exact equality, no tolerance relaxation",
        "new_regression_test": "test_exact_risk_audit_preserves_each_historical_pass_kernel_shape",
        "patient_inference_repeated_after_fix": False,
        "measured_computation_functions_ast_unchanged": True},
    "checks": {"synthetic_final_passed": 7, "earlier_6_checks_not_added": True,
        "raw_metadata_tamper_rejections": 3, "raw_checks_separate_from_synthetic": True, "offline_raw_audit": audited},
    "P3_initial_distribution_patient_conditioned": False,
    "initial_distribution_description": "globally learned shared mu/sigma; subsequent slot iterations are patient-input conditioned",
    "P3_native_same_input_prediction_variability": "observed_71_of_71_historical_validation_patients",
    "P3_slot_initialization_downstream_effect": "observed_after_10_iterations_with_native_gumbel_draws_held_fixed",
    "P3_gumbel_downstream_effect": "observed_with_native_slot_draws_held_fixed",
    "P3_fix_both_removes_repeat_variability_in_this_protocol": True,
    "P3_patient_conditioned_initializer_predictive_advantage": "not_established",
    "correction_efficacy_or_generalization": "not_tested",
    "server_SPE09_status": "reported_completed_not_independently_verified", "server_raw_results_received": False,
    "summaries": report["summaries"], "raw_receipt_sha256": _sha(RAW / "receipt.json"),
    "next": "按顺序到P4：区分槽编号的置换与真实语义身份漂移，先核查可用语义证据；不新增模块或训练。SPE09原始回传继续单独验收。"}
_write(OUT / "measurement_receipt.json", measurement)

source = Path(next(p for p in report["source_inputs_sha256"] if Path(p).name == "SlotSPE.py")).parent.parent
excerpts = []
for name, needles in (("models/slot_attention.py", ("self.slots_mu =", "self.slots_logsigma =", "slots = mu + sigma", "noise = torch.rand_like")),
        ("models/SlotSPE.py", ("hard_keep, _ = parallel_topk_st", "slots_wsi = self.slot_attention_wsi", "slots_omic = self.slot_attention_omic"))):
    path = source / name
    lines = path.read_text(encoding="utf-8").splitlines()
    excerpts.append({"path": str(path), "sha256": _sha(path), "locations": [
        {"line": i+1, "text": line.strip()} for i, line in enumerate(lines) if any(s in line for s in needles)]})
_write(OUT / "source_evidence.json", {"original_sources_unchanged": True, "excerpts": excerpts,
    "interpretation": "Shared learned Gaussian is not conditioned on current patient; native slot refinement does use patient inputs. Both Gaussian and Gumbel sampling remain active in eval."})

lines = ["P3：初始化与评估随机性核查——第四阶段（2026-10-20）", "",
    "1. 路线和任务号", "SlotSPE原版问题存在性，SPE-08/P3。", "",
    "2. 本轮已经完成什么", "代码事实：初始槽从共享、可学习的高斯分布采样；mu/sigma是全局训练得到的参数，初始采样不以当前患者内容为条件。之后10轮槽更新会使用患者输入。",
    "eval仍有两处随机抽样：槽高斯初始化、Top-K的Gumbel噪声。不能把整体波动都叫作初始化问题。",
    "实际复用10月18日本机原版fold0/seed3、3轮/111步checkpoint，289/71划分中的71名验证患者，固定256真实patch、顺序、batch8。",
    "本轮实际完成20次整模型原版评估，并做80次原生模块重放（其中20次用于整模型/模块公式对齐）。四个观察条件各20次；同一批71人，不能按重复次数扩充样本量。",
    "控制使用原版模块入口处保存的CPU/CUDA随机状态：只替换指定模块本次所用的抽样状态，模块类、forward及权重不变；编码输入保存且各次逐元素一致。",
    "参考是预先约定的第0次评估（eval_seed3080000），没有按分数挑选固定种子。", "",
    "条件                        验证C-index范围             人均风险极差中位数       最大风险极差"]
for arm, label in (("native", "原版两处均随机"), ("fix_slot_draws", "只固定槽抽样"), ("fix_gumbel_draws", "只固定Gumbel抽样"), ("fix_both_draws", "两处抽样均固定")):
    s = report["summaries"][arm]
    c, r = s["C_index_validation_exploratory"], s["patient_risk_range"]
    lines.append(f"{label:<18} {c['min']:.6f}～{c['max']:.6f}       {r['median']:.9f}          {r['max']:.9f}")
lines += ["", "这里的风险极差指同一患者20次预测的最大值减最小值；不是患者间风险差，也不是临床发生概率。",
    "原版71/71患者均发生超过1e-6的风险波动；原版最大logit极差0.044635296，C-index总跨度约0.018962。",
    "只固定Gumbel时槽初值/末轮槽仍与原版各轮逐元素一致，预测仍变，说明初始化随机性经过10轮后仍有下游作用。固定Gumbel并不固定keep：病理19/71人、组学15/71人的keep仍随输入槽变化。",
    "只固定槽时初值和末轮槽在20次之间完全一致，但两模态71/71人的keep仍变化、预测仍波动，说明Gumbel随机性也有作用。",
    "两处都固定后，原始logits、风险、初/末槽和keep在20次之间均逐元素一致，并与原版第0次一致。C-index恒定0.686627；标准差中1e-16级残差只是统计浮点舍入。",
    "白话结论：P3的重复评估不稳定确实存在，且两处抽样都能影响预测。固定两处抽样可在此协议下消除波动；这不是证明分数提高，也没有证明患者条件初始化比当前全局高斯更好。",
    "两个随机来源经非线性模块共同作用；不把各对照方差差值解释成可相加的贡献百分比。", "",
    "3. 对应文件路径、报告或提交号", str(OUT / "measurement_receipt.json"), str(OUT / "source_evidence.json"),
    str(RAW / "observations.json"), str(RAW / "native_rng_states.npz"), str(RAW / "encoded_inputs.npz"),
    "同目录native/fix_slot_draws/fix_gumbel_draws/fix_both_draws.npz保留逐轮逐患者初值、末轮槽、keep、logits、risk。",
    str(ROOT / "survot_rank/research/slotspe_proof/p3_diagnostics.py"), str(ROOT / "scripts/check_slotspe_p3.py"), "本轮未提交、未push。", "",
    "4. 检查是否通过；服务器是否实际运行；原始结果是否回传", "原版20次历史logits/risk逐元素重放，最大logit差0；整模型与原生模块重放最大差0。",
    "权重、原版源码、缓存及P1/P2证据保持；控制重放不改变外部Torch随机状态，测量hook已移除。",
    "最终7项合成测量检查通过，不与此前6项相加；另有3项原始结果副本防篡改检查拒绝错误seed、空随机状态、伪造摘要，另记。",
    "最初离线验收失败原因已保留：一次合并20轮计算sigmoid，与逐轮历史计算出现最大4.768e-7的舍入差；改回逐轮同形状精确重算后验收通过，未放宽阈值、未重跑患者计算。",
    "测量源v1和修正后的离线验收源v2分别封存，测量函数AST一致。原始数组、随机状态、输入、摘要与哈希验收通过。",
    "本机真实患者评估已执行；服务器SPE09只有文字完成说明，原始包尚未回传验收。本机历史71人不可替代服务器77人。", "",
    "5. 还有什么没完成、原因是什么", "缺少患者线索是否损害预后预测、早期聚合质量是否更差、任何初始化修正是否提分，均未证明；没有相应匹配训练对照或组织标注。",
    "仅检查一个短程checkpoint、一个验证折、固定batch/顺序；不证明跨batch、跨折稳定性或外部测试表现。控制抽样用于诊断，不作为新模块或部署修正版。", "",
    "6. 已安排下一步，需要用户提供什么", measurement["next"],
    "下一项P4先复用既有材料。服务器部分仍需AI回传实际配置/命令、日志、逐患者预测、槽记录与checkpoint，不需要SSH。", ""]
human_report = OUT / "P3_真实患者初始化与随机性核查.txt"
assert not human_report.exists()
human_report.write_text("\n".join(lines), encoding="utf-8")

progress_path, human_path = ROOT / "docs/THREE_LINE_PROGRESS.json", ROOT / "docs/三条研究路线_工作总览_20261019.txt"
before, human_before = _read(progress_path), human_path.read_text(encoding="utf-8-sig")
progress = copy.deepcopy(before)
backup = ROOT / "docs/history" / ("slotspe_p3_patient_" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S"))
backup.mkdir(parents=True, exist_ok=False)
for path in (progress_path, human_path):
    shutil.copy2(path, backup / path.name)
eid = "EV-SPE-P3-PATIENT-20261020"
assert not any(e["id"] == eid for e in progress["evidence"])
paths = [str((OUT / n).relative_to(ROOT)).replace("\\", "/") for n in ("measurement_receipt.json", "source_evidence.json", human_report.name, "measurement_tests_complete.xml", "raw_negative_audit.json")]
progress["evidence"].append({"id": eid, "type": "historical_native_trained_eval_randomness_with_module_draw_controls",
    "details": "原版71人20轮精确重放，两类随机抽样分别控制；原版C-index0.682635～0.701597，两处固定后预测完全一致。初始化与Gumbel均影响预测，未证提分或患者条件初始化优势。7项合成检查+另记3项原始副本篡改拒绝，非服务器回传。",
    "paths": paths, "local_file_sha256": {p: _sha(ROOT / p) for p in paths}, "raw_export": str(RAW)})
task = next(t for t in progress["tasks"] if t["id"] == "SPE-08")
task.setdefault("historical_measurements", []).append(copy.deepcopy(task["last_measurement"]))
task["last_measurement"] = {"type": report["scope"], "patients": 71, "checkpoint_seed": 3,
    "P3_native_prediction_variability": "observed", "P3_slot_initialization_downstream_effect": "observed_with_gumbel_draws_fixed",
    "P3_gumbel_downstream_effect": "observed_with_slot_draws_fixed", "P3_fix_both_repeats": "exactly_equal_to_reference",
    "P3_patient_conditioned_initialization_advantage": "not_established", "correction_performance": "not_evaluated",
    "report": paths[0], "synthetic_final_checks": 7, "raw_metadata_rejections_separate": 3, "checks_not_additive": True,
    "new_optimizer_steps": 0, "server_raw_received": False}
task["next_action"] = measurement["next"]
task["evidence_refs"].append(eid)
task["patient_effect"] = "native_eval_variability_and_two_random_source_effects_observed_no_correction_gain_proof"
task["expected_outputs"].append(str(human_report.relative_to(ROOT)).replace("\\", "/"))
task.setdefault("historical_local_stages", []).append(copy.deepcopy(task["local_stage"]))
task["local_stage"] = {"date": "2026-10-20", "completed": "P3原版71人20次整模型重放、槽/Gumbel随机状态归因与原始验收",
    "new_training_runs": 0, "full_native_eval_passes": 20, "module_replay_passes": 80, "originals_unchanged": True,
    "patient_conditioned_initialization_harm_or_advantage_proved": False}
line = next(l for l in progress["lines"] if l["id"] == "L2")
line["latest_native_readonly_check"] = copy.deepcopy(task["last_measurement"])
line["evidence_refs"].append(eid)
line["bottleneck"] = "P3同权重评估波动及初始化/Gumbel两处影响已在历史71人复现；双固定可消除波动，提分/患者条件初始化优势仍未证。下一项P4，SPE09原始包待验收。"
progress["current_priority_note"] = "SlotSPE逐项检查：P1/P2/P3本地阶段已登记，P3两处随机影响已隔离；下一项P4。服务器SPE09原始回传单独验收。"
progress["change_log"].append({"date": "2026-10-20", "scope": "SlotSPE_P3_native_eval_randomness_controls",
    "details": "71人原版20次精确重放，分别固定槽/Gumbel；双固定消除波动。修复离线验收的逐轮CPU计算形状，未重跑患者；7项合成+另记3项原始副本检查通过，未训练/未push。",
    "backup": str(backup.relative_to(ROOT)).replace("\\", "/")})
assert [t for t in before["tasks"] if t["id"] != "SPE-08"] == [t for t in progress["tasks"] if t["id"] != "SPE-08"]
assert [l for l in before["lines"] if l["id"] != "L2"] == [l for l in progress["lines"] if l["id"] != "L2"]
progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
anchor = "2．SlotSPE：先干净原版检查问题，A/B/C用于后续归因。"
addition = "\n2026年10月20日第四阶段P3：同一历史71人、原版checkpoint20次整模型重放差0；分别固定槽和Gumbel抽样，两者都能影响预测，两处同时固定后原始预测完全一致。\n原版验证C-index0.682635～0.701597；单患者风险极差最大0.013689。随机初始化经过10轮仍有下游影响，但尚未证明患者条件初始化提分或缺少线索损害预测。\n最终7项合成检查及另记3项原始副本篡改拒绝通过；验收CPU逐轮形状错误已修正，未重跑患者。无新增训练、未push，非服务器SPE09回传。报告：docs/research/slotspe_p3_20261020/P3_真实患者初始化与随机性核查.txt。\n下一项P4；服务器原始包仍待回传。以下P1/P2等为历史阶段，患者/重复/检查次数不相加。\n"
assert anchor in human_before
human = human_before.replace(anchor, anchor+addition, 1)
human = human.replace("当前最高优先级：SlotSPE八类问题逐项核查，P1/P2本地阶段已登记，下一项P3；SPE-09原始产物验收并行。", "当前最高优先级：SlotSPE八类问题逐项核查，P1/P2/P3本地阶段已登记，下一项P4；SPE-09原始产物验收并行。", 1)
assert human.split("3．运输插拔：", 1)[1] == human_before.split("3．运输插拔：", 1)[1]
assert human.split("1．V2：", 1)[1].split(anchor, 1)[0] == human_before.split("1．V2：", 1)[1].split(anchor, 1)[0]
human_path.write_text(human, encoding="utf-8")
_write(OUT / "progress_update_validation.json", {"all_other_tasks_unchanged": True, "non_slotspe_lines_unchanged": True,
    "V2_and_transport_human_sections_unchanged": True, "backup": str(backup), "raw_audit": audited,
    "original_hashes_rechecked": True, "frozen_measurement_source_verified": True, "new_training_runs": 0})
print(json.dumps({"stage": "P3", "raw_audit": audited, "synthetic_checks": 7, "separate_raw_metadata_checks": 3,
    "report": str(human_report), "backup": str(backup)}, ensure_ascii=False))
