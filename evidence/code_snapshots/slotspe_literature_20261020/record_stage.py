"""Validate the read-only review and update only the existing SlotSPE progress."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import copy
import hashlib
import json
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"E:\DCT-Reg")
OUT = ROOT / "docs/research/slotspe_literature_20261020"
def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))
def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = read(OUT / "source_manifest.json")
assert len(manifest["items"]) == 15
assert sum("pages" in i for i in manifest["items"]) == 8
assert sum("members" in i for i in manifest["items"]) == 7
for item in manifest["items"]:
    assert sha(Path(item["file"])) == item["sha256"] == item["sha256_after_read"]
baseline = read(OUT / "p1_source_check.json")
for item in baseline["source_files"]:
    assert sha(Path(item["absolute_path"])) == item["sha256_before"] == item["sha256_after"]
review_names = ["capacity_review.json", "semantics_review.json", "noise_objective_review.json", "initialization_transport_review.json", "primary_lookup.json", "p1_source_check.json"]
reviews = {name: read(OUT / name) for name in review_names}
assert len(reviews["primary_lookup.json"]["entries"]) == 7
for item in reviews["capacity_review.json"]["source_artifacts"]:
    assert sha(OUT / item["path"]) == item["sha256"]

cached_remote = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
assert cached_remote == "d64530cc1ab45e00787a1f3204c90535a8991bb0"
server_note = subprocess.check_output(["git", "show", "d64530c:AGENTS.md"], cwd=ROOT)
(OUT / "server_reported_d64530c_AGENTS.txt").write_bytes(server_note)

matrix = [
    {"id": "P1", "literature": "固定候选容量/有效数量问题在对象任务有依据", "original_slotspe": "默认8候选和Top-2掩码固定；cross仍读8槽", "patient_failure": "not_established", "next_evidence": "先查既有checkpoint真实占用、覆盖和读取路径；数量不变不等于预测受损"},
    {"id": "P2", "literature": "冗余竞争及对象碎片化有实证，DIAS公开代码不含删槽/重初始化", "original_slotspe": "共享更新允许重复槽；训练后实际有害冗余未证", "patient_failure": "not_established", "next_evidence": "真实attention重叠、组织/通路内容和预先指定槽干预联合核查"},
    {"id": "P3", "literature": "SmoothSA明确cold-start样本线索不足，BO-QSA为可学习query与双层优化", "original_slotspe": "初始槽不以患者内容为条件，迭代会读取输入；已有整体随机波动", "patient_failure": "initialization_specific_harm_not_established", "next_evidence": "分开记录初始化/Gumbel随机来源与逐轮attention，不能把所有波动归于初始化"},
    {"id": "P4", "literature": "CoSA类型绑定与PSA允许变换的可辨识性不是固定数组编号", "original_slotspe": "匿名槽无固定语义身份保证", "patient_failure": "matched_content_instability_not_established", "next_evidence": "先允许槽匹配，再检查内容稳定；无可信组织标签不称病理语义错误"},
    {"id": "P5", "literature": "Foreground背景纹理干扰，CODA扩散条件register吸收残余注意力", "original_slotspe": "补零敏感已证；真实无关组织占用并挤出预后信息待证", "patient_failure": "real_tissue_displacement_not_established", "next_evidence": "组织依据与可回查patch，不能把间质/免疫直接标作无关"},
    {"id": "P6", "literature": "MSF图像金字塔/VAE目标，MUFASA ViT多层语义，均不同于病理物理倍率", "original_slotspe": "无显式坐标/倍率接口；真实特征提取配方仍须核实", "patient_failure": "not_established", "next_evidence": "实际特征层/倍率/空间元数据及对应患者信息遗漏，现有单层文件不能补造多尺度数据"},
    {"id": "P7", "literature": "OCEBO目标编码器漂移不直接证明生存任务目标冲突", "original_slotspe": "最终融合损失到selector直接梯度缺失已证；辅助生存梯度仍连通", "patient_failure": "auxiliary_vs_fusion_conflict_not_established", "next_evidence": "已有B/C的匹配患者归因及必要的目标梯度/训练折诊断，不混入新模块"},
    {"id": "P8", "literature": "MESH解释模块内SA与OT关联，没有两阶段冲突实证", "original_slotspe": "后续为cross-attention，无第二跨模态OT", "patient_failure": "not_applicable_original_slotspe", "next_evidence": "本线不追加OT；具体DCT组合假设需独立范围证据"}
]
audit = {
    "task_id": "SPE-08", "date": "2026-10-20", "stage": "literature_identity_applicability_and_P1_source",
    "stage_complete": True, "all_eight_patient_problems_proved": False,
    "source_root": manifest["source"], "local_pdfs": 8, "local_archives": 7,
    "missing_local_papers_primary_pages_checked": 7,
    "review_depth": "本地8份PDF核查身份和完整相关页，7个ZIP静态核查；额外7篇只有一手摘要/指定段落，不称全文复现。",
    "matrix": matrix,
    "new_model_changes": False, "patient_data_read": False, "patient_jobs_started": False, "packages_installed": False,
    "server_report": {"commit": cached_remote, "source": "AGENTS.md section 独立干净原版 SlotSPE 启动实测", "execution_status": "reported_completed_not_independently_verified", "raw_results_received": False, "all_eight_covered": False},
    "reports": review_names,
    "human_report": "第一阶段_文献与原版对应核查.txt",
    "next_stage": "P1：先复用现有原版checkpoint确认真实槽占用与内容覆盖；不自动新增训练预算。",
    "negative_result_rule": "观测不到记not_observed；只有结构/材料不足记not_established或insufficient_evidence，不能强迫八项全成立。"
}
write(OUT / "audit.json", audit)

progress_path = ROOT / "docs/THREE_LINE_PROGRESS.json"
human_path = ROOT / "docs/三条研究路线_工作总览_20261019.txt"
progress_before = read(progress_path)
progress = copy.deepcopy(progress_before)
now = datetime.now(timezone(timedelta(hours=8)))
backup_dir = ROOT / "docs/history" / ("slotspe_literature_" + now.strftime("%Y%m%d_%H%M%S"))
backup_dir.mkdir(parents=True, exist_ok=False)
for path in [progress_path, human_path]:
    shutil.copy2(path, backup_dir / path.name)

evidence_id = "EV-SPE-LITERATURE-20261020"
evidence_paths = ["docs/research/slotspe_literature_20261020/" + name for name in ["audit.json", "source_manifest.json", "p1_source_check.json", "第一阶段_文献与原版对应核查.txt"]]
evidence = {"id": evidence_id, "type": "literature_source_and_static_archive_review", "details": "8PDF/7ZIP原件哈希不变；补查7篇缺失文献一手页面；P1确认8候选/2掩码但cross仍读全部槽。未新增患者实验或模块。", "paths": evidence_paths, "local_file_sha256": {p: sha(ROOT / p) for p in evidence_paths}}
progress["evidence"] = [e for e in progress["evidence"] if e["id"] != evidence_id] + [evidence]
task = next(t for t in progress["tasks"] if t["id"] == "SPE-08")
task["previous_completed_local_stage"] = copy.deepcopy(task)
task.update(status="partially_completed_local", next_action=audit["next_stage"], last_checked="2026-10-20", evidence_level="literature_source_existing_synthetic_patient_failure_pending", report="docs/research/slotspe_literature_20261020/audit.json")
task["local_stage"] = {"date": "2026-10-20", "completed": "文献适用条件、归档代码完整性与P1静态事实", "patient_failure_proof_completed": False, "originals_unchanged": True, "new_patient_runs": 0}
task["evidence_refs"].append(evidence_id)
task["expected_outputs"].append("docs/research/slotspe_literature_20261020/第一阶段_文献与原版对应核查.txt")
task07 = next(t for t in progress["tasks"] if t["id"] == "SPE-07")
task07["status"] = "reported_completed_awaiting_raw_artifacts"
task07["server_reported_commit"] = cached_remote
task07["server_execution_verified"] = False
task07["raw_results_received"] = False
task07["next_action"] = "验收d64530c登记的SPE-09原始manifest/梯度/预测/槽干预/checkpoint；文字记录不等于原始证据通过。"
line = next(l for l in progress["lines"] if l["id"] == "L2")
line["latest_literature_audit"] = {"task": "SPE-08", "date": "2026-10-20", "report": task["report"], "evidence_ref": evidence_id, "patient_failure_proof_completed": False}
line["bottleneck"] = "八类需逐项从文献映射到真实槽行为；先P1，同时等待SPE-09原始患者产物，不能用文字登记替代验收。"
line["current_clean_original_handoff"].update(server_execution="reported_completed_not_independently_verified", server_reporting_commit=cached_remote, raw_results_received=False)
for key in ["task_ids", "evidence_refs"]:
    value = "SPE-08" if key == "task_ids" else evidence_id
    if value not in line[key]:
        line[key].append(value)
progress["updated_date"] = "2026-10-20"
progress["current_priority"] = ["SPE-08"] + [t for t in progress["current_priority"] if t != "SPE-08"]
progress["current_priority_note"] = "SlotSPE八类问题逐项核查：第一阶段文献和P1源码已完成；先P1真实槽诊断，同时等待服务器原始产物。"
progress["change_log"].append({"date": "2026-10-20", "scope": "SlotSPE_eight_problem_literature_stage1", "details": "保存8PDF/7ZIP核查及7篇外部一手页面边界；更新SPE-08阶段完成与患者待证，SPE-07记服务器文字报告完成但原始未验收。其他线不变。", "backup": str(backup_dir.relative_to(ROOT)).replace("\\", "/")})
non_slotspe_tasks_before = [t for t in progress_before["tasks"] if not t["id"].startswith("SPE-")]
non_slotspe_lines_before = [l for l in progress_before["lines"] if l["id"] != "L2"]
assert non_slotspe_tasks_before == [t for t in progress["tasks"] if not t["id"].startswith("SPE-")]
assert non_slotspe_lines_before == [l for l in progress["lines"] if l["id"] != "L2"]
write(progress_path, progress)

human = human_path.read_text(encoding="utf-8-sig")
human = human.replace("更新：2026年10月19日", "更新：2026年10月20日", 1)
human = human.replace("当前最高优先级：SlotSPE干净原版单臂问题复现，SPE-07/SPE-09。", "当前最高优先级：SlotSPE八类问题逐项核查，先P1；SPE-09原始产物验收并行。", 1)
anchor = "2．SlotSPE：先干净原版检查问题，A/B/C用于后续归因。"
addition = "\n2026年10月20日阶段更新：SPE-08已读8份PDF和7个ZIP并补查7篇一手页面；文献身份、适用条件及P1源码核查完成，八项患者故障证明未完成。\nP1默认8候选/Top-2，但cross仍读全部8槽；不能说最终只剩2槽。DIAS公开包缺删槽与重初始化；P6的图像尺度、ViT层次与病理倍率不能混用。\n报告：docs/research/slotspe_literature_20261020/第一阶段_文献与原版对应核查.txt；原件与6个原版源码/示例哈希不变。本轮无患者任务和新模块。\n已读取origin/main缓存d64530c的服务器完成说明；原始预测、梯度、槽干预和checkpoint尚未回传验收，不登记为八项已证明。\n下一阶段先P1：复用既有checkpoint导出真实槽占用与读取内容，不自动扩大训练预算。以下2026年10月19日内容保留为前阶段记录。\n"
assert anchor in human
human = human.replace(anchor, anchor + addition, 1)
human_path.write_text(human, encoding="utf-8")

validation = {"date": "2026-10-20", "source_files_rehashed": 15, "original_baseline_files_rehashed": 6, "all_source_bytes_unchanged": True, "all_review_json_parse": True, "capacity_extracted_source_hashes_match": True, "non_slotspe_tasks_unchanged": True, "non_slotspe_lines_unchanged": True, "patient_jobs_started": False, "model_code_changed": False, "new_training_runs": 0, "new_synthetic_experiments": 0, "backup": str(backup_dir), "scope": "材料/结构化报告校验，不是模型或患者实验通过数"}
write(OUT / "validation.json", validation)
print(json.dumps(validation, ensure_ascii=False))
