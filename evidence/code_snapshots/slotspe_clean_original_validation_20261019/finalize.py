"""Record an inspected local delivery; never execute patient work."""
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(r"E:\DCT-Reg")
BASELINE = Path(r"C:\Users\栋栋\Desktop\特征解耦论文\SlotSPE-main\SlotSPE-main")
VALIDATION = Path(__file__).parent
DEST = ROOT / "docs/research/slotspe_clean_original_20261019"
ARCHIVE = ROOT / "docs/deliveries/slotspe_clean_original_audit_20261019_v1.zip"
REL_ARCHIVE = ARCHIVE.relative_to(ROOT).as_posix()
now = datetime.now().astimezone().isoformat(timespec="seconds")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

def junit(path):
    suites = ET.parse(path).getroot().findall(".//testsuite")
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    assert counts["failures"] == counts["errors"] == counts["skipped"] == 0
    return counts

first, final = junit(VALIDATION/"tests.xml"), junit(VALIDATION/"final_tests.xml")
assert first["tests"] == 34 and final["tests"] == 4
assert not DEST.exists() and not ARCHIVE.exists()
DEST.mkdir(parents=True)
shutil.copy2(VALIDATION/"tests.xml", DEST/"full_suite_before_hardening.xml")
shutil.copy2(VALIDATION/"final_tests.xml", DEST/"final_critical_checks.xml")
baseline_files = ["models/SlotSPE.py", "models/slot_attention.py", "models/transformer.py", "models/omics_encoder.py",
    "utils/loss_func.py", "utils/core_utils.py", "utils/process_args.py", "dataset/dataset_survival.py", "README.md", "requirements.txt"]
baseline_hashes = {name: sha(BASELINE/name) for name in baseline_files}
old_zip = ROOT/"docs/deliveries/slotspe_proof_abc_20261019_v2.zip"
assert sha(old_zip) == "18baac2c686307f0b89684f85768f60b75415830b009fde857ea1a55c75f89ec"
raw = read(ROOT/"docs/research/slotspe_eight_issues_20261019/synthetic_checks.json")
for name, digest in raw["source_hashes_before"].items():
    assert sha(BASELINE/name) == digest

report = {"task_id":"SPE-09", "recorded_at":now, "status":"local_delivery_ready_server_artifacts_pending",
    "baseline_repository":str(BASELINE), "baseline_files_sha256":baseline_hashes,
    "local_evidence": {"scope":raw["scope"], "source_unchanged":raw["source_unchanged"], "patient_data_read":False,
        "trained_checkpoint_loaded":False, "optimizer_steps":0,
        "self_intervention":raw["findings"]["unselected_self_full_prediction"], "eval_repeat":raw["findings"]["evaluation_randomness"],
        "raw_json":"docs/research/slotspe_eight_issues_20261019/synthetic_checks.json"},
    "validation": {"full_suite_before_final_hardening":first, "final_native_and_legacy_e2e":final, "counts_are_additive":False,
        "scope":"invented metadata/tensors only; includes standalone ZIP, native class checks, safe default plan, raw-record audit and tamper rejection"},
    "server_plan":{"priority":"highest_SlotSPE", "experiment":"original_audit", "arms":["original"], "training_runs":1,
        "folds":[0], "seeds":[3], "epochs":3, "patches":256, "batch_size":8, "eval_repeats":20,
        "config":"configs/slotspe_clean_original_server.json", "checkpoint_selection":"fixed_last_epoch",
        "repaired_models_constructed":False, "transport_added":False, "abc_training":False},
    "protocol_boundary":"Unchanged original model and losses; diagnostic cache uses complete paired cases, fixed real patches and train-only bins. Original cohort-wide binning is tested separately from unchanged AST on copies, not used to contaminate training.",
    "required_native_outputs":["trained_selector_gradients.json", "patient_slot_intervention.json", "eval_repeats.json", "validation_predictions.csv", "natural_gate_observations.json", "native_model_audit.json", "last.pt", "curve.json", "protocol.json", "prepared/manifest.json", "logs", "receipts"],
    "observation_policy":"Every item may be observed or not_observed; controlled zeros and toy gate scores are never natural occurrence rates.",
    "server_execution":"unknown_no_raw_return", "patient_effect":"not_evaluated", "attribution_conclusion":"pending_separate_matched_correction_comparison",
    "archive":REL_ARCHIVE, "remote_published":False, "previous_archive_preserved":True}
dump(DEST/"evidence_and_contract.json", report)

tracked = [ROOT/"docs/SLOTSPE_PROOF_TASK.json", ROOT/"docs/SlotSPE证明任务_服务器一键运行.txt", ROOT/"docs/THREE_LINE_PROGRESS.json", ROOT/"docs/三条研究路线_工作总览_20261019.txt"]
backup = ROOT/"docs/history"/("slotspe_clean_original_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
backup.mkdir(parents=True)
for path in tracked:
    shutil.copy2(path, backup/path.name)

guide = tracked[1].read_text(encoding="utf-8-sig")
header = '''当前最高优先级：干净原版SlotSPE单臂问题复现（SPE-07 / SPE-09，2026-10-19）

本轮先运行original一个训练臂：fold0、seed3、3轮、256真实patch、batch8、末轮同权重重复评估20次。
交接包：docs/deliveries/slotspe_clean_original_audit_20261019_v1.zip；解压到新目录。
包中原版源码直接从用户指定桌面仓库逐字节封存，哈希见DELIVERY_MANIFEST.json。
本轮不构造修正版模型，不启动A/B/C或运输；原模型forward、槽类、decoder类及参数都有原版一致性检查。
任务包默认配置也已绑定单臂短程任务；没有--run时只打印计划。

服务器AI：使用现成Python环境，先确认数据目录和GPU任务；将SlotSPE作为本轮最高优先级安排，不中断其他已在运行的任务。
设置SLOTSPE_METADATA_ROOT和SLOTSPE_FEATURES_ROOT为服务器真实目录，PYTHON为现成解释器（需要时）。然后顺序执行：

    bash run_slotspe_proof.sh --config configs/slotspe_clean_original_server.json --doctor
    bash run_slotspe_proof.sh --config configs/slotspe_clean_original_server.json --plan
    bash run_slotspe_proof.sh --config configs/slotspe_clean_original_server.json --run

计划必须显示training_arms=[original]、training_runs=1、same_weights_eval_controls=0。
模型保持原版；病例准备为配对完整输入、固定真实patch和训练折分箱。它是诊断协议，不是原论文完整基准配方。
原版全队列分箱的验证结局依赖，单独在内存副本中执行原源码函数检查，不把该问题带入可比较训练。

五项检查分别报告：
1. 最终融合NLL到选槽器的梯度，主/辅助分开，逐参数None/zero/nonzero：trained_selector_gradients.json。
2. 真实验证患者末轮槽干预：固定keep和selected槽、固定cross，只改变未选槽；保存前后self池化、logits和risk：patient_slot_intervention.json。
3. 相同权重、患者、patch及顺序重复20次；另同RNG重放：eval_repeats.json、validation_predictions.csv、native_model_audit.json。
4. 补零敏感性：真实患者缓存的副本上加16/64零行，保存前后logits/risk。人为补零不是实际发生率；病例排除和原有零行数量另报。
5. 自然输入门控：20次真实验证前向仅旁观原门控、记录所有患者与两模态，是否近零如实报告：natural_gate_observations.json。
   同时在独立原decoder副本上做合成分数反例，写在native_model_audit.json，不能拿它当患者问题发生率。

原版未观察到某问题就写not_observed；不改阈值、样本或源码强迫失败。错误和失败目录原样保留。
本窗口已有的数值是本地合成/随机权重证据，见包内docs/research/slotspe_eight_issues_20261019/synthetic_checks.json；服务器证据仍待新批次原始回传。

完整批次返回：
    python -B scripts/pack_slotspe_return.py --root '/实际结果批次' --output '/新目录/slotspe_clean_return.zip'
回传ZIP和打印的SHA256；包含末轮checkpoint、配置、manifest、日志、receipts、曲线与所有诊断JSON/CSV，不包含prepared/cache的患者输入张量。
验收：
    python -B scripts/audit_slotspe_return.py --root '/解压后的slotspe_server_return' --delivery '/本批次交付ZIP'
需要原始产物核验后才能说服务器复现。只看C-index或“运行成功”不足以证明这些计算问题。

以下保留历史两臂和A/B/C说明，供后续修正与归因；本轮入口以上述原版单臂为准。
================================================================

'''
tracked[1].write_text(header+guide, encoding="utf-8")
task = read(tracked[0])
task["previous_delivery_before_clean_audit"] = copy.deepcopy(task["delivery"])
task["entrypoint_argv"] = ["bash", "run_slotspe_proof.sh", "--config", "configs/slotspe_clean_original_server.json", "--run"]
task["plan_argv"] = task["entrypoint_argv"][:-1]+["--plan"]
task["preflight_argv"] = task["entrypoint_argv"][:-1]+["--doctor"]
task["config"] = "configs/slotspe_clean_original_server.json"
task["current_training_runs"] = 1
task["current_priority"] = "highest_SlotSPE"
task["requested_clean_original"] = report["server_plan"]
task["requested_reproduction"]["current_status"] = "retained_two_arm_followup_after_clean_original_audit"
task["delivery_status"] = "clean_original_only_local_delivery_ready_server_artifacts_pending"
task["local_validation"] = report["validation"]
task["server_patient_run_status"] = "awaiting_clean_original_server_ai_execution_and_raw_artifacts"
task["delivery"] = {"path":REL_ARCHIVE, "remote_published_this_work":False, "server_execution_verified":False,
    "sha256_sidecar":REL_ARCHIVE+".sha256", "baseline":"byte_exact_user_original_snapshot", "report":"docs/research/slotspe_clean_original_20261019/evidence_and_contract.json"}
task["execution"] = "用户要求SlotSPE最高优先级，干净原版先复现；服务器AI执行，本窗口准备任务并验收，无SSH请求。"
dump(tracked[0], task)

before = read(tracked[2])
p = copy.deepcopy(before)
p["current_priority"] = ["SPE-07", *[x for x in p["current_priority"] if x != "SPE-07"]]
p["current_priority_note"] = "用户明确SlotSPE最高优先级；先干净原版单臂验证问题，再按原始结果安排修正归因。其他线已有任务保留，不自动中断或新增。"
spe = next(t for t in p["tasks"] if t["id"] == "SPE-07")
spe["previous_two_arm_next_action"] = spe["next_action"]
spe["next_action"] = "服务器AI先跑干净原版单臂：fold0/seed3、3轮、256真实patch、20次评估，回传逐患者/梯度/干预/门控原始产物。"
spe["delivery"] = REL_ARCHIVE
spe["status"] = "awaiting_clean_original_server_ai_execution_and_artifacts"
spe["clean_original_plan"] = report["server_plan"]
spe["evidence_refs"].append("EV-SPE-CLEAN-20261019")
spe["completed_local_parts"].append("原版单臂入口、自然门控旁观、末轮补零及RNG重放、原始数值一致性验收和原版逐字节封包")
assert not any(t["id"] == "SPE-09" for t in p["tasks"])
p["tasks"].append({"id":"SPE-09", "title":"交付干净原版单臂服务器检查与严格回传验收", "status":"completed_local", "priority":"P0", "owner_window":"SlotSPE问题窗口", "depends_on":["SPE-08"],
    "next_action":"交服务器AI运行SPE-07；未收到原始产物前不登记服务器证明或患者归因。", "report":"docs/research/slotspe_clean_original_20261019/evidence_and_contract.json", "delivery":REL_ARCHIVE,
    "evidence_refs":["EV-SPE-CLEAN-20261019"], "evidence_level":"implementation_and_synthetic_only", "patient_effect":"not_evaluated", "last_checked":"2026-10-19"})
evidence_paths = [DEST/"evidence_and_contract.json", DEST/"full_suite_before_hardening.xml", DEST/"final_critical_checks.xml"]
p["evidence"].append({"id":"EV-SPE-CLEAN-20261019", "type":"clean_native_delivery_and_synthetic_e2e", "details":"完整34项检查通过；加固后4项关键单臂/旧模式闭环复验通过，不相加；实际服务器未核实。", "paths":[x.relative_to(ROOT).as_posix() for x in evidence_paths],
    "local_file_sha256":{x.relative_to(ROOT).as_posix():sha(x) for x in evidence_paths}})
line = next(x for x in p["lines"] if x["id"] == "L2")
line["current_clean_original_handoff"] = {"task":"SPE-07/SPE-09", "delivery":REL_ARCHIVE, "training_arms":["original"], "server_execution":"not_verified", "raw_results_received":False}
p["execution_boundary"]["recorder_scope"] = "用户新授权本窗口执行SlotSPE问题核查、原版诊断交接及验收；其他线只记录。"
p["execution_boundary"]["patient_jobs"] = "SlotSPE干净原版短程问题检查交服务器AI执行；本窗口未启动患者训练，其他任务不因本轮扩大。"
p["coordination"]["recorder_role"] = "当前执行SlotSPE问题核查、交接及回传验收；其他线只记录。"
p["coordination"]["current_slot_spe_authorization"] = {"recorded_at":now, "source":"用户要求证据、服务器复现、SlotSPE最高优先级、干净原版模型", "scope":"准备并检查原版单臂任务包，交服务器AI；无新增模型机制，无本地患者任务。"}
p["change_log"].append({"date":"2026-10-19", "scope":"SlotSPE_original_only_priority", "details":"新增干净原版单臂入口并设最高优先级，旧两臂/ABC包保留；新包从指定原版仓库逐字节封存，原始结果仍待服务器回传。", "backup":backup.relative_to(ROOT).as_posix()})
assert [x for x in before["tasks"] if not x["id"].startswith("SPE-")] == [x for x in p["tasks"] if not x["id"].startswith("SPE-")]
assert [x for x in before["lines"] if x["id"] != "L2"] == [x for x in p["lines"] if x["id"] != "L2"]
dump(tracked[2], p)
human = tracked[3].read_text(encoding="utf-8-sig")
human = human.replace("更新：2026年10月19日", "更新：2026年10月19日\n当前最高优先级：SlotSPE干净原版单臂问题复现，SPE-07/SPE-09。", 1)
marker = "2．SlotSPE：A/B/C代码已做好，等待服务器证明与归因。"
assert human.count(marker) == 1
human = human.replace(marker, "2．SlotSPE：先干净原版检查问题，A/B/C用于后续归因。\n最新入口：configs/slotspe_clean_original_server.json；只训练original，fold0/seed3、3轮、256真实patch、20次评估。\n交接包："+REL_ARCHIVE+"；原模型源码按指定桌面仓库逐字节封存。\nSPE-09本地完成原版入口、自然门控记录及原始数值验收；SPE-07等待服务器AI和原始回传。\n后续保留原两臂/A/B/C方案，先验收原版现象。\n")
tracked[3].write_text(human, encoding="utf-8")

build = subprocess.run([sys.executable, "-B", str(ROOT/"scripts/build_slotspe_proof_delivery.py"), "--original-only", "--clean-source-root", str(BASELINE), "--output", str(ARCHIVE)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
built = json.loads(build.stdout)
ARCHIVE.with_suffix(ARCHIVE.suffix+".sha256").write_text(built["sha256"]+"  "+ARCHIVE.name+"\n", encoding="ascii")
with zipfile.ZipFile(ARCHIVE) as z:
    name = next(n for n in z.namelist() if n.endswith("/DELIVERY_MANIFEST.json"))
    manifest = json.loads(z.read(name))
    for relative, digest in manifest["files_sha256"].items():
        assert hashlib.sha256(z.read(manifest["prefix"]+relative)).hexdigest() == digest
    for relative in baseline_files:
        assert z.read(manifest["prefix"]+"third_party/SlotSPE/"+relative) == (BASELINE/relative).read_bytes()
    default = json.loads(z.read(manifest["prefix"]+"configs/slotspe_proof_server.json"))
    assert default["experiment"] == "original_audit" and default["epochs"] == 3 and default["folds"] == [0] and default["seeds"] == [3]
receipt = {**built, "recorded_at":now, "baseline_10_files_byte_exact":True, "safe_default_original_only":True,
    "local_validation":report["validation"], "server_execution_verified":False, "raw_server_results_received":False, "remote_published":False,
    "previous_archive_sha256":sha(old_zip), "backup":backup.relative_to(ROOT).as_posix()}
dump(DEST/"delivery_receipt.json", receipt)
print(json.dumps(receipt, ensure_ascii=False))
