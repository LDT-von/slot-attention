from __future__ import annotations
import copy
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(r"E:\DCT-Reg")
SCRATCH = Path(__file__).parent
DEST = ROOT / "docs/research/slotspe_eight_issues_20261019"
ORIGINAL = Path(r"C:\Users\栋栋\Desktop\特征解耦论文\SlotSPE-main\SlotSPE-main")
now = datetime.now().astimezone().isoformat(timespec="seconds")
DEST.mkdir(parents=True, exist_ok=False)
for name in ["run_synthetic_audit.py", "synthetic_checks.json"]:
    shutil.copy2(SCRATCH/name, DEST/name)
raw = json.loads((DEST/"synthetic_checks.json").read_text(encoding="utf-8"))
old_root = ROOT/".review/slotspe_problem_audit_20261016"
old = json.loads((old_root/"diagnostic_results.json").read_text(encoding="utf-8"))
protocol = json.loads((old_root/"data_protocol_results.json").read_text(encoding="utf-8"))
old_hash_match = all(raw["source_hashes_before"][key] == value for key, value in old["source_hashes_before"].items())
assert old_hash_match and raw["all_checks_passed"] and raw["source_unchanged"]

def issue(id, status, finding, evidence, limit):
    return dict(id=id, status=status, finding=finding, evidence=evidence, limitation=limit)

issues = [
    issue("P1", "结构限制已确认_预测损害未证", "默认候选槽8、保留2；输入8/16/64个token时数量不变。主模型没有按患者推断激活数量。", ["utils/process_args.py:82-89", "models/SlotSPE.py:27", "models/SlotSPE.py:173-194", "models/slot_attention.py:75", "synthetic_checks.json:fixed_cardinality"], "固定容量不自动构成预测缺陷；SlotAttention单模块支持num_slots覆盖，主模型未按输入调用。动态内容与动态数量不同。"),
    issue("P2", "允许重复槽的结构已确认_训练后冗余待证", "同一患者的槽若从相同状态出发，10轮后仍完全相同；共享更新没有使相同槽主动分离的确定机制。", ["models/slot_attention.py:77-110", "models/SlotSPE.py:324-328", "synthetic_checks.json:slot_symmetry"], "相同初始化是人为控制边界，不是普通随机初始化的发生率。随机权重cosine高不能证明已训练模型功能冗余、碎片化或丢失预后信息。"),
    issue("P3", "随机性及单次评估波动已确认", "初始化参数为共享[1,1,D]高斯；每次forward重采样。Gumbel在eval仍采样。本轮固定权重和合成输入20次前向最大logit范围0.1732008。", ["models/slot_attention.py:38-41", "models/slot_attention.py:77-80", "models/slot_attention.py:121-123", "models/slot_attention.py:173", "utils/core_utils.py:348-408", "synthetic_checks.json:evaluation_randomness"], "槽迭代会读取当前输入，不能说最终槽缺乏输入信息。真实末轮checkpoint患者稳定性仍待服务器回传。"),
    issue("P4", "同编号语义无保证_实际语义漂移未证", "共享初始化和共享更新具有槽置换等变性；重命名两侧槽后完整预测最大差1.19e-7。槽编号没有固定病理名称。", ["models/slot_attention.py:38-40", "models/slot_attention.py:77-110", "synthetic_checks.json:slot_symmetry", "synthetic_checks.json:full_prediction_slot_relabeling"], "编号任意不等于学习失败；患者特异事件设计本身容许集合表示。不能用编号频率或直接同编号跨fold t-SNE证明语义身份。"),
    issue("P5", "输入无效token风险已确认_真实无关组织占槽待证", "loader可插入零patch，槽聚合没有padding mask；零输入经带偏置MLP非零。本轮补16/64个零token导致最终logit变化0.052293/0.083113。另有未选槽self泄漏。", ["dataset/dataset_survival.py:222-236", "dataset/dataset_survival.py:285-292", "models/omics_encoder.py:18-23", "models/slot_attention.py:69-100", "models/SlotSPE.py:296-300", "synthetic_checks.json:unmasked_zero_tokens"], "合成零token不等于已发现真实患者无关组织；非肿瘤、基质和免疫组织不能自动归为无关。TopK槽也不等于删除patch。"),
    issue("P6", "显式空间多尺度接口缺口已确认_患者损害未证", "主路径只收patch特征矩阵，没有坐标、倍率或层级接口。同一特征多重集重排后预测差1.19e-7；不能用该入口区分相同patch特征的不同空间布局。", ["dataset/dataset_survival.py:278-301", "models/SlotSPE.py:253-255", "models/slot_attention.py:95-100", "synthetic_checks.json:patch_order_invariance"], "当前入口不显式处理尺度，不代表上游patch编码没有形态或尺度信息；更不证明新增多尺度后会提分。"),
    issue("P7", "最终目标到选槽器的直接梯度缺失已确认", "完整模型丢弃decoder的连续slot_gate，只将hard_keep.bool用于self；两种TopK实现下主NLL对8个选槽参数梯度均None，辅助目标仍连通。另有编码空间重构的常量退化解。", ["models/SlotSPE.py:289-300", "models/SlotSPE.py:304-328", "utils/core_utils.py:239-246", "synthetic_checks.json:selector_gradient", str(old_root/"diagnostic_results.json")], "模型已有主生存与单模态生存辅助监督；直接主梯度缺失不代表选槽器完全不训练，也不自动证明这种监督分工导致预测损害。"),
    issue("P8", "不适用原版SlotSPE", "原版后续交互为cross-attention，没有用户所说的跨模态OT阶段，因此不能把双阶段OT冲突算成原版问题。", ["models/SlotSPE.py:196-199", "models/SlotSPE.py:293-294", "models/transformer.py:292-299"], "Slot Attention内部归一化与OT的理论联系，不是另一个跨模态OT阶段存在或冲突的证据。")
]

audit = dict(task_id="SPE-08", recorded_at=now, original_repository=str(ORIGINAL),
    scope="原版源码与合成结构核查；无患者特征读取、训练、checkpoint评估或新模块",
    inputs={str(ORIGINAL/"docs/slot_attention_issues.md"): hashlib.sha256((ORIGINAL/"docs/slot_attention_issues.md").read_bytes()).hexdigest()},
    source_hashes=raw["source_hashes_before"], original_source_unchanged=True,
    historical_evidence_source_hashes_match_current=old_hash_match,
    synthetic_checks=dict(passed=sum(raw["checks"].values()), total=len(raw["checks"]), independent_patient_experiments=0,
        report="synthetic_checks.json", runner="run_synthetic_audit.py"),
    issues=issues,
    concrete_findings=[
        dict(id="F1", title="self掩码后仍平均未选槽", level="新合成干预_原模块", source="models/transformer.py:89-100,128-132; models/SlotSPE.py:296-301", metrics=raw["findings"]["unselected_self_full_prediction"], explanation="全掩码query行softmax变均匀，残差还保留原槽，最后mean再次纳入。固定mask和cross后最终预测变化。真实患者末轮槽干预未完成。"),
        dict(id="F2", title="最终NLL不能经选择权重直接训练选槽器", level="源码与新合成梯度", evidence="synthetic_checks.json:selector_gradient", explanation="bool转化切断连续门控路线；辅助单模态生存损失仍训练选槽器，必须通过B/C患者对照判断这种分工影响。"),
        dict(id="F3", title="eval仍随机且原评估逐病例单次前向", level="源码与新合成重复评估", metrics=raw["findings"]["evaluation_randomness"]),
        dict(id="F4", title="补零token未被排除", level="数据加载源码与新合成反例", metrics=raw["findings"]["unmasked_zero_tokens"]),
        dict(id="F5", title="低温全槽softmax后掩码导致门控近零", level="原decoder可达分数模式反例", source="models/SlotSPE.py:63-68", scores=[1.0,0,0,0,0,0,0,0], temperature=0.01,
             trials=32, collapse_count=raw["findings"]["low_temperature_gate_counterexample"]["below_1e6_count"], minimum_gate_sum=min(r["gate_sum"] for r in raw["findings"]["low_temperature_gate_counterexample"]["trials"]), explanation="Gumbel选集可能不含未加噪声的最大分槽，其余分数先经低温softmax变极小，再被分母1e-8压成近零。该输出用于辅助decoder；发生频率及训练影响未在真实checkpoint测量。"),
        dict(id="F6", title="重构目标允许常量编码退化", level="历史合成反例原始JSON复核_本轮不重跑", source=str(old_root/"diagnostic_results.json"), result=old["experiments"]["constant_latent_reconstruction_counterexample"], explanation="可训练编码空间的目标不是信息保留保证；主/辅助生存目标仍可反对坍缩。"),
        dict(id="F7", title="全队列分箱让验证结局影响训练标签", level="历史真实元数据内存反事实原始JSON复核_本轮不重跑", source=str(old_root/"data_protocol_results.json"), changed_training_labels=protocol["validation_time_counterfactual"]["training_patient_discrete_labels_changed"], total_training_cases=protocol["counts"]["original_train_dataset"], explanation="原qcut结果被后续全队列等宽pd.cut覆盖。历史一名验证时间极端干预改变55/303训练标签；非正常数据偏差估计或提分证明。")
    ],
    paper_code_boundary={"dense_cross": "论文3.2明确为保留跨模态协同而使用全部槽，此项是设计，不能作为self泄漏缺陷。", "paper_url":"https://arxiv.org/html/2512.01116v1"},
    corrections_to_supplied_material=[
        "本仓库入口是survival.py、dataset/dataset_survival.py；附件slot_spe/model.py、slot_spe/trainer.py是示例路径。",
        "实际slots_mu/logsigma为[1,1,D]共享参数，不是文档代码块的[1,num_slots,D]。",
        "默认8槽、比例0.25，所以TopK为2；不是固定7槽或默认TopK=4。默认slot_iters=10，不是3。",
        "示例脚本读取UNI离线特征，不能按DINOv2四层特征或冻结EMA目标编码器诊断。",
        "原模型已有融合生存、两路单模态生存和三路重构；不是只优化重建，也没有本轮所说的图像像素分割输出。",
        "TopK编号频率不证明语义一致；高cosine不单独证明功能冗余；跨fold t-SNE须先处理槽置换和表示坐标差异。",
        "未选槽不是被丢弃patch；无真值实例掩码和对象分割输出，FG-ARI/mBO及+2%～+8%提升数字不能作为本任务指标或预期结果。",
        "P1/P2/P4/P6需保留结构事实与患者问题之间的证据缺口，缺某模块不等于已有故障。"
    ],
    primary_literature=[
        {"name":"SlotSPE", "url":"https://arxiv.org/html/2512.01116v1", "checked_scope":"self稀疏读取与cross全槽设计、总体监督；没有用论文患者分数当本轮结果"},
        {"name":"AdaSlot", "url":"https://openaccess.thecvf.com/content/CVPR2024/html/Fan_Adaptive_Slot_Attention_Object_Discovery_with_Dynamic_Slot_Number_CVPR_2024_paper.html", "boundary":"也可先建候选槽后离散选择；关键是激活数量能否变化"},
        {"name":"CoSA", "url":"https://proceedings.iclr.cc/paper_files/paper/2024/hash/ba4caa85ecdcafbf9102ab8ec384182d-Abstract-Conference.html", "boundary":"对象语义字典不等于跨患者通路语义已验证"},
        {"name":"Probabilistic Slot Attention", "url":"https://papers.neurips.cc/paper_files/paper/2024/file/a9a3f0e4a95cb273867931369c8fc3b1-Paper-Conference.pdf", "boundary":"理论条件下仍允许槽置换和仿射变换；非固定同编号病理身份保证"},
        {"name":"MESH", "url":"https://proceedings.mlr.press/v202/zhang23ba.html", "boundary":"内部运输解释不证明两阶段跨模态运输冲突"}
    ],
    other_literature_entries="本轮未逐项实测全部15条会议与标题，不登记为已验证",
    server_execution="not_verified", server_raw_results_received=False, patient_effect="not_evaluated_this_audit",
    attribution="现有A/B/C代码验收与本轮原版诊断分开；尚无患者结果支持修正提分或新增模块必要性",
    next_existing_task="SPE-07按既定短程预算等待服务器AI原始产物验收；本轮不增加服务器实验或模块。")
(DEST/"audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

f = raw["findings"]
lines = ["SlotSPE八类问题核查：原版源码、合成反例与患者证据边界", f"任务：SPE-08；记录：{now}",
    f"原仓库：{ORIGINAL}", "本轮读取用户八问题目录与工作流；直接执行未修改原版模块。没有患者特征读取、训练、优化器更新或checkpoint加载。",
    f"14项诊断断言通过，九个核心源码/示例配置文件哈希前后不变；其中八个与2026-10-16原始诊断完全一致。", "",
    "一、最具体的故障与目标路线", "1. self分支所谓未选槽没有真正退出最终汇总。",
    "   全掩码query行softmax后是1/8均匀注意力；残差保留原槽，SlotSPE.py:300再mean全部槽。",
    f"   固定keep集合、冻结cross，改变未选WSI槽后，最终logit最大变化{f['unselected_self_full_prediction']['max_logit_difference']:.8f}，risk最大变化{f['unselected_self_full_prediction']['max_risk_difference']:.8f}。",
    "   这是随机权重、合成输入和槽干预，真实患者末轮干预仍未完成。",
    "2. 最终融合NLL到选槽器的直接路线被bool切断。两种TopK下8个参数梯度全为None；辅助监督仍连通，个别梯度数值为0不等于图断开。",
    "   此项证明监督路线分工，是否需要接通及患者效果由既定B/C对照决定。",
    f"3. eval仍重新采样。合成固定输入/权重20次最大logit范围{f['evaluation_randomness']['logit_max_range']:.8f}；恢复同一随机状态差值0。",
    "4. loader补零没有输入mask。补16/64零patch时，最终logit分别变化0.052293/0.083113；零向量经MLP也非零。真实患者发生率与影响未测。",
    "5. 原decoder先全槽低温softmax再mask，选集不含原最高分槽时可能被1e-8分母压成近零门控。",
    f"   合成分数[1,0,0,0,0,0,0,0]、温度0.01，32次中{f['low_temperature_gate_counterexample']['below_1e6_count']}次gate_sum<1e-6；它首先影响辅助decoder，不能直接当最终头归零。", "",
    "二、逐项回答八类问题是否存在"]
for x in issues:
    lines.extend([f"{x['id']}：{x['status']}", f"  发现：{x['finding']}", f"  位置：{'；'.join(x['evidence'])}", f"  边界：{x['limitation']}"])
lines += ["", "三、已有证据复核，不冒充本轮新实验", "原版八个核心源码SHA256与2026-10-16诊断一致。",
    "历史重构反例三路损失可为0，但不能证明患者训练坍缩；历史元数据干预改变55/303训练标签，不能当正常C-index偏差。原产物仍在原位。",
    "", "四、论文与材料纠偏", "SlotSPE论文3.2明确cross分支使用全部槽以保留跨模态协同；cross全槽是设计，不是未选self槽漏进mean的同一个缺陷。"]
lines.extend(audit["corrections_to_supplied_material"])
lines += ["仅核对了下列一手资料支持的边界，全部15篇逐项精读/核验尚未进行："]
lines.extend(f"{p['name']}：{p['url']}" for p in audit["primary_literature"])
lines += ["", "五、证据与下一步", "问题证明：以上源码事实、结构反例和计算敏感性已记录。代码检查：14项通过，原版未修改。",
    "患者效果：本轮没有运行患者任务；服务器真实执行情况未核实，未收到原始结果。归因结论：尚不能证明哪项修正提高患者效果或必须新增某模块。",
    "既定下一步仍是SPE-07：fold0/seed3，原版与组合修正版各3轮、256真实patch、20次评估；由服务器AI执行并回传原始产物。没有自动扩大五折或新增模块。",
    "原始结果：synthetic_checks.json；可重现脚本：run_synthetic_audit.py；机器可读结论：audit.json。",
    "复现命令：D:\\python\\python.exe -B run_synthetic_audit.py（在本机；依赖已存在的torch/einops，不安装依赖）。"]
(DEST/"核查结论.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")

# Update only SlotSPE fields in the existing shared entrypoints, with backups.
progress_path = ROOT/"docs/THREE_LINE_PROGRESS.json"
human_path = ROOT/"docs/三条研究路线_工作总览_20261019.txt"
before = json.loads(progress_path.read_text(encoding="utf-8-sig"))
p = copy.deepcopy(before)
backup = ROOT/"docs/history"/("slotspe_eight_issue_audit_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
backup.mkdir(parents=True, exist_ok=False)
for path in [progress_path, human_path]:
    shutil.copy2(path, backup/path.name)
assert not any(t["id"] == "SPE-08" for t in p["tasks"])
p["tasks"].append(dict(id="SPE-08", title="核实八类槽问题在指定原版SlotSPE中的存在性", status="completed_local", priority="P0", owner_window="SlotSPE问题窗口", depends_on=[],
    next_action="源码/合成核查已完成；患者发生率与效果仍等待SPE-07原始回传，不新增模块或扩大实验。",
    acceptance=["八类逐项标明确认/结构边界/患者待证/不适用，给源码位置。", "保留不能成立的推断与实际缺口；原版源码不改。", "合成反例、历史真实输入诊断、服务器转述、患者效果分层。"],
    expected_outputs=["源码与合成核查报告及原始JSON"], evidence_refs=["EV-SPE-EIGHT-20261019"], last_checked="2026-10-19", checks_passed=14, evidence_level="source_and_synthetic_only", patient_effect="not_evaluated", report="docs/research/slotspe_eight_issues_20261019/audit.json"))
paths = ["audit.json", "核查结论.txt", "synthetic_checks.json", "run_synthetic_audit.py"]
p["evidence"].append(dict(id="EV-SPE-EIGHT-20261019", type="original_source_audit_and_synthetic_counterexamples", details="14项诊断断言通过；九文件源码哈希不变；无患者任务。八类假设逐项判定，cross全槽按论文明确为设计。",
    paths=[str((DEST/name).relative_to(ROOT)).replace("\\", "/") for name in paths],
    thread_id="01a15216-57f0-7993-b340-75195a2808fb", local_file_sha256={name:hashlib.sha256((DEST/name).read_bytes()).hexdigest() for name in paths}))
line = next(x for x in p["lines"] if x["id"] == "L2")
line["latest_source_audit"] = dict(task_id="SPE-08", evidence_ref="EV-SPE-EIGHT-20261019", checks=14, recorded_at=now,
    patient_evidence=False, result="具体self泄漏、评估随机性、门控/补零风险已复现；固定槽容量与语义等假设保留证据缺口，P8不适用。")
p["coordination"]["current_slot_spe_authorization"] = dict(recorded_at=now, source="用户本轮明确要求核实指定原版仓库的槽问题", scope="允许原版只读代码核查和无患者输入合成诊断；不新增机制，不启动患者实验。", existing_cross_line_recorder_scope_unchanged=True)
for source in p["coordination"]["sources"]:
    if source["line"] == "SlotSPE":
        source["snapshot"] = "SPE-01/02本地完成、SPE-03已发布cc81df1；SPE-08原版八问题源码/14项合成核查完成，SPE-07仍待服务器原始产物，患者效果未证。"
        source["evidence_level"] = "direct_local_original_source_and_synthetic_audit_no_patient_results"
p["change_log"].append(dict(date="2026-10-19", scope="SlotSPE_SPE-08", details="按用户新授权核查八类槽问题，保存原版哈希、14项合成证据与逐项边界；未改变其他线任务、患者预算、原版源码或已发布归因包。", backup=str(backup.relative_to(ROOT)).replace("\\", "/")))
assert [x for x in p["tasks"] if not x["id"].startswith("SPE-")] == [x for x in before["tasks"] if not x["id"].startswith("SPE-")]
assert [x for x in p["lines"] if x["id"] != "L2"] == [x for x in before["lines"] if x["id"] != "L2"]
progress_path.write_text(json.dumps(p, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
human = human_path.read_text(encoding="utf-8-sig")
marker = "2．SlotSPE：A/B/C代码已做好，等待服务器证明与归因。"
assert human.count(marker) == 1
human = human.replace("本窗口只收集、核对和汇总进度，不改源码、不设计方法、不安排或启动实验。", "此前本窗口承担三线进度记录；用户随后明确授权本轮核实SlotSPE槽问题，已完成原版只读审计及合成检查。其他线仍只记录进度。")
human = human.replace("SPE-01/02本地完成；SPE-03独立包就绪但未推送。", "SPE-01/02本地完成；SPE-03已隔离发布origin/main@cc81df1。")
addition = "\nSPE-08新增核查：对指定桌面原版的八类问题逐项定位，14项合成诊断通过，九个源码/示例文件哈希不变。\n具体复现self未选槽影响最终预测、eval随机性，并确认补零无mask及低温门控近零风险；固定容量/重复槽/语义/空间能力按证据边界登记，P8不适用。\n报告：docs/research/slotspe_eight_issues_20261019/核查结论.txt；原始数值与复现脚本在同目录。\n本轮无患者特征读取、训练或checkpoint评估，没有服务器回传；既定SPE-07和患者A/B/C任务不变。\n"
human = human.replace(marker, marker+addition)
human = human.replace("本次仅更新这两份进度入口，并备份修改前内容；未新增代码、实验、清理、推送或其他窗口消息。", "上次三线汇总仅更新进度入口；本轮按用户新授权新增SPE-08原版源码核查、合成诊断与报告，并备份入口。未修改原版模型，未启动患者任务、清理或再次推送。")
human_path.write_text(human, encoding="utf-8")
print(json.dumps(dict(report=str(DEST/"核查结论.txt"), checks=14, backup=str(backup), original_source_unchanged=True, other_lines_unchanged=True), ensure_ascii=False))
