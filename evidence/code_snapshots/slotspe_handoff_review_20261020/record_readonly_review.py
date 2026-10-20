"""Record source/file reviews only. Never imports torch or executes a model."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(r"E:\DCT-Reg")
SOURCE = Path(r"C:\Users\栋栋\Desktop\特征解耦论文\SlotSPE-main\SlotSPE-main")
STAMP = datetime.now().astimezone().isoformat(timespec="seconds")
DIRS = {i: ROOT / f"docs/research/slotspe_p{i}_20261020" for i in range(5, 9)}
PROGRESS = ROOT / "docs/THREE_LINE_PROGRESS.json"
HUMAN = ROOT / "docs/三条研究路线_工作总览_20261019.txt"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def ref(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def excerpt(relative, start, end):
    path = SOURCE / relative
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return {**ref(path), "start": start, "end": end,
            "source": "\n".join(f"{n}: {lines[n-1]}" for n in range(start, end + 1))}


def main():
    before = read(PROGRESS)
    human_before = HUMAN.read_text(encoding="utf-8")
    original_p5 = DIRS[5] / "P5_只读核验报告.txt"
    targets = [DIRS[5] / "P5_交接复核与勘误.txt", DIRS[5] / "readonly_handoff_receipt.json"]
    targets += [DIRS[i] / f"P{i}_只读核验报告.txt" for i in (6, 7, 8)]
    targets += [DIRS[i] / "source_review.json" for i in (6, 7, 8)]
    targets += [DIRS[8] / "progress_update_validation.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite an existing review output")
    backup = ROOT / "docs/history" / ("slotspe_handoff_readonly_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    backup.mkdir(parents=True, exist_ok=False)
    for path in (PROGRESS, HUMAN):
        shutil.copy2(path, backup / path.name)

    synthetic_path = ROOT / "docs/research/slotspe_eight_issues_20261019/synthetic_checks.json"
    synthetic = read(synthetic_path)
    original_sources = {str(SOURCE / name): sha(SOURCE / name)
                        for name in synthetic["source_hashes_before"]}
    assert all(original_sources[str(SOURCE / name)] == digest
               for name, digest in synthetic["source_hashes_before"].items())
    old_gradient_path = ROOT / ".review/slotspe_problem_audit_20261016/diagnostic_results.json"
    old_gradient = read(old_gradient_path)
    assert not old_gradient["trained_checkpoint_loaded"]
    assert old_gradient["optimizer_steps"] == 0
    for row in synthetic["findings"]["selector_gradient"]:
        assert len(row["main_gradient"]) == 8 and all(v is None for v in row["main_gradient"].values())
        assert len(row["auxiliary_gradient"]) == 8 and all(v is not None for v in row["auxiliary_gradient"].values())

    history_root = ROOT / ".review/slotspe_proof_20261018/patient"
    expected = {"cache/paired_blca_fold0.pt": "06ff3816afebd0cca1ff53cb3bc05e5fc34e2989e521c85222631c50bc8f1b61",
                "seed_3/original/last.pt": "3f8ed9878cf451854efe93a3d2fd73f9a66cf341522362dc626b061902a9407c",
                "manifest.json": "ac7921b8a3e1ef290a292c0bc2bbdec575ae07ebbb39fc0fe444e3e5eea85fe3"}
    historical = [ref(history_root / name) for name in expected]
    assert all(item["sha256"] == expected[Path(item["path"]).relative_to(history_root).as_posix()]
               for item in historical)
    manifest = read(history_root / "manifest.json")
    splits = dict(Counter(row["split"] for row in manifest["patients"]))
    # Metadata is not a new tensor-level zero-row audit.
    manifest_summary = {"patients": len(manifest["patients"]), "splits": splits,
                        "minimum_full_patch_count": min(row["full_patch_count"] for row in manifest["patients"]),
                        "tensor_zero_counts_recomputed_this_review": False}
    names = ["slotspe_abc_validation_20261019", "slotspe_clean_original_validation_20261019",
             "slotspe_clean_publication_20261019", "slotspe_eight_issues_20261019",
             "slotspe_literature_20261020", "slotspe_p1_20261020", "slotspe_p2_20261020",
             "slotspe_p3_20261020", "slotspe_p4_20261020", "slotspe_problem_audit_20261016",
             "slotspe_proof_20261018", "slotspe_publication_20261019", "slotspe_transport_push_20261019"]
    markers = []
    for name in names:
        marker = ROOT / ".review" / name / "ARCHIVED.txt"
        markers.append({**ref(marker), "content": marker.read_text(encoding="utf-8-sig"),
                        "marker_alone_proves_duplicate_or_deletable": False})
    raw_roots = {1: ROOT / ".review/slotspe_p1_20261020/seed3_native_v2/raw",
                 2: ROOT / ".review/slotspe_p2_20261020/seed3_native_v1/raw",
                 3: ROOT / ".review/slotspe_p3_20261020/seed3_native_v1",
                 4: ROOT / ".review/slotspe_p4_20261020/seed3_native_v1/raw"}
    raw_inventory = [{"issue": f"P{i}", "raw_root": str(path),
                      "direct_npz_count": len(list(path.glob("*.npz"))),
                      "docs_npz_count_recursive": len(list((ROOT / f"docs/research/slotspe_p{i}_20261020").rglob("*.npz"))),
                      "report_receipt": ref(ROOT / f"docs/research/slotspe_p{i}_20261020/measurement_receipt.json"),
                      "statistics_recomputed_this_review": False}
                     for i, path in raw_roots.items()]
    server_paths = [ROOT / "实验档案/结果/SPE09_slotspe_clean_original_20261019",
                    ROOT / "results/slotspe_proof/slotspe_clean_original_blca_run_20261020"]
    server_inventory = [{"path": str(path), "exists": path.exists()} for path in server_paths]
    assert not any(item["exists"] for item in server_inventory)
    common = {"task_id": "SPE-08", "recorded_at": STAMP,
              "method": "exact_source_read_and_existing_metadata_hash_review_only",
              "new_training_runs": 0, "new_model_forward_passes": 0, "new_backward_passes": 0,
              "patient_tensors_loaded_this_review": False, "new_synthetic_checks_run": 0,
              "server_execution_verified": False, "server_raw_results_received": False,
              "historical_synthetic_record": ref(synthetic_path),
              "original_source_hashes": original_sources,
              "patient_predictive_harm_established": False, "correction_efficacy_established": False}

    corrections = [
        {"lines": "33", "old": "manifest.csv", "correct": "实际依据为manifest.json；已重算完整SHA，见本收据。"},
        {"lines": "34,146", "old": "缓存可读/checkpoint可加载=可重算复核通过", "correct": "存在和哈希一致不等于已经独立重算P1-P4数值；本轮没有重算。"},
        {"lines": "63,154", "old": "71名验证患者零行统计后写已检查360患者", "correct": "统计范围不一致，且未随报告落盘原始统计；零行结果本轮未重新核验。manifest总人数360=289训练+71验证。"},
        {"lines": "100", "old": "零行会被分配0权重", "correct": "没有输入mask保障；带偏置MLP可将零输入映射为非零，softmax及eps归一化不会自动排除该token。不能断言零注意力。"},
        {"lines": "68", "old": "无法用现有缓存直接验证零行干预", "correct": "无自然零行只限制自然发生率观察；人工拼零仍可在配对随机状态下检验计算敏感性。其结果不能代表真实无关组织。"},
        {"lines": "166,169", "old": "1名患者的零行注入比较C-index", "correct": "单患者无可比较患者对，不能算有效C-index。个体干预应报logits/risk差；队列排序需多个患者及明确配对规则。"},
        {"lines": "125", "old": "P2/P3/P4都是槽层面的干预", "correct": "P3还包括整模型20次重复评估及槽/Gumbel随机数控制，不能统称只做槽层干预。"},
        {"lines": "13个ARCHIVED.txt", "old": "P1-P4与docs目录是相同内容副本", "correct": ".review保存原始NPZ而docs保存报告；两者互补。不能据此认定可删，原标记保留并由本收据纠偏。"}]
    inventory = {**common, "issue": "P5_handoff_and_record_integrity", "original_p5_report": ref(original_p5),
                 "original_p5_report_unchanged": True, "p5_original_directory_files_before": [p.name for p in DIRS[5].iterdir() if p.is_file()],
                 "historical_files": historical, "manifest_metadata_summary": manifest_summary,
                 "p1_p4_raw_inventory": raw_inventory, "archive_markers": markers,
                 "server_paths_checked": server_inventory, "corrections": corrections,
                 "P5_natural_zero_rows": "reported_zero_in_71_validation_not_recomputed_and_raw_statistics_not_retained_in_P5_directory",
                 "P5_real_irrelevant_tissue_occupancy_or_harm": "not_established"}
    write(DIRS[5] / "readonly_handoff_receipt.json", inventory)
    write(DIRS[5] / "P5_交接复核与勘误.txt", f"""SlotSPE P5交接复核与勘误
记录：{STAMP}；任务SPE-08/P5；本文件补充原报告，不覆盖原报告。

1. 路线和任务号
SlotSPE问题存在性；SPE-08/P5。仅核对文件、哈希、源码和既有JSON。

2. 本轮已经完成什么
原P5目录只有P5_只读核验报告.txt，没有同轮统计原始记录、脚本或独立复算收据。
历史缓存、seed3原版last.pt、manifest.json均存在，完整哈希与历史记录一致。
manifest记录360人=289训练+71验证，最小原始patch数445；此为元数据核对，不是重新统计缓存零行。
13个ARCHIVED.txt均存在；P1/P2/P3/P4原始目录分别含71/71/6/71个直接NPZ，对应docs目录均不含NPZ。
因此P1-P4不能按“同名重复副本”清理。所有原始材料和标记均保留，本文件记录分类纠偏。

原P5报告需纠正：
""" + "\n".join(f"- 原位置{item['lines']}：{item['correct']}" for item in corrections) + f"""

3. 对应文件
原报告：{original_p5}
完整哈希、目录清单及源码版本：{DIRS[5] / 'readonly_handoff_receipt.json'}
P1-P4原始目录、历史缓存与checkpoint均见收据的绝对路径。不提交、不推送。

4. 检查与运行状态
文件存在、完整哈希和原版源码版本核对通过。未重新运行合成诊断、前向、反向或训练。
没有重新加载患者张量；原报告“验证71人无零行”在本次仍属转交报告中的统计，未独立重算。
P1-P4既有测量收据和原始文件存在，本轮未重算全部数值，不能写成本轮数值验收通过。
两个约定SPE09目录不存在；服务器有完成转述，尚无本批次原始回传验收。

5. 未完成与原因
真实无关组织是否占槽、是否损害预测均未证实；当前材料缺逐patch组织语义标注和匹配患者对照。
原P5零行统计未落盘原始产物，71/360范围不一致。自然样本无补零与人工补零敏感性是不同问题。
源码还存在缺失slide文件回退为整袋零特征（dataset_survival.py:229-234），并非仅训练patch不足会产生零行。
训练采样不足补零条件在:286-292；SlotAttention输入无mask在slot_attention.py:69-100。

6. 已安排下一步与用户材料
本轮将P6/P7/P8补成源码复核文件，记录原有合成证据而不重跑。
现有服务器任务先验收原始回传包及SHA；P5不因“缓存无零行”被判定已解决，也不启动新患者任务。
真正研究组织占槽，需要可追溯的patch映射、组织标注及其与预后问题的定义；不能把所有基质/免疫区直接叫无关组织。
""")

    p6 = {**common, "issue": "P6", "conclusion": "explicit_spatial_scale_interface_missing_patient_harm_unproved",
          "source_excerpts": [excerpt("dataset/dataset_survival.py", 278, 301), excerpt("models/SlotSPE.py", 253, 265), excerpt("models/slot_attention.py", 82, 100)],
          "existing_synthetic_patch_order_record": synthetic["findings"]["patch_order_invariance"],
          "cache_schema_in_other_AI_chat": "reported_feature_only_not_reloaded_this_review",
          "upstream_feature_multiscale_content": "not_established_by_interface_review"}
    p7 = {**common, "issue": "P7", "conclusion": "direct_fused_loss_selector_gradient_disconnected_auxiliary_connected",
          "source_excerpts": [excerpt("models/SlotSPE.py", 43, 70), excerpt("models/SlotSPE.py", 283, 328), excerpt("utils/core_utils.py", 232, 251)],
          "existing_synthetic_gradient_records": synthetic["findings"]["selector_gradient"],
          "historical_random_weight_real_input_record": {**ref(old_gradient_path), "trained_checkpoint_loaded": False,
              "optimizer_steps": 0, "record": old_gradient["experiments"]["selection_gradient_real_inputs"]},
          "new_gradient_experiment": False, "does_not_mean_all_selector_training_absent": True,
          "abc_patient_attribution": "pending"}
    p8 = {**common, "issue": "P8", "conclusion": "not_applicable_to_original_no_downstream_crossmodal_OT",
          "source_excerpts": [excerpt("models/SlotSPE.py", 1, 8), excerpt("models/SlotSPE.py", 196, 199), excerpt("models/SlotSPE.py", 283, 301),
                              excerpt("models/transformer.py", 211, 238), excerpt("models/transformer.py", 292, 299)],
          "upstream_slot_attention_OT_theory_is_not_a_second_crossmodal_OT_stage": True,
          "transport_plugin_evidence_used": False}
    for i, value in ((6, p6), (7, p7), (8, p8)):
        write(DIRS[i] / "source_review.json", value)

    details = {
        6: ("显式空间/倍率接口缺口已确认，患者预测损害未证实。",
            "原版dataset返回wsi、genes、label、time、censor；主forward消费patch特征及组学，没有显式坐标/倍率/金字塔输入读取。\n"
            "源码：dataset/dataset_survival.py:278-301；models/SlotSPE.py:253-265；models/slot_attention.py:82-100。\n"
            "既有2026-10-19合成记录：相同特征多重集重排、控制RNG，最大logit差1.1920928955078125e-7。本轮仅读取记录。",
            "不能由无显式坐标接口推断离线patch特征没有局部形态或尺度信息；也未证明真实患者因该接口缺口损失预后信息。\n"
            "其他AI声称读取缓存键，本轮不加载张量，只把该项按转述保留。没有空间布局/倍率对照和匹配患者结果。",
            "先保留能力边界；真实效果需可追溯的空间/尺度数据与同条件对照，不新增多尺度模块。"),
        7: ("最终融合损失到选槽器的直接梯度路线断开；辅助生存梯度仍连通。患者损害及接通收益未证实。",
            "decoder返回(logits,连续slot_gate,hard_keep)，主forward丢弃连续gate，只把hard_keep.bool送入self注意力；cross直接使用槽向量。\n"
            "最终融合logits不以可微选择权重依赖pred_keep_slot；辅助单模态NLL通过decoder连续gate训练该选槽器。\n"
            "源码：models/SlotSPE.py:43-70,289-301,324-328；utils/core_utils.py:232-251。\n"
            "重读旧合成JSON：两种TopK各8个选槽参数主NLL梯度全None；辅助梯度均非None，个别数值为0。\n"
            "2026-10-16真实4人输入/随机未训练权重的旧记录也给出主梯度8个None、辅助连通；它不是已训练患者结论或本轮新实验。",
            "直接主梯度缺失不等于所有目标与任务无关。原版包含最终生存、两路单模态生存及重构监督，不能说选槽器完全不训练。\n"
            "也不能把结构路线事实扩写为已发生泛化损害、修正必然提分或需要新创新模块。尚无已验收患者B/C归因。",
            "已有A/B/C代码继续保留；B/C同前向，辅助目标/梯度、选择、数值与评估一致，仅改变最终融合监督经选择权重回传路线。\n"
            "先验收服务器trained_selector_gradients等原始产物；此处不启动B/C训练。"),
        8: ("不适用原版SlotSPE：原版没有后续跨模态OT阶段。",
            "原版构造IterativeCrossAttTransformer，跨模态前向调用cross_attention(slots_wsi,slots_omic)。\n"
            "实现为双向q/k点积、softmax、加权v、投影、GRU和残差，并非后续运输求解器。\n"
            "源码：models/SlotSPE.py:3-6,196-199,293-301；models/transformer.py:211-238,292-299。\n"
            "本轮按可执行调用路径复核，而非仅凭全文未搜到OT字样。",
            "SlotAttention内部竞争归一化与OT的理论联系，不会自动生成第二个跨模态OT阶段；不能据此证明双阶段冲突。\n"
            "加运输插件后的新模型属于另一条任务，不能倒推成原版SlotSPE缺陷。",
            "原版P8登记not_applicable，停止为其安排患者故障实验。后续若检查另一个实际有OT的模型，须单独定义范围和对照。")}
    for i, (conclusion, completed, missing, next_step) in details.items():
        write(DIRS[i] / f"P{i}_只读核验报告.txt", f"""SlotSPE P{i}只读源码复核报告
记录：{STAMP}；原版仓库：{SOURCE}
结论：{conclusion}

1. 路线和任务号
SlotSPE八类问题存在性核实；SPE-08/P{i}。

2. 本轮已经完成什么
{completed}

3. 对应文件、报告或提交号
本报告：{DIRS[i] / f'P{i}_只读核验报告.txt'}
源码摘录、完整SHA和旧证据引用：{DIRS[i] / 'source_review.json'}
历史合成记录：{synthetic_path}
既有2026-10-19八项综合报告已含P6/P7/P8；本轮补独立落盘与复核，不登记为首次新实验。
未提交、未推送。

4. 检查是否通过；服务器实际运行与原始回传
九份原版源码/示例文件SHA与历史合成记录一致；源码检查及既有JSON一致性核对通过。
新合成测试0、新前向0、新反向0、新训练0；本轮未加载患者缓存张量。
旧实验指标仅作为标明日期/输入/权重状态的既有记录，未重新执行或独立重算。
服务器有用户转述的完成消息，两个约定目录尚无原始产物，实际运行未独立核验。

5. 未完成事项及原因
{missing}

6. 已安排下一步及所需材料
{next_step}
当前用户只需转交已有服务器原始包与SHA供验收，无需提供SSH。不能用运行成功消息代替原始产物。
""")

    # Update only the established SlotSPE entrances; preserve other tasks and lines.
    progress = copy.deepcopy(before)
    task = next(t for t in progress["tasks"] if t["id"] == "SPE-08")
    task.setdefault("historical_local_stages", []).append(copy.deepcopy(task["local_stage"]))
    task["local_stage"] = {"date": "2026-10-20", "completed": "P5交接/归档勘误；P6-P8独立源码复核落盘",
                           "new_training_runs": 0, "new_model_forward_passes": 0, "new_backward_passes": 0,
                           "originals_unchanged": True, "patient_failure_proof_completed": False}
    task["next_action"] = "优先验收SPE09现有原始回传；P5自然零行统计待原始记录，真实无关组织占槽/损害未证；P6患者空间尺度损害、P7患者B/C归因待证；P8原版不适用。不新增训练/模块。"
    task["last_checked"] = "2026-10-20"
    task["last_readonly_review"] = {"report": "docs/research/slotspe_p5_20261020/readonly_handoff_receipt.json",
                                     "scope": "source_metadata_and_file_hashes_no_new_models_or_arrays",
                                     "P5_report_corrected_by_addendum": True,
                                     "P6": p6["conclusion"], "P7": p7["conclusion"], "P8": p8["conclusion"],
                                     "server_raw_results_received": False}
    statuses = {
        "P1": "默认每模态8候选/Top-2；cross仍读全槽；固定容量损害未证。",
        "P2": "历史短程模型读取高重叠且干预有计算差异；有害冗余/碎片化未证。",
        "P3": "历史同权重同输入eval波动已有本地原始记录；固定两处抽样可去除该协议波动，提分未证。",
        "P4": "无固定槽编号语义保证；真实组织/跨患者语义漂移及损害未证。",
        "P5": "无padding mask及补零分支存在；零行历史统计缺该轮原始记录，真实无关组织占槽/损害未证。",
        "P6": "缺显式空间/倍率接口；上游尺度内容及患者预测损害未证。",
        "P7": "最终融合主损失到选槽器直接梯度断开，辅助仍连通；患者损害及B/C收益未证。",
        "P8": "原版无后续跨模态OT，双阶段冲突不适用。"}
    task["issue_status_review_20261020"] = {"statuses": statuses, "P1_P4_numbers_recomputed_this_review": False,
                                           "new_experiments": 0, "check_counts_not_additive": True}
    report_paths = [DIRS[5] / "P5_交接复核与勘误.txt", DIRS[5] / "readonly_handoff_receipt.json"]
    report_paths += [path for i in (6, 7, 8) for path in (DIRS[i] / f"P{i}_只读核验报告.txt", DIRS[i] / "source_review.json")]
    relative_paths = [p.relative_to(ROOT).as_posix() for p in report_paths]
    task["expected_outputs"] += [p for p in relative_paths if p not in task["expected_outputs"]]
    eid = "EV-SPE-P5-P8-READONLY-20261020"
    task["evidence_refs"].append(eid)
    progress["evidence"].append({"id": eid, "type": "source_file_hash_and_handoff_review_no_new_patient_experiment",
        "details": "P5报告勘误及13个归档标记复核；P6接口缺口、P7直接主梯度断开且辅助连通、P8不适用。九份原版哈希匹配；历史缓存仍在，未重算P1-P4，服务器原始包未回传。",
        "paths": relative_paths, "local_file_sha256": {p: sha(ROOT / p) for p in relative_paths}})
    line = next(line for line in progress["lines"] if line["id"] == "L2")
    line["evidence_refs"].append(eid)
    line["bottleneck"] = "P1-P8证据边界已登记；P5报告/归档分类有勘误。历史缓存仍在，服务器原始包未回传；真实组织占槽、空间尺度损害和P7患者归因仍待证。"
    line["latest_readonly_review"] = copy.deepcopy(task["last_readonly_review"])
    progress["file_organization"]["slotspe_archive_marker_review"] = {
        "report": "docs/research/slotspe_p5_20261020/readonly_handoff_receipt.json", "markers_retained": 13,
        "P1_P4_raw_and_docs_are_duplicates": False, "archive_markers_authorize_deletion": False,
        "raw_checkpoints_cache_predictions_reports_retained": True}
    progress["current_priority_note"] = "SlotSPE优先：P1-P8源码/历史证据边界已登记；先验收SPE09原始回传，P5/P6患者问题及P7修正归因仍待证，P8原版不适用。"
    progress["change_log"].append({"date": "2026-10-20", "scope": "SlotSPE_P5_handoff_errata_P6_P7_P8_readonly_reports",
        "details": "仅源码/文件/既有JSON核对及落盘；保留P5旧报告和13个归档标记，纠正raw与报告重复的误判。无新增实验、训练、推送，其他线不变。",
        "backup": backup.relative_to(ROOT).as_posix()})
    human = human_before.replace(
        "当前最高优先级：SlotSPE八类问题逐项核查，P1～P4本地阶段已登记，下一项P5；SPE-09原始产物验收并行。",
        "当前最高优先级：SlotSPE P1～P8证据边界已登记；P5交接勘误、P6/P7/P8独立源码报告已补齐，患者问题与归因仍待证；优先验收SPE-09原始产物。", 1)
    anchor = "2．SlotSPE：先干净原版检查问题，A/B/C用于后续归因。\n"
    assert human.count(anchor) == 1
    addition = """2026年10月20日交接复核：P5此前只有一份报告，无该轮原始零行统计。历史缓存/原版权重/manifest完整SHA匹配；本轮未加载张量、未重算P1-P4数值。
P5报告已另附勘误：零行无mask不会自动零注意力；单患者无法算C-index；“能加载”不等于“已复算”，71/360统计范围须补原始记录。
13个ARCHIVED标记原样保留；P1-P4的.review原始NPZ与docs报告不是相同副本，不能据此删除。
P6缺显式空间/倍率接口，患者损害未证；P7直接融合主梯度到选槽器断开但辅助连通，修正患者收益未证；P8原版没有后续跨模态OT，登记不适用。
独立报告：docs/research/slotspe_p6_20261020/P6_只读核验报告.txt、slotspe_p7_20261020/P7_只读核验报告.txt、slotspe_p8_20261020/P8_只读核验报告.txt。
勘误与收据：docs/research/slotspe_p5_20261020/P5_交接复核与勘误.txt、readonly_handoff_receipt.json。
本轮新合成检查/前向/反向/训练均0，原版未修改、未push。两个约定服务器结果目录仍不存在，先验收原始回传，不能只凭完成转述。
以下为P1-P4历史测量阶段记录，患者、重复及检查次数不相加；本轮未重复执行这些测量。

"""
    human = human.replace(anchor, anchor + addition, 1)
    assert human != human_before
    assert all(t == next(x for x in before["tasks"] if x["id"] == t["id"])
               for t in progress["tasks"] if t["id"] != "SPE-08")
    assert all(line == next(x for x in before["lines"] if x["id"] == line["id"])
               for line in progress["lines"] if line["id"] != "L2")
    assert task["last_measurement"] == next(x for x in before["tasks"] if x["id"] == "SPE-08")["last_measurement"]
    assert human.split("1．V2：")[1].split(anchor)[0] == human_before.split("1．V2：")[1].split(anchor)[0]
    assert human.split("3．运输插拔：")[1] == human_before.split("3．运输插拔：")[1]
    ids = {item["id"] for item in progress["evidence"]}
    assert all(set(t.get("evidence_refs", [])) <= ids for t in progress["tasks"])
    assert all(sha(path) == digest for path, digest in original_sources.items())
    assert sha(original_p5) == inventory["original_p5_report"]["sha256"]
    assert all(sha(item["path"]) == item["sha256"] for item in markers)
    # The backup is already present; only these two established entrypoints are replaced.
    PROGRESS.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    HUMAN.write_text(human, encoding="utf-8")
    validation = {"recorded_at": STAMP, "all_other_tasks_unchanged": True, "non_slotspe_lines_unchanged": True,
                  "V2_and_transport_human_sections_unchanged": True, "last_actual_measurement_P4_preserved": True,
                  "original_source_hashes_match_history_and_remain_unchanged": True,
                  "P5_original_report_unchanged": True, "all_13_archive_markers_unchanged": True,
                  "historical_cache_checkpoint_manifest_hashes_match": True,
                  "evidence_refs_resolve": True, "new_model_execution": False,
                  "new_synthetic_checks": 0, "new_training_runs": 0, "new_patient_tensors_loaded": False,
                  "server_raw_received": False, "backup": str(backup),
                  "reports_and_evidence_sha256": {p: sha(ROOT / p) for p in relative_paths},
                  "progress_after_sha256": sha(PROGRESS), "human_after_sha256": sha(HUMAN)}
    write(DIRS[8] / "progress_update_validation.json", validation)
    print(json.dumps({"status": "recorded_readonly_review", "report_files": len(report_paths),
                      "backup": str(backup), "original_files_unchanged": True,
                      "new_patient_model_execution": False, "server_raw_received": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
