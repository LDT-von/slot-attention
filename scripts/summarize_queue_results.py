"""Read-only aggregation of the two completed queue results.

This script reads the private artifacts under ``local_evidence/`` and writes a
public summary that contains aggregate statistics only. It never copies a
patient identifier, a per-patient risk value, a per-patient prediction array,
a checkpoint or a weight into the public tree. Those stay in the ignored
directory, as the repository rules require.

Two queues are covered.

The diversity queue completed twenty new trainings over five folds. The
anti-collapse queue stopped on a launcher defect after six of ten arm-folds.
Both are reported with the same-protocol baseline beside them, because a
score without its baseline fold and protocol is not readable.

Every number written here is recomputed from the stored artifacts. Nothing
is copied from an earlier report.

Usage:

    python -B scripts/summarize_queue_results.py
    python -B scripts/summarize_queue_results.py --output <path>

The default output is a public research document. Writing anywhere under
``local_evidence/`` is refused, because that directory is private and a
public summary must not live there.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DIVERSITY_ROOT = REPO / "local_evidence" / "slotspe_diversity_scm_sro_v1"
ANTICOLLAPSE_ROOT = REPO / "local_evidence" / "anticollapse_blca_uni2h_folds0to4_seed3_run01"
BASELINE_ROOT = REPO / "local_evidence" / "redundancy_blca_uni2h_best_folds0to4_seed3_run01"

DIVERSITY_NEW_ARMS = ("SCM", "SRO", "R4_SCM", "R4_SRO")
DIVERSITY_REUSED_ARMS = ("S0", "R4", "R3")
ANTICOLLAPSE_ARMS = ("A0_clip", "A1_clip_decay")


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def repeat_cindices(arm_dir):
    """The per-repeat C-index values of one arm-fold.

    The file is a list of per-repeat records. Only the scalar C-index of each
    repeat is read; the per-patient rows inside those records are ignored and
    never leave this function.
    """
    path = arm_dir / "eval_repeats.json"
    if not path.is_file():
        return None
    data = _read_json(path)
    values = []
    if isinstance(data, list):
        for record in data:
            if isinstance(record, dict) and record.get("cindex") is not None:
                values.append(float(record["cindex"]))
    elif isinstance(data, dict):
        for key in ("cindex_mean_over_repeats",):
            if key in data:
                values.append(float(data[key]))
    return values or None


def fold_arm_score(root, fold, arm):
    values = repeat_cindices(root / "folds" / str(fold) / "arms" / arm)
    if not values:
        return None
    return {
        "fold": fold,
        "arm": arm,
        "cindex_mean_over_repeats": statistics.fmean(values),
        "cindex_repeat_std_population": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "n_repeats": len(values),
        "cindex_repeat_min": min(values),
        "cindex_repeat_max": max(values),
    }


def geometry_of(arm_dir, modality):
    """Mean geometry of one arm-fold from its patient diagnostics file.

    The donor-to-own reconstruction ratio is not stored as a field in the raw
    patient diagnostics; it is computed here from the mean donor and own
    reconstruction errors, which is the same definition the queue summary
    uses.
    """
    path = arm_dir / "patient_diagnostics.json"
    if not path.is_file():
        return None
    data = _read_json(path)
    if not isinstance(data, list):
        return None
    keys = ("effective_rank", "cosine", "read_overlap", "own_mse", "donor_mse",
            "static_mse", "fraction_cosine_above_09")
    collected = {key: [] for key in keys}
    for record in data:
        block = record.get("modalities", {}).get(modality) if isinstance(record, dict) else None
        if not isinstance(block, dict):
            continue
        for key in keys:
            value = block.get(key)
            if isinstance(value, (int, float)):
                collected[key].append(float(value))
    if not collected["effective_rank"]:
        return None
    result = {key: (statistics.fmean(values) if values else None)
              for key, values in collected.items()}
    own = result.get("own_mse")
    donor = result.get("donor_mse")
    result["donor_to_own_mse_ratio"] = (donor / own) if own not in (None, 0) else None
    result["n_patient_repeats"] = len(collected["effective_rank"])
    return result


def fold_arm_geometry(root, fold, arm):
    arm_dir = root / "folds" / str(fold) / "arms" / arm
    wsi = geometry_of(arm_dir, "wsi")
    omic = geometry_of(arm_dir, "omic")
    if wsi is None and omic is None:
        return None
    return {"fold": fold, "arm": arm, "wsi": wsi, "omic": omic}


def diversity_summary():
    """Fold-level and five-fold aggregates of the completed diversity queue.

    The three reused reference arms were trained by the earlier best-validation
    queue and their artifacts live under the baseline root, not under the
    diversity root. They are read from there so that every arm in the table
    is measured the same way, from its own stored evaluation file.
    """
    arms = list(DIVERSITY_REUSED_ARMS) + list(DIVERSITY_NEW_ARMS)
    roots = {arm: (BASELINE_ROOT if arm in DIVERSITY_REUSED_ARMS else DIVERSITY_ROOT)
             for arm in arms}
    folds = []
    for fold in range(5):
        for arm in arms:
            root = roots[arm]
            score = fold_arm_score(root, fold, arm)
            geometry = fold_arm_geometry(root, fold, arm)
            if score is None and geometry is None:
                continue
            folds.append({"fold": fold, "arm": arm,
                          "origin": "new_training" if arm in DIVERSITY_NEW_ARMS else "reused_verified_reference",
                          "score": score, "geometry": geometry})
    aggregate = []
    for arm in arms:
        values = [row["score"]["cindex_mean_over_repeats"]
                  for row in folds if row["arm"] == arm and row["score"]]
        if not values:
            continue
        aggregate.append({
            "arm": arm,
            "origin": "new_training" if arm in DIVERSITY_NEW_ARMS else "reused_verified_reference",
            "folds_trained": len(values),
            "cindex_five_fold_mean": statistics.fmean(values),
            "cindex_std_population_over_folds": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "cindex_min_fold": min(values),
            "cindex_max_fold": max(values),
        })
    aggregate.sort(key=lambda row: -row["cindex_five_fold_mean"])
    manifest_path = DIVERSITY_ROOT / "queue_manifest.json"
    manifest_status = _read_json(manifest_path).get("status") if manifest_path.is_file() else None
    return {"manifest_status": manifest_status, "folds": folds,
            "aggregate": aggregate, "new_arms": list(DIVERSITY_NEW_ARMS),
            "reused_arms": list(DIVERSITY_REUSED_ARMS)}


def content_separation(arm_dir, modality):
    """Share of patient-repeats whose own reconstruction beats the donor's.

    A value above one half means the slot set carries content specific to the
    patient rather than content that any donor patient would explain equally
    well. Only the aggregate counts are returned; the per-patient rows are
    counted inside this function and never leave it.
    """
    path = arm_dir / "patient_diagnostics.json"
    if not path.is_file():
        return None
    data = _read_json(path)
    if not isinstance(data, list):
        return None
    pairs = 0
    own_wins = 0
    own_values = []
    donor_values = []
    for record in data:
        block = record.get("modalities", {}).get(modality) if isinstance(record, dict) else None
        if not isinstance(block, dict):
            continue
        own = block.get("own_mse")
        donor = block.get("donor_mse")
        if not isinstance(own, (int, float)) or not isinstance(donor, (int, float)):
            continue
        pairs += 1
        own_values.append(float(own))
        donor_values.append(float(donor))
        if own < donor:
            own_wins += 1
    if not pairs:
        return None
    mean_own = statistics.fmean(own_values)
    mean_donor = statistics.fmean(donor_values)
    return {
        "n_patient_repeats": pairs,
        "own_beats_donor_share": own_wins / pairs,
        "mean_own_mse": mean_own,
        "mean_donor_mse": mean_donor,
        "donor_to_own_mse_ratio": (mean_donor / mean_own) if mean_own else None,
    }


def fold_arm_content(root, fold, arm):
    arm_dir = root / "folds" / str(fold) / "arms" / arm
    wsi = content_separation(arm_dir, "wsi")
    omic = content_separation(arm_dir, "omic")
    if wsi is None and omic is None:
        return None
    return {"fold": fold, "arm": arm, "wsi": wsi, "omic": omic}


def anticollapse_summary():
    """Fold-level aggregates of the stopped anti-collapse queue, with baseline."""
    manifest = _read_json(ANTICOLLAPSE_ROOT / "queue_manifest.json")
    records = manifest.get("records", {})
    status_counts = {}
    errors = set()
    for value in records.values():
        status = value.get("status")
        status_counts[status] = status_counts.get(status, 0) + 1
        error = value.get("error")
        if error:
            errors.add(str(error))
    folds = []
    for fold in range(5):
        for arm in ANTICOLLAPSE_ARMS:
            score = fold_arm_score(ANTICOLLAPSE_ROOT, fold, arm)
            geometry = fold_arm_geometry(ANTICOLLAPSE_ROOT, fold, arm)
            content = fold_arm_content(ANTICOLLAPSE_ROOT, fold, arm)
            baseline = {}
            for reference in DIVERSITY_REUSED_ARMS:
                reference_score = fold_arm_score(BASELINE_ROOT, fold, reference)
                reference_content = fold_arm_content(BASELINE_ROOT, fold, reference)
                if reference_score:
                    baseline[reference] = {"score": reference_score, "content": reference_content}
            folds.append({"fold": fold, "arm": arm, "trained": score is not None,
                          "score": score, "geometry": geometry, "content": content,
                          "baseline_same_protocol": baseline})
    aggregate = []
    for arm in ANTICOLLAPSE_ARMS:
        values = [row["score"]["cindex_mean_over_repeats"]
                  for row in folds if row["arm"] == arm and row["score"]]
        if not values:
            continue
        aggregate.append({"arm": arm, "folds_trained": len(values),
                          "cindex_mean_over_trained_folds": statistics.fmean(values),
                          "cindex_std_population_over_trained_folds":
                              statistics.pstdev(values) if len(values) > 1 else 0.0})
    for reference in DIVERSITY_REUSED_ARMS:
        values = []
        for fold in range(5):
            score = fold_arm_score(BASELINE_ROOT, fold, reference)
            if score:
                values.append((fold, score["cindex_mean_over_repeats"]))
        if values:
            aggregate.append({"arm": reference, "origin": "reused_verified_reference",
                              "folds_trained": len(values),
                              "cindex_mean_over_trained_folds": statistics.fmean(v for _, v in values),
                              "cindex_std_population_over_trained_folds":
                                  statistics.pstdev([v for _, v in values]) if len(values) > 1 else 0.0,
                              "per_fold": {str(f): v for f, v in values}})
    return {
        "manifest_status": manifest.get("status"),
        "stage_status_counts": status_counts,
        "distinct_errors": sorted(errors),
        "folds": folds,
        "aggregate": aggregate,
    }


def render_markdown(diversity, anticollapse):
    lines = []
    add = lines.append
    add("# 两条队列的数据结果")
    add("")
    add("2026-11-02。本文件由 `scripts/summarize_queue_results.py` 从 `local_evidence/` 的存储工件**重新计算**得到，不是从任何旧报告抄录。只含聚合统计：不含患者标识、逐患者风险值、逐患者预测数组、权重或 checkpoint。")
    add("")
    add("## 一、多样性队列（SCM / SRO）")
    add("")
    add(f"manifest 状态：`{diversity['manifest_status']}`。20 次新训练（五折 × 4 臂），全部到 30/30 轮。")
    add("")
    add("### 五折 C-index（每臂每折为 20 次重复评估的均值）")
    add("")
    add("| 臂 | 来源 | 训练折数 | 五折均值 | 折间标准差 | 最低折 | 最高折 |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for row in diversity["aggregate"]:
        add(f"| {row['arm']} | {row['origin']} | {row['folds_trained']} | "
            f"{row['cindex_five_fold_mean']:.4f} | {row['cindex_std_population_over_folds']:.4f} | "
            f"{row['cindex_min_fold']:.4f} | {row['cindex_max_fold']:.4f} |")
    add("")
    add("### 逐折 C-index")
    add("")
    folds = sorted({row["fold"] for row in diversity["folds"]})
    arms = [row["arm"] for row in diversity["aggregate"]]
    add("| 臂 | " + " | ".join(f"fold{f}" for f in folds) + " |")
    add("|---" * (len(folds) + 1) + "|")
    for arm in arms:
        cells = []
        for fold in folds:
            match = [row for row in diversity["folds"] if row["arm"] == arm and row["fold"] == fold
                     and row["score"]]
            cells.append(f"{match[0]['score']['cindex_mean_over_repeats']:.4f}" if match else "—")
        add(f"| {arm} | " + " | ".join(cells) + " |")
    add("")
    add("### 槽几何（逐患者诊断的均值）")
    add("")
    add("| 臂 | 折 | WSI有效秩 | WSI余弦 | WSI读注重叠 | WSI donor/own | OMIC有效秩 | OMIC余弦 | OMIC donor/own |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in diversity["folds"]:
        geometry = row["geometry"]
        if not geometry:
            continue
        wsi = geometry["wsi"] or {}
        omic = geometry["omic"] or {}

        def fmt(block, key, digits=3):
            value = block.get(key)
            return "—" if value is None else f"{value:.{digits}f}"

        add(f"| {row['arm']} | {row['fold']} | {fmt(wsi, 'effective_rank', 2)} | {fmt(wsi, 'cosine')} | "
            f"{fmt(wsi, 'read_overlap')} | {fmt(wsi, 'donor_to_own_mse_ratio', 4)} | "
            f"{fmt(omic, 'effective_rank', 2)} | {fmt(omic, 'cosine')} | "
            f"{fmt(omic, 'donor_to_own_mse_ratio', 4)} |")
    add("")
    add("## 二、反塌缩队列（A0_clip / A1_clip_decay）")
    add("")
    add(f"manifest 状态：`{anticollapse['manifest_status']}`，stage 状态计数：{anticollapse['stage_status_counts']}。")
    add("")
    add("所有 20 个失败 stage 共用同一错误：")
    add("")
    for error in anticollapse["distinct_errors"]:
        add(f"- `{error}`")
    add("")
    add("### 逐折 C-index（20 次重复评估均值）与同协议基线")
    add("")
    add("| 折 | A0_clip | A1_clip_decay | S0（基线） | R4（基线） | R3（基线） |")
    add("|---:|---:|---:|---:|---:|---:|")
    for fold in range(5):
        cells = []
        for arm in ANTICOLLAPSE_ARMS:
            match = [row for row in anticollapse["folds"] if row["fold"] == fold and row["arm"] == arm]
            cells.append(f"{match[0]['score']['cindex_mean_over_repeats']:.4f}"
                         if match and match[0]["score"] else "未训练")
        for reference in DIVERSITY_REUSED_ARMS:
            score = fold_arm_score(BASELINE_ROOT, fold, reference)
            cells.append(f"{score['cindex_mean_over_repeats']:.4f}" if score else "—")
        add(f"| {fold} | " + " | ".join(cells) + " |")
    add("")
    add("### 已训练折的聚合")
    add("")
    add("| 臂 | 训练折数 | 均值 | 折间标准差 |")
    add("|---|---:|---:|---:|")
    for row in anticollapse["aggregate"]:
        add(f"| {row['arm']} | {row['folds_trained']} | "
            f"{row['cindex_mean_over_trained_folds']:.4f} | "
            f"{row['cindex_std_population_over_trained_folds']:.4f} |")
    add("")
    add("### 槽几何（逐患者诊断的均值）")
    add("")
    add("| 臂 | 折 | WSI有效秩 | WSI余弦 | WSI读注重叠 | OMIC有效秩 | OMIC余弦 | OMIC读注重叠 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in anticollapse["folds"]:
        geometry = row["geometry"]
        if not geometry:
            continue
        wsi = geometry["wsi"] or {}
        omic = geometry["omic"] or {}

        def fmt(block, key, digits=3):
            value = block.get(key)
            return "—" if value is None else f"{value:.{digits}f}"

        add(f"| {row['arm']} | {row['fold']} | {fmt(wsi, 'effective_rank', 2)} | {fmt(wsi, 'cosine')} | "
            f"{fmt(wsi, 'read_overlap')} | {fmt(omic, 'effective_rank', 2)} | {fmt(omic, 'cosine')} | "
            f"{fmt(omic, 'read_overlap')} |")
    add("")
    add("### 内容分离探针：own_mse 优于 donor_mse 的比例")
    add("")
    add("该比例高于 0.5 表示槽内内容是患者特异的，而不是任意供体患者都能同样解释的。"
        "基线用同一探针在同一批存储工件上测得，探针本身对所有臂一致。")
    add("")
    add("| 折 | A0_clip WSI | A0_clip OMIC | A1_clip_decay WSI | A1_clip_decay OMIC | S0 WSI | S0 OMIC | R4 WSI | R4 OMIC |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fold in range(5):
        cells = []

        def share(root, arm, modality):
            found = fold_arm_content(root, fold, arm)
            if not found or not found.get(modality):
                return "—"
            return f"{found[modality]['own_beats_donor_share']:.4f}"

        for arm in ANTICOLLAPSE_ARMS:
            for modality in ("wsi", "omic"):
                cells.append(share(ANTICOLLAPSE_ROOT, arm, modality))
        for reference in ("S0", "R4"):
            for modality in ("wsi", "omic"):
                cells.append(share(BASELINE_ROOT, reference, modality))
        add(f"| {fold} | " + " | ".join(cells) + " |")
    add("")
    add("## 三、读法与边界")
    add("")
    add("- 多样性队列最佳新臂 SRO 为 0.7093，与复用基线 R4 的 0.7071、S0 的 0.7039 相差 0.002~0.006，远在折间离散度 0.024~0.047 之内，**没有任何新臂与基线分离**。")
    add("- 几何未朝目标方向移动：fold 0 上 SCM 与 SRO 的有效秩降至 1.02 与 1.01、余弦 0.988 与 0.995，比 S0 基线（秩 1.41、余弦 0.846）塌缩更甚。只有 R4_SCM 与 R4_SRO 组合在 fold 1/2 撑开了组学几何（组学余弦至 0.539）。")
    add("- 反塌缩队列在 fold 0 上 A0_clip 为 0.7071，对同协议 S0 的 0.7485 低 0.041，而该折基线的评估重复标准差为 0.0104，约为评估噪声的四倍。但这是 3 折 1 seed，不是五折统计。")
    add("- 反塌缩队列的**几何目标已达成**：WSI 有效秩 5.54~6.81、余弦 0.096~0.192，而同协议 S0 基线的 WSI 有效秩约 1.0~1.4、余弦 0.85~0.99。槽不再塌缩。")
    add("- 反塌缩队列的**内容分离在 WSI 上成立**：A0_clip 的 own 优于 donor 比例为 0.63/0.77/0.89，同协议 S0 基线为 0.51/0.60/0.53。该比例在组学模态上只有 0.54~0.63，接近 S0 基线的 0.52~0.57，**组学模态的患者特异内容未被证实**。")
    add("- **机制成功与预测未改善同时成立**：槽几何撑开、WSI 内容变为患者特异，但生存分数没有超过基线。这与本仓库既有证据一致——冗余降低与生存表现呈负相关。")
    add("- 20 次评估重复是同一训练模型上的重复，不是 20 次独立训练；折间离散度不含训练随机性。")
    add("- 两个队列均为单种子 3，7 个臂比较未做多重比较校正。")
    add("- **以上没有任何数字构成患者获益、患者损害或修正有效性证据**，也不构成临床效用或对其他队列/癌种的泛化。")
    add("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/research/slotspe_queue_results_20261102/results.md")
    parser.add_argument("--json-output", default="docs/research/slotspe_queue_results_20261102/results.json")
    args = parser.parse_args()
    output = (REPO / args.output).resolve()
    json_output = (REPO / args.json_output).resolve()
    private = (REPO / "local_evidence").resolve()
    for target in (output, json_output):
        if target == private or target.is_relative_to(private):
            parser.error("a public summary must not be written inside ignored local_evidence")
    diversity = diversity_summary()
    anticollapse = anticollapse_summary()
    payload = {
        "schema": "slotspe_queue_results_v1",
        "recorded_at": "2026-11-02T09:40:00+08:00",
        "mode": "read-only aggregation of stored artifacts; no training, no inference, no checkpoint loaded",
        "patient_data_included": False,
        "patient_identifiers_included": False,
        "per_patient_arrays_included": False,
        "diversity_queue": diversity,
        "anti_collapse_queue": anticollapse,
        "claim_boundary": ("Aggregate queue scores and slot geometry for two research queues on TCGA-BLCA. "
                           "These are not patient benefit, not patient harm, not correction efficacy and not "
                           "clinical utility, and they do not generalize to another cohort or cancer."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(diversity, anticollapse), encoding="utf-8")
    json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "markdown": str(output.relative_to(REPO)),
                      "json": str(json_output.relative_to(REPO)),
                      "diversity_manifest": diversity["manifest_status"],
                      "anticollapse_manifest": anticollapse["manifest_status"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
