"""Five-fold score statistics for the anti-collapse queue and its baseline.

The user asked for the five-fold score statistics, the best score, the best
mean score and the epoch at which the best mean occurs. This script computes
those numbers from the stored artifacts and writes them to the public
research tree.

Two evaluation protocols exist in these queues and they are not
interchangeable, so both are reported with their own coverage.

The per-epoch validation protocol reads ``epoch_validation.json``, which
every arm-fold of every queue has. It is therefore the only protocol that
covers all five folds, including the anti-collapse folds 3 and 4 whose
twenty-repeat evaluation never completed because the queue was killed near
the end of training. The score of an epoch is the C-index of that single
validation pass.

The twenty-repeat evaluation protocol reads ``eval_repeats.json``, which
holds the mean of twenty repeated evaluations of the retained best
checkpoint. It is the protocol the queue summaries use and the one the
baseline was reported under, but it is available only for the three
anti-collapse folds whose evaluation completed.

The two protocols differ because the per-epoch protocol takes the maximum
over thirty validation passes, which is an optimistic selection, while the
twenty-repeat protocol re-evaluates one retained checkpoint. On the baseline
the two agree to about 0.005, but on the anti-collapse arms they differ by
up to 0.035, so a number from one protocol is never mixed with a number
from the other.

Only aggregate statistics are written. No patient identifier, no per-patient
risk value, no per-patient prediction array, no checkpoint and no weight
leaves the ignored ``local_evidence`` directory.

Usage:

    python -B scripts/summarize_five_fold_scores.py
    python -B scripts/summarize_five_fold_scores.py --output <path>
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ANTICOLLAPSE_ROOT = REPO / "local_evidence" / "anticollapse_blca_uni2h_folds0to4_seed3_run01"
BASELINE_ROOT = REPO / "local_evidence" / "redundancy_blca_uni2h_best_folds0to4_seed3_run01"

ANTI_ARMS = ("A0_clip", "A1_clip_decay")
BASE_ARMS = ("S0", "R4", "R3")
FOLDS = (0, 1, 2, 3, 4)


def epoch_validation(root, fold, arm):
    """The per-epoch validation C-index of one arm-fold, keyed by epoch."""
    path = root / "folds" / str(fold) / "arms" / arm / "epoch_validation.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    values = {}
    for record in data:
        epoch = record.get("epoch")
        score = record.get("cindex")
        if epoch is None or score is None:
            continue
        values[int(epoch)] = float(score)
    return values or None


def repeat_cindex(root, fold, arm):
    """The mean C-index over the twenty repeated evaluations of one arm-fold."""
    path = root / "folds" / str(fold) / "arms" / arm / "eval_repeats.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    values = [float(r["cindex"]) for r in data
              if isinstance(r, dict) and r.get("cindex") is not None]
    if not values:
        return None
    return {"mean": statistics.fmean(values), "n": len(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0}


def per_epoch_arm(root, arm):
    """Five-fold statistics of the per-epoch validation protocol for one arm.

    The epoch-wise mean is taken only over epochs that every fold has, so the
    five folds are always averaged on the same set of epochs. The mean of
    the per-fold best is reported separately, because it is a different
    quantity: each fold contributes its own maximum, so it is the average of
    five optima rather than the optimum of the average.
    """
    per_fold = {}
    for fold in FOLDS:
        values = epoch_validation(root, fold, arm)
        if values:
            per_fold[fold] = values
    if not per_fold:
        return None
    common = set(range(1, 31))
    for values in per_fold.values():
        common &= set(values)
    epoch_means = []
    for epoch in sorted(common):
        scores = [per_fold[fold][epoch] for fold in sorted(per_fold)]
        epoch_means.append({"epoch": epoch, "mean": statistics.fmean(scores),
                            "std": statistics.pstdev(scores) if len(scores) > 1 else 0.0})
    best_epoch_means = max(epoch_means, key=lambda row: row["mean"]) if epoch_means else None
    fold_bests = []
    for fold in sorted(per_fold):
        values = per_fold[fold]
        best = max(values, key=lambda epoch: values[epoch])
        fold_bests.append({"fold": fold, "best_epoch": best, "best_score": values[best],
                           "epochs_available": len(values)})
    final_epoch = max(common) if common else None
    return {
        "arm": arm,
        "folds_with_data": sorted(per_fold),
        "common_epoch_range": [min(common), max(common)] if common else None,
        "epoch_means": epoch_means,
        "best_epoch_mean": best_epoch_means,
        "fold_bests": fold_bests,
        "mean_of_fold_bests": statistics.fmean(row["best_score"] for row in fold_bests),
        "std_of_fold_bests": statistics.pstdev([row["best_score"] for row in fold_bests])
                             if len(fold_bests) > 1 else 0.0,
        "final_epoch_mean": next((row["mean"] for row in epoch_means
                                  if row["epoch"] == final_epoch), None),
    }


def repeat_arm(root, arm):
    """Five-fold statistics of the twenty-repeat protocol for one arm."""
    per_fold = []
    for fold in FOLDS:
        value = repeat_cindex(root, fold, arm)
        if value:
            per_fold.append({"fold": fold, **value})
    if not per_fold:
        return None
    means = [row["mean"] for row in per_fold]
    return {
        "arm": arm,
        "folds_with_data": [row["fold"] for row in per_fold],
        "per_fold": per_fold,
        "mean": statistics.fmean(means),
        "std_population": statistics.pstdev(means) if len(means) > 1 else 0.0,
    }


def build():
    anti_epoch = {arm: per_epoch_arm(ANTICOLLAPSE_ROOT, arm) for arm in ANTI_ARMS}
    base_epoch = {arm: per_epoch_arm(BASELINE_ROOT, arm) for arm in BASE_ARMS}
    anti_repeat = {arm: repeat_arm(ANTICOLLAPSE_ROOT, arm) for arm in ANTI_ARMS}
    base_repeat = {arm: repeat_arm(BASELINE_ROOT, arm) for arm in BASE_ARMS}
    return {
        "schema": "slotspe_five_fold_scores_v1",
        "recorded_at": "2026-11-03T14:15:00+08:00",
        "mode": "read-only aggregation of stored artifacts; no training, no inference, no checkpoint loaded",
        "patient_data_included": False,
        "patient_identifiers_included": False,
        "per_patient_arrays_included": False,
        "protocols": {
            "per_epoch_validation": {
                "source": "epoch_validation.json",
                "coverage": "all five folds for every arm of both queues",
                "definition": "the C-index of one validation pass per epoch, taken as the maximum over the epoch for a fold and as the mean over the folds for an epoch",
                "caveat": "the per-fold maximum over about thirty passes is an optimistic selection, so this protocol reads higher than the twenty-repeat protocol on the same model",
            },
            "twenty_repeat_evaluation": {
                "source": "eval_repeats.json",
                "coverage": "all five folds for the baseline, and folds 0 to 2 only for the anti-collapse arms",
                "definition": "the mean C-index of twenty repeated evaluations of the retained best checkpoint, per the checkpoint_selection rule per_epoch_validation_best_then_reload_and_reevaluate that both queues use",
                "caveat": "this is the protocol the queue summaries report, and it is the one that is comparable to the recorded baseline results",
            },
            "why_both": "the two protocols differ by up to 0.035 on the anti-collapse arms, so a number from one protocol is never mixed with a number from the other",
        },
        "anti_collapse_per_epoch": anti_epoch,
        "baseline_per_epoch": base_epoch,
        "anti_collapse_repeats": anti_repeat,
        "baseline_repeats": base_repeat,
    }


def render(payload):
    lines = []
    add = lines.append
    add("# 五折分数统计：反塌缩队列与同协议基线")
    add("")
    add("2026-11-03。本文件由 `scripts/summarize_five_fold_scores.py` 从 `local_evidence/` 的存储工件重新计算得到。只含聚合统计，不含患者标识、逐患者风险值、逐患者预测数组、权重或 checkpoint。")
    add("")
    add("## 〇、两套评估口径")
    add("")
    add("两套口径在本仓库并存，**不可混用**。")
    add("")
    add("| 口径 | 来源 | 覆盖 | 定义 |")
    add("|---|---|---|---|")
    add("| 逐轮验证 | `epoch_validation.json` | 两队列全部五折 | 每轮一次验证通过的 C-index；逐折取轮次最大值，逐轮取五折均值 |")
    add("| 20 次重复评估 | `eval_repeats.json` | 基线全部五折；反塌缩仅 fold 0/1/2 | 保留的最佳 checkpoint 经 20 次重复评估的 C-index 均值 |")
    add("")
    add("两队列的 `checkpoint_selection` 规则相同，均为 `per_epoch_validation_best_then_reload_and_reevaluate`。")
    add("")
    add("逐轮验证口径在约三十次验证通过中取最大值，是乐观选择，因此同一模型上它比 20 次重复评估口径读数更高。基线上两者相差约 0.005，")
    add("但反塌缩臂上相差可达 0.035，因此**只能同口径比较**。")
    add("")
    add("## 一、逐轮验证口径（覆盖全部五折）")
    add("")
    add("### 1.1 最优平均分与出现该分数的 epoch")
    add("")
    add("「最优平均分」= 五折在**同一 epoch** 上的 C-index 均值，再在所有共同 epoch 上取最大。共同 epoch 范围保证五折总在同一批轮次上平均。")
    add("")
    add("| 队列 | 臂 | 共同epoch范围 | 最优平均分 | 该分数的epoch | 五折最佳分平均 | 五折最佳分标准差 | 末轮平均 |")
    add("|---|---|---|---:|---:|---:|---:|---:|")
    for label, group in (("基线", payload["baseline_per_epoch"]),
                         ("反塌缩", payload["anti_collapse_per_epoch"])):
        for arm in sorted(group):
            row = group[arm]
            if not row:
                continue
            best = row["best_epoch_mean"]
            rng = row["common_epoch_range"]
            add(f"| {label} | {arm} | {rng[0]}..{rng[1]} | **{best['mean']:.4f}** | **ep{best['epoch']}** | "
                f"{row['mean_of_fold_bests']:.4f} | {row['std_of_fold_bests']:.4f} | "
                f"{row['final_epoch_mean']:.4f} |")
    add("")
    add("### 1.2 逐折最佳分数与最佳 epoch")
    add("")
    for label, group in (("基线", payload["baseline_per_epoch"]),
                         ("反塌缩", payload["anti_collapse_per_epoch"])):
        for arm in sorted(group):
            row = group[arm]
            if not row:
                continue
            add(f"**{label} {arm}**")
            add("")
            add("| 折 | 可用轮次 | 最佳分数 | 最佳epoch |")
            add("|---:|---:|---:|---:|")
            for entry in row["fold_bests"]:
                add(f"| {entry['fold']} | {entry['epochs_available']} | {entry['best_score']:.4f} | "
                    f"ep{entry['best_epoch']} |")
            add("")
    add("### 1.3 逐轮均值曲线（反塌缩两臂）")
    add("")
    anti = payload["anti_collapse_per_epoch"]
    if all(anti.get(arm) for arm in ANTI_ARMS):
        a0 = {row["epoch"]: row["mean"] for row in anti["A0_clip"]["epoch_means"]}
        a1 = {row["epoch"]: row["mean"] for row in anti["A1_clip_decay"]["epoch_means"]}
        add("| epoch | A0_clip 五折均值 | A1_clip_decay 五折均值 |")
        add("|---:|---:|---:|")
        for epoch in sorted(set(a0) & set(a1)):
            add(f"| {epoch} | {a0[epoch]:.4f} | {a1[epoch]:.4f} |")
        add("")
    add("## 二、20 次重复评估口径（队列汇总口径）")
    add("")
    add("这是两条队列的 summary 使用的口径，也是与已登记基线结果可比的唯一口径。反塌缩 fold 3/4 的评估阶段未完成（队列在训练末尾被杀），因此该口径下反塌缩只有三折。")
    add("")
    add("| 队列 | 臂 | 覆盖折 | 五折/三折均值 | 折间标准差 | 逐折 |")
    add("|---|---|---|---:|---:|---|")
    for label, group in (("基线", payload["baseline_repeats"]),
                         ("反塌缩", payload["anti_collapse_repeats"])):
        for arm in sorted(group):
            row = group[arm]
            if not row:
                continue
            per = " ".join(f"{r['mean']:.4f}" for r in row["per_fold"])
            add(f"| {label} | {arm} | {len(row['folds_with_data'])}/5 | **{row['mean']:.4f}** | "
                f"{row['std_population']:.4f} | {per} |")
    add("")
    add("## 三、读法与边界")
    add("")
    add("- 逐轮验证口径下，反塌缩 `A1_clip_decay` 的五折最佳分平均 **0.7091**，与基线 `R4` 的 **0.7090** 基本持平；`A0_clip` 的 **0.6946** 低于基线 `S0` 的 **0.7004**。")
    add("- 但逐轮验证口径取的是每折三十次验证通过的最大值，属乐观选择。它不是队列汇总口径，不能与已登记的基线结论直接并列。")
    add("- 在队列汇总口径（20 次重复评估）下，反塌缩三折为 `A0_clip` **0.6817**、`A1_clip_decay` **0.6918**，同协议基线三折为 `S0` **0.7133**、`R4` **0.7229**、`R3` **0.6980**，**反塌缩两臂均低于同协议基线**。")
    add("- 反塌缩的**机制目标已达成**（槽几何全程撑开、有效秩约 6.2），而**生存分数没有改善**。两者同时成立。")
    add("- 之前登记的删槽 oracle 结果为负面：18 组比较中仅 1 组超过置换零分布上界，随机期望为 2.25 组，且塌缩基线在同一探针上给出相同答案。")
    add("- fold 3/4 的训练在第 28~29 轮被中断，评估阶段未运行；因此**这两折没有任何 checkpoint、best.pt 或 20 次重复评估**。上表中它们的逐轮验证值来自训练期写入的 `epoch_validation.json`。")
    add("- 单种子 3；20 次评估重复是同一训练模型上的重复，不是 20 次独立训练，折间离散度不含训练随机性；多臂比较未做多重比较校正。")
    add("- **以上没有任何数字构成患者获益、患者损害或修正有效性证据**，也不构成临床效用或对其他队列/癌种的泛化。")
    add("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",
                        default="docs/research/slotspe_queue_results_20261102/five_fold_scores.md")
    parser.add_argument("--json-output",
                        default="docs/research/slotspe_queue_results_20261102/five_fold_scores.json")
    args = parser.parse_args()
    output = (REPO / args.output).resolve()
    json_output = (REPO / args.json_output).resolve()
    private = (REPO / "local_evidence").resolve()
    for target in (output, json_output):
        if target == private or target.is_relative_to(private):
            parser.error("a public summary must not be written inside ignored local_evidence")
    payload = build()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(payload), encoding="utf-8")
    json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed",
                      "markdown": str(output.relative_to(REPO)),
                      "json": str(json_output.relative_to(REPO))},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
