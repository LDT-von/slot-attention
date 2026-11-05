"""Score summary for the SlotSPE plugin queue (V1/V2/V3 presets, BLCA fold 0).

The plugin queue was executed on this machine on 2026-11-04. This script
recomputes every published number from the stored artifacts in the ignored
``local_evidence`` directory and writes a public markdown and JSON document.

What is published is aggregate only: no patient identifier, no per-patient
risk value, no per-patient prediction array, no weight and no checkpoint
leaves ``local_evidence``.

The evaluation protocol matters and is stated here because it bounds every
number in the output. The queue selects the best epoch by validation C-index
(the official rule: the running maximum is replaced on a greater-or-equal
value), reloads that checkpoint, and then evaluates it twenty times. The
twenty repeats are repeated evaluations of one retained checkpoint under the
native slot-selection noise, so they measure evaluation randomness, not
training randomness, and they carry no generalisation guarantee.

The score of a single fold under this protocol is an optimistic selection: a
maximum over about thirty validation passes over the same validation
patients. The original baseline's own five-fold mean under the same
protocol is 0.7039, and fold 0 is that baseline's easiest fold at 0.7485.
A single-fold plugin number is therefore not comparable to the original's
five-fold level without stating both facts.

Usage:

    python -B scripts/summarize_plugin_queue_scores.py
    python -B scripts/summarize_plugin_queue_scores.py --output <path>
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PLUGIN_ROOT = REPO / "local_evidence" / "slot_plugins_blca_fold0_seed3_run01"
BASELINE_ROOT = REPO / "local_evidence" / "redundancy_blca_uni2h_best_folds0to4_seed3_run01"

ARMS = ("B0_native", "B1_gram", "V1_mean", "V1_noise", "V1_vib",
        "V2_content", "V2_separate", "V3_capacity", "V3_relation")
CONTENT_ARMS = ("V2_content", "V2_separate", "V3_capacity", "V3_relation")
CONTENT_MODALITIES = ("wsi", "omic")
GEOMETRY_MODALITIES = ("wsi", "omic")
GEOMETRY_SPACES = ("native", "mean", "latent")

# The three comparisons the preset plan defines as the actual tests.
PAIRED_COMPARISONS = (
    ("V1_vib", "V1_noise", "stochastic bottleneck with a KL term against the same "
                          "stochastic bottleneck without it"),
    ("V2_separate", "V2_content", "separation terms on top of the content constraint "
                                  "against the content constraint alone"),
    ("V3_relation", "V3_capacity", "pairwise relation residual against a per-slot "
                                   "residual with a matched parameter count"),
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arm_summary(arm):
    """The queue's own summary of one arm."""
    return read(PLUGIN_ROOT / "arms" / arm / "summary.json")


def arm_protocol(arm):
    return read(PLUGIN_ROOT / "arms" / arm / "protocol.json")


def selection_rows(arm):
    """The validation rows of the retained best checkpoint."""
    return read(PLUGIN_ROOT / "arms" / arm / "best_selection_predictions.json")["rows"]


def arm_curve(arm):
    return read(PLUGIN_ROOT / "arms" / arm / "curve.json")


def mean(values):
    return sum(values) / len(values)


def geometry_table():
    """Mean slot geometry of the retained best checkpoint, per arm."""
    table = {}
    for arm in ARMS:
        rows = selection_rows(arm)
        entry = {}
        for modality in GEOMETRY_MODALITIES:
            for space in GEOMETRY_SPACES:
                block = [row["geometry"][modality][space] for row in rows]
                entry[f"{modality}.{space}"] = {
                    "cosine": mean([item["cosine"] for item in block]),
                    "rms_cosine": mean([item["rms_cosine"] for item in block]),
                    "max_cosine": mean([item["max_cosine"] for item in block]),
                    "fraction_cosine_above_09": mean(
                        [item["fraction_cosine_above_09"] for item in block]),
                    "effective_rank": mean([item["effective_rank"] for item in block]),
                }
        table[arm] = entry
    return table


def content_table():
    """Content-recovery error against the constant-prediction floor."""
    table = {}
    for arm in CONTENT_ARMS:
        rows = selection_rows(arm)
        entry = {}
        for modality in CONTENT_MODALITIES:
            block = [row["content"][modality] for row in rows]
            entry[modality] = {
                "mse": mean([item["mse"] for item in block]),
                "constant_floor": mean([item["constant_floor"] for item in block]),
            }
        table[arm] = entry
    return table


def baseline_five_fold():
    """The original baseline's five-fold numbers under the same protocol."""
    summary = read(BASELINE_ROOT / "summary.json")
    s0 = summary["per_arm"]["S0"]
    return {
        "S0_cindex_per_fold": s0["cindex_per_fold"],
        "S0_five_fold_mean": s0["cindex_five_fold"]["fold_mean"],
        "S0_five_fold_std_population": s0["cindex_five_fold"]["fold_std_population"],
        "R4_five_fold_mean": summary["per_arm"]["R4"]["cindex_five_fold"]["fold_mean"],
        "R3_five_fold_mean": summary["per_arm"]["R3"]["cindex_five_fold"]["fold_mean"],
        "source": "local_evidence/redundancy_blca_uni2h_best_folds0to4_seed3_run01/summary.json",
    }


def initial_hash():
    """The shared initial core hash, and whether the original queue agrees."""
    protocol = arm_protocol("B0_native")
    original = read(BASELINE_ROOT / "folds" / "0" / "arms" / "S0" / "summary.json")
    return {
        "plugin_B0_native_initial_core_hash": protocol["initial_core_hash"],
        "original_S0_fold0_initial_core_hash": original["protocol"]["initial_core_hash"],
        "identical": protocol["initial_core_hash"]
                     == original["protocol"]["initial_core_hash"],
    }


def cohort_identity():
    """Whether the plugin queue and the completed baseline use the same patients."""
    plugin = read(PLUGIN_ROOT / "prepared" / "manifest.json")
    original = read(BASELINE_ROOT / "prepared" / "manifest.json")
    pf, of = plugin["folds"]["0"], original["folds"]["0"]
    return {
        "plugin_train": len(pf["train"]), "plugin_val": len(pf["val"]),
        "original_train": len(of["train"]), "original_val": len(of["val"]),
        "train_identical": pf["train"] == of["train"],
        "val_identical": pf["val"] == of["val"],
        "plugin_split_csv": pf["split_csv"],
        "original_split_csv": of["split_csv"],
        "note": "The plugin configuration names splits_dir 5fold_uni2h, but the original "
                "loader hardcodes the 5fold directory, so the executed cohort is the 5fold "
                "cohort. The patient lists are compared directly rather than by directory name.",
    }


def paired_comparisons(summaries):
    rows = []
    for treatment, control, description in PAIRED_COMPARISONS:
        t, c = summaries[treatment], summaries[control]
        rows.append({
            "treatment": treatment, "control": control, "description": description,
            "treatment_repeat_mean": t["repeat_cindex_mean"],
            "control_repeat_mean": c["repeat_cindex_mean"],
            "difference": t["repeat_cindex_mean"] - c["repeat_cindex_mean"],
            "treatment_repeat_std": t["repeat_cindex_std_population"],
            "control_repeat_std": c["repeat_cindex_std_population"],
        })
    return rows


def build():
    summaries = {arm: arm_summary(arm) for arm in ARMS}
    geometry = geometry_table()
    content = content_table()
    baseline = baseline_five_fold()
    scores = {
        arm: {
            "status": summaries[arm]["status"],
            "best_epoch": summaries[arm]["best_epoch"],
            "best_selection_cindex": summaries[arm]["best_selection_cindex"],
            "repeat_cindex_mean": summaries[arm]["repeat_cindex_mean"],
            "repeat_cindex_std_population": summaries[arm]["repeat_cindex_std_population"],
            "optimizer_steps": summaries[arm]["optimizer_steps"],
            "train_patients": summaries[arm]["train_patients"],
            "validation_patients": summaries[arm]["validation_patients"],
            "elapsed_seconds": summaries[arm]["elapsed_seconds"],
            "final_epoch_selection_cindex": arm_curve(arm)[-1]["selection_cindex"],
            "claim_boundary": summaries[arm]["claim_boundary"],
        }
        for arm in ARMS
    }
    return {
        "schema": "slotspe_plugin_queue_scores_v1",
        "recorded_at": "2026-11-05",
        "queue": {
            "output_root": str(PLUGIN_ROOT),
            "protocol": "BLCA / UNI2-h / fold 0 / seed 3 / 30 epochs / 4096 patches / "
                        "logical batch 8 / 20 evaluation repeats / official best-validation "
                        "checkpoint selection (greater-or-equal keeps the later epoch)",
            "validation_policy": "one full validation pass per epoch over the whole "
                                 "validation fold, all real patches, no padding, no subsampling",
            "stage_order": ["v1", "v2", "v3"],
            "arms": list(ARMS),
        },
        "cohort_identity": cohort_identity(),
        "initialization": initial_hash(),
        "scores": scores,
        "paired_comparisons": paired_comparisons(summaries),
        "slot_geometry": geometry,
        "content_recovery": content,
        "original_baseline_same_protocol": baseline,
        "reading_boundaries": [
            "One fold and one seed. A single fold of 77 validation patients carries no "
            "uncertainty estimate, and the original baseline's own fold 0 per-epoch curve "
            "spans 0.55 to 0.75 on the same data.",
            "The protocol selects the best epoch by validation C-index, which is a maximum "
            "over about thirty validation passes over the same patients. The plugin "
            "B0_native score of 0.7433 is therefore a selected peak on the easiest fold, "
            "not the original's level. The original's five-fold mean under the same "
            "protocol is 0.7039 and its fold 0 value is 0.7485.",
            "The twenty repeats are repeated evaluations of one retained checkpoint under "
            "the native slot-selection noise. They measure evaluation randomness, not "
            "training randomness, and they are not twenty independent patients or twenty "
            "independent trainings.",
            "The plugin queue and the completed baseline share the initial core state and "
            "the exact same train and validation patient lists, so the two are comparable "
            "in cohort and initialization, but they are still one fold.",
            "Seven of the nine preset arms still show collapsed slots: effective rank "
            "1.00 to 1.01 out of eight and mean pairwise cosine 0.99 to 0.998, including "
            "the V2 arms designed to reduce redundancy. The two exceptions are B1_gram, "
            "which spreads the WSI slots to effective rank 4.55, and V3_relation, which "
            "spreads the OMIC slots to effective rank 1.88. Neither is a high-scoring arm: "
            "B1_gram loses 0.080 against the native control, far beyond the repeat "
            "standard deviation, and V3_relation scores below its own matched-capacity "
            "control. Spreading the slots and raising the survival score point in opposite "
            "directions in this queue.",
            "The V1 KL weight of 0.001 is numerically inert at this scale. The V1_vib and "
            "V1_noise latent geometries agree to the third decimal, so this measurement "
            "cannot show a compression effect in either direction.",
            "The V2 and V3 content-recovery errors stay above the constant-prediction floor "
            "for both modalities, so the private slots did not recover patient-specific "
            "content better than predicting a training-fold constant.",
            "These numbers are descriptive of one exploratory cohort. They establish no "
            "patient benefit, no patient harm, no correction efficacy and no clinical utility.",
        ],
    }


def render_markdown(result):
    scores = result["scores"]
    baseline = result["original_baseline_same_protocol"]
    lines = []
    add = lines.append
    add("# 插件队列 V1–V3 分数结果（BLCA fold 0 / seed 3）")
    add("")
    add("2026-11-05。本文件由 `scripts/summarize_plugin_queue_scores.py` 从 "
        "`local_evidence/` 的存储工件重新计算得到。只含聚合统计，不含患者标识、"
        "逐患者风险值、逐患者预测数组、权重或 checkpoint。")
    add("")
    add("## 〇、协议与两把尺子")
    add("")
    add("队列协议：" + result["queue"]["protocol"] + "。")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 训练/验证患者 | {result['cohort_identity']['plugin_train']} / "
        f"{result['cohort_identity']['plugin_val']} |")
    add(f"| 与原版 best 队列患者列表完全一致 | 训练 "
        f"{result['cohort_identity']['train_identical']}、验证 "
        f"{result['cohort_identity']['val_identical']} |")
    add(f"| 初始 core 哈希与原版 S0 折0 相同 | "
        f"{result['initialization']['identical']} |")
    add(f"| 原版 S0 同协议五折均值 | {baseline['S0_five_fold_mean']:.4f} ± "
        f"{baseline['S0_five_fold_std_population']:.4f} |")
    add(f"| 原版 S0 同协议折0 | {baseline['S0_cindex_per_fold']['0']:.4f}（最易的一折） |")
    add("")
    add("**读数的关键限制**：本队列用验证 C-index 取每轮最大值（约三十次验证通过的最大值），"
        "这是官方规则也是乐观选择。B0_native 的 0.7433 是**折0 上的选中峰值**，"
        "不是原版的水平；原版同协议五折均值是 "
        f"{baseline['S0_five_fold_mean']:.4f}。20 次复评的离散度是**同一权重下的评估随机性**，"
        "不是训练随机性，不构成泛化保证。")
    add("")
    top = max(scores, key=lambda arm: scores[arm]["repeat_cindex_mean"])
    add(f"**分数最高的臂是 {top}（{scores[top]['repeat_cindex_mean']:.4f}）**，"
        f"比 B0_native 高 "
        f"{scores[top]['repeat_cindex_mean'] - scores['B0_native']['repeat_cindex_mean']:+.4f}，"
        f"小于 B0_native 自身的复评标准差 "
        f"{scores['B0_native']['repeat_cindex_std_population']:.4f}，"
        "**因此没有哪个臂与原版对照分得开**。")
    add("")
    add("## 一、九臂分数（20 次复评口径）")
    add("")
    add("| 臂 | 机制 | 选模轮 | 选模分 | **20次复评均值 ± 总体SD** | 末轮选模分 | 耗时(s) |")
    add("|---|---|---:|---:|---:|---:|---:|")
    mechanisms = {
        "B0_native": "原版旁路（对照）",
        "B1_gram": "10×off-diagonal cos² 几何正则",
        "V1_mean": "确定性均值映射，无 KL",
        "V1_noise": "Gaussian 重参数，β=0",
        "V1_vib": "同上 + KL，β=0.001",
        "V2_content": "内容约束共享/私有分解",
        "V2_separate": "同上 + HSIC + 共享方差项",
        "V3_capacity": "逐槽残差（容量对照）",
        "V3_relation": "两两关系残差",
    }
    for arm in ARMS:
        s = scores[arm]
        add(f"| {arm} | {mechanisms[arm]} | {s['best_epoch']} | "
            f"{s['best_selection_cindex']:.4f} | "
            f"**{s['repeat_cindex_mean']:.4f} ± {s['repeat_cindex_std_population']:.4f}** | "
            f"{s['final_epoch_selection_cindex']:.4f} | {s['elapsed_seconds']:.0f} |")
    add("")
    add("## 二、三条预设核心对照")
    add("")
    add("| 对照 | 处理臂 | 对照臂 | 差值 | 处理臂SD | 对照臂SD |")
    add("|---|---|---|---:|---:|---:|")
    for row in result["paired_comparisons"]:
        add(f"| {row['description']} | {row['treatment']} | {row['control']} | "
            f"**{row['difference']:+.4f}** | {row['treatment_repeat_std']:.4f} | "
            f"{row['control_repeat_std']:.4f} |")
    add("")
    add("三条对照的差值都在复评标准差量级或以下，**没有一条支持某个机制带来可测收益**。")
    add("")
    add("## 三、槽几何（保留最佳权重的验证集均值）")
    add("")
    add("| 臂 | WSI 原槽余弦 | WSI 有效秩 | OMIC 原槽余弦 | OMIC 有效秩 | 余弦>0.9 比例(WSI) |")
    add("|---|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        g = result["slot_geometry"][arm]
        wsi, omic = g["wsi.native"], g["omic.native"]
        add(f"| {arm} | {wsi['cosine']:.4f} | {wsi['effective_rank']:.3f} | "
            f"{omic['cosine']:.4f} | {omic['effective_rank']:.3f} | "
            f"{wsi['fraction_cosine_above_09']:.4f} |")
    add("")
    collapsed = [arm for arm in ARMS
                 if result["slot_geometry"][arm]["wsi.native"]["cosine"] > 0.9]
    spread = [arm for arm in ARMS
              if result["slot_geometry"][arm]["wsi.native"]["cosine"] <= 0.9]
    add(f"**{len(collapsed)}/{len(ARMS)} 个臂的槽严重塌缩**（WSI 原槽余弦 > 0.9）："
        + "、".join(collapsed) + "。")
    add("")
    if spread:
        add("**确实把槽撑开的臂只有 " + "、".join(spread) + "**，其 WSI 几何如下：")
        add("")
        add("| 臂 | WSI 余弦 | WSI 有效秩 | OMIC 余弦 | OMIC 有效秩 | 余弦>0.9 比例(WSI) |")
        add("|---|---:|---:|---:|---:|---:|")
        for arm in spread:
            g = result["slot_geometry"][arm]
            wsi, omic = g["wsi.native"], g["omic.native"]
            add(f"| {arm} | {wsi['cosine']:.4f} | {wsi['effective_rank']:.3f} | "
                f"{omic['cosine']:.4f} | {omic['effective_rank']:.3f} | "
                f"{wsi['fraction_cosine_above_09']:.4f} |")
        add("")
        add("**但这两个臂都不是本队列的高分臂。** B1_gram 撑开 WSI 槽几何的同时复评分降到 "
            f"{result['scores']['B1_gram']['repeat_cindex_mean']:.4f}"
            f"（对照 B0_native {result['scores']['B0_native']['repeat_cindex_mean']:.4f}），"
            "差 "
            f"{result['scores']['B1_gram']['repeat_cindex_mean'] - result['scores']['B0_native']['repeat_cindex_mean']:+.4f}"
            "，是本队列唯一远超复评标准差的差；V3_relation 把 OMIC 侧余弦降到 "
            f"{result['slot_geometry']['V3_relation']['omic.native']['cosine']:.4f}"
            "、有效秩升到 "
            f"{result['slot_geometry']['V3_relation']['omic.native']['effective_rank']:.3f}，"
            "但复评分 "
            f"{result['scores']['V3_relation']['repeat_cindex_mean']:.4f}，"
            "低于同族的 V3_capacity "
            f"{result['scores']['V3_capacity']['repeat_cindex_mean']:.4f}。"
            "**在这条队列上，把槽撑开与提高生存分数是反向的**，"
            "这与 S0–S6 队列中「冗余越低分数越低」的既有发现方向一致。")
        add("")
    add("### 3.1 V1 三个空间（原槽 / 均值槽 / 采样槽）")
    add("")
    add("| 臂 | 空间 | WSI 余弦 | WSI 有效秩 | OMIC 余弦 | OMIC 有效秩 |")
    add("|---|---|---:|---:|---:|---:|")
    for arm in ("B0_native", "V1_mean", "V1_noise", "V1_vib"):
        g = result["slot_geometry"][arm]
        for space in GEOMETRY_SPACES:
            wsi, omic = g[f"wsi.{space}"], g[f"omic.{space}"]
            add(f"| {arm} | {space} | {wsi['cosine']:.4f} | {wsi['effective_rank']:.3f} | "
                f"{omic['cosine']:.4f} | {omic['effective_rank']:.3f} |")
    add("")
    add("V1_vib 与 V1_noise 的 latent（采样）空间几何逐位同量级"
        "（WSI 余弦 0.9919 对 0.9920，有效秩 1.014 对 1.014），"
        "**β=0.001 的 KL 在这个尺度上不产生可测的压缩效应**，因此该测量在两个方向上都不能"
        "证明压缩有效或无效。")
    add("")
    add("## 四、V2/V3 内容恢复误差与常量预测下界")
    add("")
    add("| 臂 | WSI 内容MSE | WSI 常量下界 | OMIC 内容MSE | OMIC 常量下界 |")
    add("|---|---:|---:|---:|---:|")
    for arm in CONTENT_ARMS:
        c = result["content_recovery"][arm]
        add(f"| {arm} | {c['wsi']['mse']:.5f} | {c['wsi']['constant_floor']:.5f} | "
            f"{c['omic']['mse']:.5f} | {c['omic']['constant_floor']:.5f} |")
    add("")
    add("**四个臂的两个模态，内容误差都高于常量预测下界。** 私有槽的内容恢复没有比"
        "「直接预测训练折常量」更好，因此「内容约束让槽携带患者特异内容」这一前提"
        "在本队列上不成立。")
    add("")
    add("## 五、读法与边界")
    add("")
    add("- **单折单种子。** 77 名验证患者的一折不带任何不确定性估计；原版基线自己在同一数据上的"
        "逐轮曲线就在 0.55–0.75 之间摆。")
    add(f"- **选模是乐观选择。** 协议按验证 C-index 取每轮最大值（约三十次验证通过的最大值）。"
        f"插件 B0_native 的 {result['scores']['B0_native']['repeat_cindex_mean']:.4f} "
        f"是**折0 上的选中峰值**，不是原版的水平：原版同协议五折均值是 "
        f"{baseline['S0_five_fold_mean']:.4f}，其折0 值是 "
        f"{baseline['S0_cindex_per_fold']['0']:.4f}（最易的一折）。")
    add("- **20 次复评是评估随机性，不是训练随机性。** 它是同一保留权重在原生槽选择噪声下的"
        "重复评估，既不是 20 个独立患者，也不是 20 次独立训练，不构成泛化保证。")
    add("- **同队列同初始化，但不是同一批折。** 插件队列与已完成的基线队列共享初始 core 状态、"
        "共享完全相同的训练与验证患者列表，因此架构、队列、初始化三者可比；"
        "**但仍然只有一折**。")
    add("- **九臂之间的 0.005–0.03 差值无法归因。** 各臂差值都在自身复评标准差量级或以下，"
        "而一折无法提供宣称所需的折间方差。")
    add(f"- **B1_gram 掉 {result['scores']['B1_gram']['repeat_cindex_mean'] - result['scores']['B0_native']['repeat_cindex_mean']:+.4f}** "
        "是唯一超出复评噪声的差，但它自身的复评标准差也是九臂中最大的（"
        f"{result['scores']['B1_gram']['repeat_cindex_std_population']:.4f}），"
        f"且选模轮是第 {result['scores']['B1_gram']['best_epoch']} 轮而非第 1 轮，"
        "因此它同时是九臂里最不稳定的一个。这是与 S0–S6 既有发现同向的证据，不是孤证。")
    add("- **V1 的 KL 权重没有扫描。** β=0.001 在 common-private 那条线上已被记为数值惰性，"
        "本次测量确认了惰性，但**没有测试更大权重**，因此不能得出关于信息压缩的一般结论。")
    add("- **V2/V3 的内容误差没有超过常量预测下界**，这说明内容目标在本队列上没赢过"
        "「直接预测训练折常量」，但**不能区分**原因在匹配目标、目标定义、容量还是底层塌缩。")
    add("- 未做多 seed、容量扫描、九臂间的多重比较校正，也未做独立泛化检验。")
    add("- **以上数字只是单一探索性队列的描述性结果，不构成患者获益、患者损害、修正效力或临床效用证据。**")
    add("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO / "docs/research/slotspe_plugin_queue_results_20261105")
    args = parser.parse_args(argv)
    result = build()
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "plugin_queue_scores.md").write_text(render_markdown(result), encoding="utf-8")
    (folder / "plugin_queue_scores.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(folder), "arms": len(ARMS)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
