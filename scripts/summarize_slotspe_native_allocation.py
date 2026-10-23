"""Export aggregate-only results from a completed invented-data SlotSPE run."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="completed generated-data summary JSON")
    parser.add_argument("--output", required=True, help="new public report folder")
    args = parser.parse_args()
    source = (ROOT / args.input).resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("schema") != "native_slotspe_generated_allocation_smoke_v1" or data.get("status") != "completed_generated_data_only":
        raise ValueError("not a completed generated-data run")
    seeds = data["plan"]["seeds"]
    arms = list(data["variants"])
    runs = {(r["seed"], r["variant"]): r for r in data["runs"]}
    if len(runs) != len(seeds) * len(arms) or len(runs) != len(data["runs"]):
        raise ValueError("missing or duplicated generated-data runs")
    if data["patient_runs"] != 0 or data["server_runs"] != 0:
        raise ValueError("this exporter accepts invented-data results only")
    if not all(r["optimizer_steps"] == data["plan"]["steps"] and r["all_finite"] for r in runs.values()):
        raise ValueError("unfinished or nonfinite run")
    if not data["protected_files_unchanged"]:
        raise ValueError("source hash gate failed")
    result = {
        "schema": "native_slotspe_generated_allocation_public_summary_v1",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "source_summary_sha256": sha(source),
        "evidence_tier": "generated_data_full_model_smoke_and_short_learning_only",
        "plan": data["plan"], "variants": data["variants"],
        "assignment_config": data["assignment_config"], "runtime": data["runtime"],
        "source_sha256": data["source_sha256"], "code_sha256": data["code_sha256"],
        "native_shape_gates": data["native_shape_gates"],
        "per_seed": [], "means": [],
        "counts": {"synthetic_training_runs": len(runs), "optimizer_steps": data["completed_optimizer_steps"],
                   "native_shape_full_forwards": len(arms), "native_shape_full_backwards": len(arms),
                   "toy_full_forwards": len(runs) * (data["plan"]["steps"] + 2 * data["plan"]["eval_repeats"]),
                   "toy_full_backwards": data["completed_optimizer_steps"],
                   "patient_runs": 0, "server_runs": 0},
        "protected_files_unchanged": True,
        "raw_synthetic_file_sha256": data["raw_file_sha256"],
        "metric_definitions": {
            "nll": "native discrete survival NLL summed over invented examples / batch size; all generated examples uncensored; lower better",
            "paired_delta": "final holdout NLL(Si)-NLL(S0) within same generated-data seed; not initial-to-final improvement",
            "read_overlap": "mean over generated examples, heads and unordered slot pairs of sum_token min(final-iteration per-head reading weights)",
            "slot_cosine": "mean cosine over generated examples and unordered final slot pairs",
            "eval": "mean of three separate native stochastic forward metrics; logits are not averaged before NLL",
        },
        "limits": [
            "Three seeds change both generated dataset and initialization: three paired synthetic scenarios, not patient evidence.",
            "24 randomly sampled minibatches are not three strict epochs; no outcome-based hyperparameter/checkpoint selection.",
            "S3/S4 replace every omic SA call, including WSI-to-omic reconstruction.",
            "S4-S1 contrasts full MESH perturbation+cost updates against Sinkhorn; S2/S3 lack same-side Sinkhorn controls.",
            "Lower reading overlap/cosine does not by itself establish useful content recovery or predictive benefit.",
            "Native-dimension gate uses encoding8 and a small invented input; short training uses dimension32 and 3 iterations.",
            "Real patient training integration and efficacy remain unestablished.",
        ],
    }
    for arm in arms:
        selected = [runs[seed, arm] for seed in seeds]
        delta = [r["final_holdout"]["nll"] - runs[r["seed"], "S0"]["final_holdout"]["nll"] for r in selected]
        mean = {"variant": arm, "initial_nll": statistics.mean(r["initial_holdout"]["nll"] for r in selected),
                "final_nll": statistics.mean(r["final_holdout"]["nll"] for r in selected),
                "paired_nll_delta_mean": statistics.mean(delta),
                "paired_nll_delta_std_ddof1": statistics.stdev(delta),
                "paired_nll_deltas": delta,
                "final_mechanism": {key: statistics.mean(r["final_holdout"][key] for r in selected)
                                    for key in selected[0]["final_holdout"] if key != "nll"}}
        result["means"].append(mean)
        for r, difference in zip(selected, delta):
            result["per_seed"].append({"seed": r["seed"], "variant": arm,
                                       "initial_holdout": r["initial_holdout"],
                                       "final_holdout": r["final_holdout"],
                                       "paired_final_nll_delta_to_S0": difference,
                                       "optimizer_steps": r["optimizer_steps"], "all_finite": r["all_finite"]})
    dest = (ROOT / args.output).resolve()
    if not dest.is_relative_to(ROOT / "docs/research"):
        raise ValueError("public destination must be under docs/research")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "aggregate_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")
    lines = [
        "原版 SlotSPE：五组本地测试与误路由清理", "",
        "一、路线/任务", "SPE-08：保持原版SlotSPE宿主，检查S0–S4槽内分配。用户授权清理误会并进行本地模拟/烟雾测试。", "",
        "二、完成的工作与结果", "删除的是本轮误切换到DCT后新增的13个未跟踪文件（含错误执行包），已先备份并逐字节核验；当前进度移除两条错误入口，既有历史证据保留。",
        "独立适配器只替换原模型的病理SA/组学SA内部内容分配，不增加参数。S0双原版；S1双Sinkhorn；S2病理MESH；S3组学MESH；S4双MESH。",
        "Top-K、预测读取、原生损失/系数、高斯初槽、Gumbel和dropout保留；MESH独立随机流不干扰原生RNG。组学SA的重建复用调用同样受替换。", "",
        "下表均为生成数据；每组3个配对场景，每场景64训练/32留出样本，24个batch8更新，三次随机评估均值。NLL越低越好；读取重叠和cos仅描述机制。",
        "版本 | 留出NLL | 与S0配对差 | WSI读取重叠 | OMIC读取重叠 | WSI槽cos | OMIC槽cos",
    ]
    for row in result["means"]:
        m = row["final_mechanism"]
        lines.append(f'{row["variant"]} | {row["final_nll"]:.6f} | {row["paired_nll_delta_mean"]:+.6f} | {m["wsi_0_read_overlap"]:.6f} | {m["omic_0_read_overlap"]:.6f} | {m["wsi_0_slot_cosine"]:.6f} | {m["omic_0_slot_cosine"]:.6f}')
    lines += ["", "各场景的最终留出NLL配对差（Si−S0；负数较低）："]
    for row in result["means"][1:]:
        lines.append(row["variant"] + ": " + ", ".join(f"seed{seed} {difference:+.6f}" for seed, difference in zip(seeds, row["paired_nll_deltas"])))
    lines += ["", "各版本相对S0的留出NLL降低场景数：" + "；".join(
        f'{row["variant"]} {sum(value < 0 for value in row["paired_nll_deltas"])}/{len(seeds)}'
        for row in result["means"][1:]) + "。",
        "本轮本地结果：MESH明显改变读取分配，槽表示的变化较小；S2/S4平均留出损失更高，未建立稳定预测收益。普通Sinkhorn在这三个场景均略低于S0，仍只是小规模合成任务观察，不据此选定患者优胜模型。"]
    lines += [
        "", "三、路径/提交", "适配器：survot_rank/research/slotspe_proof/native_allocation.py",
        "检查入口：scripts/check_slotspe_native_allocation.py；汇总入口：scripts/summarize_slotspe_native_allocation.py",
        "汇总：本目录aggregate_summary.json；原始生成张量、初始化、新训练权重及logits留在被忽略的local_evidence。没有提交/推送。", "",
        "四、检查与实际执行", "适配器20项检查+既有分配模块12项检查，共32项通过；默认计划入口只打印计划。",
        "五组均通过256维、8槽、10次迭代的完整训练前向/损失反向，输入为小规模生成张量；不更新参数。",
        f'短程模拟训练{len(runs)}组，实际{data["completed_optimizer_steps"]}次参数更新；脚本合计455次完整前向、365次反向（不含单元测试）。全程CPU，新建权重，无患者/外部权重读取、无服务器执行。',
        "原版源码哈希及默认配置独立核对未改。清理和检查收据见同目录JSON。", "",
        "五、缺口/原因", "短程训练是32维、8槽、3迭代，三个种子同时改变合成数据和初始化；24次随机小批量不是严格3个epoch。全为生成事件、无删失，不能代表真实生存数据。",
        "本任务可由平均内容预测，未专门构造只有去重后才可学习的困难任务；结果不能证明独立内容恢复，也不能据此否定其他任务的潜在收益。",
        "完整MESH方案与普通Sinkhorn的差异包含噪声和成本优化；S2/S3没有同侧Sinkhorn控制。患者损害和修正收益仍未建立。", "",
        "六、下一步", "本地结果先判断机制是否确实改变，再结合配对留出损失决定是否值得进入真实数据对照。真实五臂训练器、站点数据协议与结果回传仍需接入。",
        "正式比较按同折同种子差值及其不确定性判断；不能用基线跨折标准差直接否定小的配对收益。",
    ]
    (dest / "本地五组测试与清理报告.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result["means"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
