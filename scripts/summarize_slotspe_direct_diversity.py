"""Export aggregate-only generated experiment results after raw-array checks."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "local_evidence/direct_diversity_synthetic_20261022_run01"
PUBLIC = ROOT / "docs/research/slotspe_native_assignment_training_20261022"
LABELS = {
    "R0": "原生基线",
    "R1": "直接Gram去相关",
    "R2": "Slot-BERT论文损失移植",
    "R3": "论文损失＋集合重构",
    "R4": "仅集合重构",
    "R5": "Gram去相关＋集合重构",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true")
    if not parser.parse_args().export:
        print("--export verifies the fixed generated-only receipt and exports aggregates; no model execution")
        return
    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment

    summary = json.loads((RAW / "summary.json").read_text(encoding="utf-8"))
    if (summary["schema"] != "native_slotspe_generated_direct_diversity_v1" or
            summary["status"] != "completed_generated_only_matched_six_arm" or
            summary["patient_runs"] != 0 or summary["server_runs"] != 0 or
            summary["plan"]["data"] != "new invented 8-component tensors only; no patient input"):
        raise ValueError("only the completed fixed generated-data protocol may be exported")
    for name, expected in summary["raw_file_sha256"].items():
        if digest(RAW / name) != expected:
            raise ValueError(f"raw evidence changed: {name}")
    for name, expected in summary["protected_sha256_before"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"protected source or configuration changed: {name}")
    for name, expected in summary["implementation_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"executed implementation changed: {name}")

    def set_mse(prediction, target):
        result = []
        for pred, truth in zip(prediction.astype(np.float64), target.astype(np.float64)):
            cost = ((pred[:, None] - truth[None]) ** 2).mean(-1)
            rows, columns = linear_sum_assignment(cost)
            result.append(float(cost[rows, columns].mean()))
        return float(np.mean(result))

    verified_draws = 0
    state_pairs = {}
    payloads = {}
    for run in summary["runs"]:
        seed, arm = run["seed"], run["variant"]
        if seed not in payloads:
            payloads[seed] = torch.load(RAW / f"generated_seed_{seed}.pt", weights_only=True, map_location="cpu")
        data = payloads[seed]
        states = run["initial_state_sha256"]
        if seed in state_pairs and states != state_pairs[seed]:
            raise ValueError("paired initial states differ")
        state_pairs[seed] = states
        raw = torch.load(RAW / f"matched_{arm}_seed_{seed}.pt", weights_only=True, map_location="cpu")
        if len(raw["losses"]) != 128 or len(raw["final_holdout"]) != 3:
            raise ValueError("incomplete fixed run")
        for phase in ("initial_holdout", "final_holdout"):
            for draw in raw[phase]:
                repeat = draw["repeat"]
                for modality in ("wsi", "omic"):
                    slots = draw["slots"][modality].double().numpy()
                    if slots.shape != (32, 8, 32) or not np.isfinite(slots).all():
                        raise ValueError("slot coverage or finiteness differs")
                    norm = np.linalg.norm(slots, axis=-1, keepdims=True)
                    if norm.min() <= 1e-8:
                        raise ValueError("zero slot found")
                    unit = slots / norm
                    gram = unit @ unit.transpose(0, 2, 1)
                    pairs = gram[:, ~np.eye(8, dtype=bool)]
                    expected = run[phase]["per_draw_metrics"][modality][repeat]
                    independent = {"cosine": pairs.mean(), "absolute_cosine": np.abs(pairs).mean(),
                                   "rms_cosine": np.sqrt((pairs ** 2).mean()),
                                   "effective_rank": (64 / (gram ** 2).sum((1, 2))).mean()}
                    prediction = draw["probe_predictions"][modality].numpy()
                    target = data["targets"][modality][64:].numpy()
                    independent["own_set_mse"] = set_mse(prediction, target)
                    independent["donor_set_mse"] = set_mse(np.roll(prediction, 1, axis=0), target)
                    static = data["static_train_dictionary"][modality].numpy()
                    independent["static_dictionary_set_mse"] = set_mse(np.broadcast_to(static, target.shape), target)
                    for name, value in independent.items():
                        if not np.isclose(value, expected[name], rtol=1e-6, atol=1e-7):
                            raise ValueError(f"independent raw check failed {arm}/{seed}/{phase}/{repeat}/{modality}/{name}")
                    verified_draws += 1

    public = {
        "schema": "native_slotspe_direct_diversity_public_v1",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "source_summary_sha256": digest(RAW / "summary.json"),
        "status": summary["status"], "plan": summary["plan"],
        "aggregate_final": summary["aggregate_final"],
        "paired_final_vs_R0": summary["paired_final_vs_R0"],
        "per_seed_final": [{"seed": run["seed"], "variant": run["variant"],
                            "final_holdout": {k: v for k, v in run["final_holdout"].items() if k != "per_draw_metrics"}}
                           for run in summary["runs"]],
        "counters": summary["counters"],
        "validation": {"analytic_tests_passed_after_fix": 26, "initial_test_run": "25 pass, 1 dtype check fail; fixed before training",
                       "independent_numpy_scipy_modality_draw_checks": verified_draws,
                       "paired_initial_states": True, "raw_file_hash_checks": len(summary["raw_file_sha256"]),
                       "source_and_config_unchanged": summary["protected_files_unchanged"],
                       "executed_implementation_unchanged": summary["implementation_unchanged_during_run"],
                       "extra_native_test_root_forwards": 4, "extra_native_regularizer_backwards": 2,
                       "extra_native_test_optimizer_steps": 0},
        "sample_units": {"unique_generated_holdout_examples": 96, "case_draws_per_arm_modality": 288,
                         "repeats_are_independent_examples": False},
        "protected_sha256": summary["protected_sha256_after"],
        "implementation_sha256": summary["implementation_sha256"],
        "raw_file_sha256": summary["raw_file_sha256"],
        "patient_runs": 0, "server_runs": 0, "new_commit_or_push": False,
        "boundary": "R0-R5 are fixed-weight exploratory candidates on an easy eight-component toy; native outputs and eight slots retained. Content probe is not a survival-information guarantee.",
    }
    write_json(PUBLIC / "direct_diversity_summary.json", public)
    a = public["aggregate_final"]
    lines = ["原生SlotSPE直接表示去冗余：本地生成数据结果", "", "1. 路由/任务",
             "SlotSPE / SPE-08。继续S0–S6之后的直接表示约束探索，R0–R5不是新患者队列。", "",
             "2. 已完成工作", "6组×3种子，固定128步；每个种子64训练、32留出样本，评估3次原生随机重放。",
             "这是显式含8种身份特征的容易区分的生成输入；病理32 tokens、8条组学通路；两侧始终各8槽。",
             "投影32维、3轮SA；不使用目标查询、槽编号、置零、剪成单槽或输出正交化。原生主/辅损失照常计算。",
             "", "各组末步结果（三种子均值，均为原生输出槽）：", "",
             "组别 | 方法 | 病理cos | 组学cos | 病理绝对cos | 组学绝对cos | 合成留出NLL",
             "---|---|---:|---:|---:|---:|---:"]
    for arm, label in LABELS.items():
        w, o = a[arm]["modalities"]["wsi"], a[arm]["modalities"]["omic"]
        lines.append(f"{arm} | {label} | {w['cosine']:.6f} | {o['cosine']:.6f} | {w['absolute_cosine']:.6f} | {o['absolute_cosine']:.6f} | {a[arm]['native_main_nll']:.6f}")
    lines += ["", "本轮结论：直接表示去相关能大幅压低该合成协议下的原生槽cos；R1与R5两侧均在3/3种子下降。"]
    for modality, chinese in (("wsi", "病理"), ("omic", "组学")):
        base = a["R0"]["modalities"][modality]
        result = a["R1"]["modalities"][modality]
        lines.append(f"R1 {chinese} signed cos相对下降{100 * (1-result['cosine']/base['cosine']):.2f}%；绝对cos从{base['absolute_cosine']:.6f}到{result['absolute_cosine']:.6f}，RMS为{result['rms_cosine']:.6f}，不能称完全正交。")
    lines += ["", "内容检查：own/static <1才优于与样本无关的训练均值字典；donor/own >1表示当前解码探针对自身槽有一定依赖。",
              "错配同时交换随机初态，须结合summary中的同噪声换输入距离解释，不能当作纯输入因果效应。", "",
              "组别 | 病理own/static | 组学own/static | 病理donor/own | 组学donor/own", "---|---:|---:|---:|---:"]
    for arm in LABELS:
        w, o = a[arm]["modalities"]["wsi"], a[arm]["modalities"]["omic"]
        lines.append(f"{arm} | {w['own_to_static_ratio']:.4f} | {o['own_to_static_ratio']:.4f} | {w['donor_to_own_ratio']:.4f} | {o['donor_to_own_ratio']:.4f}")
    lines += ["", "权重预设：R1 Gram10；R2论文损失1；R3论文损失1+重构5；R4重构5；R5 Gram10+重构5。",
              "各损失数值尺度不同；这是固定权重候选测试，不能据此给论文方法作普遍强弱排名。",
              "所有组的同型解码探针均接受同样步数训练：先对断开梯度的槽更新探针，再冻结探针更新主模型。",
              "重构的病理目标由各样本输入独立聚类得到，组学目标为原始通路集合；生成类型真值仅用于评价静态字典。",
              "Slot-BERT Eq11包括对角项：-mean(log_softmax(-(normalize(S)normalize(S)^T-I)/0.5))。",
              "这是论文损失移植，不是完整Slot-BERT；本文权重1并非论文某消融最佳权重0.01。",
              "论文Figure6描述无损失时通常>0.6、加入后接近0的示例热图，未给出可用于本任务的全测试集平均降幅。",
              "来源：https://arxiv.org/html/2501.12477v4#S3.SS5 ; https://arxiv.org/html/2501.12477v4#S4.SS9",
              "源码与其他文献边界见同目录direct_diversity_literature.txt。", "",
              "3. 路径/提交", "实现：survot_rank/research/slotspe_proof/slot_regularization.py；执行/导出脚本：scripts/check_slotspe_direct_diversity.py、scripts/summarize_slotspe_direct_diversity.py。",
              "公共机器记录：本目录direct_diversity_summary.json。生成张量、权重、逐样本槽、失败现场规则和执行代码快照均在忽略的local_evidence/direct_diversity_synthetic_20261022_run01。未提交或推送。", "",
              "4. 检查/实际执行/原始回执",
              f"26项测试通过；此前1项整数dtype检查失败已在训练前修正；独立NumPy/SciPy复算{verified_draws}个模态重放的cos、abs、RMS、参与率与集合MSE全部匹配。",
              "执行计数：" + json.dumps(public["counters"], ensure_ascii=False),
              "每步一次core backward和一次detached probe backward。额外两次测试套件合计4次原生生成输入前向、2次正则反向、0次优化更新。",
              "原版源码、默认患者队列及S0–S6配置哈希前后一致；患者执行0、服务器执行0，没有新患者原始回执。", "",
              "5. 缺口与原因",
              "cos下降只证明角度去相似；绝对cos/RMS另行给出。不能把保留8槽误写成保留8种有效内容。",
              "本轮六组的两侧重构探针均不及使用生成类型信息建立的静态训练均值字典（own/static均>1），donor/own接近1。现有探针没有证明稳定的样本内容分工；这也不能反推槽完全不含输入信息。",
              "直接去相关可能更多保留随机初态方向。R1病理同噪声换输入距离低于R0；仅角度收益不足以采用为真实患者训练方案。",
              "96个独立生成留出样本对应每臂每模态288次重放，不是288位独立患者。不同于历史S0–S6简单合成协议，不横向混合数值。",
              "参与率为K²/sum(G²)，不等于Shannon有效秩。NLL仅为小合成任务代理，非真实患者性能或C-index。", "",
              "6. 下一步/用户输入",
              "以角度指标和内容门槛共同筛选；本轮不足以直接选定患者训练方案。后续须先改善样本内容保留，再为真实数据单独制定匹配训练和跨折协议。",
              "本轮按本地模拟授权执行完毕，未启动患者或服务器任务。", ""]
    (PUBLIC / "直接去冗余本地结果.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"runs": len(summary["runs"]), "independent_checks": verified_draws,
                      "public_summary_sha256": digest(PUBLIC / "direct_diversity_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
