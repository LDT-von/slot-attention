# Addendum：eval_fix 是否系统性损害精度？

**日期**: 2026-10-21
**状态**: 完成（结论已定，边界已声明）
**产出**: `post_training_analysis.json`

---

## 背景

此前在单个 checkpoint（fold_0/seed_3, 3 epochs）上测得 eval_fix 使 C-index
下降 0.0055，据此怀疑该修复有精度代价。单点无法区分「系统性损失」与
「随机波动」，故训练 18 个 fold/seed pair 复核。

---

## Q1：eval_fix 有无系统性代价？

18 个 pair（5 folds × 3-4 seeds），每个 pair 内 20 次评估重复。

| 指标 | 值 |
|---|---|
| 样本量 | 18 |
| 平均 ΔC | **+0.00269** |
| 标准误 | 0.00215 |
| bootstrap 95% CI | **[-0.00120, +0.00679]** |
| 符号分布 | 11 正 / 7 负 / 0 零 |
| Δ 取值范围 | [-0.01337, +0.02073] |

**结论：`NO_SYSTEMATIC_EFFECT`**

CI 跨过 0，两种符号均出现。原先 fold_0/seed_3 的 -0.0055 是单点抽样波动，
不是该修复的性质。

### 措辞边界（重要）

本结果只能支持「**未检测到**系统性代价」，不能支持「证明无代价」：

- CI 上界为 +0.0068，样本量 18 的功效不足以排除小幅正效应。
- 若存在小于 0.007 的真实代价，本实验无法检出。

正确表述：**eval_fix 的期望效应估计为 +0.003，且与零不可区分。**

---

## Q2：所用 checkpoint 是否处于模型工作区间？

**否。这是本次实验最重要的发现，也是 Q1 结论的主要限制。**

训练统一使用 30 epochs（沿用配置默认值），但每个 pair 的 val C-index 峰值
出现在 epoch 2–24，中位数 6。

| 指标 | 值 |
|---|---|
| 峰值 C-index 均值 | **0.7168** |
| epoch-30 C-index 均值 | **0.5905** |
| 平均跌幅 | **-0.1263** |
| 掉超过 0.01 C 的 pair | **18 / 18** |

训练 NLL 降至 0.02–0.80，val NLL 升至 0.95–2.95，属系统性过拟合。

### 因此，Q1 实际回答的是

> 在**已过拟合**的 checkpoint 上，eval_fix 是否引入额外代价 → 否。

模型在其最佳 epoch 上从未被测量。**Q1 的结论不能外推到最优状态。**

这是本次实验的已知缺口，未补测（见文末）。

---

## Q3：C-index 在 fold 间的离散度

| fold | 均值 | seed 间 sd | n |
|---|---|---|---|
| fold_0 | 0.5978 | 0.1056 | 4 |
| fold_1 | 0.5351 | 0.0290 | 4 |
| fold_2 | 0.6126 | 0.0470 | 4 |
| fold_3 | 0.6277 | 0.0128 | 3 |
| fold_4 | 0.6713 | 0.0286 | 3 |
| **全体** | **0.6044** | **0.0683** | 18 |

fold_0 的 seed 间 sd（0.1056）大于 fold 间差距。**单 fold 的任何结论都不应
被采信**——这从数据上确认了"单 checkpoint 不足以下结论"的判断。

---

## 方法学记录

- 队列系统 `survot_rank.research.slotspe_proof.queue`，双 GPU 并行。
- 每个 pair 训练 `original` 与 `readout_fix` 两臂，共享初始化、分箱、
  batch 顺序、step seed 与完整 epoch 预算。
- `same_weights_eval_fix` 复用 original 权重（0 优化器步）做确定性评估，
  是 Q1 的直接测量对象。
- 21 个产出 pair 全部通过完整性校验（summary / eval_repeats / checkpoint）。

---

## 本轮两处判断错误（自我记录）

1. **沿用 epochs=30 而未质疑**。看到配置默认值为 30 即照搬，未考虑
   原始 run 的 3 epochs 才是合理设置。事后 18/18 pair 证明 30 全部过拟合。
2. **上一轮暗示 eval_fix 可能存在代价**。在只有 1 个 checkpoint 时，
   措辞倾向于「存在一个待发现的问题」。扩充样本后证明该问题不存在。
   两处均属**未经验证即下判断**。

---

## 未补测的缺口

- **最佳 epoch 上的 eval_fix 行为未知。** 需按各 pair 自身峰值 epoch 重训
  （因逐 checkpoint 加载属选择行为，会引入选择偏差）。
- Q1 的 CI 上界 +0.0068，对小于此量级的真实代价不敏感。

## 产物

- `docs/research/slotspe_p6_addendum_20261020/post_training_analysis.json`
- `results/slotspe_proof/slotspe_addendum_gpu0_20261020T145000/`（15 pairs）
- `results/slotspe_proof/slotspe_addendum_gpu1_20261020T150000/`（6 pairs）
- `results/slotspe_proof/_train_monitor_report.txt`（队列时序日志）