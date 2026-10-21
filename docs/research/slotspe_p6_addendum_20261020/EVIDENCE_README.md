# SlotsPE eval_fix 精度影响：证据包

**生成日期**: 2026-10-21
**文件**: `slotspe_evalfix_evidence.json`（203 KB,自包含）

---

## 这是什么

一个自包含的 JSON 文件,包含对以下问题的完整证据:

> **eval_fix 修复是否会系统性地损害 C-index 精度?**

包含 18 个 fold/seed pair 的逐次评估 C-index、每个 pair 的完整 30-epoch 训练曲线、
以及 provenance 哈希。**所有已发布的数字都可以从这个文件重新计算。**

---

## 结论（先看这个）

| | |
|---|---|
| 判定 | **`NO_SYSTEMATIC_EFFECT`** |
| 样本量 | 18 pairs |
| 平均 ΔC | **+0.00269** |
| bootstrap 95% CI | **[-0.00120, +0.00679]** |
| 符号分布 | 11 正 / 7 负 |

**eval_fix 没有可检测的系统性代价。**

原先在单个 checkpoint（fold_0/seed_3）上测得的 -0.0055 是抽样波动,不是该修复的性质。

### ⚠️ 必须一起读的限制

**所有被测的 checkpoint 都已越过各自的最佳点。**

| | |
|---|---|
| 峰值 C 均值 | 0.7168 |
| epoch-30 C 均值 | 0.5905 |
| 平均跌幅 | -0.1263 |
| 最佳 epoch 范围 | 2–24（中位 6） |
| 跌幅 >0.01 的 pair | 18 / 18 |

**因此上面的判定描述的是「过拟合状态下」eval_fix 的行为。模型在其最佳 epoch 的行为
从未被测量。** 这是本研究的主要局限,不是可忽略的脚注。

---

## 如何独立验证

全部数据在 `per_pair` 数组中。每个元素:

```
per_pair[i] = {
  fold, seed, label,
  delta_cindex,                        # 可直接验算
  arms.original,                       # 20 次评估的 C-index 分布
  arms.same_weights_eval_fix,
  curve,                               # 30 epoch 逐 epoch 明细
  curve_peak,                          # 最佳 epoch / 峰值 / 跌幅
  checkpoint_sha256_original,          # provenance
  same_initial_state_sha256,
  ...
}
```

**验算 1 — 每 pair 的 ΔC:**

```python
d = per_pair[i]
assert abs(d.delta_cindex
           - (d.arms['same_weights_eval_fix'].mean - d.arms['original'].mean)) < 1e-12
```

**验算 2 — 总判定:**

```python
import numpy as np
delta = np.array([p['delta_cindex'] for p in per_pair])

rng = np.random.default_rng(0)
idx = rng.integers(0, len(delta), size=(10000, len(delta)))
ci = np.percentile(delta[idx].mean(axis=1), [2.5, 97.5])

print(delta.mean())      # 0.00269
print(ci)                # [-0.00120, 0.00679]
print(ci[0] < 0 < ci[1])  # True -> NO_SYSTEMATIC_EFFECT
```

**验算 3 — 过拟合结论:**

```python
for p in per_pair:
    vals = [c['val_cindex_exploratory'] for c in p['curve']
            if c['val_cindex_status'] == 'computed']
    i = int(np.argmax(vals))
    assert vals[i] - vals[-1] > 0.01   # 18/18 成立
```

---

## 哪些数据被省略了,以及为什么

原始输出共 9.2 GB。省略的内容如下:

| 省略项 | 体积 | 理由 |
|---|---|---|
| `eval_repeats.json` 中的 `risks` / `logits`（逐患者风险值） | 243 KB/臂 → 主要体积来源 | C-index 每次评估的**值**已保留。逐患者风险向量不是重算任何结论所必需,保留会把包从 1 MB 撑到 130 MB。 |
| `last.pt`, `shared_initial_state.pt` | 821 文件,8.9 GB | 仅保留 sha256 provenance 哈希。 |
| `patient_slot_intervention.json` | 2.8 MB/臂 | RESULTS.md 中无任何结论引用它。 |

**若需要上述任一项**,原始数据在训练机上:
```
results/slotspe_proof/slotspe_addendum_gpu0_20261020T145000/
results/slotspe_proof/slotspe_addendum_gpu1_20261020T150000/
```

---

## 未补测的缺口

- **最佳 epoch 上 eval_fix 的行为未知。** 需按各 pair 自身峰值 epoch 重训
  （直接加载逐 checkpoint 评估属选择行为，会引入选择偏差）。
- CI 上界为 +0.0068，因此**小于 0.007 的真实代价无法检出**。

正确表述是「**未检测到**系统性代价」,而非「证明无代价」。

---

## 环境

```
repo:      DCT-Reg (github.com/LDT-von/DCT-Reg)
docs:      docs/research/slotspe_p6_addendum_20261020/
数据:       BLCA, 5 folds × 3–4 seeds
每 pair:    original / readout_fix 双臂共享初始化、分箱、batch 顺序、step seed
评估:      same_weights_eval_fix 复用 original 权重（0 优化器步），确定性评估
```

完整结论文档见同目录 `RESULTS.md`。