# Readfix v2 执行结果交接（server → 上游 AI）

本文件是本机 readfix v2 全流程执行的**事实交接**，供上游 AI 继续使用。所有数字均为本机实测原始值，未做筛选或美化。

- 执行方：本机隔离副本 `/data1/slotspe_full_slot_fixprobe_20261111_01`
- 分支 / HEAD：`readfix_v2_sandbox` @ `245da18ab4042669b98c7f7bd6b25fbbf793731b`（上游 `codex/full-slot-research-20261111` 同提交）
- 执行日期：2026-11-11 ~ 2026-11-13
- GPU：物理 GPU1（UUID `GPU-ae871418-cdb0-08a8-e680-ad1b87af08e7`，RTX 5090），`CUDA_VISIBLE_DEVICES=1`，配置逻辑 `cuda:0`
- 物理 GPU0 当时被其他实验室 PAGHC 进程占用（约 15.3GB），本轮**未干预**，全部执行落在物理 GPU1
- Python 3.12.4 / torch 2.10.0+cu128 / CUDA 12.8

---

## 1. 一句话结论

readfix v2 修复**有效**：edc2313 的归一化断言崩溃彻底消失，探针与两臂正式运行全部退出码 0。
但**分数不可宣称提升**：76 人验证集上的置换零假设 SD 约 0.08，而 joint 的 best 只比全 epoch 均值高约 0.073（约 1 个噪声标准差）；且同队列旧 pilot（槽更少、预算更小）已得 0.5658。
**唯一独立于分数的确凿机制发现**：`p_common`（WSI 共同槽）在训练中出现竞争饿死，原始读取质量完全下溢为 0，且随训练加剧。

---

## 2. 准入与探针

| 项 | 结果 |
|---|---|
| pytest（`test_slot_read_normalization` / `test_full_slot_diagnostics` / `test_pilot_split` / `test_model`） | **53 passed，exit 0** |
| `--check-prepared` | 通过，`model_executed=false`，`identity_sha256=8e6839d7f923706859af2deb876f26b401ae6065b3834da8219e49fee0654135` |
| prepared manifest SHA256 | `8fc4ccc92668da836aea79f253384bc70040542d6b02fbdb4a8e592e42454091` |
| 探针配置 | `configs/common_private_full_joint_readfix_probe_v2.json`（全量 304/76、4096 patch、batch2、原槽数 p/g 各 2+2、width64、3 次迭代，**预算仅 role2+task2**） |
| 探针结果 | 退出码 0，`status=completed`，恰好 2 role + 2 task 日志 |

探针门槛逐项通过：

1. 退出码 0 且 `status=completed` ✅
2. 恰好 2 个 role + 2 个 task epoch 日志 ✅
3. 全部 `train_loss` / `NLL` / `role_target_MSE` / 梯度均值有限 ✅
4. 归一化策略记录为 `masked_add_eps_l1_v2`（eps=1e-8，仅加在真实 token 上）✅
5. 每个 epoch 的**训练与验证**都有加 eps 前 `raw_reading_mass` 统计 ✅
6. padding 零读取、极低质量/完全下溢回归由新增测试覆盖并通过 ✅
7. `checkpoint_reload_verified` / `checkpoint_optimizer_and_generator_reload_verified` / `reference_frozen` 均为 `true` ✅
8. 无 `early_stop` ✅

低读取质量计数**非零**，按协议作为需保留的观察记录，**未作为停机条件**。

---

## 3. 两臂正式结果（readfix v2）

协议：`blca` fold0 / seed3 / `5fold_uni2h` / 304 训练 + 76 验证 / 4096 patch / batch2 / 槽 p,g 各 2 common + 2 residual / width64 / 3 次迭代 / role12 / task≤40 / patience10 / readout_bound 2.0 / task_lr 0.001 / role_lr 0.003。

| 臂 | best epoch | best C-index | best NLL | last epoch | last C-index | last NLL | task 轮数 | C 均值 | C 标准差 |
|---|---|---|---|---|---|---|---|---|---|
| joint | 25 | **0.5803** | 0.8259 | 35 | 0.5590 | 0.8647 | 35 | 0.5074 | 0.0378 |
| frozen | 15 | **0.5336** | 0.7966 | 25 | 0.5242 | 0.8098 | 25 | 0.5020 | 0.0414 |

其他状态：

| 项 | joint | frozen |
|---|---|---|
| 退出码 | 0 | 0 |
| `status` | completed | completed |
| optimizer steps (role/task) | 1824 / 5320 | 1824 / 3800 |
| `early_stop` | patience | patience |
| 峰值 CUDA 分配字节 | 482,392,576 | 474,513,408 |
| 配置 SHA256 | `a592fd23372aaa46c5e189ac49dbe316a901931c8762c939f62dbcff0320a685` | `6fbf49c3248f4a6ecfc656a3bc13ff08410d65d29c887b646ee8453baef847bc` |

两臂均通过 best/last checkpoint 重载评估一致性（C_index 相等、NLL `abs_tol=1e-12` 相等）、optimizer state 与 train generator state 重载一致性、`reference_frozen=true`。

**确定性验证**：joint 的 role epoch1 `train_loss=0.413880` 与探针 role epoch1 **逐位一致**，确认同初始化同数值轨迹可复现。

### 完整 task 曲线（joint，C-index）

```
ep 1 0.5021  ep 2 0.4588  ep 3 0.4410  ep 4 0.3815  ep 5 0.5123
ep 6 0.5047  ep 7 0.5047  ep 8 0.4656  ep 9 0.5098  ep10 0.5055
ep11 0.4936  ep12 0.5166  ep13 0.4877  ep14 0.5302  ep15 0.5200
ep16 0.5174  ep17 0.5123  ep18 0.5072  ep19 0.5234  ep20 0.5191
ep21 0.5234  ep22 0.5149  ep23 0.5446  ep24 0.4613  ep25 0.5803 <- best
ep26 0.5259  ep27 0.5327  ep28 0.5344  ep29 0.4936  ep30 0.4928
ep31 0.5709  ep32 0.4528  ep33 0.5429  ep34 0.5166  ep35 0.5590 <- last
```

### 完整 task 曲线（frozen，C-index）

```
ep 1 0.4104  ep 2 0.4095  ep 3 0.4197  ep 4 0.4469  ep 5 0.4503
ep 6 0.4673  ep 7 0.4919  ep 8 0.5072  ep 9 0.5132  ep10 0.5191
ep11 0.5259  ep12 0.5327  ep13 0.5327  ep14 0.5319  ep15 0.5336 <- best
ep16 0.5293  ep17 0.5285  ep18 0.5268  ep19 0.5268  ep20 0.5234
ep21 0.5251  ep22 0.5242  ep23 0.5242  ep24 0.5251  ep25 0.5242 <- last
```

frozen 曲线明显更平稳（ep11 之后在 0.5234~0.5336 窄带内），joint 呈锯齿状（0.3815~0.5803）。

---

## 4. 机制诊断（本次最有价值的产出，独立于分数）

### 4.1 WSI（`p`）侧共同槽竞争饿死

joint 的加 eps 前原始读取质量统计（train 侧 3648 观测 = 304 患者 × 3 迭代 × 4 槽；val 侧 456 观测 = 76 × 3 × 2）：

| epoch | train p 最小质量 | train p 低于阈值计数 | val p_common 低于阈值/456 | val p_common 最小质量 | C-index |
|---|---|---|---|---|---|
| role 1 | 1.92e-10 | 3 | 0 | 3.25e-02 | — |
| role 4 | **0.0** | 40 | 7 | **0.0** | — |
| role 8 | 0.0 | 90 | 19 | 0.0 | — |
| role 12 | 0.0 | 53 | 8 | 0.0 | — |
| task 1 | 0.0 | 59 | 11 | 0.0 | 0.5021 |
| task 26 | 0.0 | 79 | 18 | 0.0 | 0.5259 |
| **task 27** | 0.0 | **338** | **88** | 0.0 | 0.5327 |
| task 28 | 0.0 | 392 | 95 | 0.0 | 0.5344 |
| task 31 | 0.0 | 342 | 86 | 0.0 | 0.5709 |
| task 35 | 0.0 | 340 | 82 | 0.0 | 0.5590 |

要点：

- WSI 侧最小原始读取质量在 **role 4 之后稳定为 0.0**（完全下溢）。
- 低于阈值槽数在 **task 27 轮发生突跳**：train 由 ~50 升至 ~340/3648（约 9%），val `p_common` 由 18 升至 88/456（约 19%）。
- **塌缩吃的是 `p_common`（共同槽），不是冗余槽**。`p_residual` 始终维持在 0~12/456 的低位。
- 槽范数仍然健康：`p_common` mean_norm 约 6.14，**非零有限** → 属**读取质量塌缩**，不是数值 NaN、不是死槽。
- 角色表示层面 `p_common` 的 `centered_patient_effective_rank` 约 1.73~1.96，`g_common` 约 1.73~1.92 —— 共同侧秩本就偏低，与饿死方向一致。

### 4.2 Omics（`g`）侧全程健康

最小原始读取质量稳定在 32~46，低于阈值计数**恒为 0**（train 与 val 两侧、全部 12 role + 全部 task epoch）。

### 4.3 frozen 臂的对照意义

frozen 臂在 task 阶段冻结槽后，`p_common` 低于阈值计数**恒定为 8/456 全程不变** —— 槽**不再继续恶化**，但也**没有被救回**。这说明恶化来自 task 阶段槽与任务的联合优化，而 role 阶段结束时已存在固定的一批饿死槽。

### 4.4 与同队列旧 pilot 的对照

| | 旧 pilot | readfix v2 joint |
|---|---|---|
| C-index | **0.5658** | 0.5803 |
| 槽配置 | p/g 各 **1+1** | p/g 各 **2+2** |
| width / 迭代 | 32 / 2 | 64 / 3 |
| 预算 | role3 + task3 | role12 + task≤40 |

旧 pilot 收据：`/data1/slotspe_split_guard_0934bb6/local_evidence/common_private_pilot_blca_f0_s3_splitfixed_20261110_01/receipt.json`
（同队列 `5fold_uni2h`、同 304/76、同 seed3、`test_used=false`、`roles_frozen_during_task=true`、`reference_frozen=true`）

**加倍槽数、宽度、迭代，预算放大约 13 倍，C-index 差异仅 0.015 —— 远小于 76 人验证集噪声尺度 0.08。**

---

## 5. 必须遵守的限制（请勿在下游表述中越界）

1. **本协议没有独立测试证据**。`5fold_uni2h_outer_test_seed3/blca/fold_0.csv` 的 outer test 与 `5fold_uni2h` fold0 的 76 位验证患者**完全是同一集合**。两臂 receipt 均记 `test_used=false`。
2. **best 与 last 均为开发验证结果**，不能与旧队列 S0 或旧 3+3 末轮数字直接比较并称提升。
3. **readfix v2 改变了数值轨迹**，所有结果必须标为 readfix v2，不得与旧协议混称。
4. **best 是乐观偏置点**：joint best=0.5803 是 35 个 epoch 的最大值，均值仅 0.5074（sd 0.0378）。76 人验证集置换零假设 SD 约 0.08，best 与均值相差约 1 个噪声标准差。**报告中枢请用 mean±sd，不要单报 max。**
5. **readfix v2 只保证归一化合法，不解决竞争饿死**。不得据此声称低质量槽已恢复有用信息。
6. **单折单 seed 无法分辨槽机制是否有增益**：与同队列旧 pilot 的差异落在噪声内，既不支持"更好"也不支持"更差"。
7. 本轮**未评估患者预测损害或训练修正收益**，不含因果或生物学解释。角色名不蕴含已识别信息因子。
8. 本轮**未新增**槽平衡/熵约束/重初始化机制（受 AGENTS.md「审计线内不新增机制」约束），因此**未尝试治理**该塌缩。

---

## 6. 产物路径与哈希

### 6.1 输出目录（均在 `.gitignore` 忽略的 `local_evidence/` 下，未 stage）

```
local_evidence/full_slot_joint_readfix_probe_v2_20261111_01/   # 探针
local_evidence/full_slot_joint_readfix_v2_20261111_01/         # 正式 joint
local_evidence/full_slot_frozen_readfix_v2_20261111_01/        # 正式 frozen
local_evidence/full_slot_logs_readfix_v2_20261111_01/          # 日志
```

### 6.2 产物 SHA256

joint：

```
receipt.json   6723f1fdd730ef16c3a851995f93c3d5de31a5df786f7ced4570f372df226808
role_last.pt   f9d418466d78eb235086d1b47ef53c6acd0d2e46fdee16ae3a99a3a6ee77e237
task_best.pt   8640238cd74d368faac3168bb1939348ab6beb3f01a6161af96fc28a382675f7
task_last.pt   221a75f487626c85c756c66a1ebcb83185a5979108e17a0c52b7f2d259414dcb
```

frozen：

```
receipt.json   bbafe3788656332712dd9b470a2853b0038a9b5d6af007f6d84739d8893e474b
role_last.pt   de497fdd6b8cd3826ff227948b777a04f36c3e05dc996ca490011cc955650320
task_best.pt   1595e577295c6a68e49cbeb73afc405432eed9aea3499b735a80fb2318b1f210
task_last.pt   276a062d12174c40f279a5b9e13adc9f8a2a0f4b06575c503ac878b3758b21fb
```

探针 `local_evidence/full_slot_joint_readfix_probe_v2_20261111_01/`：

```
receipt.json   71e8dbef6fa3e8a472644f085a8a01da41c458651ca0d3a40fffe39c6f025882
role_last.pt   b36f5ad484622e327f29e69949966423cb53aabc10a3661f24a23569eb72ac9b
task_best.pt   c2eecea0bd40c48f82b9f87b966e26a3b28885a04e178721f56bb9ba877d5204
task_last.pt   15882ee2469eaa0a4be0a896c1b30a66a730378255a553b84a001b2462d35202
```

探针与 frozen 的全部 epoch 指标、逐迭代/逐槽读取质量计数、梯度均值、角色诊断（有效秩/范数/跨患者方差）、槽重叠统计、ablation 统计，均已完整写入本机 `docs/PROGRESS.json` 的 `latest_readfix_v2_full_slot_joint_frozen_completed` 与 `latest_readfix_v2_probe_passed` 条目（含完整 task 曲线）。

### 6.3 源码 SHA256（两臂一致，readfix v2）

```
research_models/common_private_synergy/components/slots.py           6aca9fca71d4d752995b17480cce40720851dc8871ef6d8a18647d35fb257218
research_models/common_private_synergy/full_slot_diagnostics.py      d8ea86e1f35089414e15622c8ac84bf4290b913ec2cadc4fb1909e2688676e12
research_models/common_private_synergy/full_slot_epoch.py             c61ecf0d5507e2ea1da0ed996b4d0165b2359d4d125b5115d5d47b2be9abcbaa
research_models/common_private_synergy/full_slot_engine.py            0b12d831cc2e82d731b8bc6d31b4b837918fcf9e275157312a537436b8b03c8d
research_models/common_private_synergy/components/contracts.py        437f60d9667351d89d174076e5fd3d50abba88fb8d5ae59a196ac6cb70f08cee
research_models/common_private_synergy/components/role_model.py       fb97256d7fabb3a839aeb890bbb64c3b2166a856562b55f26f08df547ba28fe2
research_models/common_private_synergy/components/role_reference.py   d5ad449251bc8fd9c8eaa4c50a2f2297b7fd83172708d38c789ec5afa4c43da4
research_models/common_private_synergy/components/losses.py           9cf45cda80c4564ee7560ac51ce5927690cb8da7ebbc53e3fe56b67ecae455c2
research_models/common_private_synergy/patient_pilot.py               12791e5fa3f83690ae98aeb83a31f600885088748f31d78840f3a4cdba8985e3
research_models/common_private_synergy/patient_pilot_engine.py       ddaf57da63d0278b2f2d40a6af89e7e7d696a69fb6cc241bfe4f4a4ef6922b7b
research_models/common_private_synergy/pilot_split.py                 f8a8395576eed476d12cda2f3a77b49049368a187918a47e521a8e2cb1520df1
scripts/run_common_private_full_slot.py                             786bc9402d663273a0df79f105c1ea4adf43cbc499f817eb46bfaa76df3ac1d7
slotspe_proof/patient.py                                            8390698ba93e4b9a7635e02b255dfb9f8a1dd35542a08124470a402ed7b97a59
```

`third_party/SlotSPE/` 未改动（原始可执行源码保持 byte-for-byte）。

---

## 7. 建议的下一步（等待用户拍板，尚未执行）

| 方向 | 内容 | 代价 / 前提 |
|---|---|---|
| **A 机制优先** | 治 `p_common` 竞争饿死（槽平衡 / 熵约束 / 死槽重初始化） | 需用户**明确豁免** AGENTS.md「审计线内不新增机制」约束，或置于审计线外的独立研究线 |
| **B 统计优先** | 5 折 × 多 seed，报 mean±sd 而非 max，把噪声从 ~0.08 压到可分辨后再与简化基线比较 | 算力 5×N |
| **C 成本最低** | 同队列同协议的**去槽机制消融**（单池化基线）；若 full-slot 不胜出即直接证伪当前槽机制增益 | 1~2 臂即可 |

在用户指定前**不启动**新的训练、推理或患者干预。

---

## 8. 本轮保持只读、未改动的位置

- `/data1/slotspe_full_slot_20261111_01` —— edc2313 失败 checkout，含原 joint/frozen 失败产物与日志，**原样保留**
- `/data1/SlotSPE` —— 主仓库，**未改动**
- `/data1/slotspe_split_guard_0934bb6` —— split guard 仓库，**未改动**
- `/data1/slotspe_split_guard_0934bb6/local_evidence/common_private_pilot_prepare_splitfixed_20261110_01` —— 缓存，**只读**
- `third_party/SlotSPE/` —— 原始源码，**未改动**
