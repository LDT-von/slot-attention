# 读取归一化修复 v2：探针后完整两臂

决定为 A＋E，探针满足下列数值门槛后自动继续 F；不需再次询问同一执行授权。使用物理 GPU1，启动进程前设置 `CUDA_VISIBLE_DEVICES=1`，配置继续保持逻辑 `cuda:0`。先核对当前 GPU1 空闲及设备映射，不干预其他任务。

用户转来的服务器报告：edc2313 两臂在 role 第1轮152次更新后验证失败，task0、epoch_logs空、best空；41项Linux测试通过。报错为 attention 归一化断言，报告的预归一化质量约1.34e-16；这些服务器原始产物尚未由上游本地重新审计。该失败没有患者性能分数。原始失败目录、checkout及缓存保持只读。

## 修复含义

原角色编码器将 assignment 除以 clamp_min(1e-8) 分母，低质量槽的读取和因此小于1。新策略 `masked_add_eps_l1_v2` 为真实 token 加固定1e-8，再除以实际总和；padding权重仍严格为0。所有真实token均在时，数学上等于原版的加eps后L1归一化。没有新增槽平衡损失、熵约束或重初始化。该修复改变数值轨迹，也会改变历史 pilot 在新代码上的结果；不得混称为旧数值协议，不保证解决竞争饿死。槽范数非零不排除功能性死槽。

保留非法归一化、负数和非有限值的检查。新日志在每个角色/任务 epoch 同时记录训练和验证的 `raw_reading_mass`：加eps前全部迭代的最小值、低于1e-8的总计数/比例、逐迭代/槽计数。阈值表示旧clamp工作区间，不是死槽定义；**非零低质量计数本身不阻止探针通过**。后续槽重叠统计使用修复后合法分布，不能据此声称低质量槽已恢复有用信息。

## 执行

从交付消息指定的新提交创建新的隔离副本（可使用已建立的 fixprobe 沙箱，但不得混入任何未记录修改）。按 `SERVER_FULL_SLOT_RUN_CN.md` 核对源码、现有Python/CUDA环境和原缓存身份；原版SlotSPE不改。分别用新目录保留探针和正式运行，不能覆盖旧失败目录，也不能从失败模型续训。

```bash
export CUDA_VISIBLE_DEVICES=1
PREPARED=/data1/slotspe_split_guard_0934bb6/local_evidence/common_private_pilot_prepare_splitfixed_20261110_01
python -B -m pytest -q --confcutdir=research_models/common_private_synergy \
  research_models/common_private_synergy/deferred_checks/test_slot_read_normalization.py \
  research_models/common_private_synergy/deferred_checks/test_full_slot_diagnostics.py \
  research_models/common_private_synergy/deferred_checks/test_pilot_split.py \
  research_models/common_private_synergy/tests/test_model.py
python -B scripts/run_common_private_patient_pilot.py --check-prepared --prepared-root "$PREPARED"
python -B scripts/run_common_private_full_slot.py --run \
  --config configs/common_private_full_joint_readfix_probe_v2.json --prepared-root "$PREPARED" \
  --output-root local_evidence/full_slot_joint_readfix_probe_v2_20261111_01
```

这是全容量、全部304/76人的joint探针，仅将预算改为role2＋task2；不缩patch、人数、槽数或宽度，不用分数筛选修复。GPU物理/逻辑映射、代码/配置/manifest哈希和进程退出码均须登记。

探针门槛：退出0且status completed；恰有2个role和2个task日志；有限损失/角色误差与梯度，验证诊断合法产出且没有padding读取；训练和验证每轮都有预epsilon质量计数；task_best/last重载及参考冻结检查通过。零padding与极低质量/完全下溢回归须由上述测试覆盖。低质量计数允许非零，不要求探针C-index或NLL达到任何目标。门槛任一失败则保留并停止，报告不含患者信息的错误，不自动改参重试。

门槛全部通过后，记录探针已通过，再从相同初始化和修复源码分别从头执行：

```bash
python -B scripts/run_common_private_full_slot.py --run \
  --config configs/common_private_full_joint_v1.json --prepared-root "$PREPARED" \
  --output-root local_evidence/full_slot_joint_readfix_v2_20261111_01
python -B scripts/run_common_private_full_slot.py --run \
  --config configs/common_private_full_frozen_v1.json --prepared-root "$PREPARED" \
  --output-root local_evidence/full_slot_frozen_readfix_v2_20261111_01
```

正式预算仍是role12、task最多40、patience10、batch2、4096patch，joint后frozen顺序运行。每条命令须按原说明保存私有日志、真实退出码和回执；使用 `set -euo pipefail`，失败即停。探针已看的验证分数不用于改变方案。原配置文件值保持一致，但源码和归一化协议改变，报告必须明确v2。

立即把旧失败按六项写入当前隔离副本的 `docs/PROGRESS.json`，不等待确认措辞；之后记录探针开始/失败/通过和两臂开始/完成/失败。只回传去标识汇总、全部epoch指标/诊断和哈希，模型及逐患者数据留服务器。用户报告的outer test与原76人验证同集合，仍只按开发验证解释；不得与旧队列S0或旧末轮成绩直接比较并称提升。
