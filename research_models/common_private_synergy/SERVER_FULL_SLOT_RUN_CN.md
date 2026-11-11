# 5090 完整角色网络：服务器执行说明

本说明交给已有服务器环境的协作者或 AI。用户已授权在原有数据和服务器上执行下述有限研究，不需再次询问同一许可。保持原始 SlotSPE、原有缓存、旧工作树和历史结果；新实验单独隔离，输出仅在 ignored `local_evidence/`。不要上传患者标识、特征、权重、预测或逐患者数组。不要更换数据、自动安装依赖、搜索槽比例或加入运输模块。

## 研究问题与范围

比较角色预训练后冻结槽（frozen）和任务参与塑造槽且保留角色损失（joint）。两侧各2共有+2剩余槽、width64、3次循环、有限8维 PCA/CCA 目标、共有秩2、有界加性及中心化交互生存头；这是独立经验角色模型，并非原版 SlotSPE 所有模块的逐字复现。BLCA DSS / UNI2-h / `5fold_uni2h` / fold0 / seed3，4096 patch、batch2；角色12轮，任务最多40轮、patience 10。两臂相同的数据、种子、mask、参数容量、预算上限和选模规则；早停可能导致实际轮数不同。

验证 seed 固定为19003，每个任务 epoch 计算 C-index 与 NLL。按 C-index 最大、精确同分 NLL 最小、再取较早 epoch 选择 best；另存 last。记录每轮角色误差、槽读取/表示/梯度和槽移除。这个76人验证集已经被查看，所有 best/last 均是**开发结果**，不能称作独立测试、泛化证明或患者收益。每角色多槽的几何改善或槽消融本身不证明坍塌已解决。不要自行增加分支、修改超参数、重试失败或用 test 调参。

## 0. 隔离与环境检查

以用户提供的分支名和完整提交 SHA 为准。现有验收快照 `/data1/slotspe_split_guard_0934bb6` 和它的缓存原地只读；从现有仓库创建独立目录并获取指定提交，不能覆盖已有目录。下例中的 `FULL_SLOT_COMMIT` 和 `FULL_SLOT_BRANCH` 由交付消息填入；先核对来源，不使用其他提交或强制覆盖：

```bash
set -euo pipefail
FULL_SLOT_COMMIT='<交付消息中的完整 commit SHA>'
FULL_SLOT_BRANCH='<交付消息中的分支名>'
FULL_SLOT_RUN=/data1/slotspe_full_slot_20261111_01
test ! -e "$FULL_SLOT_RUN"
git clone --no-local --no-checkout /data1/slotspe_split_guard_0934bb6 "$FULL_SLOT_RUN"
cd "$FULL_SLOT_RUN"
git fetch https://github.com/LDT-von/slot-attention.git "refs/heads/$FULL_SLOT_BRANCH"
test "$(git rev-parse FETCH_HEAD)" = "$FULL_SLOT_COMMIT"
git checkout --detach "$FULL_SLOT_COMMIT"
test "$(git rev-parse HEAD)" = "$FULL_SLOT_COMMIT"
git status --short --branch
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
```

查找并使用此前已成功运行该 pilot、支持 RTX 5090 的现有 Python 环境；不要自动安装或升级。核对环境 `python` 的版本、torch 版本、CUDA 是否可用及设备名，确认空闲显存。系统 `python3` 可用不代表支持 CUDA。不要显示凭据、私钥或患者路径之外的隐私内容。

```bash
python -c 'import sys,torch; print(sys.version); print("torch",torch.__version__,"CUDA",torch.version.cuda,"available",torch.cuda.is_available()); print("GPUs",[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
python -B scripts/run_common_private_patient_pilot.py --doctor
python -B scripts/run_common_private_full_slot.py --config configs/common_private_full_joint_v1.json
python -B scripts/run_common_private_full_slot.py --config configs/common_private_full_frozen_v1.json
```

如果环境或显存不满足，停止并回报，不自行修改方案。

## 1. 人工数据测试与真实缓存准入

```bash
python -B -m pytest -q --confcutdir=research_models/common_private_synergy \
  research_models/common_private_synergy/deferred_checks/test_full_slot_diagnostics.py \
  research_models/common_private_synergy/deferred_checks/test_pilot_split.py \
  research_models/common_private_synergy/tests/test_model.py

PREPARED=/data1/slotspe_split_guard_0934bb6/local_evidence/common_private_pilot_prepare_splitfixed_20261110_01
python -B scripts/run_common_private_patient_pilot.py --check-prepared --prepared-root "$PREPARED"
```

Linux 上符号链接测试须实际运行；发生跳过或失败要如实报告并停止。`--check-prepared` 只是元数据准入，不读特征；真正 `--run` 还会验证原版来源、CSV 及缓存哈希。旧缓存身份不符必须拒绝，不能改 manifest 冒充新缓存。测试生成输入；不代表患者性能。

## 2. 先联合臂，再冻结臂

仅当上述检查均通过且两次输出路径均不存在时执行。使用同一块空闲GPU，顺序执行，不并发；保存终端日志及进程真实退出码。建议先 `mkdir -p local_evidence/full_slot_logs_20261111_01`，不要预先创建以下两个输出目录。执行前记录当前时间、提交、GPU、依赖和配置 SHA256，执行后原样保留所有产物。若任一步失败，停止并保留失败回执，不覆盖、不自动重试。

```bash
sha256sum configs/common_private_full_joint_v1.json configs/common_private_full_frozen_v1.json
test ! -e local_evidence/full_slot_joint_20261111_01
test ! -e local_evidence/full_slot_frozen_20261111_01
mkdir -p local_evidence/full_slot_logs_20261111_01
set -o pipefail
python -B scripts/run_common_private_full_slot.py --run \
  --config configs/common_private_full_joint_v1.json --prepared-root "$PREPARED" \
  --output-root local_evidence/full_slot_joint_20261111_01 \
  2>&1 | tee local_evidence/full_slot_logs_20261111_01/joint.log
python -B scripts/run_common_private_full_slot.py --run \
  --config configs/common_private_full_frozen_v1.json --prepared-root "$PREPARED" \
  --output-root local_evidence/full_slot_frozen_20261111_01 \
  2>&1 | tee local_evidence/full_slot_logs_20261111_01/frozen.log
```

代码会为每轮保存汇总和完整 best/last（模型、角色参照、适配器、分箱、优化器及随机状态），并重载验证；目前没有自动断点续训入口。输出停留服务器。只回传去标识汇总：两臂状态/退出码、完整配置和代码哈希、训练/验证人数、所有 epoch 指标与诊断、best 和 last 的 epoch/C-index/NLL、参数数与显存峰值、每臂 receipt 和 checkpoint 的 SHA256、测试及跳过信息。训练结束必须解释 joint 的任务梯度、角色误差、槽余弦/读取重叠和删槽结果；不可只回最高分。失败时报告失败环节与不含患者信息的错误，敏感 traceback 留在私有输出目录。

未确认真正未参与选模的独立测试集前，不得报告独立测试分数。不能将历史 S0 或旧 3+3 末轮结果当作同协议匹配对照。
