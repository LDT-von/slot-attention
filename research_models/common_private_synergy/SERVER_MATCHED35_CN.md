# 共同起点两臂各 35 轮：服务器 AI 执行指令

用户已批准本协议的实现、合成检查、原分支交付和后续服务器执行交接。使用本次交付消息指定的完整提交，先完成下列检查，再顺序运行单个有限协议；无需重新申请已有研究许可。所有真实患者数据和模型产物留在服务器 ignored `local_evidence/`。本说明定义执行，不表示服务器已经运行。

## 固定问题与运行预算

问题：持续角色拟合、优化器初始化、采样和容量相同时，生存梯度进入角色网络是否额外加剧低质量读取？

- BLCA DSS / UNI2-h / `5fold_uni2h` / fold0 / seed3；现有 304 训练与 76 开发验证，4096 patch、batch2；训练仅从真实 patch 无放回抽样，短 bag 补零并 mask。
- 每模态 2 common + 2 residual 槽，width64、3 次循环、target8/common_rank2、hidden32、4 个 hazard logits、readout_bound2；保留 `masked_add_eps_l1_v2`、真实 token 上 epsilon=1e-8。新增观测不改变读取规则。
- **从新初始化只运行一次 12 轮共享角色预训练**，role_lr=0.003。角色参照、适配器与生存分箱只在训练侧拟合；生成一个共同起点，包含完整模型/任务头、固定参照、适配器、分箱、训练采样生成器及全局随机状态。
- 从该共同起点分叉，先 joint，再 continued_role_only，各固定 35 个 task epoch；task_lr=0.001、角色系数1。每臂重新创建同样参数集合与超参数的 Adam，初始状态为空，不继承预训练角色优化器的动量。恢复共同起点的采样与全局随机状态后才开始各臂；记录和核对初始状态及训练输入抽样轨迹的哈希。
- joint 的编码器、角色池化和解码接收角色与生存梯度；continued_role_only 在解码角色进入任务头时 detach，编码器/池化/解码仍接受角色损失，任务头正常接收生存梯度。对照臂不是 frozen，也不是停止训练任务头。
- 关闭基于验证分数的 patience；两臂各 35 轮，**本次新执行总计 12+35+35=82 轮**。历史已完成的 84 轮属于另一协议，不能计入本轮，也不从历史 checkpoint 续训。不因分数低、读取质量低或某臂先变好而改变预算。
- 每轮固定验证 seed19003 和同一 ordered validation bags；best 按最大 C-index、精确同分较低 NLL、再较早 epoch，另保留 last。主要机制比较采用固定末轮和全部轨迹；best 是开发选模结果，不是独立测试。

兼容说明：远端 `5e6ef83` 已提供旧 `full_slot_research_v1` 下同名的 continued_role_only，它冻结任务头；该既有模式继续保留。本协议必须使用新入口与 `full_slot_matched35_v1`，其 continued_role_only 正常训练任务头，仅截断到角色网络的生存梯度。两种行为会在 receipt 的 `task_gradient_policy` 明示，不能混用旧单臂配置。

## 1. 固定提交与隔离目录

在已有支持 5090 且此前通过检查的 Python 环境中执行；缺依赖或环境不兼容则保留输出并回报，不自动安装或升级。下方唯一需要从交付消息填入的是 `MATCHED35_COMMIT`。如目标目录已经存在，先核对是否正在运行本协议；不要覆盖、删除或另起重复训练。

```bash
set -euo pipefail
MATCHED35_COMMIT='<本次交付消息中的完整提交 SHA>'
MATCHED35_BRANCH='codex/full-slot-research-20261111'
MATCHED35_CHECKOUT='/data1/slotspe_matched35_20261113_01'
test ! -e "$MATCHED35_CHECKOUT"
git clone --no-checkout --single-branch --branch "$MATCHED35_BRANCH" \
  https://github.com/LDT-von/slot-attention.git "$MATCHED35_CHECKOUT"
cd "$MATCHED35_CHECKOUT"
git cat-file -e "$MATCHED35_COMMIT^{commit}"
git checkout --detach "$MATCHED35_COMMIT"
test "$(git rev-parse HEAD)" = "$MATCHED35_COMMIT"
test -z "$(git status --porcelain)"
mkdir -p local_evidence/matched35_logs_20261113_01
LOGDIR=local_evidence/matched35_logs_20261113_01
{
  date -Is
  git rev-parse HEAD
  sha256sum configs/common_private_matched35_v1.json
  nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader
} | tee "$LOGDIR/preflight.txt"
```

核对物理 GPU1 当前空闲且确为计划设备；历史 UUID 为 `GPU-ae871418-cdb0-08a8-e680-ad1b87af08e7`，以当前实际查询为准。若存在其他任务占用，等待可用或回报，不终止别人的进程，不更换设备/配置继续。之后设置唯一可见设备，配置使用逻辑 `cuda:0`：

```bash
export CUDA_VISIBLE_DEVICES=1
python -c 'import sys,torch; print(sys.version); print("torch",torch.__version__,"CUDA",torch.version.cuda,"available",torch.cuda.is_available()); assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; print("logical cuda:0",torch.cuda.get_device_name(0))' \
  | tee "$LOGDIR/environment.txt"
```

## 2. 合成回归与缓存准入

以下测试只生成虚构输入。保留完整测试输出、通过/失败/跳过数量；Linux 上 split 符号链接检查须实际运行。任一检查失败就停止，不通过改超参数或删断言绕过。

```bash
python -B -m pytest -q --confcutdir=research_models/common_private_synergy \
  research_models/common_private_synergy/deferred_checks/test_matched35.py \
  research_models/common_private_synergy/deferred_checks/test_matched_read_diagnostics.py \
  research_models/common_private_synergy/deferred_checks/test_slot_read_normalization.py \
  research_models/common_private_synergy/deferred_checks/test_full_slot_diagnostics.py \
  research_models/common_private_synergy/deferred_checks/test_pilot_split.py \
  research_models/common_private_synergy/tests/test_model.py \
  2>&1 | tee "$LOGDIR/synthetic_tests.log"
PREPARED=/data1/slotspe_split_guard_0934bb6/local_evidence/common_private_pilot_prepare_splitfixed_20261110_01
python -B scripts/run_common_private_patient_pilot.py --check-prepared --prepared-root "$PREPARED" \
  2>&1 | tee "$LOGDIR/cache_admission.log"
python -B scripts/run_common_private_matched35.py --config configs/common_private_matched35_v1.json \
  2>&1 | tee "$LOGDIR/plan.json"
```

确认 plan 不执行模型，且共同角色阶段12轮、两任务臂各35轮、早停关闭、304/76与固定配置吻合。`--run` 会重新检查缓存/划分身份和源文件哈希，不能修改既有 manifest 使旧缓存通过。原缓存与原版 `third_party/SlotSPE/` 保持原样。

## 3. 一次执行完整配对协议

输出根目录必须不存在。整个入口管理共享预训练和两臂顺序运行，不能把旧单臂入口执行两次代替共同起点。

```bash
OUT=local_evidence/full_slot_matched35_20261113_01
test ! -e "$OUT"
set +e
python -B scripts/run_common_private_matched35.py --run \
  --config configs/common_private_matched35_v1.json \
  --prepared-root "$PREPARED" --output-root "$OUT" \
  2>&1 | tee "$LOGDIR/run.log"
MATCHED35_PIPESTATUS=("${PIPESTATUS[@]}")
MATCHED35_EXIT=${MATCHED35_PIPESTATUS[0]}
set -e
printf '%s\n' "$MATCHED35_EXIT" > "$LOGDIR/exit_code.txt"
if [ "$MATCHED35_EXIT" -ne 0 ]; then
  printf 'Matched35 failed; preserve all private artifacts and report the failure.\n'
  exit "$MATCHED35_EXIT"
fi
if [ "${MATCHED35_PIPESTATUS[1]}" -ne 0 ]; then
  printf 'Logging failed; verify private run artifacts before claiming completion.\n'
  exit "${MATCHED35_PIPESTATUS[1]}"
fi
```

失败产物、回执和 traceback 保留在私有目录；不自动重试、不从失败点续训、不修改预算或损失系数。如果日志显示正在训练，不启动第二份同协议。输出结构为 `$OUT/shared_role/`、`$OUT/joint/`、`$OUT/continued_role_only/`，配对总回执为 `$OUT/paired_receipt.json`。任务结束后记录退出码、总耗时、GPU 映射、配置/源文件/各 receipt/checkpoint SHA256。

## 4. 完成验收与回传

完整成功应同时满足以下条件；出现不符即报告具体缺口，不把部分运行记为已完成：

1. 恰好一个共享 role12 阶段与 joint35、continued_role_only35，共82轮；两臂 last 均为 task epoch35，未触发分数早停。
2. 两臂共同起点完整模型（含任务头）、参照/适配器/分箱及 RNG 哈希匹配；同 epoch 的训练顺序和 patch 抽样轨迹匹配；任务优化器均从空状态、相同参数组初始化。只对这次内部对照作机制解释，历史 joint/frozen 仅为背景。
3. 逐轮 train/validation 保留原始读取质量和低于1e-8计数/分母，新增 Q/K 范数、真实 token 的竞争 logit 差、稳定 log 质量、epsilon 均匀混合比例，并按模态×角色×槽×迭代汇总。低质量数可以非零；它不是数值失败门槛，也不是“死槽”定义。
4. 同时保存角色 MSE、raw slot 与 decoded role 诊断、读取 overlap、角色/生存/合并梯度；continued_role_only 的角色网络生存梯度通路被截断，角色梯度及任务头训练仍存在。新 `combined` 取清空诊断反向残留后的正式训练目标梯度，历史 `combined` 口径限制见追加勘误；不得将新旧日志混作同口径趋势。
5. 任务验证保留逐槽删除后的 NLL、带符号 `ΔNLL=NLL_deleted−NLL_full` 和原有绝对 logit 变化。删除在同一完整模型输出与固定头上进行，保留原角色池化删除语义；它会改变池化并可能离开训练表示分布，不能称条件 CPI 或独有生物信息贡献。
6. best/last 可重载复现对应验证结果、参考保持固定、源码与缓存身份保持一致；所有损失与所需诊断有限。CUDA 逐位重现若未实际测试，不声称已证明。


完成后可直接运行以下只读验收，输出仅包含状态和轮数。失败时保留原产物并报告，不自动重新训练：

```bash
python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
pair = read(root / 'paired_receipt.json')
assert pair['status'] == 'completed' and pair['total_epochs'] == 82
for key in ('common_start_verified', 'matched_input_draws_verified', 'task_optimizer_reset_verified'):
    assert pair[key] is True, key
shared = read(root / 'shared_role/receipt.json')
assert shared['status'] == 'completed' and len(shared['epoch_logs']) == 12
for arm in ('joint', 'continued_role_only'):
    receipt = read(root / arm / 'receipt.json')
    assert receipt['status'] == 'completed' and len(receipt['epoch_logs']) == 35
    assert receipt['last']['epoch'] == 35 and 'early_stop' not in receipt
    for key in ('reference_frozen', 'checkpoint_reload_verified', 'checkpoint_optimizer_and_generator_reload_verified'):
        assert receipt[key] is True, (arm, key)
    for epoch in receipt['epoch_logs']:
        assert epoch['phase'] == 'task'
        assert epoch['train_read_diagnostics'] and epoch['validation']['read_diagnostics']
        assert epoch['validation']['ablation_NLL'] and epoch['validation']['ablation_delta_NLL']
print(json.dumps({'status':'verified', 'shared_role_epochs':12, 'task_epochs_per_arm':35, 'total_epochs':82}))
PY
```

只在 `docs/PROGRESS.json` 追加六项完整记录（route_task、completed_work、paths_commit、checks_actual_execution_raw_receipt、gaps_reasons、next_action_user_inputs），保留全部旧条目。回传状态/退出码、82轮完整聚合指标与诊断、best/last、匹配检查和文件哈希；患者标识、单患者交叉表、原始数组、预测、权重不推送。若工具还不能安全导出某项聚合，列为缺口，不补造数据。

判读预先固定：联合臂比角色梯度臂额外恶化，才支持该配置下生存梯度的额外作用；两臂共同恶化时保留持续角色拟合及共同优化条件的解释；差异不明显只能称本协议未观察到清楚额外效应。35个 epoch 的相关轨迹不当独立重复，不把轨迹 SD 当泛化误差，不把验证最优值当独立测试。新诊断不用于选槽、调系数或延长训练。

历史报告的“14个源文件”和“置换SD≈0.08”采用 [追加勘误](READFIX_V2_ERRATA_20261113_CN.md)。保留历史报告，不把本轮诊断伪写回过去日志。实现与合成检查记录在交付方本地 `docs/PROGRESS.json` 的 `full_slot_matched35_protocol_delivery_20261113`。远端 PROGRESS 的历史版本保持原样，本轮未追加该实现记录；服务器执行后应将真实回执另作追加。
