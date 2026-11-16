# 先复现82轮导出，再执行lambda=0.5的35轮有限协议

用户已明确授权此顺序的实现、验收和服务器交接。必须使用交付消息中的完整提交SHA；本说明不代表服务器已运行新条件。原实验、缓存和原版SlotSPE保留，患者产物只留ignored local_evidence。

## 固定协议

- BLCA/DSS/UNI2-h，fold0/seed3，原304训练/76开发验证、4096patch/batch2、width64、每模态2+2槽、3次迭代、target8/common_rank2、原生存分箱和参照。
- 不新增role预训练；从已完成matched35的同一个shared_role/role_last.pt恢复完整模型/任务头、适配器、参照、分箱与训练/全局/CUDA随机状态。核对实际checkpoint哈希。
- 只新增lambda=0.5一个条件，固定35个task epoch，task_lr=0.001、角色loss系数1、任务头生存梯度系数1，关闭分数早停。fresh Adam与旧两臂的参数组一致，丢弃role优化器状态。
- 在解码角色进入头之前使用 `stopgrad(r)+0.5*(r-stopgrad(r))`；没有给整个生存loss乘0.5。前向数值与任务头梯度保持，只有返回角色网络的生存梯度减半。Adam更新不保证减半，也不预设机制指标单调。
- lambda0/1复用已核验的同条件端点；新条件开始前校验共同状态，每轮校验输入draw哈希。保存逐轮验证、best/last和完整诊断，best仍按C-index最大、精确同分较低NLL、再较早epoch。
- 主要比较固定ep35与全35轮的p_common低质量计数/比例、epsilon均匀占比、log质量、竞争logit差、对应角色MSE及有符号逐槽删除NLL；best分数另列开发选模结果。ep27窗口不作新的预定主终点，不用新增诊断调参或延长训练。

## 1. 固定交付版本，先只读复现

下列命令在现有支持5090的环境中执行。先确认没有同协议正在运行；目标目录已存在则停止核查，不覆盖或另起重复训练。

```bash
set -euo pipefail
LAMBDA05_COMMIT='<交付消息中的完整SHA>'
CHECKOUT=/data1/slotspe_lambda05_20261116_01
test ! -e "$CHECKOUT"
git clone --no-checkout --single-branch --branch codex/full-slot-research-20261111 \
  https://github.com/LDT-von/slot-attention.git "$CHECKOUT"
cd "$CHECKOUT"
git checkout --detach "$LAMBDA05_COMMIT"
test "$(git rev-parse HEAD)" = "$LAMBDA05_COMMIT"
test -z "$(git status --porcelain)"
LOGDIR=local_evidence/lambda05_logs_20261116_01
mkdir -p "$LOGDIR"
OLD=/data1/slotspe_matched35_20261113_01
RUN="$OLD/local_evidence/full_slot_matched35_20261113_01"
ORIGINAL="$CHECKOUT/scripts/export_matched35_aggregates.py"
REVIEW=local_evidence/matched35_reproduced_20261116_01
python -S -B scripts/audit_matched35_aggregates.py \
  --aggregates research_models/common_private_synergy/exports/matched35_82epoch_aggregates.json \
  --output "$REVIEW/published_aggregate_audit.json" 2>&1 | tee "$LOGDIR/aggregate_audit.log"
python -S -B scripts/reproduce_matched35_export.py --run-root "$RUN" \
  --original-exporter "$ORIGINAL" --output "$REVIEW/aggregates.json" \
  --audit-output "$REVIEW/receipt_to_export_audit.json" \
  2>&1 | tee "$LOGDIR/reproduction.log"
```

必须退出0：原脚本SHA为bac2938e…、复现输出SHA为a8148305…，四份receipt哈希与完成记录一致、完整82轮及每个已导出字段与源receipt相符，audit的aggregate_numeric_crosscheck_complete为true。此步不导入torch、不加载checkpoint、不读患者cache、不使用GPU。失败保留新输出并停止，不改数据/原脚本来通过。原导出器不适用于任意未知输入；封装固定输入哈希并检查隐私。

将 [完整聚合追加更正](MATCHED35_REVIEW_ADDENDUM_20261116_CN.md) 纳入报告解释；旧报告和哈希不更改。保存原执行前Linux合成测试日志及符号链接用例结果；旧日志若缺失，列为缺口，下步测试只能称事后复查。

## 2. 合成检查与原缓存准入

CUDA合成检查也会使用GPU，必须先核对物理GPU1空闲并固定可见设备，避免检查触碰GPU0。先运行查询，核对后才进入下一代码块；GPU1被占用则停在这里等待/报告，不运行GPU检查。

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv,noheader | tee "$LOGDIR/gpu_before_tests.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader \
  | tee "$LOGDIR/processes_before_tests.txt"
```

确认物理GPU1可用后：

```bash
export CUDA_VISIBLE_DEVICES=1
python -S -B -m unittest discover -s tests -p 'test_matched35*.py' -v \
  2>&1 | tee "$LOGDIR/json_tests.log"
python -B -m pytest -q -rs --confcutdir=research_models/common_private_synergy \
  research_models/common_private_synergy/deferred_checks/test_lambda05_protocol.py \
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
python -S -B scripts/run_common_private_lambda05.py \
  --config configs/common_private_lambda05_v1.json | tee "$LOGDIR/plan.json"
```

仅现有环境，不自动安装或升级。所有合成测试只用虚构输入；Linux符号链接检查必须真实运行。plan默认不读患者、不执行模型，应显示new_role_epochs=0、total_new_epochs=35、角色梯度系数0.5与头系数1。任一必要检查失败停止，保留回执。

## 3. 物理GPU1空闲时执行唯一新条件

先核对GPU和进程；不干预其他任务，不更换协议设备继续。

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv,noheader | tee "$LOGDIR/gpu.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader \
  | tee "$LOGDIR/gpu_processes.txt"
```

确认物理GPU1可用后，映射为唯一可见逻辑cuda:0；然后执行：

```bash
export CUDA_VISIBLE_DEVICES=1
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))' \
  | tee "$LOGDIR/environment.txt"
OUT=local_evidence/full_slot_lambda05_20261116_01
test ! -e "$OUT"
set +e
python -B scripts/run_common_private_lambda05.py --run \
  --config configs/common_private_lambda05_v1.json --prepared-root "$PREPARED" \
  --run-root "$RUN" --original-exporter "$ORIGINAL" --aggregates "$REVIEW/aggregates.json" \
  --output-root "$OUT" 2>&1 | tee "$LOGDIR/run.log"
RUN_PIPESTATUS=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${RUN_PIPESTATUS[0]}" > "$LOGDIR/exit_code.txt"
test "${RUN_PIPESTATUS[0]}" -eq 0
test "${RUN_PIPESTATUS[1]}" -eq 0
```

入口会重新核验旧JSON链、原缓存/划分和源码、共享checkpoint的准确字节；第一步更新前验证共同状态，逐轮验证draw，完成后再核验来源。新模型只在复现链通过后加载；不要通过旧单臂入口或更改配置绕过。

## 4. 完成验收与回传

成功必须恰有35个task日志、零新role更新、5320个task优化步骤，last.epoch=35且无early_stop；endpoint_start_verified_before_updates和endpoint_start_and_draws_verified为true，两个系数为0.5/1，best/last重载与参照冻结检查通过。原共同起点和端点输入回执保持原字节。

回传：固定提交、配置/源码哈希、退出码/耗时/GPU映射、复现链audit、全部35轮去标识聚合（C-index/NLL、角色MSE、原始读取质量、八项读诊断、角色/生存/combined梯度、有符号删除NLL与原有logit变化）、best/last和checkpoint哈希。survival槽梯度日志表示已应用0.5路由后的分量，不误称未缩放分量。

患者ID、原始特征、逐患者数组、预测和权重留服务器。实际结果以六项结构追加PROGRESS，保留旧条目；不向连接器重新上传含历史隐私的完整PROGRESS或完整历史设计稿。回传安全聚合使用新增白名单工具，固定输入receipt哈希；不加载模型：

```bash
RECEIPT_SHA=$(sha256sum "$OUT/receipt.json" | cut -d ' ' -f 1)
python -S -B scripts/export_lambda05_aggregates.py --receipt "$OUT/receipt.json" \
  --expected-sha256 "$RECEIPT_SHA" --output local_evidence/lambda05_aggregates_20261116_01.json \
  2>&1 | tee "$LOGDIR/export.log"
```

失败保留产物，不自动重试/续训/调参。

本轮lambda是同一开发折的机制强度探查，不是独立重复或独立测试，也不预设能提高C-index。后续是否扩展由这一个有限条件的全部结果再决定。
