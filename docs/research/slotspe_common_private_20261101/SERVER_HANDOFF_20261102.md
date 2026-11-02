# 服务器接手：当前代码能运行什么

用户本次要求push并让服务器端接手训练，以节省本地对话额度。本交付不启动远端进程、不修改既有队列。独立分支：`codex/common-private-mechanism-review`；原`codex/common-private-survival`分支保持不动。

**可直接运行的是合成数据研究入口。当前新网络没有患者数据适配器，不能把以下命令当作患者训练命令。**不要继续使用远端旧prototype患者分数充当本方案效果，也不要把本方案混入SCM/SRO或anti-collapse队列。

## 固定版本与检出

最终提交SHA以本次push交付回复为准；服务器核对SHA后使用独立检出。不要reset、clean或覆盖已有工作区。

```bash
git fetch origin codex/common-private-mechanism-review
git worktree add --detach ../slotspe-cp-research origin/codex/common-private-mechanism-review
cd ../slotspe-cp-research
git rev-parse HEAD
python -B scripts/check_delivery_manifest.py --check --require-tracked
python -B scripts/check_xor_readout_capacity.py --plan
```

环境使用已有兼容的Python/PyTorch/NumPy，不在运行队列环境里盲目升级依赖。本机验收为Python/PyTorch现有环境、CPU两线程；未声称CUDA已验收。

## 可直接接手的训练：读出容量/固定头诊断

```bash
python -B -m pytest -q tests/test_crossmodal_shared_bottleneck.py tests/test_xor_readout_diagnostics.py
python -B scripts/check_xor_readout_capacity.py --output-root local_evidence/xor_readout_server_run01
```

输出目录必须不存在；历史回执和缓存保持不动。入口只生成假数据，用CPU两线程，不加载患者文件、患者checkpoint或服务器GPU。总共18次oracle读出训练、3次旧模型精确重放、3次冻结编码器后的读出续训，各500更新。3次重放不是3次独立新训练。此命令从新目录完整执行，不是从本机中断点恢复。

本机已有18/18 oracle、3/3 replay、2/3续训；原始回执仍写running，工具会话已不存在，Python进程检查未发现存活任务。本机没有为了凑齐最后结果重跑。交接验收记录见`xor_readout_handoff_receipt.json`；不要把未完成那一项当作通过。

## 已知结论，避免服务器重走弯路

- 正确因子直接进入同容量头、或经phi再均值时，平均NLL约.905/.920；同一任务的原端到端模型约1.144。说明干净因子条件下读出能学协同。
- 每模态只有一个含信号的噪声槽时，读出NLL约1.143，而理想条件Bayes解码约.882；重复同一比特到全部槽时约.924。不能把重复降噪的收益误称为槽互补。
- 共享KL能改变实际共享样本中的私有因素可读性，但尚无稳定预测优势；私有均值槽仍低秩，XOR互补未通过。不能据此直接定型患者网络。
- MIB/CEB/DisentangledSSL/PIBD已有紧邻机制，不能把对称KL或共有/独有分支作为原创本身。

## 患者训练仍缺哪些具体工作

当前`CrossmodalSharedBottleneck`输入是token张量，没有队列或患者CLI。接手者应先完成真实WSI/组学token、有效mask、训练折归一化和生存时间箱适配，以及匹配的原版/V2基线计划与doctor；将实际split身份、patch抽样、seed、checkpoint选择、评估口径固定后，才有可执行的患者训练任务。

现有旧患者队列实际多用`5fold`，不能仅凭目录名或默认设置写`5fold_uni2h`。患者材料、权重、预测和逐患者数组只留忽略目录。用户希望把训练转交服务器，不等于允许修改既有队列、扩大折数或掩盖未完成适配。本次不提供虚构的患者`--run`命令。

接手研究的下一最小问题是区分“单个含信号槽被其他噪声槽稀释”“编码没有保留私有因素”“解码优化不足”。先利用这批已知答案与固定头扰动，再决定是否加入给定G的内容保持；不要继续堆叠尚未验收的PCA、注意力或反塌缩改动。
