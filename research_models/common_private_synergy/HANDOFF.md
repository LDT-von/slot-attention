# 给下一位AI/开发者的交接

本轮用户要求：严格审查、准备计划和部分重要零件，**不跑深度学习**。此限制包括本轮的模型构建、模型前向/反向、优化器、特征提取及任何患者推理。后续用户若另行授权具体测试，再按新范围执行；不要把“有待执行测试文件”当作执行授权。

## 先读什么

1. 仓库 `README.md`、`docs/PROGRESS.json`，然后本目录 `ARCHITECTURE.md`。
2. `components/contracts.py` 与 `components/views.py`：输入/视图/统计来源契约。
3. `components/slots.py`、`readout.py`、`losses.py`：独立零件。
4. `architecture_spec.json`：机器可读结构与接入阻断项；不是可运行训练配置。
5. `validation_receipt.json`：本轮实际检查范围。历史 `tests/test_model.py` 和旧model.py只对应上一轮骨架。

## 文件和接口

| 文件 | 接口/内容 | 当前验证范围 |
| --- | --- | --- |
| contracts.py | ArchitectureConfig；CorrelationContract.validate_batch | 标准库已知答案；无模型实例化 |
| views.py | plan_patch_views；plan_omics_views；FeatureContext；TrainingStatisticsProvenance | 索引/遮蔽/来源元数据10项检查；无真实特征读写 |
| adapters.py | FeatureTokenSchema；PatientPairContract；TimeBinContract；AdapterContract | 元数据契约检查；不读取特征或患者标识 |
| slots.py | RoleSlotEncoder(input_dim, config)；AllSlotPool(width)；SummaryProjector(width,p) | 源码编译与静态审查，未导入/运行 |
| readout.py | AdditiveInteractionReadout(config) | 加性与空组源码审查；未运行 |
| losses.py | discrete_hazard_nll；CorrelationPolicy；DeCURWeights；decur_losses；variance_floor | 源码编译；8项张量检查已写但未执行 |
| assembly.py | CommonPrivateSynergyModel；ObjectiveWeights；build_model | 只完成源码组装和编译；未初始化、前向、反向或训练 |
| checks/ | 视图、秩契约、PID差值、加性反例、删失算术 | 纯标准库，不能证明torch实现正确 |
| deferred_checks/ | loss与encoder/pool/readout的后续张量验收候选 | 本轮严禁执行，不属于通过数 |

`ArchitectureConfig` 中 wsi_dim/omic_dim 必须由已核实特征schema指定；默认width=64、总槽8、共有7、bins4、iterations3、interaction_rank16只是源码接口示例，未经效果验证。不要把示例当作患者实验协议。

`RoleSlotEncoder.forward(x, real_mask)` 输出 `slots[B,8,D]`、`common_slots[B,Kc,D]`、`private_slots[B,8-Kc,D]`、`read_attention[B,8,N]` 和mask。mask是显式必需参数；预测对照若原版没有mask，必须独立标注这一变化。两模态实例分开，两视图在各模态内共享实例。当前没有原始RNA/图像adapter，omics必须先保留gene/pathway身份。

每个模态/角色各用一个AllSlotPool和一个SummaryProjector，同一实例跨视图共享。原summary用于预测，低维投影用于SSL。把投影整理成mapping：`common_wsi,private_wsi,common_omic,private_omic`，行顺序一致；`patient_keys`只用于验证同批行唯一性，不能公开真实患者标识或日志。组件只能查键是否重复，不能证明实际配对正确。

`decur_losses` 默认所有权重0；全关返回 `weighted_total=None`，调用者应跳过加法，禁止造一份伪tensor或为了求zero损失执行全部分支。启用时设 `CorrelationPolicy(min_patients=8,max_projection_dim=4)` 对应每role p2；这只通过必要秩门槛。源代码的max预算包括拼接后intra维度。较大batch需实际收集同一个有梯度的患者批次，不得用累积步数乘batch虚报。所有四项启用检查dc+dp_w+dp_o≤B−1。

编码/读出支持0或8共有槽，但当前DeCUR intra helper要求共有/私有投影均非空；**8:0/0:8的SSL适配尚未写**。这些比例可用于无SSL结构控制，不能把空tensor塞进loss。共有组为空时读出返回`common_active=False`且两共有logits为None，必须跳过共有辅助NLL。纯私有的共同rank约束仍需2dp≤B−1。

`discrete_hazard_nll(logits,completed_intervals,event)` 要求float32/64 logits、int64计数、bool事件；关闭autocast。旧survival_nll的time_bin不能不经转换直接传入本接口。删失在区间中途只记之前完整区间；该离散近似丢弃部分区间暴露时间，边界规则和分箱都必须写入adapter文档。

读出不再喂融合G到联合MLP；`common_wsi_logits/common_omic_logits` 可分别用于共有任务压力，权重独立。总任务以主logits计算。交互关闭是严格logit加性；开启的乘积项仍不等于纯协同。相同state_dict结构不等于激活参数容量公平，必须补同等活跃容量的单模态加性头控制。

## 下一步可以接手的工作顺序

1. 先审核上述接口并补齐适配器规格：omics真实类型/身份编码、输入宽度、split、pad行为、每患者token预算、独立患者batch、时间分箱/删失映射。`components/adapters.py` 只锁定元数据边界；真实值和映射仍需由未来 adapter 提供。未知项保持未定，不从历史报告猜值。
2. `components/assembly.py` 已提供组装容器，保持单模态来源约束、两视图共享实例、分项loss日志及全部辅助关闭的基线。它无自动训练、副作用或患者IO；不能直接复用旧model.py的融合head。下一步只做获授权的CPU张量验收。
3. 后续得到张量检查授权后，执行有限CPU契约验收；loss及slots/readout的mask、空组、pool梯度和置换、读出四角差、交互关闭lazy skip、保存加载候选已写入deferred_checks，但全部未执行。仍需补角色query梯度、编码器分组内置换、有限差分及跨设备/精度条件。未过不能跑学习实验。
4. 通过后才申请明确范围的已知答案合成学习，锁定共享/独有/XOR/混合多因素/无关噪声与非线性复制。先证明仪器区分已知答案，再解释学到的表示。
5. anti-collapse已有variance零件，但患者轴之外仍需槽复制诊断。SRO/SCM/R4只按ARCHITECTURE中的输入接口接入，保留旧定义和单项控制；不得在新代码里改写原版SlotSPE。

## 本轮可重做的无深度学习检查

从仓库根目录执行：

```powershell
python -B research_models/common_private_synergy/checks/run_static.py
```

该脚本只编译源码不执行，运行标准库检查并确认进程未导入torch。不要改成`pytest`扫描整个目录：旧tests和deferred_checks包含torch前向/反向。不运行整仓测试、不安装依赖、不启动已有队列。

## 实验设计待实现，不能自动排队

基线R0=新7:1结构+单视图+任务损失+严格加性，无辅助损失。先比较双视图任务均值（预算控制）再分项SSL；共有独立NLL独立开关；交互+匹配活跃容量加性控制；anti-collapse独立开关；SRO、SCM、R4分别加到同一个已锁定参照臂。非默认累积全开，每阶段保留冻结参照。角色比例需匹配维度、资源、激活参数并记录差异。原版SlotSPE属于外部基线，不能把所有架构差异归因于一个模块。

所有后续源文件留在本目录；若新建患者产物，仅在明确授权后放忽略的local_evidence。进度统一追加到仓库docs/PROGRESS.json，原报告/回执保持历史字节。当前没有commit或push，不连接服务器。源码库存清单不能替代科学验收。
