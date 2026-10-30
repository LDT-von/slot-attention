# SlotSPE 问题研究

研究对象是原版 SlotSPE。先查清槽机制的不足与发生条件，再判断已有改进能否迁移、是否值得做患者对照。


## SCM/SRO 损失适配（2026-10-30）

新增 SCM（槽余弦间隔）与 SRO（读取重叠）及各自与 R4 的组合；门槛0.90、最大权重0.10、前5轮线性增加。S0/R4/R3 先审计后复用，新队列拟执行五折×四臂，seed3不扩种子。入口默认只生成计划，患者运行另需明确授权。

[参数与服务器提示词](docs/research/slotspe_diversity_20261029/参数与服务器交接.txt) · [配置](configs/slotspe_diversity_queue.json) · [代码验收与限制](docs/research/slotspe_diversity_20261029/code_validation_receipt.json)。交付检查116通过；额外旧模块回归80通过、4项因本机磁盘预算或缺服务器输入失败，未宣称全仓测试通过。

## 当前交付

现行实现统一在 `slotspe_proof/`，旧包路径只保留兼容导入入口。12个独立检查/汇总脚本、研究导航、阅读材料与历史代码快照已纳入同一公开交付；原版源码与历史回执保持各自原始字节。临时渲染、备份、患者特征、权重和逐患者产物保存在忽略目录。

当前文件范围、大小和SHA256以 **[DELIVERY_MANIFEST.json](DELIVERY_MANIFEST.json)** 为准。它覆盖全部公开文件，仅排除自身哈希以避免循环；提交/树SHA覆盖清单本身。**[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json)** 保留为历史迁移快照，其旧哈希不用于当前完整性判断。历史整理回执的目录数同样保留，当前35个研究目录见[分类目录](docs/research/README.md)。

```powershell
python -B scripts/check_delivery_manifest.py --check --require-tracked
```

本轮403文件干净交付副本的全量测试为 **338 passed，0失败、0跳过**；测试只使用合成输入。

该检查只读源码和文档，不读取患者证据。维护者修改公开文件后，用 `--refresh` 更新当前清单，并将清单与对应文件一起提交。全量测试与交付核验的实际结果统一登记在[PROGRESS](docs/PROGRESS.json)。

## 从这里开始

| 你想知道什么 | 固定入口 |
|---|---|
| 有哪些问题、证据多强、结论如何收敛、下一步做什么 | **[研究总览](docs/RESEARCH_OVERVIEW.md)** |
| 某项结论的报告、数据摘要、脚本记录放在哪里 | **[研究材料分类目录](docs/research/README.md)** |
| 哪些任务实际完成、是否执行模型、是否回传、是否提交 | **[唯一进度记录](docs/PROGRESS.json)** |

## 当前认识

- **冗余已有测量支持**：重复表示/读取，选中后仍重叠；混槽、碎片化、具体遗漏各自待验证。
- **内容监督值得优先追查**：WSI重构头存在“固定槽字典＋目标内容查询”的构造反例；低重构损失不足以证明槽保存了当前样本变化，训练后是否采用该解未知。
- **选择的作用范围要按代码解释**：选槽器直接融合主梯度断开，辅助监督仍在；未选槽仍可能通过交互/汇总参与预测。
- **空间是条件性支线**：独立patch特征集合相同、没有空间绑定时，模型无法区分不同邻接排列；真实空间任务和上游上下文另查。
- 患者预测损害、修正的稳定患者收益均未建立。原P1–P8、补充维度、实现问题与被否定的强说法统一见研究总览。

2026-10-27进一步落实到原版算子：默认8选2，self分支平均注意力中未选value质量下界为 **9/16=56.25%**（eval/投影前，不是预测贡献）；另构造8个不同槽，使辅助生存输出分布相同、最终融合NLL严格不同，证明原版辅助选槽监督存在不可识别性。token归一化的支持量消去与后置选择不可回滚竞争也完成局部证明。[原版门控与支持量严格审查](docs/research/slotspe_eight_issues_20261019/原版门控与证据支持量_严格审查_20261027.txt)。9组纯算术检查通过；未执行模型或患者任务。

2026-10-27用户明确要求保留未选槽在预测中的作用，并研究重构大模块。原版病理与组学自重构实际只读取Top-K槽；联合重构使用另一组病理派生槽。现已证明病理重构单项存在可训练常量目标的零损失构造，并提出读取全部槽、无患者目标query、重建固定参考内容质量与特征矩的候选；患者层内容分离和条件方差边界已推导。候选尚未实现，保持原预测路径及既有V2/A-B-C。[问题与方案](docs/research/slotspe_eight_issues_20261019/重构模块问题与全槽内容重建方案_20261027.txt)。

2026-10-27用户明确V2是自己的方法，允许大胆探索。新增[跨视图重构与槽增量的数学试错](docs/research/slotspe_eight_issues_20261019/V2跨视图重构与槽增量_理论试错_20261027.txt)：在V2上研究全槽预测同患者不重叠patch子集的固定内容目标；证明有限采样界、理想条件增量恒等式与无序码本容量界，并构造8槽匹配零损失但全部复制患者信息、删槽误差误归因、协同与权重诱导塌缩反例。9项纯算术核查通过；候选未实现、患者数据未执行。

2026-10-27后续已实现独立的V2全槽跨视图候选、四臂计划和经验删槽诊断；56项生成输入检查、9项数学检查通过，包含四臂小尺寸假样本各更新一步并保存、重载及评估。该实现保持原预测路径，目标含参考锚的比例及加权特征一阶矩；验证折新目标诊断比较同患者、训练折常数和错配患者。患者实验和服务器执行均为0；[模块、命令和限制](docs/research/slotspe_eight_issues_20261019/跨视图模块合成验收与服务器交接_20261027.txt)。前面“候选未实现”描述的是当时理论阶段，不是当前状态。

2026-10-30 V2跨视图已在**满预算**下跑完并回报（30轮/4096 patch/20次评估/fold0/seed3，四臂B0_native、V2_separate、V2_crossview_compute、V2_crossview_train，303训/77验）。**该队列实际划分同样是 `5fold`**：其prepared manifest的77名验证患者与`splits/5fold/blca/fold_0.csv`完全相等，与`5fold_uni2h`只重叠15/77，目录名里的`uni2h`是误标。但本队列用**固定末轮权重**（`crossview_queue.py`的`fixed_final_epoch_no_validation_selection`），而SCM/SRO队列及其复用的S0/R4/R3基线用**每轮最优权重**（`per_epoch_validation_best_then_reload_and_reevaluate`）；折数与臂集合也不同，故两套C-index不是公平改进比较，差异不可归因于方法，跨视图数值重新定性为**同队列描述性参照**。读回执时发现**V2_separate与V2_crossview_compute是逐位相同的重复臂**：eval_repeats.json的SHA256相同、20次评估C-index全等、1540条逐患者risk全等、1485个core主干张量全等（虽参数量33443534对33568598不同），30条训练曲线唯一差别只是诊断字段view_loss_observed/view_cases。该臂按设计view_loss_weight=0.0即不接优化器，故非缺陷，但它**不是独立对照**，其"一致"是恒等式而非复现。四臂C-index：B0_native 0.4513±0.0407、V2_separate与V2_crossview_compute 0.6103±0.0481、V2_crossview_train 0.6241±0.0419；单折单种子，重复标准差远大于臂间0.014的差。删槽探针显示内容空间在满预算下张开（有效秩2.28/2.70、余弦0.291/0.465），而native与private空间更塌缩。**该删槽阴性结果的定性也已更正**：合成阳性对照本身在码本16时即出现饱和（drop侧碎片化、单例比例0.805，空假设0.1088超过真实gap 0.0352），故"gap小"可能只说明**仪器分辨不出**，不等于槽内无信息。详见 [PROGRESS](docs/PROGRESS.json) 的 `latest_crossview_full_budget_and_deletion_20261030`、`latest_crossview_split_identity_correction_20261030`、`latest_crossview_two_corrections_accepted_20261030` 与 `latest_cross_queue_reconciliation_20261030`。

2026-10-30另有**反塌缩队列**（A0_clip与A1_clip_decay两臂×五折×seed3共10次训练）**已跑出前3轮**：fold0 A0_clip逐轮验证C-index 0.6022/0.6434/0.7417，fold0 A1_clip_decay 0.6022/0.5784/0.6712，fold1 A0_clip 0.5457/0.4670/0.4915，fold1 A1_clip_decay 0.5457/0.4222/0.5389。**checkpoint选择规则经源码确认是"每轮最优"而非末轮**（`redundancy_anticollapse_engine.py:231-252`）：每轮写`epoch_validation.json`，仅当新值≥历史最优才更新保留态；训练结束写`last.pt`（末轮，仅存档不参与打分）另写`best.pt`（最优轮），重载best后跑完整20次重复评估，**最终报告的是重载最优轮上的20次重复均值**。因此上表逐轮值只是单趟监控值，**与已完成队列的20次重复均值尚不可比**（可比时点为各臂写完`summary.json`）。几何侧：fold0 A0_clip的WSI有效秩由1.28升到4.94、余弦0.182、余弦>0.9的槽对比例恰为0.0，OMIC有效秩5.320/余弦0.109，塌缩特征未复现；post-clip梯度范数每行恒为1.0，clip确实作用于优化器。数据为**TCGA-BLCA（膀胱尿路上皮癌）**，标签`survival_months_dss`，380名complete-case患者，fold0为303训/77验，协议与已完成队列同族可比。三点保留：单折逐轮C-index标准误约0.05，fold0升而fold1降尚不可归因；clip=1.0比自然梯度尺度低约8个数量级（pre-clip最高76429717.18），有效步长由clip接管；A1的Gram衰减从满权重10.0起步故epoch1与A0按构造相同。**尚无任何臂完成，无20次重复分数**；队列自身边界亦声明撑开几何不等于槽内含患者特异内容。按其自身边界，即使跑通也不能推出槽内含患者特异内容，那需要再跑删槽探针。

2026-10-26起按用户要求增加**数学前置门槛**：先证明具体结论及其假设，证明不成立则收紧或放弃。内容匹配的条件性分离与频率目标恢复界已推导；接通ST的普遍下降命题被反例否定，条件门控相对完整原版的严格优势未建立。[完整数学审查](docs/research/slotspe_eight_issues_20261019/theory_20261026/结论与验收.txt)。以下既有实现及协议保留，本轮不触发患者执行或新门控实现。

## 当前实验路线

新增[互信息与解耦模块实施计划](docs/research/slotspe_native_assignment_training_20261022/互信息与解耦模块实施计划_20261025.md)已接通代码：V1监督槽压缩、V2内容约束的共享/私有分解、V3关系残差，统一为九个固定preset，配套数学命题和逐版对照。模块、独立训练入口与生成输入检查已完成；患者实验为0，服务器连接仍按用户要求暂缓。[代码入口、命令与验收边界](docs/research/slotspe_native_assignment_training_20261022/插件版本使用与验收_20261025.md)。以下既有队列协议保持原状。

宿主为 **third_party/SlotSPE 中的原版 SlotSPE**。当前对照为 **S0–S6**：S0双原版，S1双Sinkhorn，S2病理MESH，S3组学MESH，S4双MESH；S5为双侧DIAS式去冗余再聚合核心，S6为双侧MetaSlot式原型去冗余核心。适配实现为`mesh_uniform`，不表示作者整套方法已完整复现。

已有生成数据适配检查与短程模拟，11臂患者训练入口现已接通并通过完整生成数据烟雾测试；本地尚未验收本轮完整十一臂患者回传。服务器执行转述与已验收结果分别登记。相似度目标的历史直接结果见[病理/组学槽cos复核](docs/research/slotspe_native_assignment_training_20261022/病理与组学槽相似度复核.txt)。S0–S4的分配比较、A/B/C的选择内容/梯度归因、原版患者基线是三个不同的控制问题。[七组当前方案](docs/research/slotspe_native_assignment_training_20261022/S0-S6对照定义与接入说明.txt)。新增S5/S6的匹配模拟显示S5仅轻微下降，S6当前预设过度压缩；见[有效槽结果与覆盖率](docs/research/slotspe_native_assignment_training_20261022/S5-S6本地有效槽相似度.txt)。

当前选定**11组统一单队列**：S0、S4、R1–R5、S1、S2、S3、S5，限定BLCA、UNI2-h预提取特征、fold0/seed3，不自动扩大fold或seed。S6仅保留登记；R0是S0别名。训练器、异构通路目标和S5变长内容探针已接入；当前远端配置为30轮、4096 patch、逻辑batch8、20次评估、固定末轮权重，划分为 `5fold_uni2h`；这与历史实际执行的3轮/256 patch须分开。见[可执行队列与服务器交接](docs/research/slotspe_native_assignment_training_20261022/十一组可执行队列与服务器交接.md)。

本地已完成的两个十一臂队列：**末轮口径**（`fixed_final_epoch_no_selection`，五折）与 **best口径**（`per_epoch_validation_best_then_reload_and_reevaluate`，五折x十一臂共55次全新训练，无一折复用）。**这两个已完成队列实际执行的划分是 `5fold`（不是上段配置里写的 `5fold_uni2h`）**：原版加载器在 `third_party/SlotSPE/dataset/dataset_survival.py` 的 `_load_split` 中硬编码了 `splits/5fold` 目录，`splits_dir` 设置到不了队列选择；completed best 队列的 prepared manifest 与队列身份都逐字节记录并校验了 `splits/5fold/blca/fold_*.csv`。两套划分是实质不同的队列（每折76名验证患者只重叠12–17名；5fold为381人，5fold_uni2h为380人），因此任何复用该基线的臂必须继承 `5fold`。两口径的臂间排名互相矛盾：末轮下 R3 最高（0.6426）、基线 S0 第8（0.5778）；best下 R4 最高（0.7071）、S0 第2（0.7039）但两者差异不显著。best口径中分数最高的臂并未解决槽冗余（有效秩约1.17、余弦约0.91），唯二真正降低冗余的 R5/R1 恰是分数最低的两个；跨55点 Spearman 显示冗余与性能关系方向为负。单种子、未做多重比较校正、best峰值选择自带乐观偏差；这些数值不是泛化性能，也不构成患者获益。完整口径、逐折数值与更正记录见 [PROGRESS](docs/PROGRESS.json) 的 `best_multifold_five_fold_completed_20261028`。

默认计划入口（查看计划，不启动患者训练）：

```powershell
python -B scripts/plan_slotspe_redundancy_queue.py --plan
python -B scripts/run_slotspe_redundancy.py --plan
python -B scripts/check_slotspe_s5_s6.py
python -B scripts/run_slotspe_proof.py --plan
```

文档整理阶段没有执行模型；后续生成数据模拟的实际计数见PROGRESS。患者执行继续按具体授权、输入、预算与原有门禁处理。

## Executable eleven-arm diagnostic (2026-10-22)

The current SlotSPE queue is implemented: **S0, S4, R1, R2, R3, R4, R5, S1, S2, S3, S5**, limited to **BLCA / existing UNI2-h / fold0 / seed3**. Configured defaults: 30 epochs, 4096 real patches, logical batch 8, 20 evaluation repeats, final-epoch checkpoint, splits `5fold_uni2h`. These match the original SlotSPE defaults (`--max_epochs 30 --num_patches 4096`). S6 is held; R0 aliases S0. Historical D0-D4 are a separate DCT-host design and are outside this SlotSPE queue.

**Historical execution snapshot from the patch-budget reconciliation.** At that reconciliation, no reviewed patient run had used 4096 patches: the reviewed completed runs and both failed eleven-arm attempts used 3 epochs / 256 patches. Subsequent user reports described an ongoing 4096-patch S0 run; this repository delivery did not inspect that live process or receive its raw results. At 4096, and counting per patient after merging all of a patient's slides as the original loader does, 41 of 381 patients own fewer real patches, 15 of them are more than half padding, and the worst case is 97.0% padded rows, against 1 affected patient at the 256 budget. Padding mass within the short-bag subset averages 0.4013, but over the whole cohort it averages 0.0432. The published loader passes no padding mask, so padded rows still produce keys and values; that is a mechanism risk, not a demonstrated patient-prediction harm. The current decision is to keep the official padding behaviour for S0 and the eleven arms and to run a single-arm S0 protocol check at 4096 first. Read [the reconciliation](docs/research/slotspe_patch_slot_sharing_20261023/patch_to_slot_sharing_readonly.txt) before authorizing a 4096 run.

Patch handling reproduces the original loader: all real patches are cached per patient, a fresh subset is drawn without replacement at each access, and bags shorter than the budget are zero-padded instead of excluding the patient (`third_party/SlotSPE/dataset/dataset_survival.py` lines 285-292). Earlier revisions of this harness excluded such patients and froze one fixed subset per patient; both were deviations from the original and are corrected. The published loader passes no padding mask, so padded rows still produce keys and values.

The decision to keep that behaviour for all eleven arms, with any padding mask held out as an independent correction arm, is enforced rather than merely documented. Run `python -B scripts/check_slotspe_padding_policy_guard.py --run` to verify that no arm can silently gain a mask, that the queue still contains exactly those eleven arms, and that `_sample_bag` still pads with plain zeros. It is a generated structural check and asserts nothing about patient performance.

The complete 14-stage/11-arm generated-data smoke passed at native 256 dimensions, 8 slots and 10 iterations. It uses invented features, not patient or real UNI2-h inputs. Historical focused checks: 312 passed; the current whole-repository test receipt is recorded in PROGRESS.json. Original SlotSPE source, existing default configs and A/B/C definitions are preserved.

[Implementation, metrics and server handoff](docs/research/slotspe_native_assignment_training_20261022/十一组可执行队列与服务器交接.md) | [Validation receipt](docs/research/slotspe_native_assignment_training_20261022/eleven_arm_delivery_validation.json).

```bash
python -B scripts/run_slotspe_redundancy.py --plan
# Reuse the existing SLOTSPE_METADATA_ROOT / SLOTSPE_FEATURES_ROOT environment.
python -B scripts/run_slotspe_redundancy.py --doctor
# Only --run executes patient work; use a fresh ignored local_evidence output.
python -B scripts/run_slotspe_redundancy.py --run
```

All arms share paired initialization, cached inputs, per-example random schedules and logical optimizer updates. Content targets are fitted on training inputs only; variable-slot matching includes unmatched coverage. Geometry, reading, content, survival and resources are reported together; low cosine alone is not efficacy. Full details and the authorized server scope are in the handoff.

## patch 与槽的共享：每个 patch 被全部 8 槽读（2026-10-23）

对 fold0/seed3 原生 3 轮 checkpoint、77 名验证患者**已保存**的注意力数组做纯只读复算。**每个 patch 的有效读取槽数 WSI 7.99999（最低 7.99940）、OMIC 7.99984**；两槽在同一 patch 上的最大权重差 WSI 5.8e-05，而平均 patch 权重是 3.9e-03。**每个槽读取的有效 patch 数 WSI 255.99964/256**，前 10% patch 只拿到 9.79% 质量（均匀应为 10.0%）；注意力矩阵被 u(槽)×v(patch) 秩一结构解释 0.999997 的能量。**读取层重叠 0.99918（WSI）比已登记的槽向量余弦 0.99717 更严重。**

**这不是槽轴 softmax 造成的**：原始 softmax 在一个 patch 上 8 槽的 max/min 比只有 1.031（WSI）/1.045（OMIC），即它拿到的 logits 本身就没有分化；这从数据侧独立复现了四环拆解"R1 不是主因"。接近均匀在**第 1 次迭代就已存在**，并在 10 次迭代内持续收紧（WSI 偏离 0.00571→0.00157），说明成因在 softmax 之前：8 个槽形成第一个 query 时就已几乎相同。

单 fold0/单 seed3/单冻结 checkpoint/77 人，纯几何事实，**不构成预测损害或修正收益证据**；未做因果干预；注意力经 head 平均；且全部数字来自 **3 轮/256 patch**，不是 4096 预算下的结果。详见[报告](docs/research/slotspe_patch_slot_sharing_20261023/patch_to_slot_sharing_readonly.txt)。

## 槽塌缩四环只读拆解（2026-10-23）

用一个冻结 checkpoint 做**推理期**四环拆解，每臂只拆掉原生 `MultiHeadSlotAttention.forward` 的一个环节，其余不变。参考臂与历史 77 人 logits 的最大绝对差为 **0.0**，训练权重 `_state_digest` 前后一致，训练 0 次、优化步 0、参数更新 0。

**主因是 R2（每槽 token 轴 L1 归一化）**，不是选择器。只有去掉它能让塌缩全程变慢且终值降低（WSI 逐迭代余弦第 10 次 0.9974→0.9116），终态有效秩 4.49→6.39（满分 8 槽），份额差与更新分歧上升约 47%。

三点必须同时读：**R1（槽轴 softmax 竞争）不是主因**，WSI 上与参考臂逐迭代最大差约 4e-05，omic 只差约 0.004，此前把它排在自我强化链首位的定位被本轮取代；**R3（去掉 GRU 与残差）是陷阱不是解药**，余弦恒为 1.0000 实为退化——逐维跨槽标准差约 4e-8、槽0-槽1 的 L2 约 1e-6，即 float32 舍入噪声，8 槽数值完全相同；即使去掉 R2，终态余弦仍 0.9116，塌缩是多环叠加。

另更正一处描述：原生 `F.normalize(..., dim=-1)` 归一化的是 **token 轴**，钉住"每个槽跨所有 token 的总质量=1"，并非"同一 token 上 8 槽的总质量"。

单 fold0、单 seed3、10 名患者、只读推理；未计算 C-index，不构成预测损害或修正收益证据。主因与槽数无关，因此容量扫描的预期收益下降。详见[报告](docs/research/slotspe_slot_ring_dissection_20261023/槽塌缩四环只读拆解.txt)。

## 门控配对训练与槽冗余定位（2026-10-22）

同一prepared批次、同一初态、同一步骤种子下，只改原版 `MoESlotDecoder` 的数值属性 `k`/`temperature` 训练三臂：`L1_top2`（原版参考）、`L2_soft8`（放开硬选择）、`L3_uniform8`（再把门控权重置为均匀）。参考臂逐次评估 C-index 与已登记基线完全一致（delta=0），因此两处对比可归因于门控属性本身。训练总损失三臂几乎相同（2.41856/2.42701/2.43769）。

**放开硬选择后槽冗余几乎不变。** 三臂全部28对槽余弦 WSI 0.99580–0.99730、OMIC 0.99230–0.99320，`>0.95` 的配对比例 WSI 三臂均为 1.0。中心化有效秩从 2.55（Top-2）升到 3.08（soft8）再回落 2.75（uniform8），最大偏移 0.53，远小于已登记折间标准差 0.06178。**冗余不是硬稀疏单独造成的**；三臂全部8个 `pred_keep_slot` 参数对融合主损失均不连通、仅辅助损失有梯度（`L3` 因温度极大降到 1e-16~1e-10），确认门控打分器只被辅助损失更新，主读出路径从不经过它。三臂 C-index 差（+0.01426、−0.00935）在离散度内，不构成性能或修正收益证据。

**槽冗余的塌缩发生在注意力迭代内部。** 只读分析原始 npz 显示：初始8槽两两余弦中位仅 0.5583/0.5813（清楚分开），最终槽升至 0.9996/0.9982；逐迭代的同 token 跨槽份额差与槽间更新分歧均**单调下降**，第10次迭代比第1次小一个量级。8 个槽共用同一份 `slots_mu`/`slots_logsigma`（形状 `(1,1,dim)`），槽间初始差异只来自 `randn`；`attn = dots.softmax(dim=-2)` 在槽轴竞争、GRU 与残差混合把槽拉向共同均值，且差异越小→更新越接近→差异更小的自我强化。`temperature`/`topk_ratio` 属 decoder 选槽器，作用于"选哪些槽参与读出"，与保存的 token 读取矩阵无关，**不是本次塌缩成因**（更正此前口头说法）。

本轮只读分析加三次 GPU 训练，均不改原版源码、不改默认配置，产物限于忽略的 `local_evidence/`。[观察汇总](docs/research/slotspe_mechanism_protocol_20261021/observe_结果汇总_20261021.md)、[报告](docs/research/slotspe_gate_pair_20261022/门控配对训练与冗余对比.txt)。

## 权重选择协议

原版 SlotSPE 按验证 C-index 选择最佳 epoch；验证集选择偏差是该官方协议本身的局限，不能把遵循官方行为描述为规避后期退化的手段，也不能说成此次修改新引入的偏差。当前 `slotspe_proof` 诊断队列仍声明使用固定末轮权重和20次评估；两种协议须明确区分。本次仓库同步与旧实现清理不改变正在执行的训练或权重选择。官方源码核查见[权重选择记录](docs/research/slotspe_target_review_20261020/epoch_selection_review_20261023.json)。

现行实现统一位于 `slotspe_proof/`；旧路径只保留兼容导入入口，已被替代的重复实现从活跃代码树移除。

## 材料如何保存

原版源码在 `third_party/SlotSPE/`；诊断与独立适配在 `slotspe_proof/`、`scripts/` 和 `tests/`；`survot_rank/research/slotspe_proof/` 为历史导入兼容入口。完整目录用途见分类目录。

历史报告、论文、下载包、配置与原始证据保持原路径。旧报告中的路径、哈希和当时状态保留；后来的纠正统一到研究总览查。患者特征、权重、预测、标识及逐患者数组保存在忽略的 `local_evidence/`，不进入Git。

## 后续维护规则

同一问题复用既有编号和主题目录；新证据添加独立运行/版本文件。每轮先更新PROGRESS的实际完成状态，结论有变化时同步研究总览与覆盖说明。仅在出现独立新主题或新协议时建立研究目录，不再为每次对话增加一份平行总结。

历史文件夹按测量复核、文献、方案实现、早期审查、交接来源五组导航。[机器目录](docs/research/INDEX.json)覆盖全部既有研究文件夹；[整理校验记录](docs/RESEARCH_ORGANIZATION.json)保留备份与完整性检查。
