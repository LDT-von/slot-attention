# 研究材料分类目录

本目录有35个研究文件夹。这里按用途分成五组；文件保留原路径，便于继续核对脚本、历史哈希和回执。每个问题的当前判断集中在[研究总览](../RESEARCH_OVERVIEW.md)，任务状态看[PROGRESS](../PROGRESS.json)。

先读总览，需要核验时再进入相应证据目录。目录中的日期表示当时的材料，不表示其中每句话仍是当前结论。

## A．测量、机制证据与复核（11个）

| 文件夹 | 用途 | 阅读要求 |
|---|---|---|
| [slotspe_local_redundancy_20261021](slotspe_local_redundancy_20261021/) | 历史本地71人槽向量/读取/选择后冗余复算。 | 对应P2.1；与另两组77人分开。 |
| [slotspe_sparsity_off_20261021](slotspe_sparsity_off_20261021/) | 历史71人关闭稀疏选择的冻结读出干预。 | 不是无门控匹配训练，不证明门控无用。 |
| [slotspe_gpu_trisurv_20261022](slotspe_gpu_trisurv_20261022/) | 本机GPU原版五折三种子与77人机制公开记录。 | 20条记录/15唯一配置等口径以deepcheck勘误为准。 |
| [slotspe_mechanism_deepcheck_20261022](slotspe_mechanism_deepcheck_20261022/) | 当前重构/几何/比例/初态深查、负控、独立复核。 | 合成最新回执是03；头级存在性与患者发生率分开。 |
| [slotspe_extended_audit_20261022](slotspe_extended_audit_20261022/) | 扩展问题的源码、生成输入及已有公开证据映射。 | 旧coords例与精确同初态边界需连同deepcheck阅读。 |
| [slotspe_observe_review_20261021](slotspe_observe_review_20261021/) | 服务器observe汇总的测量定义、算术与边界复核。 | 头平均读取熵不等于原始竞争精确均分。 |
| [slotspe_results_review_20261021](slotspe_results_review_20261021/) | 原P编号对齐、旧服务器诊断与训练曲线复核。 | 原编号对齐的主依据；包含不采信的强说法。 |
| [slotspe_p6_addendum_20261020](slotspe_p6_addendum_20261020/) | 指定提交同步来的阶段报告、诊断摘要与图表索引。 | 属来源证据；不能跳过results_review独立复核。 |
| [slotspe_gate_pair_20261022](slotspe_gate_pair_20261022/) | 门控配对训练与槽冗余定位的历史患者观察。 | 匹配门控对照与冻结读取观察分开；单折差异不构成修正收益证据。 |
| [slotspe_patch_slot_sharing_20261023](slotspe_patch_slot_sharing_20261023/) | 历史保存注意力的patch跨槽共享复算与patch预算核对；[标注癌区与通路的原版槽冗余实验方案](slotspe_patch_slot_sharing_20261023/标注癌区与通路的原版槽冗余实验方案_20261027.md)。 | 读取冗余来自3轮/256 patch；新方案仅设计，未完成患者匹配或推理；4096补零统计是协议风险，不是患者预测损害。 |
| [slotspe_slot_ring_dissection_20261023](slotspe_slot_ring_dissection_20261023/) | 冻结checkpoint的槽注意力迭代四环拆解观察。 | 单折单种子10人推理期对照；未建立预测损害或修正收益。 |

## B．文献问题与迁移条件（3个）

| 文件夹 | 用途 | 阅读要求 |
|---|---|---|
| [slotspe_paper_problem_map_20261021](slotspe_paper_problem_map_20261021/) | 本地8篇逐篇问题、页码、哈希与P1–P8映射。 | 文献指出问题不代表SlotSPE已发生。 |
| [slotspe_extended_literature_20261022](slotspe_extended_literature_20261022/) | 新增16条研读、作用层与迁移边界。 | 旧总量推断被deepcheck中的完整模型比例控制收紧。 |
| [slotspe_literature_20261020](slotspe_literature_20261020/) | 早期文献到原版映射、本地提取与缓存索引。 | 论文全文/缓存保持本地，不以索引代替本轮重新研读。 |

## C．实验方案、对照与实现验收（5个）

| 文件夹 | 用途 | 阅读要求 |
|---|---|---|
| [slotspe_native_assignment_training_20261022](slotspe_native_assignment_training_20261022/) | **当前原版SlotSPE S0–S6**、R0–R5直接表示探索、当前11组BLCA单折候选队列；新增[互信息与解耦计划](slotspe_native_assignment_training_20261022/互信息与解耦模块实施计划_20261025.md)及[插件代码入口](slotspe_native_assignment_training_20261022/插件版本使用与验收_20261025.md)。 | V1/V2/V3及九个preset完成生成验收，新插件患者实验为0；直接损失降cos未证明内容收益。 |
| [slotspe_mechanism_protocol_20261021](slotspe_mechanism_protocol_20261021/) | 原P1–P8协议、observe报告与阶段交接。 | 协议与实际完成状态分开；不直接重放旧服务器提示词。 |
| [slotspe_redundancy_methods_20261021](slotspe_redundancy_methods_20261021/) | MESH/DIAS/MetaSlot机制、分配原型与历史发布。 | DCT桥接支线仅作历史；当前原生方案以上一行为准。 |
| [slotspe_abc_20261019](slotspe_abc_20261019/) | A/B/C选择内容和末次权重梯度隔离控制。 | 保持A/B/C定义，不与S0–S4分配比较混用。 |
| [slotspe_clean_original_20261019](slotspe_clean_original_20261019/) | 干净原版队列、配置与发布验收。 | 方案准备/包装通过与患者执行分开。 |

## D．原版来源与早期问题审查（10个）

| 文件夹 | 用途 | 阅读要求 |
|---|---|---|
| [slotspe_target_review_20261020](slotspe_target_review_20261020/) | 指定原版目录、三方来源及八项源码复核。 | 原版身份依据；当前问题判断看总览。 |
| [slotspe_eight_issues_20261019](slotspe_eight_issues_20261019/) | 初期八项合成检查及冻结脚本。 | 有旧绝对路径；不在原目录重跑覆盖。 |
| [slotspe_p1_20261020](slotspe_p1_20261020/) | 早期容量与输入规模核查。 | 参数/patch预算探针不能替代P1容量匹配训练。 |
| [slotspe_p2_20261020](slotspe_p2_20261020/) | 早期槽干预与证据。 | 后续真实内容解释以冗余复算及最新复核为准。 |
| [slotspe_p3_20261020](slotspe_p3_20261020/) | 初态与评估随机性的早期诊断。 | 原P3冷启动与Q1评估重复性分开。 |
| [slotspe_p4_20261020](slotspe_p4_20261020/) | 槽置换、编号和语义核查。 | 置换不变/数值重复性不能证明类型绑定。 |
| [slotspe_p5_20261020](slotspe_p5_20261020/) | 输入与档案交接核验，含重要勘误。 | **必须同时看P5交接勘误；ARCHIVED.txt不是删除依据。** |
| [slotspe_p6_20261020](slotspe_p6_20261020/) | 早期顺序/尺度相关核查。 | 当前P6是倍率/层次；Q6顺序与X2几何另列。 |
| [slotspe_p7_20261020](slotspe_p7_20261020/) | 早期学习目标与梯度路径检查。 | 断路仅限选槽器直接融合支路，不能扩为全SA无监督。 |
| [slotspe_p8_20261020](slotspe_p8_20261020/) | 早期双阶段假设核验。 | 原版无后续跨模态OT，当前不适用。 |

## E．交接、来源同步与验收历史（6个）

| 文件夹 | 用途 | 阅读要求 |
|---|---|---|
| [slotspe_server_claims_20261020](slotspe_server_claims_20261020/) | 早期服务器强结论复核和原始回传要求。 | 汇报到达与原始产物独立验收分开。 |
| [slotspe_server_connection_20261020](slotspe_server_connection_20261020/) | 当时的仓库接入交接提示词。 | 历史交接，不是当前执行入口。 |
| [slotspe_server_return_20261020](slotspe_server_return_20261020/) | 脱敏公开回传、来源与缺失清单。 | 公开摘要不是完整私有原始包。 |
| [slotspe_server_acceptance_20261020](slotspe_server_acceptance_20261020/) | 来源提交、本地验收及修正要求。 | 保留当时状态，后续状态到PROGRESS查。 |
| [slotspe_merge_20261020](slotspe_merge_20261020/) | 公开证据的干净整合说明与验收。 | 不改写来源分支历史或旧哈希。 |
| [slotspe_scoped_sync_20261021](slotspe_scoped_sync_20261021/) | 指定诊断脚本和公开材料的同步清单。 | 复制/哈希一致不等于数值结论均通过验收。 |

## 文件夹之外的材料放在哪里

| 位置 | 用途 |
|---|---|
| `third_party/SlotSPE/` | 当前指定可执行原版；保持字节不变。 |
| `SlotSPE-main/` | 本地原始目录，保留来源与历史路径。 |
| `slotspe_proof/`、`scripts/`、`tests/` | 现行诊断、独立适配、执行门禁与验收代码。不是临床证据。 |
| `survot_rank/research/slotspe_proof/` | 旧导入路径的兼容入口；现行实现位于 `slotspe_proof/`。 |
| `configs/` | 现有原版/A-B-C/问题协议及分配对照配置；改动与运行另按任务授权。 |
| `evidence/code_snapshots/`、`docs/deliveries/` | 冻结代码与交付包；不自动重建旧包。 |
| 仓库根目录8份PDF、7份ZIP | 本地论文与作者代码下载；保留原文件，索引见[本地文献清单](../LITERATURE_LOCAL_INDEX.json)。 |
| 忽略的 `local_evidence/` | 患者原始材料、模型权重、逐患者结果及本次整理前备份；不进入Git。 |

机器可读目录见[INDEX.json](INDEX.json)，2026-10-22历史整理校验见[RESEARCH_ORGANIZATION.json](../RESEARCH_ORGANIZATION.json)，其中目录数和哈希保留当时含义。此目录的作用是导航，不建立第二套任务进度。
