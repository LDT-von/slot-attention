# SlotSPE 问题核查

**主线是用户原定的P1–P8研究问题，当前服务器实验尚未逐项测全。** 高槽相似直接关联P2，选槽器直接融合主梯度断路只覆盖P7的一个局部。评估随机波动、补零风险和训练后期退化单列为相关发现，不能替代原问题。

本仓库先检验槽表示、内容分配、聚合过程与学习目标，再分别研究预测后果和修正收益。表示机制的证据不必以C-index下降为前提；高相似、sanity check和人为干预的输出变化也不直接登记为预测损害。

当前入口：[进度](docs/PROGRESS.json)、[服务器新结论复核](docs/research/slotspe_server_claims_20261020/服务器结论复核.txt)、[原始回传与对照要求](docs/research/slotspe_server_claims_20261020/服务器AI_回传与对照验收.txt)、[P5勘误](docs/research/slotspe_p5_20261020/P5_交接复核与勘误.txt)。

## 原八项问题的实验入口（2026-10-21）

[实验协议与判定](docs/research/slotspe_mechanism_protocol_20261021/实验协议与判定.txt)、[服务器执行提示词](docs/research/slotspe_mechanism_protocol_20261021/服务器AI_原八项问题实验.txt)。入口为`python -B scripts/run_slotspe_problem_experiments.py --plan`；只有`--run`才执行，产物限定在忽略的`local_evidence`。

P2同时测槽向量和真实输入读取的重复程度；P3比较随机、本患者线索、多个错配患者线索在原生迭代预算下的内容读出；P7分解原生目标并测共享参数梯度关系。P4/P5/P6分别要求独立内容标签、审核过的候选干扰角色、真正对齐的倍率/层特征，缺条件就明确缺失。P1另有匹配容量训练入口，固定选中数、共享初态与预算；原权重换槽数仅作条件探针。P8不适用。

本地只完成合成测试、静态审查和计划检查，未运行真实模型或患者任务；新协议的服务器结果仍待回传。

本轮代码及公开证据复核已发布到`main`（实验载荷提交`abfaf11`）；[服务器拉取入口](docs/research/slotspe_mechanism_protocol_20261021/服务器AI_拉取本轮实验.txt)。发布收据见统一进度，以下历史阶段的“尚未提交”保留其当时状态。

## 2026-10-21服务器补充材料：汇总算术已复核，部分诊断结论拒绝采信

来源为DCT-Reg指定提交`2ad3853`、`0cbdbf9`、`9b3a5a8`，已仅同步15个SlotSPE诊断脚本和18份证据文档/JSON，未引入transport模块或改动默认队列。入口：[EVIDENCE_README](docs/research/slotspe_p6_addendum_20261020/EVIDENCE_README.md)、[来源与哈希](docs/research/slotspe_scoped_sync_20261021/import_manifest.json)、[本轮同步收据](docs/research/slotspe_scoped_sync_20261021/同步收据.txt)。

本地已逐条复算18个fold/seed的C-index重复值和30轮训练曲线：验证C-index各组已观测峰值平均0.7168，末轮平均0.5905，18/18组下降超过0.01；训练NLL下降而验证NLL上升，支持当前协议的明显晚期退化。峰值轮次中位数为6.5，不是部分来源说明中的6。

eval_fix的同权重平均C-index变化+0.00269，按来源算法复算的区间为[-0.00120,+0.00679]，11升7降；固定评估臂18/18组重复值相同。阶段6另记录原版最大logit波动0.1090、修正后0。重复性修复有支持，稳定性能收益仍未建立；对照只在末轮checkpoint进行。

源码复核发现：新增Q6的patch置换额外消耗模型随机流；槽置换mask差值未使用实际置换输出；self干预有空操作；部分p值和每臂置信区间写法错误。因此不接受“顺序无害”“未选槽不影响预测”“训练导致槽坍缩”等更强归因。详见[本轮证据复核](docs/research/slotspe_results_review_20261021/服务器结果证据复核.txt)、[独立复算收据](docs/research/slotspe_results_review_20261021/aggregate_verification.json)。原问题编号以[八项对齐勘误](docs/research/slotspe_results_review_20261021/八个原始问题对齐勘误.txt)为准：此前将评估随机性写作P3、顺序写作P6存在含义偏移。

本轮只执行公开汇总算术和源码检查，未运行模型；逐患者原始预测或权重仍未收到。16张原图和5份旧归档保留在忽略的本地证据目录。E盘仓库未修改，同步和复核材料尚未提交/推送独立库。下方日期更早的说明保留其当时状态。

## 2026-10-20最新回传：公开内容已修正整合，私有原始验收待完成

服务器来源分支`codex/server-slotspe-evidence-20261020@82582f2`的公开代码和摘要已整理为干净整合提交。实际执行原件留在忽略的本地证据目录，公开树仅保留脱敏副本和原件哈希；已补齐两脚本的`os`导入、重算摘要清单哈希，并纠正P4/P6批次数口径。Top-K实际比例仍等待原始配置核对。

整合不引入来源提交的患者标识文件及其历史；旧服务器分支历史未重写。本地既有交接报告和进展也已保留。P1测的是patch预算；P2全部28对高相似仍是待原始JSON验收的服务器摘要；P6未配对RNG，不能归因顺序。完整私有原始产物尚未收到，患者损害与修正收益仍未建立。

当前入口：[整合说明](docs/research/slotspe_merge_20261020/合并说明.txt)、[此前本地验收](docs/research/slotspe_server_acceptance_20261020/本地验收报告.txt)、[统一进展](docs/PROGRESS.json)。此前报告保留其当时状态；下一步接收已授权私有原始包，不新增模型执行。

## 2026-10-20早期服务器汇报复核：1de42c7仅有说明

核对到原[DCT-Reg提交1de42c7](https://github.com/LDT-von/DCT-Reg/commit/1de42c777054b061ed278eb3ecb02d0bfbd2cca3)。它只改了`AGENTS.md`和`.gitignore`，把`_p1246_diag.py`加入忽略列表；在该提交中未取得诊断JSON或脚本。这证明汇报已入库，不能代替原始结果验收，也不表示服务器没有实际运行。

服务器“六项患者损害全立”的结论目前不能接受：

- P2记录是每名患者**至少一对**去对角槽cos>0.95、max=0.9999，不等于8槽全部复制。
- P6的0.0235/0.085是risk差，P3的0.0035是C-index标准差，P5的0.011是logit差；三者不能共用0.01门槛或直接比较倍数。顺序对照还须核验高斯初槽与Gumbel抽样是否配对一致。
- P4的均值置换不变只是sanity check；P7直接梯度断开不等于选槽器完全不训练，也不等于整个Slot Attention缺少生存监督。
- 风险与观察时间的正Spearman相关不能单独判定方向翻转；须核对删失定义，并比较同队列的C(risk)与C(-risk)。[指标定义](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_censored.html)

## P1–P8当前结论

按用户原始定义对齐；问题机制是否存在、是否损害预测是两个层次，均应分别检验。

| 原始研究问题 | 当前证据与覆盖范围 |
|---|---|
| P1 固定槽容量不适配患者 | 当前固定8候选/Top-2有结构依据；patch预算实验没有改变槽数，尚未检验患者容量需求或碎片化。 |
| P2 槽冗余与碎片化 | 高cos是直接相关线索；尚缺槽读取重叠、独立信息与组织/通路对应，不能把高cos直接当作完整冗余或碎片化证明。 |
| P3 冷启动缺少患者线索 | 共享高斯初始化不读当前患者内容的结构事实成立；聚合质量/收敛后果未测。固定评估噪声没有加入患者线索。 |
| P4 槽语义身份不稳定 | 未直接测跨抽样/种子/患者的内容语义稳定性；槽编号置换不等于语义不稳定。 |
| P5 无关输入干扰/占槽 | 补零风险不等于真实无关内容占槽；缺少输入任务相关性的依据和直接分配测量。 |
| P6 尺度与特征层次不足 | 本轮没有多倍率/多编码器层的匹配比较；patch顺序另列Q6，不能替代P6。 |
| P7 学习目标与预后任务不一致 | 选槽器直接融合主梯度断路是局部证据，辅助监督仍连通；整体重建/生存目标关系与表示后果未测全。 |
| P8 Slot与跨模态OT双阶段约束 | 原版无后续跨模态OT，在本仓库不适用。 |

## 文件位置

- `survot_rank/research/slotspe_proof/`：原版诊断、回传验收、P1–P4测量与A/B/C代码。保留包名以维持历史接口，不依赖整套DCT仓库。
- `third_party/SlotSPE/`：指定桌面原仓库的核心源码快照及原作者说明，逐字节核对。
- `docs/research/slotspe_*`：各阶段报告、源码证据与既有检查记录；P5旧报告必须连同勘误阅读。
- `evidence/code_snapshots/`：冻结诊断脚本，保留其历史哈希。
- `docs/deliveries/`：既有独立任务包，旧包原样保留，不因整理而改写已发布内容。
- `local_evidence/`：本机原始数据、checkpoint、逐患者结果、失败/负向检查与历史上下文；Git排除。
- 根目录现有8份PDF和7个下载ZIP保留本地，索引见 `docs/LITERATURE_LOCAL_INDEX.json`；论文链接与研究笔记保存在文献核查目录。

## 安全查看计划

使用现有Python环境；本次整理没有安装依赖或运行患者任务。需要的库见 `requirements-audit.txt`。

```bash
python -B scripts/run_slotspe_proof.py --plan
python -B scripts/run_slotspe_proof.py --config configs/slotspe_attribution_server.json --plan
```

默认是干净原版：fold0、seed3、3轮、256真实patch、batch8、同权重20次评估；计划不会读取患者张量或训练。
原DCT目录中的宽网格默认配置仅保留在本机历史区；本仓库默认配置复制现有clean-original配置，原文件不改。
真实数据位置由服务器的 `SLOTSPE_METADATA_ROOT` / `SLOTSPE_FEATURES_ROOT` 提供。`--doctor`和`--run`按服务器说明与授权执行。

P1–P4脚本 `scripts/check_slotspe_p1.py` 至 `check_slotspe_p4.py` 有 `--help` 和默认计划入口；只在明确指定 `--run` 时做患者测量。
旧 `docs/research/slotspe_eight_issues_20261019/run_synthetic_audit.py` 是冻结历史脚本，包含旧绝对路径及写出位置，不要直接在历史报告目录重跑覆盖。

## 证据与迁移

本机历史患者缓存/原版权重仍存在，其SHA与既有记录一致。它们不代表SPE09服务器新回传；已核对服务器报告提交1de42c7，诊断脚本与原始产物尚未取得验收。
新仓库只推送代码、报告、检查记录及不含患者输入的任务包。患者标识/原始数组、完整论文文本与下载包保留本地，不进入Git。
历史JSON/脚本中的 `E:/DCT-Reg` 和旧桌面绝对路径保持原文以保全哈希；本机映射见忽略的 `LOCAL_MIGRATION_MANIFEST.json`。迁移核验只证明复制完整，未重新证明P1–P4数值，也不保证旧绝对路径审计器能在其他机器直接读取新位置。
`PUBLICATION_MANIFEST.json` 记录公开文件的来源与哈希；整理检查见 `docs/REPOSITORY_MIGRATION_VALIDATION.json`。

原始NPZ与报告不是重复副本；ARCHIVED.txt不构成删除授权。原DCT工作区与目标目录既有论文资料均保留。
