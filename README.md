# SlotSPE 问题核查

**已经确认具体问题：原版eval仍随机采样，导致同权重同输入的预测波动；最终融合生存损失到选槽器的直接梯度断开，辅助生存监督仍连通。** 另外，补零无mask和合成零行敏感性有证据，历史训练后槽向量与读取分布也高度相似。

本仓库逐项记录“问题存在 → 是否改变预测 → 是否损害患者预测 → 哪项修正有效”。已确认现象会明确写出；高相似、sanity check和人为干预的输出变化，不直接登记为患者预测损害。

当前入口：[进度](docs/PROGRESS.json)、[服务器新结论复核](docs/research/slotspe_server_claims_20261020/服务器结论复核.txt)、[原始回传与对照要求](docs/research/slotspe_server_claims_20261020/服务器AI_回传与对照验收.txt)、[P5勘误](docs/research/slotspe_p5_20261020/P5_交接复核与勘误.txt)。

## 2026-10-20服务器新汇报：已查到提交，原始诊断尚未回传

核对到原[DCT-Reg提交1de42c7](https://github.com/LDT-von/DCT-Reg/commit/1de42c777054b061ed278eb3ecb02d0bfbd2cca3)。它只改了`AGENTS.md`和`.gitignore`，把`_p1246_diag.py`加入忽略列表；在该提交中未取得诊断JSON或脚本。这证明汇报已入库，不能代替原始结果验收，也不表示服务器没有实际运行。

服务器“六项患者损害全立”的结论目前不能接受：

- P2记录是每名患者**至少一对**去对角槽cos>0.95、max=0.9999，不等于8槽全部复制。
- P6的0.0235/0.085是risk差，P3的0.0035是C-index标准差，P5的0.011是logit差；三者不能共用0.01门槛或直接比较倍数。顺序对照还须核验高斯初槽与Gumbel抽样是否配对一致。
- P4的均值置换不变只是sanity check；P7直接梯度断开不等于选槽器完全不训练，也不等于整个Slot Attention缺少生存监督。
- 风险与观察时间的正Spearman相关不能单独判定方向翻转；须核对删失定义，并比较同队列的C(risk)与C(-risk)。[指标定义](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_censored.html)

## P1–P8当前结论

下表的“已确认”来自源码、既有合成对照或历史本地真实输入；服务器新数值单列为转述，未冒充已验收患者结果。

| 项目 | 已确认现象与剩余证据缺口 |
|---|---|
| P1 | 固定8候选/Top-2结构已确认；固定容量的患者损害未证。服务器n=64..256变量定义待脚本，不能混作槽数实验。 |
| P2 | 历史训练后高相似已观察；服务器报告77/77至少一对cos>0.95、max=0.9999，不证明8槽全复制或患者损害。 |
| P3 | 初槽不以当前患者为条件、后续迭代读取患者；同权重eval波动已确认。服务器C-index std=0.0035待原始回传，不是通用噪声上界；冷启动损害及修正收益未证。 |
| P4 | 槽编号无固定语义保证；均值置换是sanity check，不证明实际语义漂移或患者损害。 |
| P5 | 补零无mask风险及同RNG合成敏感性已确认；服务器max logit变化0.011待原始验收，真实无关组织占槽/损害未证。 |
| P6 | 显式坐标/倍率接口缺口已确认；服务器重排risk差需核对RNG，尚不能归因顺序，也不证明空间尺度预测损害。 |
| P7 | 最终融合主损失到选槽器的直接梯度断开已确认；单模态生存辅助仍连通，广义目标冲突、患者损害及B/C收益未证。 |
| P8 | 原版无后续跨模态OT，双阶段冲突不适用。 |

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
