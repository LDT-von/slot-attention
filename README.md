# SlotSPE 问题核查

本仓库独立维护 SlotSPE 的 P1–P8 问题存在性、预测敏感性与修正归因。原版代码、失败记录及证据边界保留。代码/合成检查通过不等于患者预测改善。

当前入口：[进度](docs/PROGRESS.json)、[P5交接勘误](docs/research/slotspe_p5_20261020/P5_交接复核与勘误.txt)、[服务器说明](docs/SlotSPE证明任务_服务器一键运行.txt)。

| 项目 | 已有结论及边界 |
|---|---|
| P1 | 默认每模态8候选/Top-2；cross仍读全槽；固定容量损害未证。 |
| P2 | 历史短程模型读取高重叠且干预有计算差异；有害冗余/碎片化未证。 |
| P3 | 历史同权重同输入eval波动已有本地原始记录；固定两处抽样可去除该协议波动，提分未证。 |
| P4 | 无固定槽编号语义保证；真实组织/跨患者语义漂移及损害未证。 |
| P5 | 无padding mask及补零分支存在；零行历史统计缺该轮原始记录，真实无关组织占槽/损害未证。 |
| P6 | 缺显式空间/倍率接口；上游尺度内容及患者预测损害未证。 |
| P7 | 最终融合主损失到选槽器直接梯度断开，辅助仍连通；患者损害及B/C收益未证。 |
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

本机历史患者缓存/原版权重仍存在，其SHA与既有记录一致。它们不代表SPE09服务器新回传；当前服务器只有完成转述，原始产物尚未验收。
新仓库只推送代码、报告、检查记录及不含患者输入的任务包。患者标识/原始数组、完整论文文本与下载包保留本地，不进入Git。
历史JSON/脚本中的 `E:/DCT-Reg` 和旧桌面绝对路径保持原文以保全哈希；本机映射见忽略的 `LOCAL_MIGRATION_MANIFEST.json`。迁移核验只证明复制完整，未重新证明P1–P4数值，也不保证旧绝对路径审计器能在其他机器直接读取新位置。
`PUBLICATION_MANIFEST.json` 记录公开文件的来源与哈希；整理检查见 `docs/REPOSITORY_MIGRATION_VALIDATION.json`。

原始NPZ与报告不是重复副本；ARCHIVED.txt不构成删除授权。原DCT工作区与目标目录既有论文资料均保留。
