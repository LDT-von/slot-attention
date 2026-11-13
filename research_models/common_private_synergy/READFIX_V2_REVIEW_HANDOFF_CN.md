# Readfix v2 完成后的已有回执导出

本轮只补齐已完成实验的汇总证据。入口 `scripts/export_full_slot_receipt_review.py` 只用 Python 标准库读取两个指定的 JSON receipt，不加载 checkpoint、不读取患者缓存、不导入 torch、不执行模型、不使用 GPU。新产物写到新的 `local_evidence/`，原实验目录保持只读。

上游已经直接核实 `5c424412e8d9f0948a94fb60a8b2668096e708d8` 的结果文档、PROGRESS 和相对 `245da18` 的两文件差异；这不等于重新审计服务器原始 receipt/权重。完成报告的统计与解释复核见 PROGRESS 的 `full_slot_readfix_v2_completed_report_review_20261113`。原服务器报告保留原样，通过追加复核记录修正解释。

## 本轮要回答的问题

1. `p_common` 低质量计数增加时，实际角色 MSE、读取重叠、原始槽变化、固定头的单槽删除影响怎样变化？按全部 epoch 看，不用最佳分数代替机制证据。
2. 原始槽梯度范数中，role 与 survival 的记录如何变化？它们不是梯度方向，也不是梯度冲突或因果归因。
3. 哪个槽索引、哪次循环贡献了低质量计数？现有聚合只有分别按槽、按迭代统计，没有槽×迭代×患者交叉表；不得编造个体定位。

## 先保留这些解释边界

- 实际 decoded common role 通过固定 rank-2 投影，理论秩上界为 2；1.73–1.96 不能据此判为坍塌。与 width64 的 raw slot 有效秩分开。导出器保留旧的 ambient bound，另加 architectural bound，不更改旧诊断或历史 receipt。
- `minimum=0` 表示至少一条患者×迭代×槽观测完全下溢，不代表整个共同角色失效。低于阈值分母是该观测数，不是独立患者数。有限非零槽范数也不能证明仍有有用信息。
- `residual` 是“剩余角色”，不是“冗余槽”。经验 PCA/CCA 角色名不等于识别出真实共有/私有生物来源。
- 继续同时报告预定规则选出的 best 和 last，均标为开发验证。跨 epoch 的 mean±sample-SD 是相关、非平稳训练轨迹的描述，不是模型的泛化误差条，也不替代预定选模规则。[选择偏差原始研究](https://jmlr.org/papers/v11/cawley10a.html)。
- 当前公开材料未附“置换 SD≈0.08”的算法、种子、重复次数或回执。若服务器已有原始置换证据，补其方法与汇总、哈希；若没有，明确撤回其作为已测噪声尺度的用法。本轮不新增置换或预测作业。C-index 的可比较对取决于事件/删失和时间，不仅是人数，[指标定义](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_censored.html)。
- frozen 在固定 bag、固定槽参数下的读统计恒定，是预期行为；不能单凭它区分 task 梯度、继续 role 拟合或优化器变化的贡献。
- 现有 receipt 不包含 Q/K 范数、logit 差、log 质量、epsilon 混合权重或逐槽删除后的 NLL。此轮标为未记录；不修改历史日志、不重跑去补成“当时的观测”。

当前持续 full-slot 研究已获用户授权，AGENTS 顶部最新指令允许必要实现与机制验证，不需重复申请历史审计线的机制豁免。本轮的具体执行仍严格限于下述已有 JSON 导出；后续候选训练需先固定协议。

## 执行步骤

使用独立新 checkout，避免对 `/data1/SlotSPE`、原失败 checkout 或已完成 readfix sandbox 做 pull/checkout/改文件。若下面目标目录存在，换一个新的编号，保留已有目录。在新 clone 内核对用户消息给出的本次交付 commit；如果分支又有新提交，先切到该指定 commit 再执行。

```bash
git clone --branch codex/full-slot-research-20261111 --single-branch git@github.com:LDT-von/slot-attention.git /data1/slotspe_receipt_review_20261113_01
cd /data1/slotspe_receipt_review_20261113_01
git rev-parse HEAD
python -S -m unittest discover -s tests -p test_full_slot_receipt_export.py -v
python -S scripts/export_full_slot_receipt_review.py \
  --receipt joint \
    /data1/slotspe_full_slot_fixprobe_20261111_01/local_evidence/full_slot_joint_readfix_v2_20261111_01/receipt.json \
    6723f1fdd730ef16c3a851995f93c3d5de31a5df786f7ced4570f372df226808 \
  --receipt frozen \
    /data1/slotspe_full_slot_fixprobe_20261111_01/local_evidence/full_slot_frozen_readfix_v2_20261111_01/receipt.json \
    bbafe3788656332712dd9b470a2853b0038a9b5d6af007f6d84739d8893e474b \
  --output local_evidence/readfix_v2_receipt_review_20261113_01/aggregates.json
```

receipt SHA256 不匹配、字段缺失、非有限数、分母/维度不一致或 best/last 与完整曲线不符时，保留错误并停止，不改源 JSON、不把导出器校验绕开。输出文件存在时不覆盖。

应得到 joint 的 12 role + 35 task、frozen 的 12 role + 25 task，共 84 个 epoch；`audit` 会列出数量和重算 best/last 的一致性。这里只核验 receipt 文件哈希与内部一致性，checkpoint 哈希是 receipt 中的原报告值，不声称重新读取或重载权重。

## 回传内容

- 记录实际 commit、Python 版本、命令、退出码、9 项检查结果、导出器/两份输入 receipt/新 aggregates 的 SHA256。明确本轮 `model_executed=false`，不把以前的 GPU 执行写成本轮执行。
- 全部 84 个 epoch 的 `train_role_loss`、`train_survival_NLL`、`validation.role_target_MSE`、`validation.roles`、`validation.slots`、`validation.raw_reading_mass`、`validation.ablation_mean_abs_logit_change`、`train_raw_reading_mass`、`raw_slot_gradient_mean`，由导出文件提供。不要只回传挑选轮次或 C-index 曲线。
- 在 PROGRESS 追加一项六字段记录，可把导出文件的 `runs` 作为该项的汇总证据；保持原有条目不变。导出器仅输出预定汇总字段，仍须确认没有另行加入患者 ID、预测、逐患者数组或权重。原始文件继续保留在 ignored `local_evidence/`；仅推送公开的汇总进展条目并回传新 commit。
- 对原交接文档声称“14 个源文件”说明：当前两臂公开 `source_SHA256` 实为 13 项且一致。如果原 receipt 另有第 14 项，给出路径与哈希；否则纠正数量，不伪造补项。
- 若有已存在的置换证据按前述补齐；没有就写“未提供/未执行”，不要为了维持 SD≈0.08 的旧说法另造实验。

收到完整汇总后，上游先检查角色拟合、读退化、槽删除影响是否同向，再锁定下一轮有限机制协议。已有完整网络合成对照没有给出稳定的生存收益，因此不直接启动五折多 seed 扩展或把读取修复与残差辅助合并上线。
