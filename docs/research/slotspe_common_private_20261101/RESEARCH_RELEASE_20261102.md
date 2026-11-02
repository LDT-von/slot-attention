# 本次研究版本推送交接

发布分支：`codex/common-private-research-release-20261102`。研究基点：`f62c988e6bec9c5ea345b3dfbcce13c8bc2c87b4`；最终发布提交SHA以交付回复及远端核对为准。

**已完成的是可交接的研究原型与已记录验收，完整新方法尚未完成。** 目前没有证实共有／私有语义分工、全部槽有效利用或稳定预测优势，也没有该新网络的患者数据适配器。本版本不能直接充当患者训练交付。未启动患者训练／推理，未操作服务器或正在运行的队列。

## 本次包含与保留

- 已实现的`CrossmodalSharedBottleneck`：两模态各8槽、共有随机向量、条件私有槽及全槽读出；信息预算不等于已证明降低有害重复。
- 已提交的数学、语义、KL与读出研究回执，保留负结果和评估口径。
- 独立有界原槽残差组件及验收；仅作为信息保留控制，没有并入主原型或队列。
- 本次独立发布检出重新运行4个相关测试文件，35 passed in13.93s；未重跑全仓库测试，未声称历史CUDA问题已解决。
- 本地两份未完成草稿`task_anchor_plan_20261102.json`、`task_anchor_audit.py`保留在研究工作树，未纳入本发布。独立任务锚模块尚未实现／验收。

旧`SERVER_HANDOFF_20261102.md`是历史交接，其当时的未完成状态不能替代后续回执。本次先读本页和研究导航README。原远端两个研究分支仍为`0245876bdfa0e2f3d04b14eff54248dc1444e6f5`；本次采用新分支，不合并或覆盖其中队列修复与服务器报告。不要用本发布工作树替换正在执行的队列目录。

## 服务端只读接收与合成复核

在已有Git仓库执行，使用全新工作树目录，保持现有目录不变：

```bash
git fetch origin refs/heads/codex/common-private-research-release-20261102:refs/remotes/origin/codex/common-private-research-release-20261102
git worktree add --detach ../slotspe-cp-research-release-20261102 origin/codex/common-private-research-release-20261102
cd ../slotspe-cp-research-release-20261102
git rev-parse HEAD
python -B scripts/check_delivery_manifest.py --check --require-tracked
python -B -m pytest -q tests/test_crossmodal_shared_bottleneck.py tests/test_prior_normalized_slots.py tests/test_bounded_native_residual.py tests/test_native_preservation_audit.py
```

测试使用生成输入，不读取患者数据。不要在现有训练环境中为复核盲目升级依赖；本机CPU验收不代表CUDA／AMP通过。本页不提供不存在的患者训练命令，不重复启动已有合成训练矩阵。

## 下一研究缺口

当前共有分支主要由交叉内容与分布约束学习；联合生存loss不保证共有分支一定携带预后相关的共有信息。后续应先完成任务相关分工的数学与已知答案筛选，再决定实现／训练范围。患者阶段需补数据适配、匹配基线、训练折预处理／分箱、选权和独立测试，并另行授权。

六项交付：路线／任务=已提交研究版本推送；完成=独立发布检出与聚焦回归；路径／提交=本分支、本页及最终SHA；检查／实际执行=35项测试、清单和原文件哈希核验，无患者／服务器执行；缺口=语义分工、MI机制效力及患者入口未成；下一动作=接收与审阅本版本，依据明确的研究范围继续，不自动启动患者任务。
