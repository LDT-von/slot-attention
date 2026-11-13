# readfix v2 完成报告的追加勘误（2026-11-13）

本勘误适用于历史报告 [SERVER_READFIX_V2_RESULTS_CN.md](SERVER_READFIX_V2_RESULTS_CN.md) 与既有 PROGRESS 完成条目；历史正文、原始 receipt、文件路径和哈希保持原样。本文件更正后续应采用的表述，不覆盖历史事实，也不新增实验结果。证据入口为 `docs/PROGRESS.json` 中的 `full_slot_readfix_v2_completed_report_review_20261113` 和 `latest_readfix_v2_completed_receipt_review_84epoch_export`。

1. **源文件计数是 13 项。** 已完成的 readfix v2 两臂 receipt 和 edc2313 失败 receipt 中，`source_SHA256` 均列出 13 个源文件；先前“14 个源文件”是笔误，没有第 14 项可补。该计数只针对上述历史 receipt；本轮新增入口和诊断的源文件清单按实际记录，不能强制套用 13。
2. **撤回“C-index 置换零假设 SD 约 0.08”作为实测噪声尺度的表述。** 缺少相应置换算法、随机种子、重复次数与原始回执，不能以验证人数推断该数值。旧报告第 17、144、153、155、230 行中由此推出的“约一个噪声标准差”“差异落在噪声内”“将噪声从约 0.08 压低”等量化判断，不再作为证据使用。`docs/research/slotspe_p6_addendum_20261020/` 的 77 人队列槽置换、patch 顺序和重新初始化研究属于另一研究对象，不能代替本队列的 C-index 置换零假设。本次没有补造置换实验。
3. **跨 epoch 的 mean±SD 仅描述训练轨迹。** 旧报告第 153 行“报告中枢请用 mean±sd”不能替代预定 best/last 规则；相关且非平稳的 epoch 不是独立重复，轨迹 SD 不是泛化误差条。保留预定 best、固定末轮及完整轨迹，全部称为开发验证。不同容量、训练预算和读取规则的旧 pilot 不属于匹配性能对照。
4. **表格分组标记有边界歧义。** 84 epoch 汇总中的 A“≤25”和 B“≥25”在 25 处重叠，不能作为互斥切分规则复用。现有汇总明确列出的 A 为 joint task ep1–26、B 为 ep27–35；引用已有数值时使用这两个时间区间，注明是事后轨迹描述。未核对原分组实现前，不擅自改写为 `<25/≥25` 或 `≤25/>25`，也不重算一组新数字冒充原统计。


5. **历史 `combined` 原始槽梯度日志混入了诊断反向结果。** 本轮源码复核发现，旧 `full_slot_engine` 在原始槽调用 `retain_grad()` 后，先用 `autograd.grad()` 测量 role/survival 分量，再执行训练损失 `backward()`；这些诊断调用会累加非叶槽张量的 `.grad`，使旧 `combined` 日志不再只是实际优化目标的梯度。本轮在正式 `backward()` 前清空这些槽张量的 `.grad`，使新 `combined` 日志反映实际合并训练目标。该清零只修正非叶诊断容器，不改变模型参数 `.grad` 或优化更新；role/survival 分量直接取 `autograd.grad()` 返回值，不受此混入问题影响。历史数值、receipt 与此前相关系数保留；任何涉及旧 `combined` 的相关分析必须附上这项测量口径限制，不能当作纯合并梯度的证据。

有限非零槽范数不能证明槽仍保有有用内容。解码后 common role 的固定 rank-2 投影使其秩上界为 2；该层有效秩接近 2 不构成原始槽坍塌证据。加 ε 归一化后的均匀读取仍可以携带平均 value 信息；它引起的 overlap 或删除 logit 影响增大，不等于内容信号增加。frozen 中读取固定而删除影响改变体现读出适应，尚不能替代共同起点的 continued-role-only 对照。

下一轮协议见 [SERVER_MATCHED35_CN.md](SERVER_MATCHED35_CN.md)，交付与本地合成检查登记于交付方本地 PROGRESS 的 `full_slot_matched35_protocol_delivery_20261113`（本轮远端 PROGRESS 沿用历史版本）。该交付不代表新患者训练已在服务器执行。
