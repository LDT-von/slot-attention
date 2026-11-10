# 真实患者生存小试跑交接

**分折修复接续（2026-11-10）：**早期准备流程存在“配置声明 `5fold_uni2h`，原版加载器实际读取 `5fold`”的缺陷。当前独立入口使用 `pilot_split.py` 的显式选择器；旧队列和原版源码不改。训练前要求新格式分折身份和完整排除台账。包括真实集合正确但缺少身份记录的历史缓存，也不会自动升级为可训练缓存；保留它们作历史证据。新增 `--check-prepared` 只核对 CSV／manifest 中的集合元数据，不读取特征、不运行模型。修复与边界算术的合成验收见 [分折修复回执](split_guard_delivery_20261110.json)。

2026-11-10：用户明确选择“真实患者生存小试跑，先补齐接入”。此授权更新此前暂停患者接入的阶段边界；当前交付只在本地使用合成患者验收，尚未读取真实患者输入或执行服务器任务。

本入口运行当前 `RoleTargetModel` 的生存版本，复用角色编码、角色解码和有界中心化交互，保留两阶段训练。研究目标是工程可执行性及探索性验证。真实来源未知时，角色定义为相对训练折线性参照的可预测／剩余内容，不宣称结构来源纯度、噪声已知或 PID 分解。

## 已锁定的小试跑

- BLCA / 现有 UNI2-h / DSS / `5fold_uni2h` / fold0 / seed3。
- 每次访问从全部真实 patch 中无放回采样至 4096，不足补零；新研究模型显式 mask 补零位置。原版 SlotSPE 不变。此 mask 与原版预测协议有差异，未来效果比较必须做匹配对照。
- 两侧各 1 个共有、1 个剩余槽，宽度 32、循环 2 次。这只是显式容量控制，没有比例搜索或最优比例主张。
- 逻辑 batch 2；角色阶段 3 轮，预测阶段 3 轮；每阶段使用固定末轮权重，不按验证集选 epoch。当前为一次固定种子的验证输入采样，没有稳定性认证。
- 训练折内：病理完整真实 patch 均值、拼接通路表达，分别标准化并截断 SVD 到 8 维；在有限目标坐标上拟合普通线性 CCA，共有秩 2。该内容目标不覆盖全部病理分布、空间或高阶信息，不使用患者总体未知的结构载荷和噪声。
- 输入：病理预提取 token；组学按通路形成 token，基因位置保持原签名顺序，补到最大通路宽度并追加通路 one-hot。签名内容及顺序由准备阶段原始 CSV 哈希约束。
- 时间边界只由训练折非删失时间的四分位拟合。损失采用完整观察区间计数的离散 hazard NLL，删失所在区间的部分随访不进入损失。该标签协议与历史原版 NLL 数字不可直接混比。
- 预测头只读取四个实际角色输出，输出 4 个 hazard logits；保存末轮权重、适配器、参照、时间边界、训练损失、验证 NLL/C-index、角色目标误差和逐患者预测。患者产物只写 ignored `local_evidence/`。

配置：[common_private_patient_pilot.json](../../configs/common_private_patient_pilot.json)。入口：[run_common_private_patient_pilot.py](../../scripts/run_common_private_patient_pilot.py)。

## 服务器执行顺序

在本交付 Git 分支的仓库根目录执行。使用服务器已有环境与真实数据路径；无需重新提取特征。确认当前 checkout 的 `third_party/SlotSPE` 完整存在。

```bash
python -B scripts/run_common_private_patient_pilot.py --plan
python -B scripts/run_common_private_patient_pilot.py --doctor
```

`--plan` 不导入 torch、不读患者值；`--doctor` 检查依赖、代码哈希和既有环境变量是否存在，不读取患者值，也不代表数据已验收。依赖包含 torch、numpy、pandas、h5py、scikit-survival，不自动安装或升级。

部署本次修复时先运行生成输入的分折检查，Linux 上应实际覆盖符号链接案例。该测试只构造人工 CSV 和小型张量，包含一次 CPU 人工模型测试以核对评估前的拦截，不使用真实患者输入：

```bash
python -B -m pytest -q --confcutdir=research_models/common_private_synergy \
  research_models/common_private_synergy/deferred_checks/test_pilot_split.py
```

该模块的临时文件写入当前 checkout 的 ignored `local_evidence/pst/`，`--confcutdir` 避免调用历史根 conftest 的跨目录临时路径逻辑。若服务器未安装 pytest，先报告环境缺项，不自动安装或升级。

只有包含 `split_identity.schema=pilot_explicit_split_v1` 的新缓存可以进入当前训练入口。先进行只读检查：

```bash
python -B scripts/run_common_private_patient_pilot.py --check-prepared \
  --prepared-root local_evidence/已准备目录
```

该检查会打开分折 CSV 和 manifest 中的患者身份元数据，输出路径、人数和摘要，不输出患者 ID，不读取特征或模型权重。需要核对：请求 CSV 和实际选择器的逻辑路径、真实路径与内容哈希；保留集合及顺序；排除原因；每个 split 的“请求＝保留与排除的不相交并集”；训练／验证互斥。符号链接允许存在，但记录其解析后的文件身份。旧缓存缺少这些字段会直接拒绝，不能仅补写几项字段使其通过。

`--run` 会在读取缓存和拟合参照前完成该检查，并验证原版来源、配置、CSV 与缓存文件哈希；评估前再次核对身份及 manifest。准备／复用必须经具体执行授权，当前修复本身不启动新的患者准备、训练或推理。

否则先准备新缓存。服务器需已设置 `SLOTSPE_METADATA_ROOT` 和 `SLOTSPE_FEATURES_ROOT`。也可使用 `--data-config` 指向原有绝对路径配置 JSON。准备会读取真实输入，但不会训练模型：

```bash
python -B scripts/run_common_private_patient_pilot.py --prepare \
  --output-root local_evidence/common_private_pilot_prepare_splitfixed_20261110_01
```

完成后启动训练。两个目录都必须是新目录；已有失败产物保留，不自动覆盖或恢复：

```bash
python -B scripts/run_common_private_patient_pilot.py --run \
  --prepared-root local_evidence/common_private_pilot_prepare_splitfixed_20261110_01 \
  --output-root local_evidence/common_private_pilot_blca_f0_s3_splitfixed_20261110_01
```

在有 CPU 环境的机器上，另可运行完整合成患者冒烟。它会执行两阶段训练、保存并重建模型；不读取真实患者输入：

```bash
python -B scripts/run_common_private_patient_pilot.py --smoke \
  --output-root local_evidence/common_private_pilot_synthetic_20261110_01
```

## 判读与后续门槛

首轮验收看：是否完成；训练损失是否有限；生存头是否正常；角色在任务阶段是否确实冻结；保存重载是否一致；验证集 NLL/C-index 是否可计算。短程模型可能未收敛，不能用一次分数否定或证明最终方法。

当前没有同协议 S0 对照，没有额外 seed/fold，也没有独立测试集结果。历史验证分数不能与本次直接比较并称为提升。后续性能验证需在相同数据准备、mask、标签、预算、种子和 checkpoint 协议下增加对照。有限参照的泛化、角色泄漏、合法条件采样与纠正收益仍独立审查。

原版源码、既有患者队列、历史失败与回执保持不变；本入口没有接入运输或双视图/SCM/SRO/残差对比等附加训练项。推送范围只含可执行试跑及必要角色组件，不将患者缓存、权重、预测或标识纳入 Git。

## 历史报告的解释边界

服务器回传的 run02 是 `5fold` 家族内的 fold0 单折，不是五折实验；另一 uni2h 试跑通过隔离 metadata 根的符号链接改变了硬编码路径的实际目标。这些属于用户回传的服务器证据，未在本地重新验收患者产物。历史目录、参数和回执不改名、不覆盖。

`pd.cut(right=False)` 与内部边界的 `searchsorted(side='right')` 在边界点均进入下一桶；四例标签差异不能由这两个规则的差异来解释。新测试用已知边界独立验证这一点。原版全临床等宽分箱、训练折分位数标签和 pilot 的删失 NLL 是不同事项；应各自按真实执行版本追溯。

同一预测在两种 NLL 下的差可用于描述损失口径差。未经挑选的重复均值没有必然抬高分数的方向；独立 best 队列与末轮队列之间的全部分数差也不能只归因于选 epoch。self-pool A/B/C 与共有／剩余角色研究的消融臂是不同定义，不能混用。新训练是否必要要在数据／评估协议可信之后决定。
