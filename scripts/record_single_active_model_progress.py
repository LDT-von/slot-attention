"""Add the single-active-model consolidation entry to docs/PROGRESS.json.

The progress file is the single independent progress entry for this
repository.  This script appends one entry and updates the recorded_at
stamp.  It does not touch any historical entry, does not edit any old
report, and does not start training, inference or a patient intervention.

Run from the repository root:

    python -B scripts/record_single_active_model_progress.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT / 'docs/PROGRESS.json'

KEY = 'latest_single_active_model_consolidation_20261117'

ENTRY = {
    'recorded_at': '2026-11-17T21:55:00+08:00',
    'route_task': (
        'SlotSPE problem discovery / 按用户明确指令"改完必须只能剩下一个模型"，'
        '把 common-private 研究线收敛为唯一活跃模型 scaled_role16_v1。'
        '本阶段只做代码重组、逐字保全与本地验证，'
        '未在患者数据上训练、未推理、未干预。'),
    'status': 'single_active_model_consolidated_locally_no_patient_execution',
    'user_steering': (
        '用户先问 scaled16 的创新点、方法一到六的主要区别、各自的损失函数，'
        '再问能否只保留一个模型并让槽同时承载共有信息与互有信息，'
        '再逐项确认 task_mode / 角色目标统计量 / slot_query_bias 是否需要，'
        '最后下达"改完必须只能剩下一个模型"。'),
    'completed_work': (
        '第一，scaled16 相对原 8 槽骨架的实测改动两处。'
        '(1) 零 GRU 更新偏置：在 RoleSlotEncoder.update 的 nn.GRUCell 上把 bias_ih 与 bias_hh 清零，'
        '初始 16 槽有效秩从 6.22 升到 9.67（满值 16），槽对余弦从 0.8160 降到 0.6061。'
        '这是直接消除槽初始塌缩的机制，不是加了惩罚项；'
        '此前 R1/R5 上的正则化惩罚会掉性能，所以没有沿用。'
        '(2) continued_role_only：任务阶段把 roles detach 后再送入 BoundedRoleReadout，'
        '生存损失的反向边被切断，角色网络只由角色内容 MSE 训练。'
        '合成检查 9 已验证：joint 模式下生存梯度确实进角色网络，'
        'continued_role_only 模式下进不去，而角色梯度仍然进得去。'
        '这两项改动落在 scaled/role_model.py 与 scaled/engine.py，'
        '归档臂的源码一字未改（脚本 check_scaled16_synthetic.py 的 CHECK 8 断言归档编码器文件本身没被编辑，'
        '改动是从外部施加的）。'
        '第二，把六种方法收敛为一个。'
        '原 V0 joint-MLP 参考、matched35、full_slot、spread、symmetry、Cox 全部降级归档，'
        '只有 scaled_role16_v1 保持活跃。归档位置：'
        'research_models/common_private_synergy/archive/arms/（五个臂）、'
        'research_models/common_private_synergy/archive/legacy/（V0 参考）、'
        'archive/（各臂运行脚本）、configs/archive/（各臂配置）。'
        '第三，逐字保全与审计路径不变。'
        '每个归档文件都保留原始字节，且原路径通过软链继续可解析，'
        '所以 sha(ROOT / name) 查表与 receipt 复验完全照旧可用。'
        '新增 research_models/common_private_synergy/MODEL_REGISTRY.json（唯一活跃模型与各归档方法及其原因）、'
        'research_models/common_private_synergy/archive/ARCHIVE_MANIFEST.json（每个文件的归档路径、原路径、SHA-256）、'
        'scripts/check_single_active_model.py（唯一活跃模型审计）、pytest.ini（同一测试模块只按 receipt 路径收集一次）。'),
    'measured_facts': (
        '本机实测，均为合成张量或源码层面的事实，不是患者性能数字。'
        '零 GRU 更新偏置：16 槽初始有效秩 6.22 -> 9.67，槽对平均余弦 0.8160 -> 0.6061。'
        '梯度隔离：mode=joint 时生存梯度到达角色网络为 True、角色梯度到达为 True；'
        'mode=continued_role_only 时生存梯度到达为 False、角色梯度到达为 True。'
        '归档审计：37 个归档文件全部与重构前 git 对象逐字一致；'
        '42 个 receipt 锁定路径全部仍然可解析，其中 40 个字节与 receipt 完全一致；'
        '2 个（components/adapters.py、components/assembly.py）在重构之前就已经与 receipt 不一致，'
        '本次重构原样保留，属于既有差异而非新引入。'
        'pytest 总数 688，与重构前基线逐项相同，没有因为归档而丢测试或重复收集。'),
    'paths_commit': (
        '研究仓库 /data1/slotspe_matched35_20261113_01，分支 codex/scaled-16slot-20261117：'
        '68e74dd 零 GRU 偏置与 continued_role_only、'
        'cfcc214 归档五个臂与 V0 参考并新增注册表/清单/审计脚本/pytest.ini、'
        '7b38352 把归档审计基线钉在重构前的提交 68e74dd（重构后 HEAD 存的是软链 blob，'
        '不再承载原始文件内容，所以基线不能取 HEAD）。'
        '已推送到 origin/codex/scaled-16slot-20261117。'
        '本仓库 /data1/SlotSPE 只更新 docs/PROGRESS.json。'),
    'checks_and_actual_execution': (
        'python -B scripts/check_single_active_model.py：registry_active_models=1、'
        'archived_files=37、receipt_pinned_paths_verified=42、problems=[]、ok=true。'
        '该检查同时验证三件事：注册表只声明一个活跃模型且入口/配置/合成检查文件都存在；'
        '活跃 runner 与 scaled 包的真实 import（按 AST 解析，不是文本匹配）不触及任何归档模型模块；'
        '每个 receipt 锁定路径仍可解析且字节符合记录，每个归档文件等于重构前的 git 对象。'
        '还做了反向验证：给归档文件追加一行注释后，检查立刻报出 receipt 字节不符与 git 基线不符两处失败，还原后恢复 ok=true。'
        'pytest 全量 688 passed（重构前基线同为 688 passed）。'
        'research_models/common_private_synergy/checks/run_static.py 仍是 22 项通过、torch 未被导入。'
        'scripts/check_scaled16_synthetic.py 九项全通过（含 CHECK 8 零偏置有效秩断言与 CHECK 9 生存梯度隔离断言），'
        'CHECK 7 确认 13 个归档依赖源文件未变。'
        '三个归档臂自己的合成套件仍然通过并各自复核了自己的记录源哈希：'
        'check_spread_arm_synthetic（14 个记录哈希仍匹配）、'
        'check_symmetry_arm_synthetic（14 个记录哈希仍匹配）、'
        'check_cox_objective_synthetic。'
        '真实执行：本次未在患者数据上训练、未推理、未干预，也没有发起任何患者实验。'),
    'gaps_and_reasons': (
        '第一，本次没有患者性能数字。所有检查都是合成张量、源码哈希与导入图，'
        '它们只证明归档与收敛正确，不证明 scaled_role16_v1 的患者表现，也不证明它比任何基线好。'
        '第二，有两个类因为历史 receipt 的逐字锁定而物理上留在原处：'
        'components/role_model.py 里的 RoleTargetModel（旧 8 槽骨架）与 '
        'patient_pilot_engine.py 里的 SurvivalRoleModel。'
        '前者所在文件同时提供 scaled16 需要的 BoundedLinear 与 BoundedRoleReadout，'
        '不能整文件删除；后者所在文件被 receipt 锁定。'
        '两个类都不可从活跃 runner 到达，已在 MODEL_REGISTRY.json 里标为 legacy_retained_for_hash_lock。'
        '所以"只剩一个模型"在可执行层面成立，在文件字节层面有两个不可删除的历史类残留，这一点如实记录。'
        '第三，归档重排改变了 git 的存储形态：原路径现在是软链，'
        '所以 sha(ROOT / name) 依然有效，但用 git show HEAD:<原路径> 取原始内容会得到软链目标字符串而不是文件内容，'
        '审计基线已钉在 68e74dd。'
        '第四，README.md / ARCHITECTURE.md / HANDOFF.md 描述的是旧的 8 槽设计，'
        '它们被 validation_receipt.json 逐字锁定，不能改写；'
        '本轮以新增 MODEL_REGISTRY.json 的方式取代它们，不改历史文件。'
        '第五，患者队列身份问题没有因本轮改变：'
        '默认队列配置写的是 5fold_uni2h，但原版加载器硬编码 splits/5fold，'
        '已完成队列与 S0/R4/R3 基线都是 5fold，两者每折只共享 12 到 17 名验证患者；'
        '任何复用这些基线的臂必须继承 5fold。'),
    'next_action_and_user_inputs': (
        '下一步需要用户授权后：(1) 在真实数据上跑 scaled_role16_v1 一臂'
        '（16 槽、宽 256、batch 8、4096 patch、task_mode=continued_role_only、'
        'p_target_mode=spread、slot_update_bias=zero），'
        '每轮 CPU 载入约 82 秒、float64 显存峰值约 2.1 GiB，'
        '启动时不要再加会切断长跑的短 timeout，或配合 --resume；'
        '(2) 同时跑原始 S0 与本臂的同队列对照，队列身份必须一致，不能按配置名推断；'
        '(3) 用 scripts/export_slot_relation_tables.py 实测 16 槽的组内余弦与跨模态典范相关，'
        '确认零偏置带来的分化在真实数据上没有退化，'
        '因为共享的共有槽与互有槽必须同时成立，只看有效秩不够；'
        '(4) 归档后的三个 spread/symmetry/Cox 臂不再需要重跑，'
        '它们的作用已被 scaled16 内部吸收（spread 目标统计量、零偏置分化、离散风险 NLL），'
        '但历史数值仍可从归档目录与 local_evidence 复核。'
        '等待用户输入：是否授权患者执行，以及对照臂的队列身份按 5fold 还是 5fold_uni2h 统一。'),
}


def main() -> int:
    payload = json.loads(PROGRESS.read_text(encoding='utf-8'))
    payload[KEY] = ENTRY
    payload['recorded_at'] = ENTRY['recorded_at']
    # The historical file carries no trailing newline.  Writing one would
    # show up as a spurious diff on the last line, so the original ending is
    # preserved exactly.
    PROGRESS.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'updated': str(PROGRESS.relative_to(ROOT)),
                      'entry': KEY,
                      'total_keys': len(payload)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
