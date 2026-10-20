"""Verify all outputs and make a concise report from measured results only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,math
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    model=read('model/proof_results.json')
    numeric=read('model/numeric_controls.json')
    protocol=read('protocol/protocol_proof_results.json')
    reconstruction=read('reconstruction/reconstruction_proof_results.json')
    training=read('patient/training_protocol.json')
    manifest=read('patient/manifest.json')
    rows=read('patient/summaries.json')
    assert model['all_assertions_passed'] and numeric['all_assertions_passed']
    assert protocol['status']=='PASS' and protocol['original_files_unchanged']
    assert reconstruction['source_unchanged'] and training['source_unchanged']
    assert len(rows)==6 and {(r['seed'],r['arm']) for r in rows}=={
        (s,a) for s in [3,17,29] for a in ['original','readout_fix']}
    source=Path(model['source'])
    assert all(digest(source/p)==h for p,h in training['source_hashes_before'].items())
    assert all(digest(p)==h for p,h in protocol['source_hashes_before'].items())
    assert digest(ROOT/'fixed_model.py')==training['fixed_model_sha256']
    assert digest(ROOT/'patient/run_patient_proof.py')==training['runner_sha256']
    assert digest(ROOT/'patient/manifest.json')==training['manifest_sha256']
    same=[]
    for seed in [3,17,29]:
        pair=[r for r in rows if r['seed']==seed]
        assert pair[0]['same_initial_state_sha256']==pair[1]['same_initial_state_sha256']
        assert pair[0]['parameter_count']==pair[1]['parameter_count']
        for r in pair:
            path=ROOT/f"patient/seed_{seed}/{r['arm']}/last.pt"
            assert digest(path)==r['checkpoint_sha256']
            assert r['epochs']==3 and r['optimizer_steps']==111
            assert r['main_gradient_connected_parameter_count']==(0 if r['arm']=='original' else 8)
            if r['arm']=='readout_fix':
                assert r['main_gradient_nonzero_parameter_count']>=4
                assert r['patient_risk_std_max']<1e-10
        e=read(f'patient/seed_{seed}/same_weights_eval_fix/summary.json')
        same.append(e)
    changes=[f['interventions'][0]['original_training_labels_changed'] for f in protocol['folds']]
    original_masks=[r['fixed_set_intervention']['original_pool_max_change'] for r in model['records']]
    evaluation_ranges=[r['original_eval_max_range'] for r in model['records']]
    count=manifest['counts']
    lines=[
      'SlotSPE 问题存在性与最小修正：证明实验结果（2026-10-18）',
      '',
      '结论：已证明三项模型实现行为及一项数据协议依赖，并在独立最小修正中消除对应现象；没有证明 OT 必要，也没有证明修正提升患者泛化。',
      '',
      '1. 选槽器是否收到最终融合生存监督',
      '随机权重5个seed + 真实患者训练后3个seed，原版最终生存NLL对选槽打分器8个参数张量的梯度均为None。',
      '修正版保持相同参数数量，让ST选择值进入self分支汇总，训练后8个张量均连接并出现非零选槽梯度；逐seed的连接/非零数量见表。',
      '原版仍有单模态辅助监督，不能说整个选择器从未训练。ST是梯度估计，不是离散选择的无偏真实导数。',
      '',
      '2. self分支的Top-K是否真正排除了未选槽',
      f'固定选中集合，只扰动未选中槽，原self汇总最大变化在5个seed为{min(original_masks):.6g}到{max(original_masks):.6g}；修正后全部为0，未选槽到self汇总的梯度也为0。',
      '修正使用选中槽汇总；cross分支保留原版全槽设计。因此不能把结论写成整个原模型全局稀疏，也不能说原cross丢掉了所有互补信息。',
      '',
      '3. 已训练权重的单次验证是否随机',
      f'随机权重5seed原eval的最大logit极差为{min(evaluation_ranges):.6g}到{max(evaluation_ranges):.6g}，修正后为0。',
      '另用真实患者训练后checkpoint，每个种子在同一验证患者上重复20次；下表给出风险标准差和C-index变化。',
      '同权重eval_fix对照不再训练，排除了“只是因为修正版权重不同所以更稳定”的解释。',
      '',
      '4. 验证结局是否影响训练标签',
      '真实BLCA五折，只改一位验证患者时间且训练原始结局完全不变，原训练标签变化人数：'+', '.join(map(str,changes))+'。',
      '只用训练折拟合分箱后，同样干预导致的变化全部为0；原参考生存数组含验证患者，修正参考与验证ID交集为0。',
      '保留验证结局重排/行倒序阴性对照，不把所有数据扰动都预设为失败。不能据此声称Harrell C-index被必然抬高。',
      '',
      '5. 保留失败与否定证据',
      '第一版修正虽连通梯度，但seed17梯度范数仅约1e-15到1e-19，按预设1e-12阈值失败；该版本和失败结果均保留，未用于患者训练。',
      '最终控制保留原硬采样，单独以温度1估计选择梯度；原0.01温度仍用于硬选择算法与选中支持内gate。',
      '重构常量反例仅在关闭dropout的重构子目标下三路为0；开启原训练dropout后同一构造不是0。',
      '因此撤回“已证明正常患者训练坍缩”的任何推断。detach本身不能排除常量解；冻结非退化目标只是一项证明对照，未并入此次模型修正。',
      '',
      '真实患者诊断协议',
      f'BLCA原fold0，complete-case WSI/RNA配对；counts={count}。',
      '两臂共同使用训练侧未删失时间的分位数分箱；256个固定真实patch，331条完整通路；256维/8槽/10迭代，batch8。',
      '3个seed(3,17,29)，每臂3epochs/111次更新；相同初始化、batch顺序和步骤seed；Adam lr5e-4，无weight decay，cosine eta_min1e-6。',
      '保存固定第3轮checkpoint，不挑验证最高轮；每个checkpoint在71位验证患者上重复20次，未使用独立测试集。',
      '这与原30轮、训练随机采样4096patch、batch32的完整协议不同，只能支持短程机制诊断。',
      '',
      'seed | arm | 最后checkpoint验证C-index均值±标准差(探索) | 平均患者风险标准差 | main梯度连接/非零张量',
    ]
    for r in rows:
        lines.append(f"{r['seed']} | {r['arm']} | {r['val_cindex_mean_exploratory']:.6f} ± {r['val_cindex_std_population']:.6f} | {r['patient_risk_std_mean']:.6g} | {r['main_gradient_connected_parameter_count']}/{r['main_gradient_nonzero_parameter_count']}")
    lines.extend(['','同权重eval_fix对照（不再训练）'])
    for r in same:
        lines.append(f"seed {r['seed']}：同一原版checkpoint，平均风险标准差 {r['original_eval_patient_risk_std_mean']:.6g} → {r['patient_risk_std_mean']:.6g}；C-index标准差 {r['original_eval_val_cindex_std_population']:.6g} → {r['val_cindex_std_population']:.6g}；新增优化步0。")
    lines.extend([
      '', '解决范围与下一步判断',
      '当前实现修正已经消除所测梯度/汇总/随机推理/训练标签依赖验证结局的现象。修正代码独立放在fixed_model.py，原始模型与DCT正式方法未覆盖。',
      '这些是工程与评估问题，不能直接充当论文原创模块。跨模态事件的条件增量价值仍是研究假设，需要在干净基线上用匹配预算的患者实验证明。',
      '如果修正后并无稳定泛化收益，仍按实现行为改正报告，不把“测试通过”写成“好方法已验证”。',
      '', '证据入口',
      'model/proof_results.json：5seed梯度/汇总/随机评估证明。',
      'model/v1_rejected.json：被否定的第一版修正。',
      'model/numeric_controls.json：全选极限与极端门控数值对照。',
      'protocol/protocol_proof_results.json：5fold分箱与参考数组因果干预。',
      'reconstruction/reconstruction_proof_results.json：重构反例及不成立边界。',
      'patient/：输入manifest、缓存、协议、6个末轮checkpoint、曲线、逐患者重复预测与训练后梯度。',
      '原始源文件、CSV和划分未改变；所有主张均以当前保存产物为依据。'])
    (ROOT/'证明实验结论.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    evidence={'status':'all_completed','trained_checkpoints':len(rows),'same_weight_controls':len(same),
              'claims':{'gradient_disconnect':'proved_and_repaired','self_readout_exclusion':'proved_and_repaired',
                        'eval_randomness':'measured_and_deterministic_control_verified',
                        'validation_outcome_dependence':'proved_and_repaired',
                        'actual_patient_training_collapse':'not_proved',
                        'generalization_improvement':'not_established','ot_necessity':'not_established'},
              'source_unchanged':True,'summary_rows':rows,'same_weights_controls':same,
              'artifact_sha256':{p:digest(ROOT/p) for p in ['fixed_model.py','run_model_proofs.py',
                  'model/proof_results.json','protocol/protocol_proof_results.json',
                  'reconstruction/reconstruction_proof_results.json','patient/training_protocol.json',
                  'patient/summaries.json']+(['source_snapshot/snapshot_manifest.json']
                    if (ROOT/'source_snapshot/snapshot_manifest.json').exists() else [])}}
    (ROOT/'evidence_index.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    orange='#da7b26';green='#177d73';x=np.arange(5)
    axs[0,0].bar(x,changes,color=orange,label='原实现')
    axs[0,0].scatter(x,[0]*5,color=green,label='训练折专属分箱',zorder=3)
    axs[0,0].set(xticks=x,xticklabels=[f'折{i}' for i in x],ylabel='改变的训练标签人数',title='仅改变一名验证患者时间')
    axs[0,0].legend(fontsize=9)
    axs[0,1].bar(x,original_masks,color=orange,label='原self汇总')
    axs[0,1].scatter(x,[0]*5,color=green,label='选中槽汇总',zorder=3)
    axs[0,1].set(xticks=x,xticklabels=model['seeds'],ylabel='汇总特征最大绝对变化',title='固定选择，仅扰动未选中的槽')
    axs[0,1].legend(fontsize=9)
    for arm,color,label,offset in [('original',orange,'原模型',-.15),('readout_fix',green,'最小修正',.15)]:
        part=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['seed'])
        xx=np.arange(3)+offset
        axs[1,0].bar(xx,[r['patient_risk_std_mean'] for r in part],width=.28,color=color,label=label)
        axs[1,0].scatter(xx,[r['patient_risk_std_mean'] for r in part],color=color,s=20,zorder=3)
        axs[1,1].errorbar(xx,[r['val_cindex_mean_exploratory'] for r in part],
            yerr=[r['val_cindex_std_population'] for r in part],fmt='o',color=color,capsize=4,label=label)
    axs[1,0].set(xticks=np.arange(3),xticklabels=[3,17,29],xlabel='训练种子',ylabel='患者风险标准差均值',title='训练后同一checkpoint重复预测20次')
    axs[1,1].set(xticks=np.arange(3),xticklabels=[3,17,29],xlabel='训练种子',ylabel='C-index均值 ± 重复评估标准差',title='短程验证值：不用于证明泛化优势')
    axs[1,0].legend(fontsize=9);axs[1,1].legend(fontsize=9)
    for ax in axs.flat:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('SlotSPE 证明实验：原实现与最小修正\n上排为干预证明；下排为3轮真实患者训练诊断',fontsize=14)
    fig.savefig(ROOT/'证明实验结果.png',dpi=180,facecolor='white')
    print(json.dumps({'status':'all_completed','report':str(ROOT/'证明实验结论.txt'),
                      'figure':str(ROOT/'证明实验结果.png'),'trained_checkpoints':6},ensure_ascii=False))

if __name__=='__main__':main()
