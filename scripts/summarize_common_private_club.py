"""Aggregate locked synthetic experiments, retaining all tasks and seeds."""
import json
import statistics
from pathlib import Path


def main():
    root=Path('docs/research/slotspe_common_private_20261101')
    source=json.loads((root/'club_clean_confirmation_receipt.json').read_text())
    assert source['status']=='completed' and len(source['rows'])==45
    rows=source['rows'];tasks=source['plan']['tasks'];seeds=source['plan']['seeds'];arms=source['plan']['arms']
    pair_checks=[];summary=[]
    for task in tasks:
        for seed in seeds:
            group=[r for r in rows if r['task']==task and r['seed']==seed]
            pair_checks.append(len(group)==3 and len({r['initial_core_hash'] for r in group})==1)
        baseline={r['seed']:r for r in rows if r['task']==task and r['arm']=='unconstrained'}
        for arm in arms:
            group=[r for r in rows if r['task']==task and r['arm']==arm]
            item={'task':task,'arm':arm,'n_trainings':len(group),
                  'mean_nll':{k:statistics.mean(r['test_nll'][k] for r in group) for k in ['iid','shift']},
                  'mean_factor_increment':{k:statistics.mean(r['probes'][k]['gain_relative_to_fit_mean_baseline'] for r in group) for k in ['private_wsi','private_omic']},
                  'mean_private_geometry':{k:statistics.mean(r['geometry'][k]['mean_uncentered_singular_value_effective_rank'] for r in group) for k in ['private_mean_wsi','private_mean_omic']},
                  'mean_training_critic_heldout_ratio':statistics.mean(r['training_critic_heldout_meanmode_ratio_per_dimension'] for r in group)}
            item['paired_difference_vs_unconstrained']={}
            for condition in ['iid','shift']:
                ds=[r['test_nll'][condition]-baseline[r['seed']]['test_nll'][condition] for r in group]
                mean=statistics.mean(ds);half=2.7764451051977987*statistics.stdev(ds)/len(ds)**.5
                item['paired_difference_vs_unconstrained'][condition]={'mean':mean,'descriptive_t95':[mean-half,mean+half],
                    'warning':'5 joint data/initialization seeds; descriptive t interval, not multiplicity-adjusted claim'}
            summary.append(item)
    assert all(pair_checks)
    report={'paired_initializations_verified':sum(pair_checks),'total_neural_trainings_this_stage':78,
            'pilot_trainings':27,'clean_confirmation_trainings':45,'slot_set_capacity_trainings':6,
            'density_estimator_fits':4,'analytic_scalar_optimizations':3,'summary':summary,
            'decision':'Do not advance to patients: no established broad predictive gain, private slot rank remains near one, per-slot content complementarity unproven.',
            'scope':'all generated CPU; no patient training/inference or server queue activity'}
    (root/'club_stage_summary.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    for r in summary:print(r['task'],r['arm'],r['mean_nll'],'rank',r['mean_private_geometry'],'paired',r['paired_difference_vs_unconstrained']['iid'])


if __name__=='__main__':main()
