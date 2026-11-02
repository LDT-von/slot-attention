"""Posthoc slot audit of replayed synthetic models, no patient I/O.

Replays 18 configurations already used in clean confirmation to recover
frozen features. These are training executions, NOT 18 new independent
confirmation runs. New probe cohorts are disjoint from all network fitting.
"""
import hashlib
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.compare_common_private_club as pilot
from scripts.confirm_common_private_club import clean_generated
from slotspe_proof.slot_increment_probe import slot_increment_probe


def capture(net,task,seed):
    modes={'posterior_mean':[],'one_stochastic_draw':[]}
    for split,(offset,n) in enumerate([(17100,192),(18100,128),(19100,384)]):
        pack=clean_generated(offset+seed,n,task)
        risk=pack[4]['shared']+pack[4]['private_wsi']+pack[4]['private_omic']
        if task=='synergy':risk=3*((pack[4]['private_wsi']>0)!=(pack[4]['private_omic']>0)).float()-1.5+.3*pack[4]['shared']
        for mode in modes:
            gen=torch.Generator().manual_seed(60000+seed+split*7)
            noise=[torch.randn(n,4,4,generator=gen) for _ in range(2)]
            with torch.no_grad():
                out=net(*pack[:2],sample=mode!='posterior_mean',noise=noise)
                slots=[f(u).detach() for f,u in zip(net.phi,out['private'])]
            modes[mode].append({'common':out['common'].detach(),'phi_slots':slots,
                                'private_wsi':pack[4]['private_wsi'],'private_omic':pack[4]['private_omic'],
                                'risk':risk})
    return modes


def score(modes):
    result={}
    for mode,parts in modes.items():
        result[mode]={}
        for j,factor in enumerate(['private_wsi','private_omic']):
            result[mode][factor]=slot_increment_probe(*[(r['common'],[r['phi_slots'][j]],r[factor]) for r in parts])
        result[mode]['joint_latent_risk']=slot_increment_probe(*[(r['common'],r['phi_slots'],r['risk']) for r in parts])
    return result


def main():
    torch.set_num_threads(2);pilot.generated=clean_generated
    root=Path('docs/research/slotspe_common_private_20261101')
    output=root/'slot_complementarity_audit_receipt.json'
    cache=Path('local_evidence/synthetic_slot_features_20261102')
    if output.exists() or cache.exists():raise RuntimeError('preserve existing artifacts; no silent overwrite')
    cache.mkdir(parents=True)
    plan={'tasks':['independent','correlated','synergy'],'seeds':[31,47],
          'arms':['unconstrained','conditional_kl','direct_club'],'steps':300,
          'meaning':'posthoc replay, not new independent confirmation','probe_n':[192,128,384],
          'modes':['posterior_mean','one_stochastic_draw'],'test_slot_selection':'validation only',
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    payload={'status':'running','plan':plan,'rows':[],'patient_execution':False,'server_execution':False}
    output.write_bytes((json.dumps(payload,indent=2)+'\n').encode())
    parent=json.loads((root/'club_clean_confirmation_receipt.json').read_text())['rows']
    for task in plan['tasks']:
        for seed in plan['seeds']:
            for arm in plan['arms']:
                net,critics,record=pilot.train(task,seed,arm,300)
                old=next(r for r in parent if (r['task'],r['seed'],r['arm'])==(task,seed,arm))
                assert record['initial_core_hash']==old['initial_core_hash']
                assert abs(record['trace'][-1]['base_loss']-old['trace'][-1]['base_loss'])<1e-6
                net.eval();modes=capture(net,task,seed)
                fname=cache/f'{task}_{seed}_{arm}.pt'
                torch.save({'synthetic_only':True,'modes':modes,'task':task,'seed':seed,'arm':arm},fname)
                result=score(modes)
                row={'task':task,'seed':seed,'arm':arm,'initial_hash_and_last_base_loss_match_parent':True,
                     'synthetic_cache_path':str(fname),'synthetic_cache_sha256':hashlib.sha256(fname.read_bytes()).hexdigest(),
                     'probes':result}
                payload['rows'].append(row)
                print(json.dumps({'task':task,'seed':seed,'arm':arm,'gains':{mode:{k:v['gain_relative_to_fit_mean_baseline'] for k,v in r.items()} for mode,r in result.items()}}),flush=True)
                output.write_bytes((json.dumps(payload,indent=2)+'\n').encode())
    payload['status']='completed';output.write_bytes((json.dumps(payload,indent=2)+'\n').encode())


if __name__=='__main__':main()
