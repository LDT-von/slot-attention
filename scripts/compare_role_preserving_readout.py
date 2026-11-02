"""Matched readout x content/budget test, new synthetic seeds, no patients."""
import hashlib
import json
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.compare_content_budget_slots import generated
from slotspe_proof.content_budget_slots import ContentBudgetSlotSurvival,content_budget_terms
from slotspe_proof.role_preserving_readout import RolePreservingSlotSurvival
from slotspe_proof.common_private_survival import objective,survival_nll
from slotspe_proof.common_private_diagnostics import linear_probe
from slotspe_proof.incremental_probe import incremental_probe


def core_hash(net):
    h=hashlib.sha256()
    for name,value in net.state_dict().items():
        if not name.startswith('phi.'):
            h.update(name.encode());h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def pooled(net,out,readout):
    if readout=='role_concat':return net.pooled_private(out['private'])
    return [f(v+e[None]).mean(1) for f,v,e in zip(net.phi,out['private'],net.role_offsets)]


def run(task,seed,readout,content_budget,cache):
    torch.default_generator.manual_seed(seed)
    cls=RolePreservingSlotSurvival if readout=='role_concat' else ContentBudgetSlotSurvival
    net=cls(8,8,width=16,slots=4,common_dim=4,private_dim=4)
    initial=core_hash(net);opt=torch.optim.Adam(net.parameters(),lr=.003)
    pack=generated(110000+seed,512,task)
    schedule=torch.Generator().manual_seed(120000+seed);noise_rng=torch.Generator().manual_seed(130000+seed)
    for step in range(500):
        idx=torch.randperm(512,generator=schedule)[:32]
        noise=[torch.randn(32,4,4,generator=noise_rng) for _ in range(2)]
        out=net(*[x[idx] for x in pack['xs']],noise=noise)
        targets=[{key:v[idx] for key,v in t.items()} for t in pack['targets']]
        term=content_budget_terms(out,targets,budget=1.)
        loss=objective(out,pack['time'][idx],pack['event'][idx],beta=0.)+content_budget*(term['content']+.3*term['budget_violation'])
        if not torch.isfinite(loss):raise RuntimeError('nonfinite synthetic loss')
        opt.zero_grad();loss.backward();opt.step()
    net.eval();frozen=[]
    packs=[generated(offset+seed,n,task) for offset,n in [(140000,192),(150000,128),(160000,384)]]
    with torch.no_grad():
        for pack in packs:
            out=net(*pack['xs'],sample=False)
            frozen.append({'common':out['common'].detach(),'private':[v.detach() for v in out['private']],
                           'pooled':[v.detach() for v in pooled(net,out,readout)],'bits':pack['all_bits'],'risk':pack['risk'],
                           'time':pack['time'],'event':pack['event']})
        noise_rng=torch.Generator().manual_seed(170000+seed);nll=[];ce=[]
        for _ in range(5):
            noise=[torch.randn(384,4,4,generator=noise_rng) for _ in range(2)]
            out=net(*packs[2]['xs'],noise=noise)
            nll.append(float(survival_nll(out['logits'],packs[2]['time'],packs[2]['event'])))
            ce.append(torch.cat([F.binary_cross_entropy_with_logits(logits,t['labels'],reduction='none').mean(0) for logits,t in zip(out['content_logits'],packs[2]['targets'])]))
        rate=torch.cat([r.mean(0) for r in out['normalized_kl_per_slot']])
    fname=cache/f'{task}_{seed}_{readout}_{content_budget}.pt'
    torch.save({'synthetic_only':True,'frozen':frozen,'task':task,'seed':seed,'readout':readout,'constraints':content_budget},fname)
    own=[];off=[]
    for j in range(2):
        for k in range(4):
            for target in range(8):
                result=linear_probe(frozen[0]['private'][j][:,k],frozen[0]['bits'][:,target],frozen[2]['private'][j][:,k],frozen[2]['bits'][:,target])
                (own if target==j*4+k else off).append(result['test_r2'])
    risk_probe=incremental_probe(*[(r['common'],torch.cat(r['pooled'],-1),r['risk']) for r in frozen])
    return {'task':task,'seed':seed,'readout':readout,'content_budget_on':content_budget,'shared_core_initial_hash':initial,
            'parameter_count':sum(p.numel() for p in net.parameters()),'phi_parameter_count':sum(p.numel() for p in net.phi.parameters()),
            'test_expected_nll':sum(nll)/5,'expected_content_ce_per_slot':torch.stack(ce).mean(0).tolist(),
            'rate_nats_per_slot':rate.tolist(),'own_bit_linear_r2':sum(own)/len(own),'other_bits_linear_r2':sum(off)/len(off),
            'risk_probe':risk_probe,'synthetic_cache_path':str(fname),'synthetic_cache_sha256':hashlib.sha256(fname.read_bytes()).hexdigest()}


def main():
    torch.set_num_threads(2)
    output=Path('docs/research/slotspe_common_private_20261101/role_readout_comparison_receipt.json')
    cache=Path('local_evidence/synthetic_role_readout_20261102')
    if output.exists() or cache.exists():raise RuntimeError('preserve existing artifacts')
    cache.mkdir(parents=True)
    plan={'tasks':['all_useful','selective','synergy'],'seeds':[223,251,277],
          'readouts':['mean_pool','role_concat'],'constraints':[0,1],'steps':500,
          'fixed_hyperparameters':'content 1, hinge .3, rate cap1 nat; unchanged after previous pilot',
          'capacity':'phi parameter counts matched within 5%; same head and all non-phi initialization',
          'metadata':'synthetic one-hot content identities; no clinical anchor validation',
          'selection':'new exploratory seeds, fixed final step; no model/weight selection from these tests'}
    payload={'status':'running','plan':plan,'rows':[],'patient_execution':False,'server_execution':False}
    output.write_bytes((json.dumps(payload,indent=2)+'\n').encode())
    for task in plan['tasks']:
        for seed in plan['seeds']:
            for readout in plan['readouts']:
                for constraints in plan['constraints']:
                    row=run(task,seed,readout,constraints,cache);payload['rows'].append(row)
                    output.write_bytes((json.dumps(payload,indent=2)+'\n').encode())
                    print(json.dumps({'task':task,'seed':seed,'readout':readout,'constraints':constraints,'nll':row['test_expected_nll'],
                        'own_r2':row['own_bit_linear_r2'],'other_r2':row['other_bits_linear_r2']}),flush=True)
    payload['status']='completed';output.write_bytes((json.dumps(payload,indent=2)+'\n').encode())


if __name__=='__main__':main()
