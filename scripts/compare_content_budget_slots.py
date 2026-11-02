"""Four-arm fixed typed-content / rate-budget pilot, synthetic CPU only.

No survival labels or planted factor arrays are used to construct content
targets: the adapter reads fixed token type metadata and observed amplitudes.
These type identities are synthetic oracle metadata, NOT validated WSI labels.
"""
import hashlib
import json
import math
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from slotspe_proof.content_budget_slots import ContentBudgetSlotSurvival,typed_token_targets,content_budget_terms
from slotspe_proof.common_private_survival import objective,survival_nll
from slotspe_proof.common_private_diagnostics import linear_probe
from slotspe_proof.slot_increment_probe import slot_increment_probe
from scripts.compare_common_private_club import state_hash


def generated(seed,n,task,k=4):
    rng=torch.Generator().manual_seed(seed)
    c=torch.randn(n,generator=rng)
    p=torch.randn(n,k,generator=rng);q=torch.randn(n,k,generator=rng)
    ids=torch.arange(k).repeat_interleave(3)
    xs=[]
    for factors in [p,q]:
        x=torch.zeros(n,3*k,k+4)
        x[:,:,:k]=F.one_hot(ids,k).float()[None]
        x[:,:,k]=factors[:,ids]+torch.randn(n,3*k,generator=rng)*.05
        x[:,:,k+1]=c[:,None]+torch.randn(n,3*k,generator=rng)*.05
        x[:,:,k+2:]=torch.randn(n,3*k,2,generator=rng)*.3
        xs.append(x)
    signs_p=(p>0).float()*2-1;signs_q=(q>0).float()*2-1
    weights=torch.tensor([1.,.75,-.5,.25])
    if task=='all_useful':risk=.5*c+(signs_p@weights+signs_q@weights.flip(0))/math.sqrt(2*k)
    elif task=='selective':risk=2*signs_p[:,0]
    elif task=='synergy':risk=3*((p[:,0]>0)!=(q[:,0]>0)).float()-1.5+.3*c
    else:raise ValueError(task)
    hazards=torch.sigmoid(risk[:,None]+torch.tensor([-2.,-1.,0.,1.]))
    hits=torch.rand(n,4,generator=rng)<hazards
    t=torch.where(hits.any(1),hits.int().argmax(1),4);censor=torch.randint(0,4,(n,),generator=rng)
    targets=[typed_token_targets(x,k,k) for x in xs]
    return {'xs':xs,'time':torch.minimum(t,censor),'event':(t<=censor).float(),
            'targets':targets,'risk':risk,'all_bits':torch.cat([t['labels'] for t in targets],1)}


def run(task,seed,anchor,budget):
    torch.default_generator.manual_seed(seed)
    net=ContentBudgetSlotSurvival(8,8,width=16,slots=4,common_dim=4,private_dim=4)
    initial=state_hash(net);optimizer=torch.optim.Adam(net.parameters(),lr=.003)
    train=generated(41000+seed,512,task)
    schedule=torch.Generator().manual_seed(51000+seed);random=torch.Generator().manual_seed(61000+seed)
    trace=[]
    for step in range(500):
        idx=torch.randperm(512,generator=schedule)[:32]
        noise=[torch.randn(32,4,4,generator=random) for _ in range(2)]
        out=net(*[x[idx] for x in train['xs']],noise=noise)
        targets=[{key:value[idx] for key,value in t.items()} for t in train['targets']]
        terms=content_budget_terms(out,targets,budget=1.)
        loss=objective(out,train['time'][idx],train['event'][idx],beta=0.)+anchor*terms['content']+.3*budget*terms['budget_violation']
        if not torch.isfinite(loss):raise RuntimeError('nonfinite generated loss')
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step in [0,99,299,499]:trace.append({'step':step,'loss':float(loss.detach()),'content_ce':float(terms['content'].detach()),'soft_budget_violation':float(terms['budget_violation'].detach())})
    net.eval()
    packs=[generated(offset+seed,n,task) for offset,n in [(71000,192),(81000,128),(91000,384)]]
    frozen=[]
    with torch.no_grad():
        for pack in packs:
            o=net(*pack['xs'],sample=False)
            tagged=[f(v+e[None]) for f,v,e in zip(net.phi,o['private'],net.role_offsets)]
            frozen.append({'common':o['common'],'private':o['private'],'tagged':tagged,'bits':pack['all_bits'],'risk':pack['risk']})
        random_eval=torch.Generator().manual_seed(101000+seed)
        nll=[];ce=[]
        for _ in range(5):
            noise=[torch.randn(384,4,4,generator=random_eval) for _ in range(2)]
            out=net(*packs[2]['xs'],noise=noise)
            nll.append(float(survival_nll(out['logits'],packs[2]['time'],packs[2]['event'])))
            ce.append(torch.cat([F.binary_cross_entropy_with_logits(logits,t['labels'],reduction='none').mean(0) for logits,t in zip(out['content_logits'],packs[2]['targets'])]))
        rate=torch.cat([r.mean(0) for r in out['normalized_kl_per_slot']])
    # Same independently fitted linear probe for every arm; unused content-head
    # CE is never taken as proof that an unconstrained latent has no information.
    own=[];off=[]
    for j in range(2):
        for k in range(4):
            for target in range(8):
                r=linear_probe(frozen[0]['private'][j][:,k],frozen[0]['bits'][:,target],frozen[2]['private'][j][:,k],frozen[2]['bits'][:,target])
                (own if target==j*4+k else off).append(r['test_r2'])
    increment=slot_increment_probe(*[(r['common'],r['tagged'],r['risk']) for r in frozen])
    return {'task':task,'seed':seed,'anchor_on':anchor,'budget_on':budget,'initial_hash':initial,'trace':trace,
            'test_expected_nll':sum(nll)/5,'test_expected_content_ce_per_slot':torch.stack(ce).mean(0).tolist(),
            'test_rate_nats_per_slot':rate.tolist(),'fraction_rates_at_most_1_15':float((rate<=1.15).float().mean()),
            'linear_probe_own_bit_r2_mean':sum(own)/len(own),'linear_probe_other_bits_r2_mean':sum(off)/len(off),
            'whole_vs_single_risk_probe':increment}


def main():
    torch.set_num_threads(2)
    path=Path('docs/research/slotspe_common_private_20261101/content_budget_pilot_receipt.json')
    if path.exists():raise RuntimeError('preserve existing receipt')
    plan={'tasks':['all_useful','selective','synergy'],'seeds':[173,197,211],
          'arms':[[0,0],[1,0],[0,1],[1,1]],'steps':500,'rate_budget_nats_per_slot':1.,
          'content_weight':1.,'budget_hinge_weight':.3,'same_model_all_arms':True,
          'target_source':'fixed typed-token metadata and observed values, never event labels; synthetic metadata is an optimistic assumption',
          'selection':'fixed settings, no best checkpoint, no tuning on test'}
    payload={'status':'running','plan':plan,'rows':[],'patient_execution':False,'server_execution':False}
    path.write_bytes((json.dumps(payload,indent=2)+'\n').encode())
    for task in plan['tasks']:
        for seed in plan['seeds']:
            for anchor,budget in plan['arms']:
                r=run(task,seed,anchor,budget);payload['rows'].append(r)
                path.write_bytes((json.dumps(payload,indent=2)+'\n').encode())
                print(json.dumps({'task':task,'seed':seed,'anchor':anchor,'budget':budget,'nll':r['test_expected_nll'],
                    'content_ce':sum(r['test_expected_content_ce_per_slot'])/8,'rate':sum(r['test_rate_nats_per_slot'])/8,
                    'own_r2':r['linear_probe_own_bit_r2_mean'],'off_r2':r['linear_probe_other_bits_r2_mean'],
                    'all_vs_single_gain':r['whole_vs_single_risk_probe']['gain_relative_to_fit_mean_baseline']}),flush=True)
    payload['status']='completed';path.write_bytes((json.dumps(payload,indent=2)+'\n').encode())


if __name__=='__main__':main()
