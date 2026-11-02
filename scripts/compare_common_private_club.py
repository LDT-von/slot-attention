"""Matched three-arm mechanism pilot, generated CPU only. No patient loader.

Fixed choices before running: beta=.3 reference; gamma=.03 CLUB; 50 step
warm-up; 3 critic updates/core update; 300 core updates; seeds 3/13/23.
No best-epoch or hyperparameter selection on confirmation probe-test rows.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.check_common_private_semantics import representations
from slotspe_proof.common_private_survival import CommonPrivateSurvival,objective,survival_nll
from slotspe_proof.common_private_club import (GaussianCLUB,branch_features,
    standardized_rows,separation_penalty,frozen_critic)
from slotspe_proof.incremental_probe import incremental_probe


def generated(seed,n,task,shift=False):
    rng=torch.Generator().manual_seed(seed)
    c,p,q=torch.randn(3,n,generator=rng)
    if task=='correlated':p=.8*c+.6*p
    x=torch.randn(n,12,6,generator=rng)*.3
    y=torch.randn(n,12,6,generator=rng)*.3
    x[:,:,0]+=c[:,None];y[:,:,0]+=c[:,None]
    x[:,:,1]+=p[:,None];y[:,:,2]+=q[:,None]
    risk=c+p+q
    if task=='synergy':risk=3*((p>0)!=(q>0)).float()-1.5+.3*c
    # A prespecified distribution-shift stress: reversal of an auxiliary cue.
    if shift:
        x[:,:,3]+=-risk[:,None]
        y[:,:,3]+=-risk[:,None]
    else:
        x[:,:,3]+=risk[:,None]
        y[:,:,3]+=risk[:,None]
    # Keep the cue modest to avoid making it a deterministic target channel.
    x[:,:,3]+=(torch.randn(n,1,generator=rng)*2)
    y[:,:,3]+=(torch.randn(n,1,generator=rng)*2)
    hazards=torch.sigmoid(risk[:,None]+torch.tensor([-2.,-1.,0.,1.]))
    hits=torch.rand(n,4,generator=rng)<hazards
    event_time=torch.where(hits.any(1),hits.int().argmax(1),4)
    censor=torch.randint(0,4,(n,),generator=rng)
    return x,y,torch.minimum(event_time,censor),(event_time<=censor).float(),{'shared':c,'private_wsi':p,'private_omic':q}


def state_hash(net):
    h=hashlib.sha256()
    for name,t in net.state_dict().items():h.update(name.encode());h.update(t.detach().numpy().tobytes())
    return h.hexdigest()


def train(task,seed,arm,steps):
    torch.manual_seed(seed)
    net=CommonPrivateSurvival(6,6,width=16,slots=4,common_dim=4,private_dim=4)
    initial=state_hash(net)
    critics=torch.nn.ModuleList([GaussianCLUB(4,16,seed=771+j) for j in range(2)])
    opt=torch.optim.Adam(net.parameters(),lr=.003)
    copt=torch.optim.Adam(critics.parameters(),lr=.003)
    data=generated(1000+seed,512,task)
    trace=[];start=time.perf_counter()
    for step in range(steps):
        idx=torch.randperm(512)[:32]
        out=net(data[0][idx],data[1][idx])
        ys=branch_features(net,out)
        # Critics update on detached rows even for zero-weight controls.
        sx=standardized_rows(out['common'].detach())
        sy=[standardized_rows(y.detach()) for y in ys]
        for _ in range(3):
            fitloss=torch.stack([c.fitting_loss(sx,y) for c,y in zip(critics,sy)]).mean()
            copt.zero_grad();fitloss.backward();copt.step()
        with frozen_critic(critics): penalty=separation_penalty(critics,out['common'],ys)
        base=objective(out,data[2][idx],data[3][idx],beta=.3 if arm=='conditional_kl' else 0.)
        weight=.03*min(1.,max(0.,(step-49)/50)) if arm=='direct_club' else 0.
        loss=base+weight*penalty
        if not torch.isfinite(loss):raise RuntimeError(f'nonfinite {task} {seed} {arm} {step}')
        if step in [0,50,100,200,steps-1]:
            params=list(net.parameters())
            bg=torch.autograd.grad(base,params,retain_graph=True,allow_unused=True)
            pg=torch.autograd.grad(penalty,params,retain_graph=True,allow_unused=True)
            bn=sum(float(g.square().sum()) for g in bg if g is not None)**.5
            pn=sum(float(g.square().sum()) for g in pg if g is not None)**.5
            trace.append({'step':step,'base_loss':float(base.detach()),'ratio_per_dimension':float(penalty.detach()),
                          'critic_fit_nll_per_dimension':float(fitloss.detach()),'weight':weight,
                          'base_grad_norm':bn,'applied_penalty_grad_norm':weight*pn})
        opt.zero_grad();loss.backward();opt.step()
    return net,critics,{'initial_core_hash':initial,'trace':trace,'training_seconds':time.perf_counter()-start}


def measure(task,seed,arm,steps):
    net,critics,record=train(task,seed,arm,steps)
    packs=[generated(k+seed,n,task) for k,n in [(7100,256),(8100,128),(9100,512)]]
    reps=[representations(net,p) for p in packs]
    probes={}
    for factor,branch in [('private_wsi','private_wsi_readout'),('private_omic','private_omic_readout')]:
        probes[factor]=incremental_probe(*[(r[1]['common'],r[1][branch],r[2][factor]) for r in reps])
    # Evaluate common and own private readout ability to recover the shared factor.
    common_probe=incremental_probe(*[(torch.zeros_like(r[1]['common']),r[1]['common'],r[2]['shared']) for r in reps])
    copy_probe=incremental_probe(*[(torch.zeros_like(r[1]['common']),r[1]['private_wsi_readout'],r[2]['shared']) for r in reps])
    scores={}
    with torch.no_grad():
        for label,pack in [('iid',packs[2]),('shift',generated(9100+seed,512,task,shift=True))]:
            torch.manual_seed(22003)
            scores[label]=sum(float(survival_nll(net(*pack[:2])['logits'],*pack[2:4])) for _ in range(5))/5
        # New heldout density critics distinguish training-critic blind spots.
        frozen_x=standardized_rows(reps[2][1]['common'].float())
        eval_ratio=float(torch.stack([c(frozen_x,standardized_rows(reps[2][1][key].float()))/16
                      for c,key in zip(critics,['private_wsi_readout','private_omic_readout'])]).mean())
    return {'task':task,'seed':seed,'arm':arm,'steps':steps,**record,'test_nll':scores,'probes':probes,
            'common_shared_recovery':common_probe,'private_shared_recovery':copy_probe,
            'training_critic_heldout_meanmode_ratio_per_dimension':eval_ratio}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=300)
    args=parser.parse_args();torch.set_num_threads(2)
    output=Path('docs/research/slotspe_common_private_20261101/club_comparison_pilot_receipt.json')
    if output.exists():raise RuntimeError('preserve existing receipt; choose a new explicitly reviewed run')
    rows=[]
    for task in ['independent','correlated','synergy']:
        for seed in [3,13,23]:
            for arm in ['unconstrained','conditional_kl','direct_club']:
                r=measure(task,seed,arm,args.steps);rows.append(r)
                print(json.dumps({'task':task,'seed':seed,'arm':arm,'nll':r['test_nll'],
                                  'increment':{k:v['gain_relative_to_fit_mean_baseline'] for k,v in r['probes'].items()}}),flush=True)
                payload={'schema':'common_private_direct_club_pilot_v1','status':'running','rows':rows,
                         'patient_execution':False,'network':'unchanged width16 K4/modal; paired initialization/data; no epoch selection',
                         'scope':'27 generated CPU neural trainings; 3 data/initialization seeds per task; exploratory synthetic tests',
                         'survival_eval':'mean of 5 stochastic NLLs, not mixture NLL or independent trainings'}
                output.write_bytes((json.dumps(payload,indent=2)+'\n').encode())
    payload['status']='completed'
    output.write_bytes((json.dumps(payload,indent=2)+'\n').encode())


if __name__=='__main__':main()
