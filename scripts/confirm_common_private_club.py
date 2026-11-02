"""Locked clean-factor confirmation after the cue-confounded pilot.

Reuses the exact pilot optimizer and evaluator, explicitly substitutes only
the generated data source. No hyperparameter selection. Seeds 31/47/61/79/97
have not been used for network development in the preceding receipts.
"""
import hashlib
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.compare_common_private_club as pilot


def clean_generated(seed,n,task,shift=False):
    rng=torch.Generator().manual_seed(seed)
    c,p,q=torch.randn(3,n,generator=rng)
    if shift:
        c=1.5*c+.4;p=1.5*p;q=1.5*q
    if task=='correlated':p=.8*c+.6*p
    x=torch.randn(n,12,6,generator=rng)*.3
    y=torch.randn(n,12,6,generator=rng)*.3
    x[:,:,0]+=c[:,None];y[:,:,0]+=c[:,None]
    x[:,:,1]+=p[:,None];y[:,:,2]+=q[:,None]
    risk=c+p+q
    if task=='synergy':risk=3*((p>0)!=(q>0)).float()-1.5+.3*c
    hazards=torch.sigmoid(risk[:,None]+torch.tensor([-2.,-1.,0.,1.]))
    hits=torch.rand(n,4,generator=rng)<hazards
    t=torch.where(hits.any(1),hits.int().argmax(1),4)
    censor=torch.randint(0,4,(n,),generator=rng)
    return x,y,torch.minimum(t,censor),(t<=censor).float(),{'shared':c,'private_wsi':p,'private_omic':q}


def geometry(tensor):
    normalized=torch.nn.functional.normalize(tensor,dim=-1)
    gram=normalized@normalized.transpose(1,2)
    k=tensor.shape[1]
    cosine=(gram.sum((1,2))-gram.diagonal(dim1=1,dim2=2).sum(1))/(k*(k-1))
    values=torch.linalg.svdvals(tensor.double())
    probs=values/values.sum(1,keepdim=True).clamp_min(1e-12)
    rank=(-(probs*probs.clamp_min(1e-12).log()).sum(1)).exp()
    return {'mean_pairwise_cosine':float(cosine.mean()),'mean_uncentered_singular_value_effective_rank':float(rank.mean()),
            'warning':'geometry only; no content or use inference'}


def main():
    torch.set_num_threads(2)
    path=Path('docs/research/slotspe_common_private_20261101/club_clean_confirmation_receipt.json')
    if path.exists():raise RuntimeError('preserve existing receipt')
    pilot.generated=clean_generated
    original_train=pilot.train
    def train_with_geometry(task,seed,arm,steps):
        net,critics,record=original_train(task,seed,arm,steps)
        test=clean_generated(9100+seed,128,task)
        with torch.no_grad():
            out=net(*test[:2],sample=False)
            record['geometry']={name:geometry(t) for name,t in zip(['native_wsi','native_omic','private_mean_wsi','private_mean_omic'],out['slots']+out['private'])}
        return net,critics,record
    pilot.train=train_with_geometry
    rows=[]
    plan={'tasks':['independent','correlated','synergy'],'seeds':[31,47,61,79,97],
          'arms':['unconstrained','conditional_kl','direct_club'],'steps':300,'beta_reference':.3,'gamma':.03,
          'core_model':'width16 K4/modal common4 private4; unchanged core architecture',
          'factor_semantics':'no shared risk cue; correlated p statistically associated with c, not strictly exclusive',
          'shift':'changes latent scale/location, not cue reversal',
          'selection':'all fixed-final, no test or confirmation hyperparameter selection',
          'pilot_script_sha256':hashlib.sha256(Path(pilot.__file__).read_bytes()).hexdigest()}
    payload={'status':'running','plan':plan,'rows':rows,'patient_execution':False,'server_execution':False}
    # Save design before any outcome is available.
    path.write_bytes((json.dumps(payload,indent=2)+'\n').encode())
    for task in plan['tasks']:
        for seed in plan['seeds']:
            for arm in plan['arms']:
                r=pilot.measure(task,seed,arm,plan['steps']);rows.append(r)
                print(json.dumps({'task':task,'seed':seed,'arm':arm,'nll':r['test_nll'],
                    'private_rank':[r['geometry'][k]['mean_uncentered_singular_value_effective_rank'] for k in ['private_mean_wsi','private_mean_omic']]}),flush=True)
                path.write_bytes((json.dumps(payload,indent=2)+'\n').encode())
    payload['status']='completed'
    path.write_bytes((json.dumps(payload,indent=2)+'\n').encode())


if __name__=='__main__':main()
