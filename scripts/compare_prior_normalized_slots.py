"""2x2 readout-coordinate/rate ablation; CPU generated features only."""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.confirm_common_private_club import clean_generated,geometry
from scripts.compare_common_private_club import state_hash
from slotspe_proof.common_private_survival import CommonPrivateSurvival,objective,survival_nll
from slotspe_proof.prior_normalized_slots import PriorNormalizedSlotSurvival
from slotspe_proof.incremental_probe import incremental_probe


def train(task,seed,mode,beta):
    torch.default_generator.manual_seed(seed)
    cls=PriorNormalizedSlotSurvival if mode=='normalized' else CommonPrivateSurvival
    net=cls(6,6,width=16,slots=4,common_dim=4,private_dim=4)
    initial=state_hash(net);opt=torch.optim.Adam(net.parameters(),lr=.003)
    train=clean_generated(1000+seed,512,task)
    schedule=torch.Generator().manual_seed(21000+seed)
    random=torch.Generator().manual_seed(31000+seed)
    for step in range(300):
        idx=torch.randperm(512,generator=schedule)[:32]
        noise=[torch.randn(32,4,4,generator=random) for _ in range(2)]
        out=net(train[0][idx],train[1][idx],noise=noise)
        if mode=='normalized' and step in [0,299]:assert torch.allclose(out['kl_joint'],out['normalized_kl_joint'],atol=1e-4)
        loss=objective(out,train[2][idx],train[3][idx],beta=beta)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite generated training')
        opt.zero_grad();loss.backward();opt.step()
    return net,initial


def capture(net,pack):
    with torch.no_grad():
        out=net(*pack[:2],sample=False)
        pooled=[f(u).mean(1).double() for f,u in zip(net.phi,out['private'])]
    return out,pooled


def main():
    torch.set_num_threads(2)
    path=Path('docs/research/slotspe_common_private_20261101/prior_normalized_ablation_receipt.json')
    if path.exists():raise RuntimeError('preserve existing receipt')
    plan={'tasks':['independent','correlated','synergy'],'seeds':[109,131,157],
          'modes':['raw','normalized'],'betas':[0.,.3],'steps':300,
          'design':'2x2; paired initial weights, input schedule and posterior base noise; no hyperparameter or checkpoint selection',
          'meaning':'new exploratory seeds, not a proof of semantic factorization or patient improvement'}
    payload={'status':'running','plan':plan,'rows':[],'patient_execution':False,'server_execution':False}
    path.write_bytes((json.dumps(payload,indent=2)+'\n').encode())
    for task in plan['tasks']:
        for seed in plan['seeds']:
            for mode in plan['modes']:
                for beta in plan['betas']:
                    net,initial=train(task,seed,mode,beta);net.eval()
                    packs=[clean_generated(offset+seed,n,task) for offset,n in [(27100,192),(28100,128),(29100,384)]]
                    reps=[capture(net,p) for p in packs];probes={}
                    for j,factor in enumerate(['private_wsi','private_omic']):
                        probes[factor]=incremental_probe(*[(r[0]['common'].double(),r[1][j],p[4][factor].double()) for r,p in zip(reps,packs)])
                    with torch.no_grad():
                        torch.default_generator.manual_seed(41000+seed)
                        nll=sum(float(survival_nll(net(*packs[2][:2])['logits'],*packs[2][2:4])) for _ in range(5))/5
                    row={'task':task,'seed':seed,'mode':mode,'beta':beta,'initial_core_hash':initial,'test_expected_nll':nll,
                         'kl_per_dimension_meanmode':float(reps[2][0]['kl_joint'].mean()/reps[2][0]['latent_dimensions']),
                         'private_geometry':[geometry(u) for u in reps[2][0]['private']],
                         'factor_probes':probes}
                    payload['rows'].append(row);path.write_bytes((json.dumps(payload,indent=2)+'\n').encode())
                    print(json.dumps({'task':task,'seed':seed,'mode':mode,'beta':beta,'nll':nll}),flush=True)
    payload['status']='completed';path.write_bytes((json.dumps(payload,indent=2)+'\n').encode())


if __name__=='__main__':main()
