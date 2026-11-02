"""Known-density and learned-estimator gates, generated CPU only."""
import json
import math
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from slotspe_proof.common_private_club import GaussianCLUB,log_ratio_from_gaussian


def dataset(seed,n,case):
    rng=torch.Generator().manual_seed(seed)
    g=torch.randn(n,1,generator=rng)
    p=torch.randn(n,1,generator=rng)
    if case=='independent': y=p
    elif case=='linear': y=.8*g+.6*p
    elif case=='quadratic': y=(g.square()-1)/2**.5+.3*p
    elif case=='heteroskedastic': y=(.4+g.abs())*p
    else: raise ValueError(case)
    return g,y


def main():
    torch.set_num_threads(2)
    rows=[]
    for case in ['independent','linear','quadratic','heteroskedastic']:
        fit=dataset(421,1024,case); val=dataset(1421,512,case); test=dataset(2421,1024,case)
        critic=GaussianCLUB(1,1)
        opt=torch.optim.Adam(critic.parameters(),lr=.003)
        best=None; best_val=float('inf')
        for step in range(600):
            idx=torch.arange((step*128)%1024,(step*128)%1024+128)
            loss=critic.fitting_loss(fit[0][idx],fit[1][idx])
            opt.zero_grad();loss.backward();opt.step()
            if (step+1)%50==0:
                with torch.no_grad(): score=float(critic.fitting_loss(*val))
                if score<best_val:
                    best_val=score;best={k:v.clone() for k,v in critic.state_dict().items()}
        critic.load_state_dict(best)
        with torch.no_grad():
            ratio=float(critic(*test));nll=float(critic.fitting_loss(*test))
        passed=abs(ratio)<.1 if case=='independent' else ratio>.1
        rows.append({'case':case,'ratio_nats_proxy':ratio,'heldout_conditional_nll':nll,'passed':passed,
                     'true_mi_nats':-.5*math.log(1-.8**2) if case=='linear' else 0 if case=='independent' else None})
    x,y=dataset(424,2048,'linear')
    oracle=float(log_ratio_from_gaussian(.8*x,torch.full_like(x,math.log(.36)),y))
    expected=.8**2/(1-.8**2)
    oracle_passed=abs(oracle-expected)<.2
    report={'learned_gates':rows,'oracle_gaussian':{'estimated_ratio':oracle,'population_ratio':expected,
                'true_mi':-.5*math.log(.36),'passed':oracle_passed},
            'all_passed':oracle_passed and all(r['passed'] for r in rows),
            'scope':'fabricated CPU; four density-estimator fits, zero survival/patient trainings',
            'limitations':'one density data seed; passing these cases does not certify the variational MI bound on moving network features'}
    path=Path('docs/research/slotspe_common_private_20261101/club_gate_receipt.json')
    path.write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))
    if not report['all_passed']:raise SystemExit(1)


if __name__=='__main__':main()
