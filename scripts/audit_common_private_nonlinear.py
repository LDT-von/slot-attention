"""Six newly generated neural runs; independent probe fit/validation/test."""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.check_common_private_semantics import generated, train, representations
from slotspe_proof.incremental_probe import incremental_probe
from slotspe_proof.common_private_survival import survival_nll


def main():
    torch.set_num_threads(2)
    results=[]
    for task in ['shared_and_private','private_only']:
        for beta in [0.,.3,1.]:
            net=train(task,3,beta,300)
            # All three probe cohorts independent of network fitting cohort.
            packs=[generated(seed,n,task) for seed,n in [(7103,256),(8103,128),(9103,512)]]
            reps=[representations(net,p) for p in packs]
            probes={}
            for factor, branch in [('private_wsi','private_wsi_readout'),('private_omic','private_omic_readout'),('shared','private_wsi_readout')]:
                parts=[(r[1]['common'],r[1][branch],r[2][factor]) for r in reps]
                probes[factor]=incremental_probe(*parts)
            # Score stochastic expected NLL using fixed common noise schedule.
            with torch.no_grad():
                torch.manual_seed(22003)
                nll=sum(float(survival_nll(net(*packs[2][:2])['logits'],*packs[2][2:4])) for _ in range(5))/5
            result={'task':task,'seed':3,'beta':beta,'steps':300,'probes':probes,'test_expected_nll':nll}
            results.append(result)
            print(json.dumps({'task':task,'beta':beta,'nll':nll,'gains':{k:v['gain_relative_to_fit_mean_baseline'] for k,v in probes.items()}}),flush=True)
    receipt={'scope':'six generated CPU trainings; no patient/checkpoint/server I/O',
             'interpretation':'exploratory one training/data seed per task; finite-function-class gains, not MI',
             'probe_cohorts_disjoint_from_neural_training':True,'results':results}
    out=Path(__file__).resolve().parents[1]/'docs/research/slotspe_common_private_20261101/nonlinear_reaudit_receipt.json'
    out.write_bytes((json.dumps(receipt,indent=2)+'\n').encode())


if __name__=='__main__': main()
