"""Known-answer controls including stochastic denoising versus mean slots."""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from slotspe_proof.slot_increment_probe import slot_increment_probe


def pack(seed,n,case):
    gen=torch.Generator().manual_seed(seed)
    p=torch.randn(n,4,generator=gen,dtype=torch.float64)
    common=torch.zeros(n,1,dtype=torch.float64)
    if case in ['duplicate','nonlinear_duplicate']:
        z=p if case=='duplicate' else p.square()
        slots=z[:,None,:].repeat(1,4,1);target=z
    elif case=='complementary':
        slots=torch.diag_embed(p);target=p
    elif case=='xor':
        bits=(p>0).double();slots=torch.diag_embed(bits)
        target=(bits[:,0]!=bits[:,1]).double()[:,None]
    elif case=='noise':
        slots=torch.randn(n,4,4,generator=gen,dtype=torch.float64);target=p
    elif case=='noisy_duplicate':
        slots=p[:,None,:]+torch.randn(n,4,4,generator=gen,dtype=torch.float64);target=p
    elif case=='posterior_mean_duplicate':
        slots=p[:,None,:].repeat(1,4,1);target=p
    else:raise ValueError(case)
    return common,[slots],target


def main():
    torch.set_num_threads(2);rows=[]
    cases=['duplicate','nonlinear_duplicate','complementary','xor','noise','noisy_duplicate','posterior_mean_duplicate']
    for seed in [103,127,151]:
        for case in cases:
            r=slot_increment_probe(*[pack(seed+k,n,case) for k,n in [(0,192),(1000,128),(2000,384)]])
            gain=r['gain_relative_to_fit_mean_baseline']
            if case in ['complementary','xor','noisy_duplicate']:passed=gain>.1
            elif case=='noise':passed=abs(gain)<.1
            else:passed=abs(gain)<.01
            rows.append({'seed':seed,'case':case,'passed':passed,**r})
            print(json.dumps({'seed':seed,'case':case,'gain':gain,'passed':passed}),flush=True)
    report={'all_passed':all(r['passed'] for r in rows),'rows':rows,
            'scope':'21 fabricated probe comparisons; zero neural/patient training',
            'critical_control':'noisy duplicates can show all-slot gain without unique information; posterior-mean copies must not'}
    path=Path('docs/research/slotspe_common_private_20261101/slot_increment_gate_receipt.json')
    if path.exists():raise RuntimeError('preserve existing receipt')
    path.write_bytes((json.dumps(report,indent=2)+'\n').encode())
    if not report['all_passed']:raise SystemExit(1)


if __name__=='__main__':main()
