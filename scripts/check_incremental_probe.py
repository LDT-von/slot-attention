"""Known-answer gate for frozen-representation increment; CPU fabricated only."""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from slotspe_proof.incremental_probe import incremental_probe


def pack(seed, n, case):
    rng=torch.Generator().manual_seed(seed)
    g=torch.rand(n,1,generator=rng,dtype=torch.float64)*4-2
    p=torch.randn(n,1,generator=rng,dtype=torch.float64)
    if case=='quadratic_copy': u,y=g.square(),g.square()
    elif case=='nonlinear_copy': u,y=torch.sin(2*g),torch.sin(2*g)
    elif case=='noise_private': u,y=p,g.square()
    elif case=='private_signal': u,y=p,p+.1*g
    elif case=='constant_private': u,y=torch.ones_like(p),g.square()
    elif case=='xor':
        g=(g>0).double();u=(p>0).double();y=(g!=u).double()
    elif case=='correlated_increment': u=.8*g+.6*p;y=u
    else: raise ValueError(case)
    return g,u,y


def main():
    torch.set_num_threads(2)
    cases=['quadratic_copy','nonlinear_copy','noise_private','constant_private',
           'private_signal','xor','correlated_increment']
    results=[]
    for seed in [21,37,59]:
        for case in cases:
            parts=[pack(seed+offset,n,case) for offset,n in [(0,256),(1000,128),(2000,512)]]
            r=incremental_probe(*parts)
            gain=r['gain_relative_to_fit_mean_baseline']
            threshold=.05 if case in cases[:4] else .15
            passed=gain<threshold if case in cases[:4] else gain>threshold
            results.append({'seed':seed,'case':case,'passed':passed,'threshold':threshold,**r})
    receipt={'all_passed':all(r['passed'] for r in results),'results':results,
             'scope':'21 fabricated probe fits; no neural training, patients, weights or servers',
             'limitations':'predefined finite grid, simple scalar factors, not universal conditional independence test'}
    dest=Path(__file__).resolve().parents[1]/'docs/research/slotspe_common_private_20261101/incremental_probe_receipt.json'
    dest.write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
    print(json.dumps({'all_passed':receipt['all_passed'],'results':[{'case':r['case'],'seed':r['seed'],'gain':r['gain_relative_to_fit_mean_baseline'],'passed':r['passed']} for r in results]},indent=2))
    if not receipt['all_passed']: raise SystemExit(1)


if __name__=='__main__': main()
