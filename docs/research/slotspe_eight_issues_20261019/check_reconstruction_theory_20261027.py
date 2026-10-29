"""Exact arithmetic checks for the written reconstruction arguments; no model execution."""
import argparse
from datetime import datetime, timezone, timedelta
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
EXPECTED = {
 'third_party/SlotSPE/models/SlotSPE.py':'eb5518a7e7dbc0e8e2fb1495c21c379ada36f78ad0c9d0d1f9ce69cb8051687e',
 'third_party/SlotSPE/models/slot_attention.py':'200f92f62f74d1387f62834d485cd96dd7adf7d104e75707d4a0a51a021ac109',
 'third_party/SlotSPE/models/omics_encoder.py':'3e2231dab47103046922ec40a9bf6945fe936081037a630b72035eee5de0e685',
 'third_party/SlotSPE/dataset/dataset_survival.py':'44677851b59ca9b60547d1623c2ff9a3e32a3e9ffb38f67f569e6c0ff3048ea3',
}

def run():
    checks=[]
    def record(name, **data): checks.append({'name':name,'passed':True,**data})
    hashes={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in EXPECTED}
    assert hashes==EXPECTED
    record('native_source_hashes',matched=len(hashes))
    # A zero-weight trainable projection with a nonzero output bias makes
    # every target identical; choosing the same output bias for the native
    # decoder yields exact cosine 1. This is algebra, not a model run.
    v=[F(3),F(4),F(0)]
    norm2=sum(x*x for x in v)
    assert norm2>0 and sum(x*y for x,y in zip(v,v))**2==norm2**2
    record('nonzero_constant_target_cosine_is_one', norm_squared=str(norm2))
    cases=0
    for n in range(1,9):
      for pad in range(0,9):
        real=[F((i+1)%5,5) for i in range(n)]
        pad_losses=[F(3,5)]*pad
        total=sum(real+pad_losses)/F(n+pad)
        grouped=F(n,n+pad)*(sum(real)/n)+F(pad,n+pad)*(F(3,5) if pad else F(0))
        assert total==grouped
        cases+=1
    record('padding_loss_weight_decomposition', cases=cases)
    # Patient-specific histogram targets and the finite-dimensional
    # separation inequality, checked for a family of exact vectors.
    examples=0
    for a in range(9):
      for b in range(9):
        t1=[F(a,8),1-F(a,8)]
        t2=[F(b,8),1-F(b,8)]
        for e1 in (F(0),F(1,16),F(-1,16)):
          for e2 in (F(0),F(1,16),F(-1,16)):
            d1=[t1[0]+e1,t1[1]-e1]
            d2=[t2[0]+e2,t2[1]-e2]
            # Here all distances are sqrt(2)*|first-coordinate difference|,
            # so the triangle/Lipschitz bound is checked without float roots.
            delta=abs(t1[0]-t2[0])
            assert abs(d1[0]-d2[0])>=max(F(0),delta-abs(e1)-abs(e2))
            examples+=1
    record('patient_target_separation_triangle', cases=examples)
    # Coordinate mass is bounded by vector l2 error (squares only).
    checks_count=0
    for j in range(-8,9):
      for k in range(-8,9):
        x,y=F(j,8),F(k,8)
        assert x*x<=x*x+y*y and y*y<=x*x+y*y
        checks_count+=1
    record('mass_coordinate_bound',cases=checks_count)
    # Two WSI states each have two possible genomic values. For arbitrary
    # deterministic predictors, verify orthogonal conditional-MSE identity.
    joint=[(0,F(0),F(1,4)),(0,F(2),F(1,4)),(1,F(1),F(1,4)),(1,F(3),F(1,4))]
    mu={0:F(1),1:F(2)}
    var=sum(p*(g-mu[x])**2 for x,g,p in joint)
    assert var==1
    for ga in range(-3,5):
      for gb in range(-3,5):
        pred={0:F(ga),1:F(gb)}
        mse=sum(p*(g-pred[x])**2 for x,g,p in joint)
        excess=sum(p*(mu[x]-pred[x])**2 for x,g,p in joint)
        assert mse==var+excess
    record('conditional_mse_identity',predictors=64,irreducible_variance=str(var))
    # Q=A, and a constant Z predicts the mean. E[A]=0 and variance 1.
    a=[F(-1),F(1)]
    r_z=sum(x*x for x in a)/2
    r_zq=sum((x-x)**2 for x in a)/2
    assert r_z==1 and r_zq==0 and r_z-r_zq==1
    record('query_bypass_conditional_variance_example',without_query=str(r_z),with_query=str(r_zq))
    # Entropy-gap demonstration with binary probabilities is asserted via
    # exact conditional distributions rather than numerical MI estimates.
    prob_b_given_g=[F(1,4),F(3,4)]
    p_b=sum(prob_b_given_g)/2
    assert p_b==F(1,2) and all(p != p_b for p in prob_b_given_g)
    record('conditional_information_nonzero_example', probabilities=[str(x) for x in prob_b_given_g])
    report=BASE/'重构模块问题与全槽内容重建方案_20261027.txt'
    return {'schema':'slotspe_reconstruction_theory_audit_v1',
      'recorded_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
      'passed':len(checks),'failed':0,'checks':checks,'protected_source_sha256':hashes,
      'checker_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
      'report_sha256':sha256(report.read_bytes()).hexdigest(),
      'execution':{'stdlib_arithmetic_only':True,'neural_forwards':0,'neural_backwards':0,
                   'patient_data_loads':0,'training_runs':0,'server_runs':0},
      'limits':['Finite checks do not replace the mathematical proofs or constitute formal verification.',
                'No patient outcomes, biological semantics, model reachability or intervention efficacy are proved.']}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    r=run()
    if args.output:
      with args.output.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':r['passed'],'failed':r['failed'],'checks':r['checks'],
                      'output':str(args.output) if args.output else None},ensure_ascii=False,indent=2))
