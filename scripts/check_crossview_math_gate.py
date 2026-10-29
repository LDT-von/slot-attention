"""Exact synthetic arithmetic for conditional cross-view and finite-table proofs.

No neural models, patient data, training, external data, or server operations.
"""
from collections import Counter, defaultdict
import argparse
from datetime import datetime, timezone, timedelta
from fractions import Fraction as F
from itertools import combinations, product
import math
import json
from pathlib import Path

def expectation(rows, fun): return sum(p*fun(x) for x,p in rows)

def empirical_mse_oracle(rows):
    groups_full=defaultdict(list); groups_drop=defaultdict(list)
    for target,slots in rows:
        full=tuple(sorted(slots))
        groups_full[full].append((target,F(1,len(rows))))
        for i in range(len(slots)):
            drop=tuple(sorted(slots[:i]+slots[i+1:]))
            groups_drop[drop].append((target,F(1,len(rows)*len(slots))))
    def risk(groups):
        total=F(0)
        for group in groups.values():
            mass=sum(p for _,p in group)
            mean=sum(y*p for y,p in group)/mass
            total+=sum(p*(y-mean)**2 for y,p in group)
        return total
    return risk(groups_full),risk(groups_drop)

def entropy_continuity(t,d):
    if t>=1-1/d: return math.log(d)
    if t==0: return 0.0
    return t*math.log(d-1)-t*math.log(t)-(1-t)*math.log1p(-t)

def entropy_certificate(n, delta=.05):
    q,u,v=3,3,2
    tf=min(1,math.sqrt((q*u*math.log(2)+math.log(4/delta))/(2*n)))
    td=min(1,math.sqrt((q*v*math.log(2)+math.log(4/delta))/(2*n)))
    penalty=sum((entropy_continuity(tf,q*u),entropy_continuity(tf,u),
                 entropy_continuity(td,q*v),entropy_continuity(td,v)))
    h=-(F(1,3)*math.log(F(1,3))+F(2,3)*math.log(F(2,3)))
    return {'n':n,'empirical_gap_nats':float(h),'penalty_nats':penalty,
            'lower_bound_nats':max(0,float(h)-penalty)}

def run():
    checks=[]
    def record(name,**kw): checks.append({'name':name,'passed':True,**kw})
    # theta~Uniform[0,1], four independent Bernoulli draws conditional theta.
    # Integral theta^k(1-theta)^(4-k) = k!(4-k)!/5! for each bit sequence.
    rows=[]
    for bits in product((0,1),repeat=4):
        k=sum(bits)
        p=F(math.factorial(k)*math.factorial(4-k),math.factorial(5))
        rows.append((bits,p))
    assert sum(p for _,p in rows)==1
    target=lambda bits:F(sum(bits[2:]),2)
    predict=lambda bits:F(1+sum(bits[:2]),4)
    r=expectation(rows,lambda bits:(target(bits)-predict(bits))**2)
    baseline=expectation(rows,lambda bits:(target(bits)-F(1,2))**2)
    assert r==F(1,8) and baseline==F(1,6) and baseline-r==F(1,24)
    record('beta_bernoulli_positive_exact',ordered_sequences=16,bayes_mse=str(r),baseline_mse=str(baseline),gain=str(baseline-r))
    # A and B are independent fair bits: their exact conditional distribution
    # gives the same predictor and risk as the unconditional constant.
    independent=list(product((0,1),repeat=2))
    group_means={a:sum(F(b) for aa,b in independent if aa==a)/2 for a in (0,1)}
    assert group_means=={0:F(1,2),1:F(1,2)}
    risk=sum((F(b)-group_means[a])**2 for a,b in independent)/4
    assert risk==F(1,4)
    record('independent_view_no_predictive_gain',conditional_mse=str(risk),constant_mse=str(risk))
    # Distinct randomly separated tokens from same identical finite bag:
    # B=1-A perfectly, but all patient full-bag means equal 1/2.
    example=[(F(0),F(1)),(F(1),F(0))]
    assert all(b==1-a for a,b in example)
    assert all((a+b)/2==F(1,2) for a,b in example)
    record('perfect_crossview_does_not_prove_patient_difference',states=2,crossview_mse='0',patient_bag_mean_variance='0')
    # Fixed bag {0,0,0,1,1,1}, split A/B without replacement.
    bag=[F(0),F(0),F(0),F(1),F(1),F(1)]; n=6
    pairs=[]
    for a in combinations(range(n),2):
        for b in combinations([i for i in range(n) if i not in a],2):
            pairs.append((sum(bag[i] for i in a)/2,sum(bag[i] for i in b)/2))
    mu=sum(bag)/n; var=sum((x-mu)**2 for x in bag)/n
    cov=sum((a-mu)*(b-mu) for a,b in pairs)/len(pairs)
    assert cov==-var/F(n-1)==F(-1,20)
    record('disjoint_finite_views_negatively_correlated',allocations=len(pairs),covariance=str(cov))
    # Patient content and its perfectly predictive slot value are equal bits.
    # A random independent donor has 1/2 expected MSE; own MSE is 0.
    patients=[F(0),F(1)]
    own=sum((mu-mu)**2 for mu in patients)/2
    donor=sum((mu-other)**2 for mu in patients for other in patients)/4
    covariance=sum((mu-F(1,2))**2 for mu in patients)/2
    assert donor-own==2*covariance==F(1,2)
    record('matched_vs_independent_donor_identity',own_mse=str(own),donor_mse=str(donor),gap=str(donor-own))
    dup=[(F(t),(t,t)) for t in (0,1)]
    full,drop=empirical_mse_oracle(dup)
    assert full==drop==0
    fixed_drop=sum((F(7,8)*t-t)**2 for t in (-1,1))/2
    assert fixed_drop==F(1,64)
    record('oracle_rejects_weak_drop_decoder_false_positive',oracle_gap=str(drop-full),fixed_decoder_mse=str(fixed_drop))
    toy=[(F(0),(0,0)),(F(1,2),(0,1)),(F(1),(1,1))]
    full,drop=empirical_mse_oracle(toy)
    assert full==0 and drop==F(1,18)
    record('finite_multiset_oracle_nonempty_example',patients=3,oracle_gap=str(drop-full))
    # Both original continuous slots reveal T, but shared quantization is
    # allowed to retain it only when flag=0, inducing artificial code gap.
    for t in (0,1):
        z1,z2=(t,0),(t,1)
        q=lambda z:z[0] if z[1]==0 else 0
        assert z1[0]==z2[0]==t and q(z1)==t and q(z2)==0
    record('quantized_gap_need_not_mean_continuous_slot_uniqueness',states=2)
    lo_small=entropy_certificate(288); lo_large=entropy_certificate(9999)
    assert lo_small['lower_bound_nats']==0 and lo_large['lower_bound_nats']>0
    record('finite_sample_certificate_can_be_vacuous',small=lo_small,large=lo_large)
    return {'schema':'crossview_math_gate_v1','checks':checks,'passed':len(checks),'failed':0,
      'execution':{'stdlib_arithmetic_only':True,'neural_forwards':0,'neural_backwards':0,
                   'patient_data_loads':0,'training_runs':0,'server_runs':0},
      'limitations':['The proven positive example assumes a Dirichlet/iid patient distribution.',
        'The conditional table is an empirical oracle, not a deployed predictor or patient outcome proof.',
        'Quantized-code information does not establish unique information in original continuous slots.']}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Create a new JSON receipt; existing files are never overwritten')
    args=parser.parse_args()
    result=run()
    result['recorded_at']=datetime.now(timezone(timedelta(hours=8))).isoformat()
    payload=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        with args.output.open('x',encoding='utf-8',newline='\n') as f: f.write(payload)
    print(payload,end='')
