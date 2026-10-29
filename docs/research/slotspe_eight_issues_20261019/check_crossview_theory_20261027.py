"""Exact finite sanity checks for the V2 cross-view theory. No model or patient data."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from fractions import Fraction as F
from hashlib import sha256
from itertools import combinations, permutations, product
import json
import math
from pathlib import Path

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
EXPECTED={
 'third_party/SlotSPE/models/SlotSPE.py':'eb5518a7e7dbc0e8e2fb1495c21c379ada36f78ad0c9d0d1f9ce69cb8051687e',
 'slotspe_proof/slot_disentanglement.py':'75ab77ac2cec6f16a4b73f37f027b574a4f41ca7f21e9cfb741fb13bfb8859ea',
 'slotspe_proof/plugin_host.py':'e47761e4a33c1e924dae29b8534383dc7d20f2d647040e069c74d6e7311465a7',
}

def cond_entropy_bits(rows, chosen):
    """Rows are equally weighted; this exact helper requires power-of-2 ratios."""
    counts=defaultdict(Counter)
    for target,slots in rows:
        key=tuple(slots[i] for i in chosen)
        counts[key][target]+=1
    total=F(0)
    for group in counts.values():
        n=sum(group.values())
        for count in group.values():
            ratio=F(n,count)
            assert ratio.denominator==1 and ratio.numerator & (ratio.numerator-1)==0
            total+=F(count,len(rows))*(ratio.numerator.bit_length()-1)
    return total

def shapley_and_last(rows,k):
    h0=cond_entropy_bits(rows,())
    hs=cond_entropy_bits(rows,tuple(range(k)))
    phi=[F(0)]*k
    orders=list(permutations(range(k)))
    for order in orders:
        prefix=()
        for i in order:
            nxt=tuple(sorted(prefix+(i,)))
            phi[i]+=(cond_entropy_bits(rows,prefix)-cond_entropy_bits(rows,nxt))/len(orders)
            prefix=nxt
    last=[cond_entropy_bits(rows,tuple(j for j in range(k) if j!=i))-hs for i in range(k)]
    assert sum(phi)==h0-hs
    return phi,last,h0-hs

def compositions(n,m):
    if m==1:
        yield (n,)
    else:
        for a in range(n+1):
            for tail in compositions(n-a,m-1): yield (a,)+tail

def run():
    checks=[]
    def record(name,**data): checks.append({'name':name,'passed':True,**data})
    hashes={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in EXPECTED}
    assert hashes==EXPECTED
    record('source_and_v2_hashes',matched=len(hashes))
    # A shared linear decoder matches 8 targets while all 8 private slots
    # contain all of (A,B). Native two-layer GELU can express linear maps by
    # GELU(x) - GELU(-x) = x; this does not execute the network itself.
    for a,b in product((0,1),repeat=2):
        slots=[(j,a,b,a if j%2 else b) for j in range(1,9)]
        targets=[(j,a if j%2 else b) for j in range(1,9)]
        def decode(z):
            e,x,y,c=z
            return (e,c)
        assert [decode(z) for z in slots]==targets
        assert all((z[1],z[2])==(a,b) for z in slots)
    record('v2_matching_zero_loss_does_not_exclude_full_information_duplication',cases=4,slots=8)
    # All eight slots duplicate target +/-1: the fixed full-set mean appears
    # deletion-sensitive although the exact remaining-data predictor is perfect.
    for t in (-1,1):
        full=F(sum([t]*8),8)
        dropped_fixed=F(sum([t]*7),8)
        assert full==t and (dropped_fixed-t)**2==F(1,64)
        assert F(sum([t]*7),7)==t
    record('fixed_decoder_deletion_sensitivity_is_not_unique_information',fixed_mse='1/64',best_drop_mse='0')
    duplicate=[(t,(t,t)) for t in (0,1) for _ in (0,1)]
    phi,last,total=shapley_and_last(duplicate,2)
    assert phi==[F(1,2),F(1,2)] and last==[0,0] and total==1
    record('duplicated_bit_shapley_and_last_increment',shapley=[str(x) for x in phi],last=[str(x) for x in last],total=str(total))
    xor=[(a^b,(a,b)) for a,b in product((0,1),repeat=2)]
    phi,last,total=shapley_and_last(xor,2)
    assert phi==[F(1,2),F(1,2)] and last==[1,1] and total==1
    assert all(cond_entropy_bits(xor,(i,))==1 for i in (0,1))
    record('xor_synergy_and_nonadditivity_of_last_increments',shapley=[str(x) for x in phi],last=[str(x) for x in last],total=str(total))
    separate=[((a,b),(a,b)) for a,b in product((0,1),repeat=2)]
    phi,last,total=shapley_and_last(separate,2)
    assert phi==[1,1] and last==[1,1] and total==2
    # For this distribution an intentionally uniform dropped predictor has
    # two-bit CE, although the optimal dropped predictor needs just one bit.
    assert cond_entropy_bits(separate,(0,))==1 and cond_entropy_bits(separate,(0,1))==0
    fake_delta=F(2); true_delta=F(1)
    assert fake_delta>true_delta
    record('imperfect_drop_decoder_overstates_increment',true_bits=str(true_delta),observed_bits=str(fake_delta))
    # beta*H(T|Z)-gamma*H(T|Z_-i) rewards an uninformative
    # representation if beta<gamma; equality leaves -gamma*CMI.
    beta,gamma=F(1),F(2)
    collapsed=beta*1-gamma*1
    duplicated=beta*0-gamma*0
    assert collapsed<duplicated and collapsed==-1
    beta=gamma=F(1)
    for h_full,h_drop in ((F(0),F(0)),(F(0),F(1)),(F(1),F(1))):
        assert beta*h_full-gamma*h_drop == -gamma*(h_drop-h_full)
    record('outer_weight_can_reward_global_collapse',beta_less_gamma_collapse=str(collapsed),
           duplicated=str(duplicated),equal_weights='minus conditional increment')
    vals=[F(0),F(1),F(0),F(1),F(0),F(1)]
    n=len(vals); a=b=2
    mu=sum(vals)/n; var=sum((v-mu)**2 for v in vals)/n
    pairs=[]
    for ai in combinations(range(n),a):
        for bi in combinations([i for i in range(n) if i not in ai],b):
            ma=sum(vals[i] for i in ai)/a
            mb=sum(vals[i] for i in bi)/b
            pairs.append((ma,mb))
    vb=sum((mb-mu)**2 for ma,mb in pairs)/len(pairs)
    cv=sum((ma-mu)*(mb-mu) for ma,mb in pairs)/len(pairs)
    mse=sum((ma-mb)**2 for ma,mb in pairs)/len(pairs)
    assert vb==F(n-b,b*(n-1))*var==F(1,10)
    assert cv==-var/F(n-1)==F(-1,20)
    assert mse==F(n,n-1)*(F(1,a)+F(1,b))*var==F(3,10)
    record('disjoint_views_finite_population_identity',allocations=len(pairs),target_variance=str(vb),view_covariance=str(cv),view_mse=str(mse))
    # Uniform over all 165 unordered length-8 multisets of 4 codes;
    # uniformly delete one occurrence. Exact joint/rational marginals.
    full=list(compositions(8,4)); remaining=list(compositions(7,4))
    assert len(full)==165 and len(remaining)==120
    joint={}; marginal=defaultdict(F)
    for counts in full:
        for j,nj in enumerate(counts):
            if nj:
                r=list(counts); r[j]-=1; r=tuple(r)
                p=F(nj,8*len(full))
                joint[(counts,r)]=p; marginal[r]+=p
    assert sum(joint.values())==1 and all(marginal[r]==F(1,120) for r in remaining)
    h_cond=sum(float(p)*math.log2(float(marginal[r]/p)) for (s,r),p in joint.items())
    lower=math.log2(F(165,120))
    assert h_cond>=lower>0
    # Cardinal thresholds: empirical target entropy is at most log2(289).
    assert math.comb(10,7)==120 and math.comb(11,8)==165
    assert math.comb(11,7)==330 and 330>289>120
    record('unordered_hard_capacity_witness',target_classes=165,remaining_multisets=120,
           average_conditional_entropy_bits=h_cond,strict_lower_bound_bits=lower,
           m5_drop_classes=330,example_train_patients=289)
    report=BASE/'V2跨视图重构与槽增量_理论试错_20261027.txt'
    return {'schema':'slotspe_v2_crossview_theory_v1',
      'recorded_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
      'passed':len(checks),'failed':0,'checks':checks,'protected_source_sha256':hashes,
      'checker_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
      'report_sha256':sha256(report.read_bytes()).hexdigest(),
      'execution':{'stdlib_arithmetic_only':True,'neural_forwards':0,'neural_backwards':0,
                   'patient_data_loads':0,'training_runs':0,'server_runs':0},
      'limitations':['Finite arithmetic checks supplement the written proofs, not formal verification.',
        'Synthetic exact examples do not show real model convergence, patient damage or benefit.',
        'Empirical decoder gaps do not equal conditional information without fit-error assumptions.']}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=run()
    if args.output:
        with args.output.open('x',encoding='utf-8',newline='\n') as f:
            f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'failed':result['failed'],'checks':result['checks'],
        'output':str(args.output) if args.output else None},ensure_ascii=False,indent=2))
