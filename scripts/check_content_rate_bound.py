"""Exact finite distributions: audit content/rate chain-rule inequalities."""
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path


def entropy(rows,names):
    mass=defaultdict(float)
    for row in rows:mass[tuple(row[k] for k in names)]+=row['prob']
    return -sum(p*math.log(p) for p in mass.values() if p>0)


def measure(case):
    rows=[];rate=0.
    for a,(g,t,n) in enumerate(itertools.product(range(2),repeat=3)):
        target=g if case=='target_known_by_common' else t
        if case=='synergy_target':target=t^n
        if case in ['minimal','synergy_target']:outputs=[(target,1.)];states=2
        elif case in ['copy_nuisance','target_known_by_common']:outputs=[(2*t+n,1.)] if case=='copy_nuisance' else [(n,1.)];states=4 if case=='copy_nuisance' else 2
        elif case in ['uninformative','mismatched_prior']:outputs=[(0,.5),(1,.5)];states=2
        elif case=='overlapping_targets':outputs=[(t,1.)];states=2
        else:raise ValueError(case)
        for v,q in outputs:
            prior=(.75 if v==0 else .25) if case=='mismatched_prior' else 1/states
            rate+=q/8*math.log(q/prior)
            rows.append({'a':a,'g':g,'t':target,'v':v,'prob':q/8})
    h=lambda *names:entropy(rows,names)
    conditional_h=h('t','g')-h('g')
    ce=h('t','v','g')-h('v','g') # optimal decoder cross entropy
    extra=h('a','t','g')+h('v','t','g')-h('t','g')-h('a','v','t','g')
    upper=rate-conditional_h+ce
    naive=rate-h('t')+ce
    assert extra<=upper+1e-12
    return {'case':case,'rate':rate,'target_conditional_entropy':conditional_h,'bayes_content_ce':ce,
            'extra_information_given_target_common':extra,'valid_upper_bound':upper,
            'incorrect_unconditional_entropy_substitution':naive}


def main():
    cases=['minimal','copy_nuisance','uninformative','mismatched_prior','target_known_by_common','overlapping_targets','synergy_target']
    rows=[measure(c) for c in cases]
    shared=next(r for r in rows if r['case']=='target_known_by_common')
    assert shared['extra_information_given_target_common']>shared['incorrect_unconditional_entropy_substitution']+.6
    report={'all_valid_bounds_passed':True,'rows':rows,'scope':'exact enumeration, zero model/patient training',
            'limitations':'conditional entropy known in finite construction; finite learned CE/rate cannot certify the population inequality numerically'}
    path=Path('docs/research/slotspe_common_private_20261101/content_rate_bound_receipt.json')
    if path.exists():raise RuntimeError('preserve existing receipt')
    path.write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps(report,indent=2))


if __name__=='__main__':main()
