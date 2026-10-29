"""Exact arithmetic witnesses for the original SlotSPE audit. No neural execution.

The general statements are proved in the accompanying text. Finite checks are
sanity checks, not a proof assistant or a claim about natural training.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from fractions import Fraction as F
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
EXPECTED = {
    "third_party/SlotSPE/models/SlotSPE.py": "eb5518a7e7dbc0e8e2fb1495c21c379ada36f78ad0c9d0d1f9ce69cb8051687e",
    "third_party/SlotSPE/models/slot_attention.py": "200f92f62f74d1387f62834d485cd96dd7adf7d104e75707d4a0a51a021ac109",
    "third_party/SlotSPE/models/transformer.py": "9dfed61da662a391ff7a29b3b514e892f1cf8287a0d350ac0afab9d5d4a54985",
    "third_party/SlotSPE/utils/loss_func.py": "bb52dca9e903dd72d18b694bcdc94452a0bd201f96412937db8706f80924d682",
}


def exact_top2(weights):
    """Analytic ideal-Gumbel top-2 law, via exponential-race sampling."""
    total = sum(weights)
    result = defaultdict(F)
    for i, wi in enumerate(weights):
        for j, wj in enumerate(weights):
            if i != j:
                result[tuple(sorted((i, j)))] += F(wi, total) * F(wj, total - wi)
    assert sum(result.values()) == 1
    return dict(result)


def run():
    checks = []
    def record(name, **details):
        checks.append({"name": name, "passed": True, **details})

    hashes = {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in EXPECTED}
    assert hashes == EXPECTED
    record("protected_native_source_hashes", matched=len(hashes))

    # Stress the algebra with every nonempty proper mask for S=2..8,
    # and three different selected-row distributions. Unselected rows are
    # exactly uniform as proved from the native finite-sentinel mask.
    mask_cases = 0
    minimum_default_mass = F(1)
    for s in range(2, 9):
        for k in range(1, s):
            for chosen in combinations(range(s), k):
                selected = set(chosen)
                for mode in ("selected_only", "tiny_leak", "arbitrary_positive"):
                    rows = []
                    for i in range(s):
                        if i not in selected:
                            rows.append([F(1, s)] * s)
                        else:
                            if mode == "selected_only":
                                raw = [F(j + 1) if j in selected else F(0) for j in range(s)]
                            elif mode == "tiny_leak":
                                raw = [F(j + 1) if j in selected else F(1, 10**20) for j in range(s)]
                            else:
                                raw = [F((i + 2) * (j + 1) + 1) for j in range(s)]
                            rows.append([x / sum(raw) for x in raw])
                    beta = [sum(row[j] for row in rows) / s for j in range(s)]
                    assert sum(beta) == 1
                    unselected_mass = sum(beta[j] for j in range(s) if j not in selected)
                    lower = F((s - k)**2, s*s)
                    assert unselected_mass >= lower
                    if mode == "selected_only":
                        assert unselected_mass == lower
                    values = [F(2*j - s + 1, j + 1) for j in range(s)]
                    pooled = sum(beta[j] * values[j] for j in range(s))
                    decomposition = F(s-k, s) * sum(values)/s + sum(
                        sum(rows[i][j]*values[j] for j in range(s)) for i in selected
                    )/s
                    assert pooled == decomposition
                    if s == 8 and k == 2:
                        minimum_default_mass = min(minimum_default_mass, unselected_mass)
                    mask_cases += 1
    assert minimum_default_mass == F(9, 16)
    record("pooled_attention_bound_and_decomposition", rational_matrix_cases=mask_cases,
           default_minimum_unselected_mass=str(minimum_default_mass))

    signs = [1]*4 + [-1]*4
    plus = exact_top2([4]*4 + [1]*4)
    minus = exact_top2([1]*4 + [4]*4)
    categories = defaultdict(F)
    for pair, probability in plus.items():
        categories[sum(signs[i] for i in pair)] += probability
    assert categories[2] == F(3,5)
    assert categories[-2] == F(3,95)
    assert categories[0] == F(7,19)
    record("finite_score_gumbel_top2_probabilities", masks=len(plus),
           two_positive=str(categories[2]), mixed=str(categories[0]),
           two_negative=str(categories[-2]))

    # u=+/-log(2), tau=.01 => global softmax weight ratio 4**100.
    huge = 4**100
    soft_plus = [F(huge, 4*(huge+1))]*4 + [F(1,4*(huge+1))]*4
    soft_minus = soft_plus[4:] + soft_plus[:4]
    eps = F(1,10**8)
    c = [F(1,3), F(-1,7), F(2,5), F(3,4)]
    for pair, probability in plus.items():
        permuted = tuple(sorted((i+4)%8 for i in pair))
        assert probability == minus[permuted]
        mass_p = sum(soft_plus[i] for i in pair)
        mass_m = sum(soft_minus[i] for i in permuted)
        assert mass_p == mass_m
        aux_p = [mass_p/(mass_p+eps)*x for x in c]
        aux_m = [mass_m/(mass_m+eps)*x for x in c]
        assert aux_p == aux_m
    record("auxiliary_output_distribution_identical_with_native_epsilon", paired_masks=28,
           temperature="1/100", epsilon=str(eps), nonzero_auxiliary_logits=True)

    # Exact slot construction: zero channel mean, equal norm, distinct,
    # and a nonzero rank-4 auxiliary linear head annihilating all differences.
    d = 256
    v = [F(0)]*d
    v[0], v[1] = F(1), F(-1)
    z = []
    for i in range(4):
        zi = v.copy()
        zi[2+2*i], zi[3+2*i] = F(1,2), F(-1,2)
        z.append(zi)
    z += [[-x for x in zi] for zi in z.copy()]
    assert len(set(tuple(x) for x in z)) == 8
    assert all(sum(zi) == 0 for zi in z)
    assert len(set(sum(x*x for x in zi) for zi in z)) == 1
    assert all(sum(zi[j] for zi in z) == 0 for j in range(d))
    assert [F(zi[0]-zi[1],2) for zi in z] == signs
    # Four independent coordinate rows in unused dimensions give rank 4.
    assert all(all(zi[j] == 0 for j in (20,21,22,23)) for zi in z)
    record("eight_distinct_slots_equal_norm_and_rank4_decoder_kernel", dimension=d, rank=4)

    # eta is symbolic exp(-1e9) in the proof. These nonzero rational eta
    # checks validate the finite-sentinel formula without rounding eta to 0.
    for eta in (F(0), F(1,10**30), F(1,10), F(1,2)):
        kappa = F(2)*(1-eta)/(2+6*eta)
        for pair in plus:
            chosen = set(pair)
            rows = []
            for i in range(8):
                raw = [F(1) if i in chosen and j in chosen else eta for j in range(8)]
                if i not in chosen:
                    rows.append([F(1,8)]*8)
                else:
                    rows.append([x/sum(raw) for x in raw])
            pooled = sum(sum(row[j]*signs[j] for j in range(8)) for row in rows)/8
            m = sum(signs[i] for i in pair)/2
            assert pooled == kappa*F(m)/4
    record("finite_mask_native_readout_counterexample", rational_eta_cases=4, masks_per_eta=28)

    gap_coefficient = categories[2]-categories[-2]
    gap_log_argument = F(2)/F(5,4)
    assert gap_coefficient == F(54,95)
    assert gap_log_argument == F(8,5) and gap_log_argument > 1
    record("strict_fused_nll_gap", exact_expression="(54/95)*log(8/5)",
           approximate_for_readability=float(gap_coefficient)*math.log(float(gap_log_argument)),
           positive_by="positive rational coefficient times log of rational greater than one")

    vals = [F(-2), F(1), F(3), F(6)]
    global_mean = sum(vals)/len(vals)
    for scale in (F(1,10**14), F(1,100), F(1,2), F(9,10)):
        u = sum((scale+eps)*x for x in vals)/(len(vals)*(scale+eps))
        assert u == global_mean
    a = [F(1,20), F(1,10), F(1,50), F(1,200)]
    mass = sum(a)
    mu = sum(x*y for x,y in zip(a,vals))/mass
    u = sum((x+eps)*y for x,y in zip(a,vals))/(mass+len(vals)*eps)
    assert u == mass/(mass+len(vals)*eps)*mu + len(vals)*eps/(mass+len(vals)*eps)*global_mean
    record("native_epsilon_support_mass_identity_and_collision", exact_scale_cases=4)

    after = (F(9,10)+eps)/(1+2*eps)
    before = F(1,2)
    assert after-before == F(2,5)/(1+2*eps) > 0
    record("selection_and_aggregation_do_not_commute", exact_gap=str(after-before))

    return {"schema":"slotspe_native_gate_math_audit_v1",
            "recorded_at":datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "checks":checks, "passed":len(checks), "failed":0,
            "protected_source_sha256":hashes,
            "checker_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
            "report_sha256":sha256((BASE/'原版门控与证据支持量_严格审查_20261027.txt').read_bytes()).hexdigest(),
            "execution":{"stdlib_arithmetic_only":True,"neural_forwards":0,"neural_backwards":0,
                         "patient_data_loads":0,"training_runs":0,"server_runs":0},
            "limitations":["No formal proof assistant used.","Finite arithmetic checks supplement the written proofs.",
                           "Gumbel probabilities use the ideal continuous law, not PyTorch RNG measurements.",
                           "No patient harm, correction efficacy, or full encoder reachability established."]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    receipt = run()
    payload = json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(payload)
    print(json.dumps({"passed":receipt['passed'],"failed":0,"checks":receipt['checks'],
                      "output":str(args.output) if args.output else None},ensure_ascii=False,indent=2))
