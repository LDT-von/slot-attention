"""Read-only reproduction of the three exact fixed-target counterexamples."""
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path


def main():
    rows = [dict(a=a, b=b, target=b, private=b, prob=.25) for a, b in itertools.product([0, 1], repeat=2)]
    def entropy(*keys):
        masses = defaultdict(float)
        for row in rows: masses[tuple(row[k] for k in keys)] += row['prob']
        return -sum(p*math.log(p) for p in masses.values())
    def information(x, y):
        return entropy(x)+entropy(y)-entropy(x, y)
    report = json.loads(Path('docs/research/slotspe_common_private_20261101/frozen_target_counterexamples_receipt.json').read_text())
    one, two, three = report['cases']
    measurements = [
        (one['target_entropy_nats'], entropy('target')),
        (one['I_V_target'], information('private', 'target')),
        (one['I_V_outcome_A'], information('private', 'a')),
        (one['same_input_alternate_outcome_B_information'], information('private', 'b')),
        (two['minimum_rate_for_perfect_recovery_of_independent_task_A_and_target_B'], entropy('a', 'b')),
        (two['remaining_conditional_input_information_bound_if_zero_ce'], two['trial_cap_nats']-entropy('target')),
        (three['minimum_I_G_V_nats'], entropy('b')),
        (three['H_T_given_G'], entropy('b', 'b')-entropy('b')),
    ]
    assert all(abs(a-b) < 1e-12 for a, b in measurements)
    assert entropy('a', 'b') > two['trial_cap_nats']
    print(json.dumps({'exact_counterexamples': 3, 'recomputed_scalar_checks': len(measurements),
                      'passed': True, 'patient_execution': False}))


if __name__ == '__main__': main()
