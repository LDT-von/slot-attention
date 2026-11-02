"""Exact finite-population information/content controls, not MI estimators.

Rows enumerate hypothetical patient states. A patient's slots are variables
of the same row, never treated as independent patient observations.
"""
import itertools
import math
from collections import defaultdict


class FiniteLaw:
    def __init__(self, rows):
        supplied = [(float(p), row) for p, row in rows]
        if not supplied or any(not math.isfinite(p) or p < 0 for p, _ in supplied):
            raise ValueError('finite nonnegative probability mass required')
        self.rows = [(p, row) for p, row in supplied if p > 0]
        if abs(sum(p for p, _ in self.rows)-1) > 1e-10:
            raise ValueError('probabilities must sum to one')

    def entropy(self, names, given=()):
        def joint(keys):
            masses = defaultdict(float)
            for p, row in self.rows: masses[tuple(row[k] for k in keys)] += p
            return -sum(p*math.log(p) for p in masses.values())
        return joint(tuple(names)+tuple(given))-joint(tuple(given))

    def mi(self, x, y, given=()):
        value = self.entropy(x, given)+self.entropy(y, given)-self.entropy(tuple(x)+tuple(y), given)
        return 0. if abs(value) < 1e-12 else value

    def tc(self, variables, given=()):
        value = sum(self.entropy([v], given) for v in variables)-self.entropy(variables, given)
        return 0. if abs(value) < 1e-12 else value

    def mse(self, target, given):
        groups = defaultdict(lambda: [0., 0., 0.])
        for p, row in self.rows:
            values = groups[tuple(row[k] for k in given)]
            values[0] += p; values[1] += p*row[target]; values[2] += p*row[target]**2
        return sum(second-first*first/mass for mass, first, second in groups.values())


CODINGS = ('factorwise', 'duplicated_patient', 'concentrated', 'conditional_reencoding',
           'within_modality_synergy', 'shared_copy', 'constant', 'independent_noise',
           'correlated_posterior_noise')
SLOTS = ('v0', 'v1', 'v2', 'v3')


def four_factor_law(coding):
    if coding not in CODINGS: raise ValueError('unknown known-answer coding')
    rows = []
    for s, p0, p1, q0, q1 in itertools.product([0, 1], repeat=5):
        private = (p0, p1, q0, q1)
        if coding == 'independent_noise': noise = list(itertools.product([0, 1], repeat=4))
        elif coding == 'correlated_posterior_noise': noise = [(0,)*4, (1,)*4]
        else: noise = [None]
        for epsilon in noise:
            if coding == 'factorwise': values = private
            elif coding == 'duplicated_patient': values = ((p0, p1), (p0, p1), (q0, q1), (q0, q1))
            elif coding == 'concentrated': values = ((p0, p1), 0, (q0, q1), 0)
            elif coding == 'conditional_reencoding': values = tuple(p ^ s for p in private)
            elif coding == 'within_modality_synergy': values = (p0 ^ p1, p1, q0 ^ q1, q1)
            elif coding == 'shared_copy': values = (s,)*4
            elif coding == 'constant': values = (0,)*4
            else: values = epsilon
            row = {'a': (s, *private), 'g': s, 's': s, 'target': private,
                   'parity': p0 ^ p1, **dict(zip(SLOTS, values))}
            for k, p in enumerate(private):
                row['p'+str(k)] = p
                row['t'+str(k)] = (2*s-1)+2*(2*p-1)
            # Four distinct time bins require each private factor; no scalar-risk reduction.
            row['hazards'] = tuple(1/(1+math.exp(-(-1.5+.6*k+.3*(2*s-1)+1.2*(2*p-1)))) for k, p in enumerate(private))
            rows.append((1/(32*len(noise)), row))
    return FiniteLaw(rows)


def rate_decomposition(law):
    # r_k is the exact aggregate marginal q(V_k), chosen separately per law.
    # This is an oracle reference, NOT the neural N(0,I) training prior.
    rate = sum(law.entropy([v])-law.entropy([v], ['a', 'g']) for v in SLOTS)
    joint = law.mi(['a'], SLOTS, ['g'])
    tc = law.tc(SLOTS, ['g'])
    posterior_tc = law.tc(SLOTS, ['a', 'g'])
    mismatch = sum(law.mi(['g'], [v]) for v in SLOTS)
    return {'rate_to_matched_marginals': rate, 'joint_conditional_information': joint,
            'conditional_total_correlation': tc, 'posterior_conditional_tc': posterior_tc,
            'conditional_marginal_reference_mismatch': mismatch,
            'identity_residual': rate-(joint+tc-posterior_tc+mismatch)}


def bayes_survival_nll(law, given, censor=(.1, .1, .1, .7)):
    """Population optimum under independent censoring, ignoring censor factors.

    Mix event probabilities and survival curves within each representation
    group. Averaging hazards would be incorrect for these mixtures.
    """
    if len(censor) != 4 or any(p < 0 for p in censor) or abs(sum(censor)-1) > 1e-12:
        raise ValueError('four-bin independent censor distribution required')
    groups = defaultdict(lambda: [0., [0.]*4, [0.]*4])
    for p, row in law.rows:
        group = groups[tuple(row[k] for k in given)]; group[0] += p
        survival = 1.
        for k, hazard in enumerate(row['hazards']):
            group[1][k] += p*survival*hazard
            survival *= 1-hazard
            group[2][k] += p*survival
    total = 0.
    for mass, event, survival in groups.values():
        for k in range(4):
            f, s = event[k]/mass, survival[k]/mass
            total -= mass*(f*sum(censor[k:])*math.log(f)+censor[k]*s*math.log(s))
    return total


def content_row(coding):
    law = four_factor_law(coding)
    all_slots = ['g', *SLOTS]
    subset_nll = [bayes_survival_nll(law, ['g', left, right])
                  for left in SLOTS[:2] for right in SLOTS[2:]]
    whole = bayes_survival_nll(law, all_slots)
    return {'coding': coding, 'enumerated_states': len(law.rows), **rate_decomposition(law),
            'target_entropy_given_common': law.entropy(['target'], ['g']),
            'target_entropy_given_all': law.entropy(['target'], all_slots),
            'all_slot_conditional_content_mse': sum(law.mse('t'+str(k), all_slots) for k in range(4))/4,
            'per_slot_conditional_content_mse': sum(law.mse('t'+str(k), ['g', SLOTS[k]]) for k in range(4))/4,
            'own_factor_entropy_given_corresponding_private_alone': sum(law.entropy(['p'+str(k)], [SLOTS[k]]) for k in range(4))/4,
            'survival_nll_common': bayes_survival_nll(law, ['g']),
            'survival_nll_all': whole,
            'oracle_best_one_slot_per_modality_nll': min(subset_nll),
            'whole_minus_oracle_best_single_nll': whole-min(subset_nll),
            'leave_one_out_target_information': [law.mi(['target'], [v], ['g', *[u for u in SLOTS if u != v]]) for v in SLOTS]}


def noisy_shared_law(correction=False, flip=.25):
    if not 0 <= flip <= .5: raise ValueError('flip probability must lie in[0,.5]')
    return FiniteLaw([(p/2, {'s': s, 'g': s ^ error, 'v': error if correction else 0})
                      for s in [0, 1] for error, p in [(0, 1-flip), (1, flip)]])
