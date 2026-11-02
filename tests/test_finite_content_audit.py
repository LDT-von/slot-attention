import math
import pytest

from slotspe_proof.finite_content_audit import FiniteLaw, CODINGS, SLOTS, four_factor_law, content_row, noisy_shared_law, bayes_survival_nll


@pytest.mark.parametrize('coding', CODINGS)
def test_rate_identity_including_correlated_posterior_correction(coding):
    row = content_row(coding)
    assert abs(row['identity_residual']) < 1e-11
    if coding == 'correlated_posterior_noise':
        assert row['rate_to_matched_marginals'] == pytest.approx(0, abs=1e-12)
        assert row['conditional_total_correlation'] == pytest.approx(3*math.log(2))
        assert row['posterior_conditional_tc'] == pytest.approx(3*math.log(2))


def test_content_reencoding_is_not_identified_by_rate_or_joint_prediction():
    good, recoded, concentrated, duplicate = [content_row(c) for c in ['factorwise', 'conditional_reencoding', 'concentrated', 'duplicated_patient']]
    for row in [recoded, concentrated, duplicate]:
        assert row['survival_nll_all'] == pytest.approx(good['survival_nll_all'])
        assert row['all_slot_conditional_content_mse'] == pytest.approx(0, abs=1e-12)
    assert recoded['rate_to_matched_marginals'] == pytest.approx(good['rate_to_matched_marginals'])
    assert recoded['own_factor_entropy_given_corresponding_private_alone'] == pytest.approx(math.log(2))
    assert good['own_factor_entropy_given_corresponding_private_alone'] == pytest.approx(0, abs=1e-12)
    assert concentrated['rate_to_matched_marginals'] == pytest.approx(good['rate_to_matched_marginals'])
    assert duplicate['rate_to_matched_marginals'] == pytest.approx(2*good['rate_to_matched_marginals'])


def test_multi_factor_task_needs_all_factorwise_slots_but_not_duplicate_copies():
    useful, duplicate = [content_row(c) for c in ['factorwise', 'duplicated_patient']]
    assert useful['whole_minus_oracle_best_single_nll'] < -.01
    assert duplicate['whole_minus_oracle_best_single_nll'] == pytest.approx(0, abs=1e-12)
    assert all(value == pytest.approx(math.log(2)) for value in useful['leave_one_out_target_information'])
    assert all(abs(value) < 1e-12 for value in duplicate['leave_one_out_target_information'])


def test_per_slot_content_can_penalize_valid_cooperative_code():
    row = content_row('within_modality_synergy')
    assert row['all_slot_conditional_content_mse'] == pytest.approx(0, abs=1e-12)
    assert row['per_slot_conditional_content_mse'] > 0
    assert row['survival_nll_all'] == pytest.approx(content_row('factorwise')['survival_nll_all'])


def test_parity_leave_one_out_overcounts_and_target_conditional_tc_penalizes_synergy():
    law = four_factor_law('factorwise')
    slots = ['v0', 'v1']
    whole = law.mi(['parity'], slots, ['g'])
    leave = sum(law.mi(['parity'], [v], ['g', *[u for u in slots if u != v]]) for v in slots)
    assert whole == pytest.approx(math.log(2)) and leave == pytest.approx(2*whole)
    assert law.tc(slots, ['g']) == pytest.approx(0, abs=1e-12)
    assert law.tc(slots, ['g', 'parity']) == pytest.approx(math.log(2))


def test_noisy_common_alone_makes_shared_correction_useful():
    null, corrected = noisy_shared_law(), noisy_shared_law(True)
    assert null.entropy(['s'], ['g']) > 0
    assert corrected.entropy(['s'], ['g', 'v']) == pytest.approx(0, abs=1e-12)
    assert corrected.mi(['g'], ['v']) == pytest.approx(0, abs=1e-12)


def test_probability_validation_rejects_bad_law():
    with pytest.raises(ValueError): FiniteLaw([(.8, {'a': 0})])
    with pytest.raises(ValueError): FiniteLaw([(-.1, {'a': 0}), (1., {'a': 1})])


def test_survival_population_control_uses_distribution_mixture_not_mean_hazards():
    components = [(.3, (.1, .2, .3, .4)), (.7, (.6, .7, .8, .9))]
    law = FiniteLaw([(p, {'g': 0, 'hazards': h}) for p, h in components])
    event = [0.]*4; survival = [0.]*4
    for weight, hazards in components:
        remaining = 1.
        for t, h in enumerate(hazards):
            event[t] += weight*remaining*h
            remaining *= 1-h; survival[t] += weight*remaining
    censor = [.1, .1, .1, .7]
    total_observed = sum(event[t]*sum(censor[t:])+censor[t]*survival[t] for t in range(4))
    assert total_observed == pytest.approx(1.)
    correct = -sum(event[t]*sum(censor[t:])*math.log(event[t])+censor[t]*survival[t]*math.log(survival[t]) for t in range(4))
    assert bayes_survival_nll(law, ['g']) == pytest.approx(correct)
    naive_h = [sum(p*h[t] for p, h in components) for t in range(4)]
    remaining = 1.; naive = 0.
    for t, h in enumerate(naive_h):
        f = remaining*h; remaining *= 1-h
        naive -= event[t]*sum(censor[t:])*math.log(f)+censor[t]*survival[t]*math.log(remaining)
    assert naive > correct+.001
