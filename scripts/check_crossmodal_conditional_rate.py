"""Gaussian conditional-information oracle and shared-readout counterexamples."""
import hashlib
import itertools
import json
import math
from pathlib import Path


def main():
    doc = Path('docs/research/slotspe_common_private_20261101')
    output = doc/'crossmodal_conditional_rate_math_receipt.json'
    if output.exists(): raise RuntimeError('preserve existing receipt')
    rows = []
    # Xw=(S,P), Xo=(S,Q), corr(P,Q)=rho, Cw=aS+bP+tau*N.
    # Reference is q(Co|Xo)=N(aS+bQ,tau^2), not q(Cw|Xw,Xo).
    for b, tau, rho in itertools.product([0., .2, 1., 3.], [.5, 1., 2.], [0., .5, .9]):
        rate = b*b*(1-rho)/(tau*tau)
        information = .5*math.log1p(b*b*(1-rho*rho)/(tau*tau))
        conditional_variance = b*b*(1-rho*rho)+tau*tau
        mismatch = .5*(math.log(tau*tau/conditional_variance)
            +(conditional_variance+b*b*(rho-1)**2)/(tau*tau)-1)
        assert mismatch >= -1e-12 and abs(rate-information-mismatch) < 1e-10
        rows.append({'private_coefficient': b, 'noise_sd': tau, 'private_correlation': rho,
                     'expected_cross_kl_nats': rate, 'true_conditional_mi_nats': information,
                     'reference_mismatch_nats': mismatch})
    bypass = []
    for eps in [1., .1, .01, .001]:
        # Fair bit B, Cw=eps*B+Nw, Co=No. This is distinct from Gaussian MI oracle above.
        rate = .5*eps*eps
        for draws in [1, 5, 20]:
            # Mean of draws of the actual G=(Cw+Co)/2, Bayes sign-error.
            error = .5*math.erfc(eps*math.sqrt(draws)/2)
            bypass.append({'epsilon': eps, 'draws': draws, 'single_direction_rate_nats': rate,
                           'joint_draw_rate_upper_bound_nats': draws*rate,
                           'actual_sample_mean_bayes_bit_error': error,
                           'noiseless_posterior_mean_bit_error': 0.})
    examples = [
        {'case': 'useful_shared', 'posterior_means': 'a*S in both views', 'cross_kl': 0.,
         'conclusion': 'matched shared signal is allowed; cross KL does not certify usefulness'},
        {'case': 'constant_collapse', 'posterior_means': 'zero in both views', 'cross_kl': 0.,
         'conclusion': 'content/task preservation must be independently verified'},
        {'case': 'shared_nuisance', 'posterior_means': 'a*H in both views, H independent of outcome', 'cross_kl': 0.,
         'conclusion': 'cross predictability cannot distinguish useful common content from common nuisance'},
        {'case': 'wrong_reference_sees_own_input', 'posterior_means': 'b*P; reference also sees P', 'cross_kl': 0.,
         'true_conditional_mi_if_b_tau_one': .5*math.log(2),
         'conclusion': 'the other-only reference restriction is essential'},
        {'case': 'xor_private_synergy', 'I_Y_P': 0., 'I_Y_Q': 0., 'I_Y_PQ': math.log(2),
         'conclusion': 'joint survival utility must remain; do not require each private slot to predict Y alone'},
    ]
    payload = {'status': 'completed', 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'gaussian_oracle_cases': len(rows), 'all_identities_passed': True, 'rows': rows,
        'sampling_bypass_and_repeat_controls': bypass, 'required_degeneracy_controls': examples,
        'neural_training': 0, 'patient_execution': False,
        'scope': 'population identities under explicit generative assumptions; finite training KL is not a measured patient MI certificate',
        'repeat_boundary': 'one-draw information bound does not transfer unchanged to multi-draw averaged prediction or deterministic posterior predictive expectation'}
    output.write_bytes((json.dumps(payload, indent=2)+'\n').encode())
    print(json.dumps({'gaussian_oracle_cases': len(rows), 'all_identities_passed': True,
                      'sampling_repeat_controls': len(bypass), 'degeneracy_controls': len(examples)}))


if __name__ == '__main__': main()
