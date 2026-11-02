"""Analytic re-encoding counterexample; no neural or patient execution."""
import hashlib
import itertools
import json
import math
from pathlib import Path


def main():
    doc = Path('docs/research/slotspe_common_private_20261101')
    output = doc/'shared_path_leakage_receipt.json'
    if output.exists(): raise RuntimeError('preserve existing receipt')
    rows = []
    for eps in [1., .1, .01, .001]:
        alignment = []; error = []; normal_fixed_noise = []; normal_scaled_noise = []
        for p, q in itertools.product([-1., 1.], repeat=2):
            cw, co = [0., eps*p, 0.], [0., 0., eps*q]
            g = [(a+b)/2 for a, b in zip(cw, co)]
            prediction = 2/eps*g[1]+4/eps*g[2]
            alignment.append(sum((a-b)**2 for a, b in zip(cw, co)))
            error.append((prediction-(p+2*q))**2)
            normal_fixed_noise.append(.5*(eps*p)**2)
            normal_scaled_noise.append(.5*(eps*p)**2/eps**2)
        mean = lambda x: sum(x)/len(x)
        assert mean(error) < 1e-20
        assert abs(mean(alignment)-2*eps**2) < 1e-12
        assert abs(mean(normal_scaled_noise)-.5) < 1e-12
        rows.append({'epsilon': eps, 'deterministic_shared_alignment_mse': mean(alignment),
            'task_reconstruction_mse': mean(error), 'shared_head_largest_absolute_weight': 4/eps,
            'analytic_gaussian_conditional_rate_fixed_noise': mean(normal_fixed_noise),
            'analytic_gaussian_conditional_rate_co_scaled_noise': mean(normal_scaled_noise)})
    payload = {'status': 'completed', 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_route': 'CommonPrivateSurvival.forward averages modality summaries; alignment MSE and no bound on readout weights',
        'rows': rows, 'checks_passed': True, 'neural_training': 0, 'patient_execution': False,
        'source_fact_vs_construction': 'This is an allowed representation/readout construction, not evidence that any trained model learned it.',
        'shared_candidate': {'variables': 'q(Cw|Xw), q(Co|Xo); each posterior must depend only on its own modality',
            'rate': 'E KL(q(Cw|Xw)||r(Cw|Xo)) and reverse',
            'identity': 'R_w = I(Cw;Xw|Xo) + E_Xo KL(q(Cw|Xo)||r(Cw|Xo)) >= I(Cw;Xw|Xo)',
            'assumptions': 'fixed population/network; Markov Cw-Xw-Xo; normalized densities and finite terms',
            'readout_requirement': 'Use actual noisy C samples in survival; a deterministic posterior mean bypass is not covered by this bound.',
            'collapse': 'q=r=constant gives zero rate. Task/cross-view information preservation and independent content controls remain necessary.',
            'scope': 'standard variational conditional-information identity, not a new theorem or guarantee of task-relevant common factors'},
        'gaussian_scope': 'The rate examples use an ideal independent unit-Gaussian private factor, not the fair-bit task construction; they illustrate noise/SNR scaling. No actual MI was estimated. Co-scaled variance is analytic and unconstrained, not an assertion about bounded log-variance code.'}
    output.write_bytes((json.dumps(payload, indent=2)+'\n').encode())
    print(json.dumps(payload, indent=2))


if __name__ == '__main__': main()
