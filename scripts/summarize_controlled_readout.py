"""Audit the locked factorial receipt without retraining or new model choice."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT/'local_evidence/controlled_slot_readout_20261102/receipt.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    assert data['status'] == 'completed' and len(data['rows']) == 48
    assert data['verified_old_repeats'] == 6
    for path, digest in data['source_sha256_before_execution'].items():
        assert sha(ROOT/path) == digest, path
    assert sha(DOC/'controlled_readout_plan_20261102.json') == data['plan_sha256']
    assert sha(DOC/'slot_pooling_rate_neural_receipt.json') == data['old_receipt_sha256']
    rows = data['rows']; seeds = data['plan']['seeds']
    def select(condition, shape, role, scale):
        subset = [r for r in rows if (r['condition'], r['phi_shape'], r['role_access'], r['scale']) == (condition, shape, role, scale)]
        assert [r['seed'] for r in subset] == seeds
        return subset
    summary = []; contrasts = []
    for condition in data['plan']['conditions']:
        for shape in data['plan']['phi_shapes']:
            for seed in seeds:
                hashes = {r['initial_state_hash'] for r in rows if (r['condition'], r['phi_shape'], r['seed']) == (condition, shape, seed)}
                assert len(hashes) == 1
            for scale in data['plan']['scales']:
                selected = {}
                for role in data['plan']['role_access']:
                    subset = select(condition, shape, role, scale); selected[role] = subset
                    summary.append({'condition': condition, 'phi_shape': shape, 'role_access': role, 'scale': scale,
                                    'mean_nll': sum(r['expected_nll'] for r in subset)/3,
                                    'nll_by_seed': [r['expected_nll'] for r in subset],
                                    'max_gradient_l2': max(r['max_gradient_l2'] for r in subset),
                                    'max_parameter_abs': max(r['max_parameter_abs'] for r in subset),
                                    'mean_donor_nll_delta': sum(r['fixed_head_nll_deltas']['private_donor'] for r in subset)/3})
                differences = [s['expected_nll']-b['expected_nll'] for s, b in zip(selected['signed'], selected['blind'])]
                mean = sum(differences)/3
                contrasts.append({'condition': condition, 'phi_shape': shape, 'scale': scale,
                                  'signed_minus_blind_by_seed': differences, 'mean': mean,
                                  'all_seed_deltas_negative': all(d < 0 for d in differences),
                                  'mean_gain_at_least_03': mean <= -.03,
                                  'noise_difference_below_02': abs(mean) < .02})
    signal = [r for r in contrasts if r['condition'] == 'single_total1' and r['phi_shape'] == 'full']
    noise = [r for r in contrasts if r['condition'] == 'noise_only' and r['phi_shape'] == 'full']
    passed = all(r['all_seed_deltas_negative'] and r['mean_gain_at_least_03'] for r in signal) and all(r['noise_difference_below_02'] for r in noise)
    interaction = {shape: next(r['mean'] for r in contrasts if (r['condition'], r['phi_shape'], r['scale']) == ('single_total1', shape, 'sum'))-next(r['mean'] for r in contrasts if (r['condition'], r['phi_shape'], r['scale']) == ('single_total1', shape, 'mean')) for shape in data['plan']['phi_shapes']}
    result = {'schema': 'slotspe_controlled_readout_summary_v1', 'status': 'completed',
              'source_receipt_sha256': sha(source), 'summary': summary, 'contrasts': contrasts,
              'role_by_scale_interaction_sum_minus_mean': interaction,
              'primary_gate_passed': passed,
              'decision': 'Do not promote signed role pooling or select a best secondary shape. Retain scale-dependent role access as a diagnostic; no extra sweep.',
              'optimizer_executions': 48, 'old_deterministic_repeats': 6, 'new_arm_executions': 42,
              'steps_each': 500, 'optimizer_updates': 24000,
              'exact_initialization_groups': 12, 'training_seconds_total': sum(r['training_seconds'] for r in rows),
              'source_pins_reverified': True, 'all_optimization_finite': all(r['finite_optimization'] for r in rows),
              'source_sha256': {'scripts/summarize_controlled_readout.py': sha(Path(__file__))},
              'patient_execution': False, 'server_execution': False,
              'limits': ['Same exploratory data, privileged stable slot roles, one informative bit per modality.',
                         'The .03/.02 thresholds are engineering gates, not significance or a clinical margin.',
                         'Within-phi contrasts isolate the forward control at matched initialization; post-training gradients and trajectories necessarily differ.',
                         'Full vs block differs by six parameters and phi initialization; no best-of-shape selection.',
                         'Failure does not prove absent information or lack of expressive capacity.']}
    for name, payload in [('controlled_readout_neural_receipt.json', data), ('controlled_readout_stage_summary.json', result)]:
        path = DOC/name
        if path.exists(): raise RuntimeError('preserve old receipts')
        path.write_bytes((json.dumps(payload, indent=2)+'\n').encode('utf-8'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
