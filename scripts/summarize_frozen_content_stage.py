"""Aggregate synthetic-only target/utility evidence with integrity checks."""
import hashlib
import itertools
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def main():
    output = DOC/'frozen_target_stage_summary.json'
    if output.exists(): raise RuntimeError('preserve existing summary')
    file = DOC/'frozen_target_neural_receipt.json'
    payload = json.loads(file.read_text()); plan = payload['plan']
    assert payload['status'] == 'completed'
    assert [(r['condition'], r['seed'], r['arm']) for r in payload['rows']] == list(itertools.product(plan['conditions'], plan['seeds'], plan['arms']))
    for path, digest in payload['source_sha256_before_execution'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    for row in payload['rows']:
        path = (ROOT/row['synthetic_cache_path']).resolve()
        assert path.is_relative_to(ROOT/'local_evidence/synthetic_frozen_targets_20261102')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['synthetic_cache_sha256']
    for condition, seed in itertools.product(plan['conditions'], plan['seeds']):
        rows = [r for r in payload['rows'] if (r['condition'], r['seed']) == (condition, seed)]
        assert len({r['initial_hash'] for r in rows}) == 1
        assert len({tuple(r['teacher_hashes']) for r in rows}) == 1
        assert len({r['parameter_count'] for r in rows}) == 1
    summaries = []
    for condition, arm in itertools.product(plan['conditions'], plan['arms']):
        rows = [r for r in payload['rows'] if (r['condition'], r['arm']) == (condition, arm)]
        metrics = ['test_expected_nll', 'test_deterministic_nll', 'own_true_bit_linear_r2',
                   'other_true_bits_linear_r2', 'common_true_bits_linear_r2',
                   'test_expected_oracle_content_ce', 'test_expected_frozen_target_ce', 'test_expected_arm_target_ce',
                   'gradient_norm_mean', 'training_seconds']
        summaries.append({'condition': condition, 'arm': arm, **{m: mean(r[m] for r in rows) for m in metrics},
            'rate_nats_per_slot_mean': mean(mean(r['rate_nats_per_slot']) for r in rows),
            'gradient_norm_max': max(r['gradient_norm_max'] for r in rows),
            'pooled_vs_common_risk_probe_gain': mean(r['pooled_risk_increment_probe']['gain_relative_to_fit_mean_baseline'] for r in rows),
            'common_risk_probe_test_mse': mean(r['pooled_risk_increment_probe']['common']['test_mse'] for r in rows)})
    paired = []
    for condition, reference in itertools.product(plan['conditions'], [a for a in plan['arms'] if a != 'frozen_content']):
        deltas = []
        for seed in plan['seeds']:
            rows = {r['arm']: r for r in payload['rows'] if (r['condition'], r['seed']) == (condition, seed)}
            deltas.append(rows['frozen_content']['test_expected_nll']-rows[reference]['test_expected_nll'])
        half = 4.30265273*stdev(deltas)/(3**.5)
        paired.append({'condition': condition, 'contrast': 'frozen_content_minus_'+reference,
            'nll_deltas': deltas, 'mean': mean(deltas), 'descriptive_t95': [mean(deltas)-half, mean(deltas)+half]})
    summary = {'schema': 'slotspe_frozen_content_stage_summary_v1',
        'source_receipt_sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
        'new_neural_trainings': 30, 'steps_per_run': 500, 'core_updates': 15000,
        'prototype_teacher_fits_neural_setup': 12, 'prototype_teacher_fits_gate': 12,
        'paired_initialization_groups': 6, 'verified_cache_hashes': 30,
        'summaries': summaries, 'paired_nll_deltas': paired,
        'training_seconds_total': sum(r['training_seconds'] for r in payload['rows']),
        'decision': 'Do not promote unconditioned frozen PCA targets to the primary model. Retain as diagnostic; audit shared-path semantic leakage before further private objectives.',
        'limits': '3 paired data/training seeds per condition, exploratory, no multiplicity correction or clinical inference; negative finite probes are not absence proofs. Source hashes match pre-execution pins.',
        'patient_execution': False, 'server_execution': False}
    output.write_bytes((json.dumps(summary, indent=2)+'\n').encode())
    print(json.dumps({'new_neural_trainings': 30, 'source_pins_and_cache_hashes': 'passed', 'paired_nll_deltas': paired}, indent=2))


if __name__ == '__main__': main()
