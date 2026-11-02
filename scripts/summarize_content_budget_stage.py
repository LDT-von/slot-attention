"""Recompute aggregate synthetic results; never open tensor caches or patient data."""
import hashlib
import itertools
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs/research/slotspe_common_private_20261101'


def avg(xs):
    return mean(xs)


def main():
    output = DOC / 'content_budget_stage_summary.json'
    if output.exists():
        raise RuntimeError('preserve existing aggregate receipt')
    files = ['content_budget_pilot_receipt.json', 'role_readout_comparison_receipt.json']
    pilot, readout = [json.loads((DOC / f).read_text(encoding='utf-8')) for f in files]
    assert pilot['status'] == readout['status'] == 'completed'
    expected_pilot = list(itertools.product(pilot['plan']['tasks'], pilot['plan']['seeds'], pilot['plan']['arms']))
    assert [(r['task'], r['seed'], [r['anchor_on'], r['budget_on']]) for r in pilot['rows']] == expected_pilot
    expected_readout = list(itertools.product(readout['plan']['tasks'], readout['plan']['seeds'],
                                             readout['plan']['readouts'], readout['plan']['constraints']))
    assert [(r['task'], r['seed'], r['readout'], r['content_budget_on']) for r in readout['rows']] == expected_readout
    for data, hash_key in [(pilot, 'initial_hash'), (readout, 'shared_core_initial_hash')]:
        for task, seed in itertools.product(data['plan']['tasks'], data['plan']['seeds']):
            rows = [r for r in data['rows'] if (r['task'], r['seed']) == (task, seed)]
            assert len({r[hash_key] for r in rows}) == 1
    for row in readout['rows']:
        path = ROOT / row['synthetic_cache_path']
        assert path.is_relative_to(ROOT / 'local_evidence')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['synthetic_cache_sha256']
    psummary = []
    for task, (a, b) in itertools.product(pilot['plan']['tasks'], pilot['plan']['arms']):
        rows = [r for r in pilot['rows'] if (r['task'], r['anchor_on'], r['budget_on']) == (task, a, b)]
        psummary.append({'task': task, 'anchor': a, 'budget': b,
            'nll': avg(r['test_expected_nll'] for r in rows),
            'content_ce': avg(avg(r['test_expected_content_ce_per_slot']) for r in rows),
            'rate_nats': avg(avg(r['test_rate_nats_per_slot']) for r in rows),
            'own_r2': avg(r['linear_probe_own_bit_r2_mean'] for r in rows),
            'other_r2': avg(r['linear_probe_other_bits_r2_mean'] for r in rows),
            'whole_vs_single_risk_gain': avg(r['whole_vs_single_risk_probe']['gain_relative_to_fit_mean_baseline'] for r in rows)})
    rsummary = []
    for task, read, on in itertools.product(readout['plan']['tasks'], readout['plan']['readouts'], readout['plan']['constraints']):
        rows = [r for r in readout['rows'] if (r['task'], r['readout'], r['content_budget_on']) == (task, read, on)]
        rsummary.append({'task': task, 'readout': read, 'constraints': on,
            'nll': avg(r['test_expected_nll'] for r in rows),
            'content_ce': avg(avg(r['expected_content_ce_per_slot']) for r in rows),
            'rate_nats': avg(avg(r['rate_nats_per_slot']) for r in rows),
            'own_r2': avg(r['own_bit_linear_r2'] for r in rows),
            'other_r2': avg(r['other_bits_linear_r2'] for r in rows),
            'pooled_vs_common_risk_gain': avg(r['risk_probe']['gain_relative_to_fit_mean_baseline'] for r in rows),
            'parameters': rows[0]['parameter_count'], 'phi_parameters': rows[0]['phi_parameter_count']})
    deltas = []
    for task, on in itertools.product(readout['plan']['tasks'], readout['plan']['constraints']):
        ds = []
        for seed in readout['plan']['seeds']:
            rows = {r['readout']: r for r in readout['rows'] if (r['task'], r['seed'], r['content_budget_on']) == (task, seed, on)}
            assert abs(rows['role_concat']['phi_parameter_count']/rows['mean_pool']['phi_parameter_count']-1) < .05
            ds.append(rows['role_concat']['test_expected_nll']-rows['mean_pool']['test_expected_nll'])
        half = 4.30265273*stdev(ds)/(3**.5)
        deltas.append({'task': task, 'constraints': on, 'concat_minus_mean_nll_by_seed': ds,
                       'mean': avg(ds), 'descriptive_t95': [avg(ds)-half, avg(ds)+half],
                       'caution': 'three paired exploratory seeds, no multiple-comparison correction; not clinical inference'})
    report = {'schema': 'slotspe_synthetic_content_budget_stage_v1',
        'source_sha256': {f: hashlib.sha256((DOC/f).read_bytes()).hexdigest() for f in files},
        'pilot_training_runs': len(pilot['rows']), 'readout_training_runs': len(readout['rows']),
        'steps_per_run': 500, 'new_core_updates': 36000, 'same_initialization_groups': 18,
        'verified_synthetic_cache_hashes': len(readout['rows']),
        'pilot': psummary, 'readout': rsummary, 'paired_readout_deltas': deltas,
        'patient_training': 0, 'patient_inference': 0, 'patient_checkpoints_read': 0,
        'server_execution': False, 'scope': 'optimistic fixed synthetic role metadata, three exploratory tasks; no clinical target adapter',
        'metric_distinction': 'pilot whole-versus-selected-single is not readout pooled-versus-common; do not compare them as the same diagnostic'}
    output.write_bytes((json.dumps(report, indent=2)+'\n').encode())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
