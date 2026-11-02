"""Read-only aggregation/integrity checks of the locked synthetic rate matrix."""
import hashlib
import itertools
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def main():
    output = DOC/'crossmodal_conditional_rate_stage_summary.json'
    if output.exists(): raise RuntimeError('preserve existing aggregate')
    file = DOC/'crossmodal_conditional_rate_neural_receipt.json'
    data = json.loads(file.read_text()); plan = data['plan']; rows = data['rows']
    assert data['status'] == 'completed'
    expected = list(itertools.product(plan['tasks'], plan['seeds'], plan['shared_weight'], plan['private_weight']))
    assert [(r['task'], r['seed'], r['shared_weight'], r['private_weight']) for r in rows] == expected
    for path, digest in data['source_sha256_before_execution'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    for row in rows:
        path = (ROOT/row['synthetic_cache_path']).resolve()
        assert path.is_relative_to(ROOT/'local_evidence/synthetic_crossmodal_rate_20261102')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['synthetic_cache_sha256']
    for task, seed in itertools.product(plan['tasks'], plan['seeds']):
        group = [r for r in rows if (r['task'], r['seed']) == (task, seed)]
        assert len({r['initial_hash'] for r in group}) == len({r['parameter_count'] for r in group}) == 1
    summary = []
    for task, sw, pw in itertools.product(plan['tasks'], plan['shared_weight'], plan['private_weight']):
        group = [r for r in rows if (r['task'], r['shared_weight'], r['private_weight']) == (task, sw, pw)]
        avg = lambda f: mean(f(r) for r in group)
        summary.append({'task': task, 'shared_weight': sw, 'private_weight': pw,
            'nll_mean': avg(lambda r: r['test_expected_nll_single_draw_average']),
            'nll_sd': stdev(r['test_expected_nll_single_draw_average'] for r in group),
            'posterior_means_nll_diagnostic': avg(lambda r: r['test_posterior_means_nll_diagnostic']),
            'shared_rate_nats': avg(lambda r: r['test_shared_directional_rate_mean_nats']),
            'private_rate_nats': avg(lambda r: mean(r['test_private_rate_nats_per_slot'])),
            'shared_factor_r2_sample': avg(lambda r: r['shared_factor_linear_r2']['common']),
            'shared_factor_r2_mean': avg(lambda r: r['shared_factor_linear_r2']['common_mean']),
            'planted_wsi_bit_r2_shared_sample': avg(lambda r: r['private_factor_from_shared_linear_r2']['common'][0]),
            'planted_wsi_bit_r2_shared_mean': avg(lambda r: r['private_factor_from_shared_linear_r2']['common_mean'][0]),
            'pooled_private_risk_increment': avg(lambda r: r['actual_pooled_private_risk_increment_probe']['gain_relative_to_fit_mean_baseline']),
            'mean_coordinate_all_vs_single_gain': avg(lambda r: r['mean_coordinate_all_vs_single_probe']['gain_relative_to_fit_mean_baseline']),
            'private_mean_rank_by_modality': [avg(lambda r: r['private_mean_coordinate_geometry'][j]['uncentered_effective_rank']) for j in range(2)],
            'private_sample_rank_by_modality': [avg(lambda r: r['private_sample_geometry'][j]['uncentered_effective_rank']) for j in range(2)],
            'private_mean_cosine_by_modality': [avg(lambda r: r['private_mean_coordinate_geometry'][j]['pair_cosine']) for j in range(2)],
            'gradient_norm_max': max(r['gradient_norm_max'] for r in group), 'parameter_count': group[0]['parameter_count']})
    contrasts = []
    for task, pw in itertools.product(plan['tasks'], plan['private_weight']):
        deltas = []
        for seed in plan['seeds']:
            group = {r['shared_weight']: r for r in rows if (r['task'], r['seed'], r['private_weight']) == (task, seed, pw)}
            deltas.append(group[.1]['test_expected_nll_single_draw_average']-group[0.]['test_expected_nll_single_draw_average'])
        half = 4.30265273*stdev(deltas)/(3**.5)
        contrasts.append({'task': task, 'private_weight': pw, 'shared_KL_on_minus_off_nll': deltas,
                          'mean': mean(deltas), 'descriptive_t95': [mean(deltas)-half, mean(deltas)+half]})
    report = {'schema': 'slotspe_crossmodal_conditional_rate_summary_v1',
        'source_receipt_sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
        'neural_trainings': 48, 'steps_per_run': 500, 'core_updates': 24000,
        'paired_initialization_groups': 12, 'cache_hashes_verified': 48,
        'source_pins_before_execution_reverified': True,
        'training_seconds_excluding_probes': sum(r['training_seconds'] for r in rows),
        'summary': summary, 'paired_shared_KL_contrasts': contrasts,
        'limitations': '3 paired data/training seeds; exploratory, no multiple-comparison correction; planted WSI factor in nuisance-shift task is statistically shared through confounding, and OOD probe failure is not absence of information. Single-draw rates do not bound unlimited mean predictions.',
        'patient_execution': False, 'server_execution': False}
    output.write_bytes((json.dumps(report, indent=2)+'\n').encode())
    print(json.dumps({'summary': summary, 'paired_shared_KL_contrasts': contrasts}, indent=2))


if __name__ == '__main__': main()
