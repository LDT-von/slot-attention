"""Accept the locked24-run result and clarified targets; select no new model."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original_path = ROOT/'local_evidence/private_rate_multifactor_20261102/receipt.json'
    effect_path = original_path.parent/'effect_probe_receipt.json'
    original = json.loads(original_path.read_text(encoding='utf-8'))
    effect = json.loads(effect_path.read_text(encoding='utf-8'))
    assert original['status'] == effect['status'] == 'completed'
    assert len(original['rows']) == len(effect['rows']) == 24
    assert effect['original_receipt_sha256'] == sha(original_path)
    assert effect['clarification_sha256'] == sha(DOC/'private_rate_target_clarification_20261102.json')
    for source in [original['source_sha256_before_execution'], effect['source_sha256']]:
        for p, h in source.items(): assert sha(ROOT/p) == h, p
    assert original['plan_sha256'] == sha(DOC/'private_rate_multifactor_plan_20261102.json')
    effect_rows = {(r['task'], r['seed'], r['arm']): r for r in effect['rows']}
    for row in original['rows']:
        assert row['status'] == 'completed'
        path = (ROOT/row['synthetic_cache_path']).resolve()
        assert path.is_relative_to(ROOT/'local_evidence/private_rate_multifactor_20261102/synthetic_cache')
        assert sha(path) == row['synthetic_cache_sha256']
    summary = []; comparisons = []
    for task in original['plan']['tasks']:
        for seed in original['plan']['seeds']:
            group = [r for r in original['rows'] if (r['task'], r['seed']) == (task, seed)]
            assert len(group) == 4
            assert len({r['initial_hash'] for r in group}) == 1
            assert all(r['input_label_hashes'] == group[0]['input_label_hashes'] for r in group)
        for arm in original['plan']['arms']:
            rows = [r for r in original['rows'] if (r['task'], r['arm']['name']) == (task, arm['name'])]
            result = {'task': task, 'arm': arm['name'], 'nll': statistics.mean(r['expected_single_draw_nll'] for r in rows),
                      'nll_by_seed': [r['expected_single_draw_nll'] for r in rows],
                      'synthetic_bin_concordance': statistics.mean(r['observed_bin_concordance'] for r in rows),
                      'private_rate_nats_per_slot_mean': statistics.mean(sum(sum(a) for a in r['test_private_rate_per_slot'])/4 for r in rows),
                      'shared_directional_rate_mean': statistics.mean(r['test_shared_rate_mean'] for r in rows),
                      'train_any_raw_threshold_exceedance_fraction': statistics.mean(r['train_any_budget_active_fraction'] for r in rows),
                      'train_raw_active_slot_fraction': statistics.mean(r['train_mean_active_slot_fraction'] for r in rows),
                      'gradient_l2_max': max(r['gradient_l2_max'] for r in rows),
                      'mean_private_donor_nll_delta': statistics.mean(r['fixed_readout_nll_deltas']['private_donor'] for r in rows),
                      'mean_private_N0_nll_delta': statistics.mean(r['fixed_readout_nll_deltas']['private_N0'] for r in rows)}
            for mode in ['sample', 'all_mean']:
                diagnostics = [r['diagnostics'][mode] for r in rows]
                clarified = [effect_rows[(task, r['seed'], arm['name'])]['effects'][mode] for r in rows]
                result[mode] = {
                    'factor_increment_over_G': statistics.mean(d['all_vs_single_factor_probe']['all_slots']['gain_relative_to_fit_mean_baseline'] for d in diagnostics),
                    'factor_all_vs_single_gain': statistics.mean(d['all_vs_single_factor_probe']['gain_relative_to_fit_mean_baseline'] for d in diagnostics),
                    'factor_all_vs_single_by_seed': [d['all_vs_single_factor_probe']['gain_relative_to_fit_mean_baseline'] for d in diagnostics],
                    'effect_increment_over_G': statistics.mean(d['all_slots']['gain_relative_to_fit_mean_baseline'] for d in clarified),
                    'effect_all_vs_single_gain': statistics.mean(d['gain_relative_to_fit_mean_baseline'] for d in clarified),
                    'raw_private_factor_increment': statistics.mean(d['raw_private_factor_increment']['gain_relative_to_fit_mean_baseline'] for d in diagnostics),
                    'G_shared_linear_r2': statistics.mean(d['G_shared_linear']['test_r2'] for d in diagnostics),
                    'G_factor_linear_r2': statistics.mean(d['G_private_factor_linear']['test_r2'] for d in diagnostics),
                    'private_nuisance_linear_r2': statistics.mean(d['private_nuisance_linear']['test_r2'] for d in diagnostics)}
            result['private_sample_rank'] = [statistics.mean(r['private_sample_geometry'][j]['uncentered_effective_rank'] for r in rows) for j in range(2)]
            result['private_all_mean_rank'] = [statistics.mean(r['private_all_mean_geometry'][j]['uncentered_effective_rank'] for r in rows) for j in range(2)]
            summary.append(result)
        for reference in ['no_private_fee', 'flat']:
            for candidate in ['leaky', 'continuous']:
                differences = []
                for seed in original['plan']['seeds']:
                    left = next(r for r in original['rows'] if (r['task'], r['seed'], r['arm']['name']) == (task, seed, reference))
                    right = next(r for r in original['rows'] if (r['task'], r['seed'], r['arm']['name']) == (task, seed, candidate))
                    differences.append(right['expected_single_draw_nll']-left['expected_single_draw_nll'])
                comparisons.append({'task': task, 'candidate': candidate, 'reference': reference,
                                    'paired_nll_differences': differences, 'mean': statistics.mean(differences)})
    result = {'schema': 'slotspe_private_rate_multifactor_summary_v1', 'status': 'completed',
              'source_receipt_sha256': sha(original_path), 'effect_receipt_sha256': sha(effect_path),
              'summary': summary, 'paired_nll_comparisons': comparisons,
              'new_neural_trainings': 24, 'core_updates': 12000, 'paired_task_seed_groups': 6,
              'unique_initial_state_hashes': len({r['initial_hash'] for r in original['rows']}),
              'parameters': sorted({r['parameter_count'] for r in original['rows']}),
              'training_seconds_excluding_probes': sum(r['training_seconds_excluding_probes'] for r in original['rows']),
              'all_training_steps_finite': all(r['finite_training_steps'] == 500 for r in original['rows']),
              'factor_instrument_gains': original['instrument_controls']['relative_gains'],
              'effect_instrument_gains': effect['instrument']['gains'],
              'cache_hashes_verified': 24, 'source_pins_verified': True,
              'decision': 'Do not promote the leaky or continuous private fee. No-fee preserves more factor content and has better factorwise NLL; stronger fees shrink actual sampled content and do not establish XOR cooperation. Retain lower raw rate as a trade-off, not success. No further eta scan.',
              'limits': ['Three paired composite data/training seeds; exploratory, no significance or noninferiority claim.',
                         'Observed-bin concordance is synthetic and may move differently from proper NLL.',
                         'Individual-factor recovery is coverage; XOR task effect relations are separately assessed after a disclosed protocol clarification.',
                         'Mean-mode probes do not establish information in actual stochastic readout.',
                         'No measured MI, biological slot identity, clinical adapter or patient benefit.'],
              'source_sha256': {'scripts/summarize_private_rate_multifactor.py': sha(Path(__file__))},
              'patient_execution': False, 'server_execution': False}
    for name, payload in [('private_rate_multifactor_neural_receipt.json', original),
                          ('private_rate_effect_receipt.json', effect), ('private_rate_multifactor_stage_summary.json', result)]:
        path = DOC/name
        if path.exists(): raise RuntimeError('preserve old public receipt')
        path.write_bytes((json.dumps(payload, indent=2)+'\n').encode())
    print(json.dumps({'new_neural_trainings':24,'updates':12000,'initial_groups':6,'unique_initials':result['unique_initial_state_hashes'],
                      'summary': [{k:r[k] for k in ['task','arm','nll','synthetic_bin_concordance','private_rate_nats_per_slot_mean','sample','all_mean']} for r in summary]}, indent=2))


if __name__ == '__main__': main()
