"""Audit the published, pinned 82-epoch aggregate using only the standard library."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_matched35_chain import (
    EXPECTED_SHA256, MATCHED_KEYS, ROLES, SHARED_CHECKPOINT_SHA256,
    _config, _hash, _json_bytes, _number, _require, _row, _scan_aggregate, _sha,
)


def _audit_exported(value):
    _scan_aggregate(value)
    _require(value['schema'] == 'matched35_aggregate_export_v1', 'wrong aggregate schema')
    _require(value['exporter_SHA256'] == EXPECTED_SHA256['original_exporter'], 'wrong exporter provenance')
    pair, shared, arms = (value[k] for k in ('paired_receipt', 'shared_role', 'arms'))
    _config(pair['config'], 'joint')
    _require(pair['status'] == 'completed' and pair['total_epochs'] == pair['expected_total_epochs'] == 82
             and pair['shared_role_epochs'] == 12, 'wrong paired epoch budget')
    for key in ('common_start_verified', 'matched_input_draws_verified', 'task_optimizer_reset_verified'):
        _require(pair[key] is True, 'paired verification missing')
    _require(pair['shared_checkpoint_SHA256'] == SHARED_CHECKPOINT_SHA256, 'wrong shared origin')
    for digest in pair['source_SHA256'].values():
        _sha(digest, 'source provenance')
    _require(shared['status'] == 'completed' and shared['epochs'] == 12
             and len(shared['role_epoch_rows']) == 12, 'wrong shared role stage')
    for i, row in enumerate(shared['role_epoch_rows'], 1):
        _require(type(row['epoch']) is int and row['epoch'] == i, 'shared role epoch order')
        for key in ('train_loss', 'train_role_loss'):
            _number(row[key], key, minimum=0)
        for role in ROLES:
            _number(row['role_target_MSE'][role], 'role MSE', minimum=0)
    curves = {}
    for name in ('joint', 'continued_role_only'):
        arm = arms[name]
        _require(arm['status'] == 'completed' and arm['epochs'] == 35 and arm['early_stop'] is None,
                 'wrong fixed task stage')
        _require(arm['shared_checkpoint_SHA256'] == SHARED_CHECKPOINT_SHA256, 'task origin differs')
        for key in ('reference_frozen', 'role_parameters_changed_during_task', 'task_optimizer_initial_state_empty',
                    'checkpoint_reload_verified', 'checkpoint_optimizer_and_generator_reload_verified'):
            _require(arm[key] is True, 'task verification missing')
        _require(arm['task_gradient_policy'] == ('joint' if name == 'joint' else 'role_only_trainable_head'),
                 'task gradient policy differs')
        rows = arm['epoch_rows']
        _require(len(rows) == 35, 'incomplete task curve')
        for i, row in enumerate(rows, 1):
            # These are the exact exported fields. Sampling seed and absolute
            # deletion logit change were omitted by the historical exporter.
            validation = {k: row[k] for k in ('C_index', 'NLL', 'role_target_MSE', 'raw_reading_mass',
                                              'read_diagnostics', 'ablation_NLL', 'ablation_delta_NLL')}
            _row({**row, 'validation': validation}, 'task', i, name, exported=True)
            for role in ROLES:
                r, s = row['roles'][role], row['slots'][role]
                _require(r['patients'] == s['patients'] == 76 and s['slots'] == 2 and s['width'] == 64,
                         'role or slot summary axis differs')
                for field in ('mean_norm_by_slot', 'zero_fraction_by_slot', 'across_patient_std_rms_by_slot',
                              'centered_patient_effective_rank_by_slot'):
                    _require(len(s[field]) == 2, 'raw slot summary vector axis differs')
                rank_bound = 2 if role.endswith('common') else 6
                _require(0 <= r['centered_patient_effective_rank'] <= rank_bound + 1e-10,
                         'decoded role rank exceeds its architectural bound')
        best = max(rows, key=lambda r: (r['C_index'], -r['NLL'], -r['epoch']))
        for label, row in (('best', best), ('last', rows[-1])):
            expected = {k: row[k] for k in ('epoch', 'C_index', 'NLL')}
            _require(arm[label] == expected == pair['arms'][name][label], 'selection disagrees with full curve')
        _require(pair['arms'][name]['receipt_SHA256'] == EXPECTED_SHA256[name + '/receipt.json'],
                 'paired receipt reference differs')
        curves[name] = rows
    left, right = arms['joint'], arms['continued_role_only']
    for key in MATCHED_KEYS:
        if key in ('split_identity', 'source_SHA256'):
            continue  # These are not exported per arm by the original script.
        _require(left[key] == right[key], 'endpoint start differs: ' + key)
    _require([r['input_draw_SHA256'] for r in curves['joint']] ==
             [r['input_draw_SHA256'] for r in curves['continued_role_only']], 'input draws differ')
    differences = [b['C_index'] - a['C_index'] for a, b in zip(curves['joint'], curves['continued_role_only'])]
    summary = {}
    for name, rows in curves.items():
        counts = [r['raw_reading_mass']['p_common']['below_threshold_count'] for r in rows]
        summary[name] = dict(best=arms[name]['best'], last=arms[name]['last'],
            p_common_low_counts_by_epoch=counts, p_common_early_count=sum(counts[:26]),
            p_common_late_count=sum(counts[26:]), p_common_total_fraction=sum(counts)/(35*456),
            epsilon_uniform_fraction_last=rows[-1]['read_diagnostics']['p_common']['metrics']['epsilon_uniform_fraction']['mean'],
            role_MSE_trajectory_means={r: statistics.mean(x['role_target_MSE'][r] for x in rows) for r in ROLES},
            deleted_delta_NLL_ranges={r: [[min(x['ablation_delta_NLL'][r][i] for x in rows),
                                          max(x['ablation_delta_NLL'][r][i] for x in rows)] for i in (0, 1)] for r in ROLES})
    return dict(schema='matched35_full_aggregate_audit_v1', status='verified', total_epochs=82,
        aggregate_numeric_crosscheck_complete=True, model_executed=False, patient_rows_exported=False,
        paired_C_index_mean=statistics.mean(differences), paired_C_index_SD_ddof1=statistics.stdev(differences),
        control_C_index_higher_epochs=sum(x > 0 for x in differences),
        best_difference=right['best']['C_index']-left['best']['C_index'],
        last_difference=right['last']['C_index']-left['last']['C_index'], arms=summary,
        limitations=['Published aggregates audited; raw receipt bytes remain private on server.',
                     'Original exporter omits validation sampling seed and absolute deletion logit change.',
                     'Original exporter copies nested diagnostics without a general privacy allowlist; use only pinned inputs.',
                     'These are correlated development trajectories, not independent test estimates.'])


def audit_aggregates(path):
    raw = Path(path).read_bytes()
    _require(_hash(raw) == EXPECTED_SHA256['aggregates'], 'published aggregate SHA256 mismatch')
    result = _audit_exported(_json_bytes(raw))
    result['aggregate_SHA256'] = _hash(raw)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.aggregates.resolve() or args.output.exists():
        raise FileExistsError('fresh audit output required')
    result = audit_aggregates(args.aggregates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'total_epochs': 82, 'model_executed': False}))


if __name__ == '__main__':
    main()
