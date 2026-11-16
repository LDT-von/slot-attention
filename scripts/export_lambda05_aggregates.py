"""Allowlisted lambda05 receipt export; no models, checkpoints or patient cache."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_full_slot_receipt_review import digest, number, scalars, vector, validation_summary, mass_summary
from scripts.verify_matched35_chain import ROLES, METRICS, _read_diagnostics, _json_bytes, _require
from research_models.common_private_synergy.lambda05 import validate_protocol


def safe_diagnostics(source, bags):
    _read_diagnostics(source, bags)
    # Never copy an unknown field, even under a legitimate diagnostic name.
    return dict(axes=['iteration', 'slot'], bag_observations=bags, observation_count=bags*6,
        raw_mass_threshold=1e-8,
        below_threshold_count_by_iteration_slot=[vector(r, 2) for r in source['below_threshold_count_by_iteration_slot']],
        below_threshold_fraction_by_iteration_slot=[vector(r, 2) for r in source['below_threshold_fraction_by_iteration_slot']],
        metrics={name: {**scalars(source['metrics'][name], ('mean', 'minimum', 'maximum')),
                       **{key: [vector(r, 2) for r in source['metrics'][name][key]] for key in
                          ('mean_by_iteration_slot', 'minimum_by_iteration_slot', 'maximum_by_iteration_slot')}}
                 for name in METRICS})


def export_receipt(path, expected_sha256):
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    _require(actual == digest(expected_sha256), 'lambda receipt hash differs')
    source = _json_bytes(raw)
    validate_protocol(source['config'])
    _require(source['status'] == 'completed' and source['test_used'] is False
             and source['synthetic_only'] is False and source['protocol'] == 'lambda05_fixed35_v1',
             'completed real lambda05 receipt required')
    _require(source['new_role_epochs'] == 0 and source['new_task_epochs'] == source['total_new_epochs'] == 35
             and len(source['epoch_logs']) == 35 and source['last']['epoch'] == 35 and 'early_stop' not in source,
             'fixed lambda35 budget incomplete')
    for flag in ('endpoint_start_verified_before_updates', 'endpoint_start_and_draws_verified',
                 'checkpoint_reload_verified', 'checkpoint_optimizer_and_generator_reload_verified', 'reference_frozen'):
        _require(source[flag] is True, 'lambda validation flag missing')
    _require(source['role_survival_gradient_scale'] == .5 and source['head_survival_gradient_scale'] == 1.
             and source['historical_chain_audit']['aggregate_numeric_crosscheck_complete'] is True,
             'lambda routing or historical full-chain audit differs')
    c = source['config']
    result = dict(schema='lambda05_aggregate_export_v1', source_receipt_SHA256=actual,
                  configuration_SHA256=digest(c['configuration_SHA256']), model_executed=False,
                  patient_rows_exported=False, role_survival_gradient_scale=.5, head_survival_gradient_scale=1.,
                  new_role_epochs=0, new_task_epochs=35, best=scalars(source['best'], ('epoch', 'C_index', 'NLL')),
                  last=scalars(source['last'], ('epoch', 'C_index', 'NLL')),
                  shared_checkpoint_SHA256=digest(source['shared_checkpoint_SHA256']), epoch_rows=[])
    for index, row in enumerate(source['epoch_logs'], 1):
        _require(row['phase'] == 'task' and type(row['epoch']) is int and row['epoch'] == index,
                 'lambda epoch order differs')
        validation = validation_summary(row['validation'], c, 76, 'task')
        validation['read_diagnostics'] = {r: safe_diagnostics(row['validation']['read_diagnostics'][r], 76) for r in ROLES}
        for key in ('ablation_NLL', 'ablation_delta_NLL'):
            validation[key] = {r: vector(row['validation'][key][r], 2) for r in ROLES}
        for role in ROLES:
            for deleted, delta in zip(validation['ablation_NLL'][role], validation['ablation_delta_NLL'][role]):
                _require(abs(deleted-validation['NLL']-delta) <= 1e-12, 'signed deletion loss mismatch')
        result['epoch_rows'].append(dict(epoch=index, phase='task',
            **scalars(row, ('train_loss', 'train_role_loss', 'train_survival_NLL')),
            input_draw_SHA256=digest(row['input_draw_SHA256']), validation=validation,
            train_read_diagnostics={r: safe_diagnostics(row['train_read_diagnostics'][r], 304) for r in ROLES},
            train_raw_reading_mass={m: mass_summary(row['train_raw_reading_mass'][m], 304, 3, 4) for m in ('p', 'g')},
            raw_slot_gradient_mean={term: {m: vector(row['raw_slot_gradient_mean'][term][m], 4) for m in ('p', 'g')}
                                    for term in ('role', 'survival', 'combined')}))
    best = max(result['epoch_rows'], key=lambda r: (r['validation']['C_index'], -r['validation']['NLL'], -r['epoch']))
    for label, row in (('best', best), ('last', result['epoch_rows'][-1])):
        _require(result[label] == dict(epoch=row['epoch'], C_index=row['validation']['C_index'], NLL=row['validation']['NLL']),
                 'lambda selection differs from curve')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.receipt.resolve():
        raise FileExistsError('fresh export output required')
    result = export_receipt(args.receipt, args.expected_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='exported', task_epochs=35, model_executed=False, patient_rows_exported=False)))


if __name__ == '__main__':
    main()
