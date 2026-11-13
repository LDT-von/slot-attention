"""Export allowlisted aggregates from existing readfix-v2 receipts, without torch.

Only the explicitly supplied JSON receipts are read. No model, checkpoint,
patient cache, network connection, training or inference is opened here.
Receipt hashes bind the export to the completed runs. Unknown fields are never
copied; vectors are restricted to slot/iteration axes, not patient axes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROLES = tuple(m + '_' + r for m in ('p', 'g') for r in ('common', 'residual'))


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('aggregate must be a finite number')
    return value


def positive_int(value):
    if type(value) is not int or value < 1:
        raise ValueError('positive integer dimension required')
    return value


def vector(value, size, *, nullable=False):
    if nullable and value is None:
        return None
    if not isinstance(value, list) or len(value) != size:
        raise ValueError('aggregate vector must match its slot or iteration axis')
    return [number(x) for x in value]


def scalars(source, names, *, nullable=()):
    return {name: (None if name in nullable and source[name] is None else number(source[name]))
            for name in names}


def digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('lowercase SHA256 required')
    return value


def mass_summary(source, patients, iterations, slots):
    result = scalars(source, ('threshold', 'observations', 'minimum', 'below_threshold_count',
                              'below_threshold_fraction'))
    for key, size in (('below_threshold_count_by_iteration', iterations),
                      ('below_threshold_count_by_slot', slots), ('minimum_by_iteration', iterations)):
        result[key] = vector(source[key], size)
    denominator = patients * iterations * slots
    count = result['below_threshold_count']
    if (result['observations'] != denominator or type(count) is not int or not 0 <= count <= denominator
            or result['minimum'] < 0 or result['threshold'] != 1e-8
            or not math.isclose(result['below_threshold_fraction'], count / denominator, abs_tol=1e-12)
            or any(type(x) is not int or x < 0 for key in ('below_threshold_count_by_iteration',
                                                         'below_threshold_count_by_slot') for x in result[key])
            or sum(result['below_threshold_count_by_iteration']) != count
            or sum(result['below_threshold_count_by_slot']) != count):
        raise ValueError('read-mass denominator/count disagreement')
    return result


def validation_summary(source, config, patients, phase):
    result = {'role_target_MSE': scalars(source['role_target_MSE'], ROLES),
              'validation_sampling_seed': number(source['validation_sampling_seed']),
              'roles': {}, 'slots': {}, 'raw_reading_mass': {}, 'ablation_mean_abs_logit_change': {}}
    for key in ROLES:
        modality, role = key.split('_')
        slots = config['allocation'][modality][0 if role == 'common' else 1]
        r = scalars(source['roles'][key], ('patients', 'mean_norm', 'across_patient_std_rms',
                                           'centered_patient_effective_rank', 'patient_rank_upper_bound'))
        # The old diagnostic reports an ambient bound; retain it and add the
        # tighter architectural bound instead of calling rank ~2 a collapse.
        r['architectural_rank_upper_bound'] = min(patients - 1, config['common_rank'] if role == 'common'
                                                  else config['target_dim'] - config['common_rank'])
        result['roles'][key] = r
        s = source['slots'][key]
        z = scalars(s, ('patients', 'slots', 'width', 'mean_norm', 'zero_slot_fraction',
                       'across_patient_std_rms', 'patient_rank_upper_bound', 'pair_cosine_mean',
                       'pair_cosine_undefined_count', 'read_overlap_mean'),
                    nullable=('pair_cosine_mean', 'read_overlap_mean'))
        for name in ('mean_norm_by_slot', 'zero_fraction_by_slot', 'across_patient_std_rms_by_slot',
                     'centered_patient_effective_rank_by_slot'):
            z[name] = vector(s[name], slots)
        if r['patients'] != patients or z['patients'] != patients or z['slots'] != slots or z['width'] != config['width']:
            raise ValueError('role/slot dimensions disagree with receipt configuration')
        result['slots'][key] = z
        result['raw_reading_mass'][key] = mass_summary(source['raw_reading_mass'][key], patients,
                                                       config['iterations'], slots)
        if phase == 'task':
            result['ablation_mean_abs_logit_change'][key] = vector(source['ablation_mean_abs_logit_change'][key], slots)
    if phase == 'task':
        result.update(scalars(source, ('C_index', 'NLL')))
    return result


def export_receipt(path, expected_sha256, mode):
    path = Path(path)
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != digest(expected_sha256):
        raise ValueError('receipt SHA256 differs from the published completed run')
    source = json.loads(payload)
    if source['schema'] != 'full_slot_research_v1' or source['status'] != 'completed' or source['test_used'] is not False:
        raise ValueError('completed full_slot_research_v1 development receipt required')
    if mode not in ('joint', 'frozen') or source['config']['task_mode'] != mode:
        raise ValueError('declared arm differs from receipt task_mode')
    normalization = {'policy': 'masked_add_eps_l1_v2', 'epsilon': 1e-8, 'epsilon_on': 'real_tokens_only'}
    if source['read_normalization'] != normalization:
        raise ValueError('readfix-v2 normalization receipt required')
    c = source['config']
    config = scalars(c, ('fold', 'seed', 'patches', 'batch_size', 'target_dim', 'common_rank', 'width',
                         'iterations', 'hidden', 'bins', 'role_epochs', 'task_epochs', 'patience',
                         'role_lr', 'task_lr', 'role_weight', 'readout_bound'))
    config['task_mode'] = mode
    config['allocation'] = {m: [positive_int(x) for x in vector(c['allocation'][m], 2)] for m in ('p', 'g')}
    for key in ('target_dim', 'width', 'iterations', 'role_epochs', 'task_epochs', 'patience'):
        positive_int(config[key])
    if type(config['common_rank']) is not int or not 0 <= config['common_rank'] <= config['target_dim']:
        raise ValueError('invalid common rank')
    train, val = positive_int(source['train_count']), positive_int(source['validation_count'])
    result = {'receipt_SHA256': actual, 'schema': source['schema'], 'status': 'completed', 'test_used': False,
              'config': config, 'train_count': train, 'validation_count': val,
              'prepared_manifest_SHA256': digest(source['prepared_manifest_SHA256']),
              'read_normalization': normalization, 'model_parameters': positive_int(source['model_parameters']),
              'optimizer_steps': scalars(source['optimizer_steps'], ('role', 'task')),
              'best': scalars(source['best'], ('epoch', 'C_index', 'NLL')),
              'last': scalars(source['last'], ('epoch', 'C_index', 'NLL')), 'epoch_logs': []}
    for key in ('checkpoint_reload_verified', 'checkpoint_optimizer_and_generator_reload_verified',
                'reference_frozen', 'role_parameters_changed_during_task'):
        if type(source[key]) is not bool:
            raise ValueError('boolean verification flag required')
        result[key] = source[key]
    result['checkpoint_SHA256_reported_only'] = {name: digest(source['artifacts_SHA256'][name])
                                               for name in ('role_last.pt', 'task_best.pt', 'task_last.pt')}
    counts, task_started = {'role': 0, 'task': 0}, False
    for row in source['epoch_logs']:
        phase = row['phase']
        if phase not in counts or (phase == 'role' and task_started):
            raise ValueError('expected role phase followed by task phase')
        task_started |= phase == 'task'
        counts[phase] += 1
        if row['epoch'] != counts[phase]:
            raise ValueError('epoch sequence must be complete and ordered')
        log = {'phase': phase, 'epoch': counts[phase],
               **scalars(row, ('train_loss', 'train_role_loss', 'train_survival_NLL'), nullable=('train_survival_NLL',)),
               'validation': validation_summary(row['validation'], config, val, phase),
               'train_raw_reading_mass': {m: mass_summary(row['train_raw_reading_mass'][m], train,
                                         config['iterations'], sum(config['allocation'][m])) for m in ('p', 'g')},
               'raw_slot_gradient_mean': {term: {m: vector(row['raw_slot_gradient_mean'][term][m],
                                                         sum(config['allocation'][m]), nullable=True)
                                                    for m in ('p', 'g')} for term in ('combined', 'survival', 'role')}}
        result['epoch_logs'].append(log)
    if counts['role'] != config['role_epochs'] or not 1 <= counts['task'] <= config['task_epochs']:
        raise ValueError('completed receipt has inconsistent phase lengths')
    curve = [{'epoch': row['epoch'], **scalars(row['validation'], ('C_index', 'NLL'))}
             for row in result['epoch_logs'] if row['phase'] == 'task']
    best = max(curve, key=lambda x: (x['C_index'], -x['NLL'], -x['epoch']))
    if best != result['best'] or curve[-1] != result['last']:
        raise ValueError('best/last do not agree with the complete task curve')
    result['audit'] = {'epoch_counts': counts, 'best_last_match_curve': True,
                       'raw_receipt_hash_verified': True, 'checkpoints_loaded_or_rehashed': False,
                       'gradient_scope': 'as recorded norms of raw-slot gradients; no gradient directions',
                       'ablation_scope': 'recorded slot removal and mean absolute logit change; no NLL/CPI',
                       'rank_scope': 'common-rank bound is architectural, not an empirical collapse threshold'}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', nargs=3, action='append', metavar=('ARM', 'PATH', 'SHA256'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('fresh output file required')
    exported, inputs = {}, []
    for mode, name, sha256 in args.receipt:
        path = Path(name).resolve()
        if output == path or path.parent in output.parents:
            raise ValueError('output must be outside each original run directory')
        if mode in exported:
            raise ValueError('duplicate arm')
        exported[mode] = export_receipt(path, sha256, mode)
        inputs.append((path, sha256))
    result = {'schema': 'full_slot_aggregate_receipt_review_v1',
              'scope': 'existing receipt aggregates only; no new patient execution or independent test',
              'exporter_SHA256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'runs': exported}
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != sha256 for path, sha256 in inputs):
        raise RuntimeError('source receipt changed during export')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(payload)
    print(json.dumps({'status': 'exported', 'arms': list(exported),
                      'output_SHA256': hashlib.sha256(output.read_bytes()).hexdigest(),
                      'model_executed': False, 'patient_rows_exported': False}))


if __name__ == '__main__':
    main()
