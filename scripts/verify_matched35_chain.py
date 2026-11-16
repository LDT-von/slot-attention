"""Verify the pinned matched35 JSON evidence chain without running any model.

Only JSON receipts, the original exporter bytes and its aggregate JSON are read.
Checkpoint files and patient caches are never opened. The CLI offers no hash
override; private test helpers use invented documents and invented hashes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


EXPECTED_SHA256 = {
    'paired_receipt.json': 'b29b84f31792c923d9caa5b1bae190709e7902fe6f3bfb43f60dcf20fa50d800',
    'shared_role/receipt.json': 'd1904ac1f1dde9665f8d56ab3baa17b8c779c4a5dc3649bcc7f9d81d8d75299c',
    'joint/receipt.json': '229cf395f0ab922506d68ba146b9cbf5366e4d47a032c036ea98a12bf09af35b',
    'continued_role_only/receipt.json': 'f51f8efd99868f08ef20f410d4f1c7176aa327c65ee13a694a5cb4bc6a757e30',
    'original_exporter': 'bac2938eab5289e275ef93aee1b5d75aaecd23ac41a8ba26dfcb85aa02369270',
    'aggregates': 'a8148305634592a4d29093270b5570c1c094a9f39403c9f667e25f691b9dcd74',
}
SHARED_CHECKPOINT_SHA256 = 'efae2d2b6c7fc72d157813f97b8e484c8f2a81dcb7c85bb6e5f1e4ed611abd64'
CONFIG_SHA256 = '5da68434e215d11b71d453124dcd5c6dccf84c78db9d26d66690db3b5c686203'
ROLES = tuple(m + '_' + r for m in ('p', 'g') for r in ('common', 'residual'))
METRICS = ('query_norm', 'key_norm_real_mean', 'key_norm_real_max', 'raw_mass',
           'log_pre_epsilon_mass', 'epsilon_uniform_fraction',
           'logit_gap_real_mean', 'logit_gap_real_min')
MATCHED_KEYS = ('shared_checkpoint_SHA256', 'initial_model_tensor_SHA256',
                'initial_train_rng_SHA256', 'initial_torch_rng_SHA256',
                'initial_cuda_rng_SHA256', 'task_optimizer_initial_groups',
                'task_start_role_parameter_SHA256', 'split_identity',
                'prepared_manifest_SHA256', 'source_SHA256')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, label, *, minimum=None, maximum=None):
    _require(type(value) in (int, float) and math.isfinite(value), label + ': finite number required')
    _require(minimum is None or value >= minimum, label + ': below permitted range')
    _require(maximum is None or value <= maximum, label + ': above permitted range')
    return value


def _integer(value, label, *, maximum=None):
    _require(type(value) is int and value >= 0, label + ': nonnegative integer required')
    _require(maximum is None or value <= maximum, label + ': count exceeds denominator')
    return value


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def _finite_tree(value):
    if isinstance(value, float):
        _require(math.isfinite(value), 'JSON contains a nonfinite number')
    elif isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'JSON contains a duplicate object key')
        result[key] = value
    return result


def _json_bytes(raw):
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object)
    _finite_tree(value)
    return value


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _sha(value, label):
    _require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
             label + ': SHA256 required')


def _vector(values, length, label, *, count=False, maximum=None):
    _require(isinstance(values, list) and len(values) == length, label + ': wrong axis length')
    for value in values:
        (_integer(value, label, maximum=maximum) if count else _number(value, label))
    return values


def _matrix(values, rows, cols, label, *, count=False, maximum=None):
    _require(isinstance(values, list) and len(values) == rows, label + ': wrong iteration axis')
    return [_vector(row, cols, label, count=count, maximum=maximum) for row in values]


def _mass_summary(value, bags, slots):
    expected = bags * 3 * slots
    _require(value['observations'] == expected and type(value['observations']) is int,
             'reading mass: wrong observation denominator')
    _require(value['threshold'] == 1e-8, 'reading mass: wrong threshold')
    low = _integer(value['below_threshold_count'], 'reading mass', maximum=expected)
    fraction = _number(value['below_threshold_fraction'], 'reading mass fraction', minimum=0, maximum=1)
    _require(_close(fraction, low / expected), 'reading mass: inconsistent fraction')
    _number(value['minimum'], 'reading mass minimum', minimum=0)
    by_iter = _vector(value['below_threshold_count_by_iteration'], 3, 'reading mass iteration counts',
                      count=True, maximum=bags*slots)
    by_slot = _vector(value['below_threshold_count_by_slot'], slots, 'reading mass slot counts',
                      count=True, maximum=bags*3)
    _require(sum(by_iter) == low == sum(by_slot), 'reading mass: inconsistent grouped counts')
    minima = _vector(value['minimum_by_iteration'], 3, 'reading mass minimum by iteration')
    _require(all(v >= 0 for v in minima) and _close(min(minima), value['minimum']),
             'reading mass: inconsistent minima')


def _read_diagnostics(value, bags):
    _require(value['axes'] == ['iteration', 'slot'], 'reading diagnostics: wrong named axes')
    _require(type(value['bag_observations']) is int and value['bag_observations'] == bags,
             'reading diagnostics: wrong bag count')
    _require(type(value['observation_count']) is int and value['observation_count'] == bags*3*2,
             'reading diagnostics: wrong denominator')
    _require(value['raw_mass_threshold'] == 1e-8, 'reading diagnostics: wrong threshold')
    for name in METRICS:
        metric = value['metrics'][name]
        lo, mean, hi = (_number(metric[key], 'reading metric ' + key)
                        for key in ('minimum', 'mean', 'maximum'))
        _require((lo <= mean or _close(lo, mean)) and (mean <= hi or _close(mean, hi)), 'reading metric: inconsistent bounds')
        matrices = {key: _matrix(metric[key], 3, 2, 'reading metric axes') for key in
                    ('mean_by_iteration_slot', 'minimum_by_iteration_slot', 'maximum_by_iteration_slot')}
        _require(_close(sum(map(sum, matrices['mean_by_iteration_slot'])) / 6, mean),
                 'reading metric: inconsistent overall mean')
        _require(_close(min(map(min, matrices['minimum_by_iteration_slot'])), lo)
                 and _close(max(map(max, matrices['maximum_by_iteration_slot'])), hi),
                 'reading metric: inconsistent extrema')
        for i in range(3):
            for j in range(2):
                _require((matrices['minimum_by_iteration_slot'][i][j] <= matrices['mean_by_iteration_slot'][i][j]
                          or _close(matrices['minimum_by_iteration_slot'][i][j], matrices['mean_by_iteration_slot'][i][j]))
                         and (matrices['mean_by_iteration_slot'][i][j] <= matrices['maximum_by_iteration_slot'][i][j]
                          or _close(matrices['mean_by_iteration_slot'][i][j], matrices['maximum_by_iteration_slot'][i][j])),
                         'reading metric: inconsistent cell bounds')
        if name in ('query_norm', 'key_norm_real_mean', 'key_norm_real_max', 'raw_mass'):
            _require(lo >= 0, 'reading metric: negative norm or mass')
        if name == 'epsilon_uniform_fraction':
            _require(0 <= lo <= hi <= 1, 'reading metric: invalid epsilon share')
    counts = _matrix(value['below_threshold_count_by_iteration_slot'], 3, 2,
                     'reading diagnostic counts', count=True, maximum=bags)
    fractions = _matrix(value['below_threshold_fraction_by_iteration_slot'], 3, 2,
                        'reading diagnostic fractions')
    for i in range(3):
        for j in range(2):
            _require(_close(fractions[i][j], counts[i][j] / bags),
                     'reading diagnostics: inconsistent cell fraction')
    return sum(map(sum, counts))


def _config(value, mode):
    expected = dict(schema='full_slot_matched35_v1', study='blca', fold=0, seed=3,
                    patches=4096, batch_size=2, target_dim=8, common_rank=2,
                    allocation={'p': [2, 2], 'g': [2, 2]}, width=64, iterations=3,
                    hidden=32, bins=4, role_epochs=12, task_epochs=35, patience=35,
                    role_lr=.003, task_lr=.001, role_weight=1., task_mode=mode,
                    readout_bound=2., fixed_task_budget=True, read_diagnostics=True,
                    ablation_nll=True, torch_threads=2, device='cuda:0',
                    configuration_SHA256=CONFIG_SHA256)
    for key, expected_value in expected.items():
        _require(value.get(key) == expected_value, 'receipt config differs: ' + key)
        if type(expected_value) in (int, bool):
            _require(type(value[key]) is type(expected_value), 'receipt config type differs: ' + key)


def _row(row, phase, epoch, mode, *, exported=False):
    _require(row['phase'] == phase and type(row['epoch']) is int and row['epoch'] == epoch,
             'wrong phase, missing, duplicate or reordered epoch')
    _sha(row['input_draw_SHA256'], 'input draw')
    for key in ('train_loss', 'train_role_loss'):
        _number(row[key], key, minimum=0)
    if phase == 'task':
        _number(row['train_survival_NLL'], 'train NLL', minimum=0)
    else:
        _require(row['train_survival_NLL'] is None, 'role phase has survival loss')
    val = row['validation']
    if not exported:
        _require(val['validation_sampling_seed'] == 19003, 'validation bags seed differs')
    for role in ROLES:
        _number(val['role_target_MSE'][role], 'role MSE', minimum=0)
        _mass_summary(val['raw_reading_mass'][role], 76, 2)
        val_count = _read_diagnostics(val['read_diagnostics'][role], 76)
        _require(val_count == val['raw_reading_mass'][role]['below_threshold_count'],
                 'validation count disagrees between diagnostics')
        _read_diagnostics(row['train_read_diagnostics'][role], 304)
    for modality in ('p', 'g'):
        _mass_summary(row['train_raw_reading_mass'][modality], 304, 4)
        total = sum(sum(map(sum, row['train_read_diagnostics'][modality + '_' + role]
                           ['below_threshold_count_by_iteration_slot'])) for role in ('common', 'residual'))
        _require(total == row['train_raw_reading_mass'][modality]['below_threshold_count'],
                 'training count disagrees between diagnostics')
        for term in ('role', 'survival', 'combined'):
            grad = row['raw_slot_gradient_mean'][term][modality]
            if phase == 'role' and term == 'survival':
                _require(grad is None, 'role phase has survival gradient')
            else:
                _vector(grad, 4, 'raw-slot gradient')
                _require(all(g >= 0 for g in grad), 'negative gradient norm')
        if phase == 'task' and mode == 'continued_role_only':
            terms = row['raw_slot_gradient_mean']
            _require(all(g == 0 for g in terms['survival'][modality]), 'control survival-to-role edge not zero')
            _require(all(_close(a, b) for a, b in zip(terms['combined'][modality], terms['role'][modality])),
                     'control combined gradient differs from role gradient')
    if phase == 'task':
        _number(val['C_index'], 'validation C-index', minimum=0, maximum=1)
        _number(val['NLL'], 'validation NLL', minimum=0)
        for role in ROLES:
            for name in (('ablation_NLL', 'ablation_delta_NLL') if exported else
                         ('ablation_NLL', 'ablation_delta_NLL', 'ablation_mean_abs_logit_change')):
                values = _vector(val[name][role], 2, 'slot deletion vector')
                if name != 'ablation_delta_NLL':
                    _require(all(v >= 0 for v in values), 'negative deleted loss or absolute change')
            _require(all(_close(deleted - val['NLL'], delta) for deleted, delta in
                         zip(val['ablation_NLL'][role], val['ablation_delta_NLL'][role])),
                     'signed slot deletion NLL disagrees with full NLL')


def _receipt(receipt, mode):
    phase = 'role' if mode == 'shared_role' else 'task'
    count = 12 if phase == 'role' else 35
    _require(receipt['status'] == 'completed' and receipt['schema'] == 'full_slot_research_v1',
             'receipt is not a completed matched35 stage')
    _require(receipt['test_used'] is False, 'independent test must remain unused')
    _require(type(receipt['train_count']) is int and receipt['train_count'] == 304
             and type(receipt['validation_count']) is int and receipt['validation_count'] == 76,
             'wrong admitted development cohort counts')
    _config(receipt['config'], 'joint' if phase == 'role' else mode)
    _require(isinstance(receipt['source_SHA256'], dict) and bool(receipt['source_SHA256']),
             'source provenance missing')
    for value in receipt['source_SHA256'].values():
        _sha(value, 'source hash')
    for key in ('initial_model_tensor_SHA256', 'initial_train_rng_SHA256',
                'initial_torch_rng_SHA256', 'initial_cuda_rng_SHA256', 'prepared_manifest_SHA256'):
        _sha(receipt[key], key)
    _require(isinstance(receipt['split_identity'], dict) and bool(receipt['split_identity']),
             'split provenance missing')
    _require(receipt['read_normalization']['policy'] == 'masked_add_eps_l1_v2'
             and receipt['read_normalization']['epsilon'] == 1e-8
             and receipt['read_normalization']['epsilon_on'] == 'real_tokens_only',
             'reading normalization differs')
    _require(receipt['reference_frozen'] is True, 'reference freeze not verified')
    _require('early_stop' not in receipt, 'fixed-budget stage stopped early')
    rows = receipt['epoch_logs']
    _require(isinstance(rows, list) and len(rows) == count, 'wrong epoch budget')
    for epoch, row in enumerate(rows, 1):
        _row(row, phase, epoch, mode)
    _require(receipt['optimizer_steps'] == {'role': 1824 if phase == 'role' else 0,
                                           'task': 5320 if phase == 'task' else 0},
             'wrong optimizer step budget')
    if phase == 'task':
        _require(receipt['task_gradient_policy'] == ('joint' if mode == 'joint' else 'role_only_trainable_head'),
                 'wrong task gradient routing policy')
        for flag in ('checkpoint_reload_verified', 'checkpoint_optimizer_and_generator_reload_verified',
                     'role_parameters_changed_during_task', 'task_optimizer_initial_state_empty'):
            _require(receipt[flag] is True, 'task state verification missing: ' + flag)
        _sha(receipt['task_start_role_parameter_SHA256'], 'role-start hash')
        groups = receipt['task_optimizer_initial_groups']
        _require(isinstance(groups, list) and bool(groups) and all(group['lr'] == .001 for group in groups),
                 'task optimizer groups missing or learning rate differs')
        best_row = max(rows, key=lambda r: (r['validation']['C_index'], -r['validation']['NLL'], -r['epoch']))
        for label, row in (('best', best_row), ('last', rows[-1])):
            expected = dict(epoch=row['epoch'], C_index=row['validation']['C_index'], NLL=row['validation']['NLL'])
            _require(receipt[label] == expected, label + ' disagrees with full curve')
    return rows


def _scan_aggregate(value):
    """Reject identifier/patient-axis content without echoing private values."""
    forbidden = {'patient_id', 'patient_ids', 'patient_identifiers', 'identifiers',
                 'predictions', 'prediction', 'weights', 'state_dict', 'features', 'patient_rows'}
    if isinstance(value, dict):
        for key, child in value.items():
            _require(key.lower() not in forbidden and 'by_patient' not in key.lower(),
                     'aggregate contains a private field')
            _scan_aggregate(child)
    elif isinstance(value, list):
        _require(len(value) not in (76, 304), 'aggregate contains a possible patient axis')
        for child in value:
            _scan_aggregate(child)
    elif isinstance(value, str):
        _require(re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}', value, re.I) is None,
                 'aggregate contains a possible patient identifier')


def _verify_chain(run_root, original_exporter, aggregates, expected_hashes):
    root = Path(run_root).resolve()
    inputs = {name: root / name for name in EXPECTED_SHA256 if name.endswith('receipt.json')}
    inputs.update(original_exporter=Path(original_exporter).resolve(), aggregates=Path(aggregates).resolve())
    blobs, hashes = {}, {}
    for name, path in inputs.items():
        if name.endswith('receipt.json'):
            _require(root in path.resolve().parents, 'receipt symlink escapes run root')
        raw = path.read_bytes()
        hashes[name] = _hash(raw)
        _require(hashes[name] == expected_hashes[name], 'SHA256 mismatch: ' + name)
        if name != 'original_exporter':
            blobs[name] = _json_bytes(raw)
    pair = blobs['paired_receipt.json']
    _require(pair['schema'] == 'matched35_pair_v1' and pair['status'] == 'completed',
             'pair receipt is not completed')
    _require(pair['test_used'] is False and pair.get('synthetic_only') is False,
             'pair is not the recorded development execution')
    _require(pair['total_epochs'] == pair['expected_total_epochs'] == 82 and pair['shared_role_epochs'] == 12,
             'pair epoch budget differs')
    for flag in ('common_start_verified', 'matched_input_draws_verified', 'task_optimizer_reset_verified'):
        _require(pair[flag] is True, 'pair assertion missing: ' + flag)
    _config(pair['config'], 'joint')
    _require(pair['shared_checkpoint_SHA256'] == SHARED_CHECKPOINT_SHA256, 'shared checkpoint provenance differs')
    stages = {mode: blobs[mode + '/receipt.json'] for mode in ('shared_role', 'joint', 'continued_role_only')}
    curves = {mode: _receipt(receipt, mode) for mode, receipt in stages.items()}
    shared, joint, control = (stages[k] for k in ('shared_role', 'joint', 'continued_role_only'))
    for key in MATCHED_KEYS:
        _require(joint[key] == control[key], 'task arms differ in common state: ' + key)
    for key in ('prepared_manifest_SHA256', 'split_identity', 'source_SHA256'):
        _require(joint[key] == shared[key] and pair['source_SHA256'] == shared['source_SHA256'],
                 'shared phase provenance differs')
    _require(shared['artifacts_SHA256']['role_last.pt'] == pair['shared_checkpoint_SHA256']
             == joint['shared_checkpoint_SHA256'], 'shared checkpoint references differ')
    _require(joint['model_parameters'] == control['model_parameters'] == shared['model_parameters'],
             'model capacities differ')
    _require([r['input_draw_SHA256'] for r in curves['joint']]
             == [r['input_draw_SHA256'] for r in curves['continued_role_only']],
             'training input draw trajectories differ')
    for mode in ('joint', 'continued_role_only'):
        ref = pair['arms'][mode]
        _require(ref['status'] == 'completed' and ref['epochs'] == 35
                 and ref['receipt_SHA256'] == hashes[mode + '/receipt.json'],
                 'pair arm receipt reference differs')
        _require(ref['best'] == stages[mode]['best'] and ref['last'] == stages[mode]['last'],
                 'pair arm selection references differ')
    aggregate = blobs['aggregates']
    _scan_aggregate(aggregate)
    full_numeric = False
    if aggregate.get('schema') == 'matched35_aggregate_export_v1':
        from scripts.audit_matched35_aggregates import _audit_exported
        _audit_exported(aggregate)
        # Compare every exported task-row field with the original receipts.
        import importlib.util
        spec = importlib.util.spec_from_file_location('_pinned_matched35_exporter', inputs['original_exporter'])
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        for mode in ('joint', 'continued_role_only'):
            _require(exporter.arm_epochs(stages[mode]) == aggregate['arms'][mode]['epoch_rows'],
                     'aggregate task rows differ from pinned raw receipt')
            for key, exported in aggregate['arms'][mode].items():
                if key not in ('epochs', 'epoch_rows'):
                    _require(stages[mode].get(key) == exported, 'aggregate arm metadata differs from receipt')
        for key, exported in aggregate['paired_receipt'].items():
            _require(pair.get(key) == exported, 'aggregate pair metadata differs from receipt')
        expected_role_rows = [dict(epoch=r['epoch'], train_loss=r['train_loss'],
                                  train_role_loss=r['train_role_loss'],
                                  role_target_MSE={role:r['validation']['role_target_MSE'][role] for role in ROLES})
                              for r in shared['epoch_logs']]
        _require(expected_role_rows == aggregate['shared_role']['role_epoch_rows'],
                 'aggregate shared role rows differ from receipt')
        full_numeric = True
    # Do not copy the unknown original aggregate schema into the audit output.
    # The pinned bytes and privacy scan are verified; receipt arithmetic below
    # is computed independently from all original stage rows.
    rows_j, rows_c = curves['joint'], curves['continued_role_only']
    differences = [c['validation']['C_index'] - j['validation']['C_index'] for j, c in zip(rows_j, rows_c)]
    mass_windows = {}
    for mode in ('joint', 'continued_role_only'):
        rows = curves[mode]
        counts = [r['validation']['raw_reading_mass']['p_common']['below_threshold_count'] for r in rows]
        mass_windows[mode] = {'all_35_count': sum(counts), 'all_35_fraction': sum(counts)/(35*456),
                             'ep1_26_count': sum(counts[:26]), 'ep1_26_mean': sum(counts[:26])/26,
                             'ep27_35_count': sum(counts[26:]), 'ep27_35_mean': sum(counts[26:])/9}
    return {'schema': 'matched35_chain_audit_v1', 'status': 'verified', 'input_SHA256': hashes,
            'phase_epochs': {'shared_role': 12, 'joint': 35, 'continued_role_only': 35},
            'total_epochs': 82, 'common_start_verified': True, 'matched_input_draws_verified': True,
            'task_optimizer_reset_verified': True, 'model_executed': False,
            'patient_rows_exported': False, 'checkpoints_loaded_or_rehashed': False,
            'aggregate_verification': 'pinned bytes, privacy scan, receipt arithmetic and every exported row field',
            'aggregate_numeric_crosscheck_complete': full_numeric,
            'scores': {mode: {label: stages[mode][label] for label in ('best', 'last')}
                       for mode in ('joint', 'continued_role_only')},
            'paired_C_index_mean': sum(differences)/35,
            'p_common_observation_denominator_per_epoch': 456, 'p_common_mass_windows': mass_windows,
            'scope': 'one development fold/seed; epochs correlated, no independent test',
            'limitations': ['No checkpoint bytes or patient-cache values were inspected.',
                            'Metric equality is not equality of all CUDA/model states.',
                            'Rounded zero epsilon/zero low-mass counts do not certify constant attention.',
                            'Signed CV of log mass is not used as a relative stability measure.']}


def verify_chain(run_root, original_exporter, aggregates):
    """Return a safe audit after checking the fixed published evidence pins."""
    result = _verify_chain(run_root, original_exporter, aggregates, EXPECTED_SHA256)
    _require(result['aggregate_numeric_crosscheck_complete'] is True, 'full aggregate crosscheck incomplete')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--original-exporter', type=Path, required=True)
    parser.add_argument('--aggregates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    root, output = args.run_root.resolve(), args.output.resolve()
    protected = [root / name for name in EXPECTED_SHA256 if name.endswith('receipt.json')]
    protected.extend((args.original_exporter.resolve(), args.aggregates.resolve()))
    if output in protected or output == root or root in output.parents:
        raise ValueError('audit output must be outside the original run and inputs')
    if output.exists():
        raise FileExistsError('preserve existing audit output')
    result = verify_chain(root, args.original_exporter, args.aggregates)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'verified', 'total_epochs': 82, 'model_executed': False,
                      'patient_rows_exported': False}))
    return result


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        # Private paths and receipt values may appear in exception messages.
        raise SystemExit('Evidence-chain verification failed; inputs were not modified ('
                         + type(error).__name__ + ').') from None
