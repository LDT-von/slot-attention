"""Export matched35 aggregates from the completed paired run. Read-only.

Reads only the receipts already written by the matched35 run. It does not
load checkpoints, read the patient cache, import torch, execute a model or
use a GPU. Patient identifiers, per-patient arrays, predictions and weights
are never copied: only slot, iteration and epoch axes are exported.

Usage:
  python -B scripts/export_matched35_aggregates.py --output <path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROLES = tuple(m + '_' + r for m in ('p', 'g') for r in ('common', 'residual'))


def digest(value):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError('lowercase SHA256 required')
    return value


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('aggregate must be a finite number')
    return value


def vector(value, size):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError('aggregate vector must match its slot or iteration axis')
    return [number(x) for x in value]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def arm_epochs(receipt):
    """Per-epoch aggregate rows for one task arm."""
    rows = []
    for log in receipt['epoch_logs']:
        validation = log['validation']
        row = {
            'epoch': log['epoch'],
            'phase': log['phase'],
            'train_loss': number(log['train_loss']),
            'train_role_loss': number(log['train_role_loss']),
            'train_survival_NLL': number(log['train_survival_NLL']) if log['train_survival_NLL'] is not None else None,
            'C_index': number(validation['C_index']),
            'NLL': number(validation['NLL']),
            'role_target_MSE': {role: number(validation['role_target_MSE'][role]) for role in ROLES},
            'input_draw_SHA256': log.get('input_draw_SHA256'),
        }
        if 'read_diagnostics' in validation:
            row['read_diagnostics'] = validation['read_diagnostics']
        if 'ablation_NLL' in validation:
            row['ablation_NLL'] = validation['ablation_NLL']
            row['ablation_delta_NLL'] = validation['ablation_delta_NLL']
            row['ablation_delta_NLL_definition'] = validation.get('ablation_delta_NLL_definition')
        if 'raw_reading_mass' in validation:
            row['raw_reading_mass'] = validation['raw_reading_mass']
        if 'roles' in validation:
            row['roles'] = validation['roles']
        if 'slots' in validation:
            row['slots'] = validation['slots']
        if 'raw_slot_gradient_mean' in log:
            row['raw_slot_gradient_mean'] = log['raw_slot_gradient_mean']
        if 'train_read_diagnostics' in log:
            row['train_read_diagnostics'] = log['train_read_diagnostics']
        if 'train_raw_reading_mass' in log:
            row['train_raw_reading_mass'] = log['train_raw_reading_mass']
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=Path('local_evidence/full_slot_matched35_20261113_01'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.run_root
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('fresh output file required')

    pair = read(root / 'paired_receipt.json')
    if pair['status'] != 'completed':
        raise ValueError('paired run is not completed: ' + pair['status'])
    shared = read(root / 'shared_role' / 'receipt.json')
    arms = {name: read(root / name / 'receipt.json') for name in ('joint', 'continued_role_only')}

    result = {
        'schema': 'matched35_aggregate_export_v1',
        'scope': 'existing matched35 receipts only; no patient execution, no checkpoint load, no independent test',
        'exporter_SHA256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'paired_receipt': {
            'status': pair['status'],
            'total_epochs': pair['total_epochs'],
            'expected_total_epochs': pair['expected_total_epochs'],
            'shared_role_epochs': pair['shared_role_epochs'],
            'common_start_verified': pair['common_start_verified'],
            'matched_input_draws_verified': pair['matched_input_draws_verified'],
            'task_optimizer_reset_verified': pair['task_optimizer_reset_verified'],
            'optimizer_policy': pair.get('optimizer_policy'),
            'shared_checkpoint_SHA256': pair.get('shared_checkpoint_SHA256'),
            'config': pair['config'],
            'source_SHA256': pair.get('source_SHA256'),
            'arms': pair['arms'],
        },
        'shared_role': {
            'status': shared['status'],
            'epochs': len(shared['epoch_logs']),
            'role_epoch_rows': [
                {'epoch': log['epoch'], 'train_loss': number(log['train_loss']),
                 'train_role_loss': number(log['train_role_loss']),
                 'role_target_MSE': {role: number(log['validation']['role_target_MSE'][role]) for role in ROLES}}
                for log in shared['epoch_logs']],
        },
        'arms': {},
    }
    for name, receipt in arms.items():
        result['arms'][name] = {
            'status': receipt['status'],
            'epochs': len(receipt['epoch_logs']),
            'best': receipt['best'],
            'last': receipt['last'],
            'early_stop': receipt.get('early_stop'),
            'task_gradient_policy': receipt.get('task_gradient_policy'),
            'role_parameters_changed_during_task': receipt.get('role_parameters_changed_during_task'),
            'checkpoint_reload_verified': receipt.get('checkpoint_reload_verified'),
            'checkpoint_optimizer_and_generator_reload_verified': receipt.get('checkpoint_optimizer_and_generator_reload_verified'),
            'reference_frozen': receipt.get('reference_frozen'),
            'shared_checkpoint_SHA256': receipt.get('shared_checkpoint_SHA256'),
            'initial_model_tensor_SHA256': receipt.get('initial_model_tensor_SHA256'),
            'initial_train_rng_SHA256': receipt.get('initial_train_rng_SHA256'),
            'initial_torch_rng_SHA256': receipt.get('initial_torch_rng_SHA256'),
            'initial_cuda_rng_SHA256': receipt.get('initial_cuda_rng_SHA256'),
            'task_optimizer_initial_state_empty': receipt.get('task_optimizer_initial_state_empty'),
            'task_optimizer_initial_groups': receipt.get('task_optimizer_initial_groups'),
            'task_start_role_parameter_SHA256': receipt.get('task_start_role_parameter_SHA256'),
            'prepared_manifest_SHA256': receipt.get('prepared_manifest_SHA256'),
            'artifacts_SHA256': receipt.get('artifacts_SHA256'),
            'epoch_rows': arm_epochs(receipt),
        }

    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(payload)
    print(json.dumps({'status': 'exported',
                      'total_epochs': result['paired_receipt']['total_epochs'],
                      'output_SHA256': hashlib.sha256(output.read_bytes()).hexdigest(),
                      'model_executed': False, 'patient_rows_exported': False}))


if __name__ == '__main__':
    main()
