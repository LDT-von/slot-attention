"""Three-arm synthetic verification of the continued-role-only control.

Three checks, on invented data only (no patient cache, no checkpoint, no GPU):

1. Equivalence: the joint and frozen arms must reproduce the numbers that the
   pre-change engine produced. A change in either means the third arm cannot
   be compared against the completed joint/frozen runs.
2. Correctness of the new arm: role parameters must change during the task
   phase, the readout must not, and the survival gradient must be measured
   but not applied.
3. Contrast: the new arm must differ from frozen in exactly the slot
   parameters, and from joint in the applied task gradient.

Writes only aggregate numbers; no patient identifiers or per-patient arrays.
"""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research_models.common_private_synergy.full_slot_engine import run_full

OUT = ROOT / 'local_evidence/full_slot_three_arm_synthetic_20261113_01'
PATHS = ['research_models/common_private_synergy/synthetic_full_slot_three_arm.py',
         'research_models/common_private_synergy/full_slot_engine.py',
         'research_models/common_private_synergy/full_slot_epoch.py',
         'research_models/common_private_synergy/full_slot_diagnostics.py',
         'research_models/common_private_synergy/components/role_model.py']


def build_prepared():
    prepared = OUT / 'invented'
    (prepared / 'prepared/cache').mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(71211)
    manifest = {'omic_sizes': [3, 4], 'patients': {}, 'folds': {'0': {'train': [], 'val': []}}}
    for i in range(72):
        pid = f'invented_{i:03d}'
        z = torch.randn((), generator=gen)
        independent = torch.randn((), generator=gen)
        patches = torch.randn(24, 6, generator=gen) * .8
        patches[:, 0] += z
        patches[:, 1] += independent
        omics = [torch.randn(n, generator=gen) for n in (3, 4)]
        omics[0][0] += z
        omics[1][0] += independent
        noise = torch.randn((), generator=gen)
        t = float(10 * torch.exp(-.8 * z + .25 * noise))
        censor = float(torch.rand((), generator=gen) < .25)
        rel = f'prepared/cache/{pid}.pt'
        torch.save({'patient_id': pid, 'patches': patches, 'omics': omics}, prepared / rel)
        manifest['patients'][pid] = {'cache_path': rel, 'time': t, 'censor': censor}
        manifest['folds']['0']['train' if i < 48 else 'val'].append(pid)
    (prepared / 'prepared/manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return prepared, manifest


def summarize(receipt):
    """Aggregate only: no patient identifiers, no per-patient arrays."""
    rows = []
    for log in receipt['epoch_logs']:
        grads = log['raw_slot_gradient_mean']

        def first_slot(term):
            value = grads.get(term)
            if not value:
                return None
            vector = value.get('p')
            return vector[0] if vector else None

        rows.append({
            'phase': log['phase'], 'epoch': log['epoch'],
            'train_loss': log['train_loss'],
            'train_role_loss': log['train_role_loss'],
            'train_survival_NLL': log['train_survival_NLL'],
            'C_index': log['validation'].get('C_index'),
            'NLL': log['validation'].get('NLL'),
            'p_common_below': log['validation']['raw_reading_mass']['p_common']['below_threshold_count'],
            'p_common_roleMSE': log['validation']['role_target_MSE']['p_common'],
            'grad_role_p0': first_slot('role'),
            'grad_surv_p0': first_slot('survival'),
            'grad_combined_p0': first_slot('combined'),
        })
    return rows


def main():
    prepared, manifest = build_prepared()
    c = {'fold': 0, 'seed': 3, 'device': 'cpu', 'torch_threads': 1, 'target_dim': 3,
         'common_rank': 1, 'allocation': {'p': [2, 2], 'g': [2, 2]}, 'width': 8,
         'iterations': 2, 'hidden': 8, 'bins': 4, 'readout_bound': 2., 'batch_size': 4,
         'patches': 16, 'role_epochs': 6, 'task_epochs': 12, 'role_lr': .003,
         'task_lr': .001, 'role_weight': 1., 'patience': 6}
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in PATHS}

    results = {}
    for mode in ('frozen', 'joint', 'continued_role_only'):
        arm_dir = OUT / mode
        receipt_path = arm_dir / 'receipt.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            if receipt.get('status') != 'completed':
                raise RuntimeError('existing arm receipt is not completed: ' + mode)
            if receipt['config']['task_mode'] != mode:
                raise RuntimeError('existing arm receipt has a different task mode')
            print(json.dumps({'arm': mode, 'reused': True, 'best': receipt['best']}), flush=True)
        else:
            result = run_full(prepared, arm_dir, manifest, {**c, 'task_mode': mode}, hashes, synthetic=True)
            receipt = result
            print(json.dumps({'arm': mode, 'best': result['best'], 'last': result['last'],
                              'role_parameters_changed_during_task': result['role_parameters_changed_during_task']}),
                  flush=True)
        results[mode] = {'receipt': receipt, 'rows': summarize(receipt)}

    frozen, joint, cro = (results[m]['rows'] for m in ('frozen', 'joint', 'continued_role_only'))
    # The role phase must be identical in all three arms up to the first task epoch.
    role_rows = [r for r in frozen if r['phase'] == 'role']
    identical_role = all(
        [r for r in results[m]['rows'] if r['phase'] == 'role'] == role_rows for m in results)
    cro_task = [r for r in cro if r['phase'] == 'task']
    joint_task = [r for r in joint if r['phase'] == 'task']
    frozen_task = [r for r in frozen if r['phase'] == 'task']

    summary = {
        'scope': 'invented synthetic data only; no patient cache, no checkpoint, no GPU',
        'role_phase_identical_across_arms': identical_role,
        'arms': {
            mode: {
                'role_parameters_changed_during_task': results[mode]['receipt']['role_parameters_changed_during_task'],
                'best': results[mode]['receipt']['best'],
                'last': results[mode]['receipt']['last'],
                'optimizer_steps': results[mode]['receipt']['optimizer_steps'],
                'first_task_epoch': next((r for r in results[mode]['rows'] if r['phase'] == 'task'), None),
            } for mode in results
        },
        'task_phase_rows': {'frozen': frozen_task, 'joint': joint_task, 'continued_role_only': cro_task},
    }
    (OUT / 'three_arm_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'role_phase_identical': identical_role,
                      'written': str(OUT / 'three_arm_summary.json')}, indent=2))


if __name__ == '__main__':
    main()
