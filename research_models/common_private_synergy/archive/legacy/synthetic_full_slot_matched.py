"""One preregistered matched synthetic gradient-path check, not patient data."""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research_models.common_private_synergy.full_slot_engine import run_full


def main():
    out = ROOT / 'local_evidence/full_slot_matched_synthetic_20261111_01'
    resume = sys.argv[1:] == ['--continue-joint']
    if sys.argv[1:] and not resume:
        raise ValueError('only --continue-joint is supported')
    out.mkdir(parents=True, exist_ok=resume)
    prepared = out / 'invented'
    (prepared / 'prepared/cache').mkdir(parents=True, exist_ok=resume)
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
        item = {'patient_id': pid, 'patches': patches, 'omics': omics}
        if resume:
            prior = torch.load(prepared / rel, map_location='cpu', weights_only=True)
            if prior['patient_id'] != pid or not torch.equal(prior['patches'], patches) or any(not torch.equal(a, b) for a, b in zip(prior['omics'], omics)):
                raise RuntimeError('invented cache does not reproduce preregistered generator')
        else:
            torch.save(item, prepared / rel)
        manifest['patients'][pid] = {'cache_path': rel, 'time': t, 'censor': censor}
        manifest['folds']['0']['train' if i < 48 else 'val'].append(pid)
    manifest_path = prepared / 'prepared/manifest.json'
    if resume:
        if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise RuntimeError('invented manifest does not reproduce preregistered generator')
    else:
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    c = {'fold': 0, 'seed': 3, 'device': 'cpu', 'torch_threads': 1, 'target_dim': 3,
         'common_rank': 1, 'allocation': {'p': [2, 2], 'g': [2, 2]}, 'width': 8,
         'iterations': 2, 'hidden': 8, 'bins': 4, 'readout_bound': 2., 'batch_size': 4,
         'patches': 16, 'role_epochs': 6, 'task_epochs': 12, 'role_lr': .003,
         'task_lr': .001, 'role_weight': 1., 'patience': 6}
    paths = ['research_models/common_private_synergy/synthetic_full_slot_matched.py',
             'research_models/common_private_synergy/full_slot_engine.py',
             'research_models/common_private_synergy/full_slot_epoch.py',
             'research_models/common_private_synergy/full_slot_diagnostics.py']
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    if resume:
        receipt = json.loads((out / 'frozen/receipt.json').read_text(encoding='utf-8'))
        if receipt['status'] != 'completed' or receipt['config'] != {**c, 'task_mode': 'frozen'}:
            raise RuntimeError('completed matched frozen arm required')
        if receipt['prepared_manifest_SHA256'] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise RuntimeError('frozen arm manifest changed')
        for path in paths[1:]:
            if receipt['source_SHA256'][path] != hashes[path]:
                raise RuntimeError('engine or diagnostics changed since frozen arm')
    for mode in (('joint',) if resume else ('frozen', 'joint')):
        result = run_full(prepared, out / mode, manifest, {**c, 'task_mode': mode}, hashes, synthetic=True)
        print(json.dumps({'arm': mode, 'best': result['best'], 'last': result['last'],
                          'role_parameters_changed': result['role_parameters_changed_during_task']}), flush=True)


if __name__ == '__main__':
    main()
