"""Three fixed fresh synthetic cohorts; paired arms, no parameter search."""
import hashlib
import json
from pathlib import Path

import torch

from .full_slot_engine import run_full
from .patient_pilot import ROOT


def main():
    root = ROOT / 'local_evidence/full_slot_synthetic_replication_20261111_01'
    root.mkdir(parents=True, exist_ok=False)
    seeds = (71212, 71213, 71214)
    c = {'fold': 0, 'seed': 3, 'device': 'cpu', 'torch_threads': 1, 'target_dim': 3,
         'common_rank': 1, 'allocation': {'p': [2, 2], 'g': [2, 2]}, 'width': 8,
         'iterations': 2, 'hidden': 8, 'bins': 4, 'readout_bound': 2., 'batch_size': 4,
         'patches': 16, 'role_epochs': 6, 'task_epochs': 12, 'role_lr': .003,
         'task_lr': .001, 'role_weight': 1., 'patience': 6}
    paths = ['research_models/common_private_synergy/synthetic_full_slot_replication.py',
             'research_models/common_private_synergy/full_slot_engine.py',
             'research_models/common_private_synergy/full_slot_epoch.py',
             'research_models/common_private_synergy/full_slot_diagnostics.py']
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    for seed in seeds:
        case = root / str(seed)
        prepared = case / 'invented'
        (prepared / 'prepared/cache').mkdir(parents=True)
        gen = torch.Generator().manual_seed(seed)
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
            t = float(10 * torch.exp(-.8 * z + .25 * torch.randn((), generator=gen)))
            censor = float(torch.rand((), generator=gen) < .25)
            rel = f'prepared/cache/{pid}.pt'
            torch.save({'patient_id': pid, 'patches': patches, 'omics': omics}, prepared / rel)
            manifest['patients'][pid] = {'cache_path': rel, 'time': t, 'censor': censor}
            manifest['folds']['0']['train' if i < 48 else 'val'].append(pid)
        (prepared / 'prepared/manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        for mode in ('frozen', 'joint'):
            result = run_full(prepared, case / mode, manifest, {**c, 'task_mode': mode}, hashes, synthetic=True)
            print(json.dumps({'data_seed': seed, 'arm': mode, 'best': result['best'],
                              'last': result['last']}), flush=True)


if __name__ == '__main__':
    main()
