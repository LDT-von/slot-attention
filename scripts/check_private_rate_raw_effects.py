"""Posthoc localization: raw-slot predictive effects, with no neural refits."""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slotspe_proof.incremental_probe import incremental_probe
from slotspe_proof.prognostic_effect_probe import effects

DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    parent = DOC/'private_rate_multifactor_neural_receipt.json'
    source = json.loads(parent.read_text(encoding='utf-8'))
    assert source['status'] == 'completed'
    output = DOC/'private_rate_raw_effect_receipt.json'
    if output.exists(): raise RuntimeError('preserve earlier localization receipt')
    payload = {'status': 'running', 'parent_receipt_sha256': sha(parent),
               'scope': 'Posthoc diagnostic after inspecting pooled effect results. Same frozen partitions and existing kernel grid; no model selection or new neural training.',
               'source_sha256': {'scripts/check_private_rate_raw_effects.py': sha(Path(__file__)),
                                 'slotspe_proof/prognostic_effect_probe.py': sha(ROOT/'slotspe_proof/prognostic_effect_probe.py')},
               'rows': [], 'new_neural_trainings': 0, 'patient_execution': False, 'server_execution': False}
    def save(): output.write_bytes((json.dumps(payload, indent=2)+'\n').encode())
    save()
    for row in source['rows']:
        if row['task'] == 'factorwise':
            values = {mode: row['diagnostics'][mode]['raw_private_factor_increment'] for mode in ['sample', 'all_mean']}
            reused = True
        else:
            path = (ROOT/row['synthetic_cache_path']).resolve()
            assert path.is_relative_to(ROOT/'local_evidence/private_rate_multifactor_20261102/synthetic_cache')
            assert sha(path) == row['synthetic_cache_sha256']
            cached = torch.load(path, map_location='cpu', weights_only=True)
            assert cached['synthetic_only'] is True
            values = {mode: incremental_probe(*[(r[mode]['common'], torch.cat([v.flatten(1) for v in r[mode]['private']], -1), effects(r['factors'], row['task'])) for r in cached['frozen']])
                      for mode in ['sample', 'all_mean']}
            reused = False
        payload['rows'].append({'task': row['task'], 'seed': row['seed'], 'arm': row['arm']['name'], 'identical_factor_target_reused': reused, 'raw_effect_increment': values}); save()
    payload['status'] = 'completed'; save()
    for task in source['plan']['tasks']:
        for arm in source['plan']['arms']:
            rs = [r for r in payload['rows'] if (r['task'], r['arm']) == (task, arm['name'])]
            print(json.dumps({'task': task, 'arm': arm['name'], **{mode: sum(r['raw_effect_increment'][mode]['gain_relative_to_fit_mean_baseline'] for r in rs)/3 for mode in ['sample', 'all_mean']}}))


if __name__ == '__main__': main()
