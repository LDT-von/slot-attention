"""Post-training clarified effect probes on hash-verified synthetic caches."""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slotspe_proof.multifactor_survival_synthetic import generate
from slotspe_proof.slot_increment_probe import slot_increment_probe
from slotspe_proof.prognostic_effect_probe import effects, effect_probe

DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def controls():
    packs = [generate(offset+653, n, 'paired_xor') for offset, n in [(840000, 192), (850000, 128), (860000, 384)]]
    result = {}
    for mode in ['role_placed', 'duplicate', 'shuffled']:
        parts = []
        for i, pack in enumerate(packs):
            n = len(pack['shared']); g = torch.zeros(n, 4); g[:, 0] = pack['shared']
            groups = [torch.zeros(n, 2, 16) for _ in range(2)]
            for j in range(2):
                if mode == 'duplicate': groups[j][:, :, :2] = pack['factors'][:, None, 2*j:2*j+2]
                else:
                    groups[j][:, 0, 0] = pack['factors'][:, 2*j]
                    groups[j][:, 1, 1] = pack['factors'][:, 2*j+1]
            target = effects(pack['factors'], 'paired_xor')
            if mode == 'shuffled': target = target[torch.randperm(n, generator=torch.Generator().manual_seed(910000+i))]
            parts.append((g, groups, target))
        result[mode] = slot_increment_probe(*parts)
    gains = {k: v['gain_relative_to_fit_mean_baseline'] for k, v in result.items()}
    assert gains['role_placed'] > .25 and abs(gains['duplicate']) < 1e-10 and abs(gains['shuffled']) < .05
    return {'passed': True, 'gains': gains, 'details': result}


def main():
    torch.set_num_threads(2)
    original_path = ROOT/'local_evidence/private_rate_multifactor_20261102/receipt.json'
    original = json.loads(original_path.read_text(encoding='utf-8'))
    assert original['status'] == 'completed' and len(original['rows']) == 24
    for p, h in original['source_sha256_before_execution'].items(): assert sha(ROOT/p) == h
    output = ROOT/'local_evidence/private_rate_multifactor_20261102/effect_probe_receipt.json'
    if output.exists(): raise RuntimeError('preserve old effect receipt')
    clarification = DOC/'private_rate_target_clarification_20261102.json'
    files = ['scripts/check_private_rate_effects.py', 'slotspe_proof/prognostic_effect_probe.py']
    payload = {'status': 'running', 'original_receipt_sha256': sha(original_path),
               'clarification_sha256': sha(clarification), 'source_sha256': {p: sha(ROOT/p) for p in files},
               'instrument': None, 'rows': [], 'new_neural_trainings': 0, 'patient_execution': False}
    def save(): output.write_bytes((json.dumps(payload, indent=2)+'\n').encode())
    save()
    payload['instrument'] = controls(); save()
    print(json.dumps({'effect_instrument': payload['instrument']['gains']}), flush=True)
    for row in original['rows']:
        if row['task'] == 'factorwise':
            diagnostic = {mode: row['diagnostics'][mode]['all_vs_single_factor_probe'] for mode in ['sample', 'all_mean']}
            reused = True
        else:
            path = (ROOT/row['synthetic_cache_path']).resolve()
            assert path.is_relative_to(ROOT/'local_evidence/private_rate_multifactor_20261102/synthetic_cache')
            assert sha(path) == row['synthetic_cache_sha256']
            cache = torch.load(path, map_location='cpu', weights_only=True)
            assert cache['synthetic_only'] is True
            diagnostic = effect_probe(cache['frozen'], row['task']); reused = False
        accepted = {'task': row['task'], 'seed': row['seed'], 'arm': row['arm']['name'],
                    'factor_diagnostics_reused_identical_target': reused, 'effects': diagnostic}
        payload['rows'].append(accepted); save()
    assert sha(original_path) == payload['original_receipt_sha256']
    payload['status'] = 'completed'; save()
    print(json.dumps({'completed': len(payload['rows']), 'new_neural_trainings': 0}), flush=True)


if __name__ == '__main__': main()
