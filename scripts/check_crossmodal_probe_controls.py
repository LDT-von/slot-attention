"""No retraining: same-width planted-XOR/row-mismatch probe controls."""
import hashlib
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slotspe_proof.incremental_probe import incremental_probe

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def main():
    torch.set_num_threads(2)
    output = DOC/'crossmodal_conditional_probe_controls_receipt.json'
    if output.exists(): raise RuntimeError('preserve existing controls')
    source = json.loads((DOC/'crossmodal_conditional_rate_neural_receipt.json').read_text())
    assert source['status'] == 'completed'
    controls = []
    for row in source['rows']:
        if row['task'] != 'private_xor' or row['private_weight'] != .3: continue
        path = (ROOT/row['synthetic_cache_path']).resolve()
        assert path.is_relative_to(ROOT/'local_evidence/synthetic_crossmodal_rate_20261102')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['synthetic_cache_sha256']
        pack = torch.load(path, map_location='cpu', weights_only=True)
        assert pack['synthetic_only'] is True
        frozen = pack['frozen']; positive = []; mismatch = []
        random = torch.Generator().manual_seed(380000+row['seed'])
        for p in frozen:
            width = sum(s.shape[-1] for s in p['phi_sample'])
            assert width == 32
            # Same input width, a known low-rank code. No claim of matching
            # unknown effective complexity or nuisance of learned features.
            oracle = p['private_bits'].repeat_interleave(width//2, dim=-1)
            positive.append((p['common'], oracle, p['risk']))
            wrong = oracle[torch.randperm(len(oracle), generator=random)]
            mismatch.append((p['common'], wrong, p['risk']))
        upper = incremental_probe(*positive); wrong = incremental_probe(*mismatch)
        controls.append({'seed': row['seed'], 'shared_weight': row['shared_weight'],
            'private_weight': row['private_weight'], 'feature_width': 32,
            'oracle_tiled_xor_probe': upper, 'mismatched_oracle_probe': wrong,
            'learned_actual_pooled_private_probe': row['actual_pooled_private_risk_increment_probe']})
    report = {'status': 'completed', 'control_pairs': len(controls), 'rows': controls,
        'positive_min_relative_gain': min(r['oracle_tiled_xor_probe']['gain_relative_to_fit_mean_baseline'] for r in controls),
        'mismatch_max_relative_gain': max(r['mismatched_oracle_probe']['gain_relative_to_fit_mean_baseline'] for r in controls),
        'neural_training': 0, 'patient_execution': False,
        'scope': 'posthoc instrument validation on exactly the cached fit/validation/test rows; same width and hyperparameter grid but low-rank oracle lacks learned nuisance. Negative learned probes remain non-detection, not absence or a whole-network synergy impossibility.'}
    output.write_bytes((json.dumps(report, indent=2)+'\n').encode())
    print(json.dumps({k: report[k] for k in ['control_pairs', 'positive_min_relative_gain', 'mismatch_max_relative_gain', 'neural_training']}))


if __name__ == '__main__': main()
