"""Run one locked CPU-only role/phi/scale factorial readout diagnostic."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_xor_readout_capacity import network
from scripts.compare_common_private_club import state_hash
from scripts.compare_crossmodal_shared_bottleneck import paired, explicit_noise
from slotspe_proof.common_private_survival import survival_nll
from slotspe_proof.controlled_slot_readout import ControlledSlotReadout
from slotspe_proof.slot_pooling_diagnostics import code_inputs, code_bayes_logits

DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(seed, condition, shape, role, scale):
    model = ControlledSlotReadout(network(seed), phi_shape=shape, role_access=role, scale=scale, seed=900000+seed)
    initial = state_hash(model)
    assert all(name.startswith(('head.', 'phi.', 'role_offsets.')) for name, _ in model.named_parameters())
    params = list(model.parameters())
    assert sum(p.numel() for p in params) == (1396 if shape == 'full' else 1402)
    optimizer = torch.optim.Adam(params, lr=.003)
    train = paired(310000+seed, 512, 'private_xor')
    batch = torch.Generator().manual_seed(320000+seed)
    noise_generator = torch.Generator().manual_seed(330000+seed)
    norms = []; trace = []; parameter_max = 0.; start = time.perf_counter()
    for step in range(500):
        idx = torch.randperm(512, generator=batch)[:32]
        noise = explicit_noise(noise_generator, 32)
        g, v = code_inputs(train['shared_factor'][idx], train['private_bits'][idx], noise['private_noise'], condition)
        loss = survival_nll(model(g, v), train['time'][idx], train['event'][idx])
        if not torch.isfinite(loss): raise RuntimeError('nonfinite synthetic loss')
        optimizer.zero_grad(); loss.backward()
        gradient_norm = math.sqrt(sum(float(p.grad.square().sum()) for p in params if p.grad is not None))
        if not math.isfinite(gradient_norm): raise RuntimeError('nonfinite readout gradient')
        norms.append(gradient_norm); optimizer.step()
        parameter_max = max(parameter_max, max(float(p.detach().abs().max()) for p in params))
        if step in [0, 99, 299, 499]:
            trace.append({'step': step, 'nll': float(loss.detach()), 'pre_update_gradient_l2': gradient_norm})
    test = paired(360000+seed, 384, 'private_xor')
    generator = torch.Generator().manual_seed(370000+seed)
    for n in [192, 128, 384]: explicit_noise(generator, n)
    donor = torch.randperm(384, generator=torch.Generator().manual_seed(410000+seed))
    permutation = torch.tensor([2, 0, 3, 1])
    scores = {}; bayes = []
    with torch.no_grad():
        for _ in range(5):
            noise = explicit_noise(generator, 384)
            g, v = code_inputs(test['shared_factor'], test['private_bits'], noise['private_noise'], condition)
            variants = {'own': v, 'private_donor': [x[donor] for x in v],
                        'private_zero': [torch.zeros_like(x) for x in v],
                        'value_only_role_permutation': [x[:, permutation] for x in v]}
            for name, values in variants.items():
                scores.setdefault(name, []).append(float(survival_nll(model(g, values), test['time'], test['event'])))
            bayes.append(float(survival_nll(code_bayes_logits(g, v, condition), test['time'], test['event'])))
    own = sum(scores['own'])/5
    return {'seed': seed, 'condition': condition, 'phi_shape': shape, 'role_access': role, 'scale': scale,
            'initial_state_hash': initial, 'parameters': sum(p.numel() for p in params),
            'steps': 500, 'training_seconds': time.perf_counter()-start, 'trace': trace,
            'expected_nll': own, 'conditional_code_bayes_nll': sum(bayes)/5,
            'per_draw_nll': scores['own'],
            'fixed_head_nll_deltas': {k: sum(v)/5-own for k, v in scores.items() if k != 'own'},
            'max_gradient_l2': max(norms), 'last100_mean_gradient_l2': sum(norms[-100:])/100,
            'max_parameter_abs': parameter_max, 'finite_optimization': True, 'encoder_parameters_in_optimizer': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output-root', default='local_evidence/controlled_slot_readout_20261102')
    args = parser.parse_args()
    plan_path = DOC/'controlled_readout_plan_20261102.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if not args.run:
        print(json.dumps(plan, indent=2)); return
    destination = (ROOT/args.output_root).resolve()
    if not destination.is_relative_to(ROOT/'local_evidence') or destination == ROOT/'local_evidence' or destination.exists():
        raise ValueError('fresh ignored local_evidence child required')
    torch.set_num_threads(2)
    old_path = DOC/'slot_pooling_rate_neural_receipt.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    assert old['status'] == 'completed'
    files = ['scripts/check_controlled_slot_readout.py', 'slotspe_proof/controlled_slot_readout.py',
             'scripts/check_xor_readout_capacity.py', 'scripts/compare_crossmodal_shared_bottleneck.py',
             'scripts/compare_common_private_club.py', 'slotspe_proof/common_private_survival.py',
             'slotspe_proof/crossmodal_shared_bottleneck.py', 'slotspe_proof/slot_pooling_diagnostics.py']
    payload = {'status': 'running', 'plan': plan, 'plan_sha256': sha(plan_path),
               'source_sha256_before_execution': {p: sha(ROOT/p) for p in files},
               'old_receipt_sha256': sha(old_path), 'rows': [], 'verified_old_repeats': 0,
               'patient_execution': False, 'server_execution': False}
    destination.mkdir(parents=True); output = destination/'receipt.json'
    def save(): output.write_bytes((json.dumps(payload, indent=2)+'\n').encode('utf-8'))
    save()
    try:
        for seed in plan['seeds']:
            for condition in plan['conditions']:
                for shape in plan['phi_shapes']:
                    shared_hash = None
                    for role in plan['role_access']:
                        for scale in plan['scales']:
                            row = run(seed, condition, shape, role, scale)
                            if shared_hash is None: shared_hash = row['initial_state_hash']
                            assert shared_hash == row['initial_state_hash']
                            if (shape, role, scale) == ('full', 'blind', 'mean'):
                                previous = next(r for r in old['rows'] if (r['seed'], r['condition'], r['readout']) == (seed, condition, 'existing_mean'))
                                assert row['expected_nll'] == previous['expected_nll'], 'old repeat drift'
                                payload['verified_old_repeats'] += 1
                            payload['rows'].append(row); save()
                            print(json.dumps({k: row[k] for k in ['seed', 'condition', 'phi_shape', 'role_access', 'scale', 'expected_nll']}), flush=True)
        assert len(payload['rows']) == 48 and payload['verified_old_repeats'] == 6
        assert all(sha(ROOT/p) == h for p, h in payload['source_sha256_before_execution'].items())
        payload['status'] = 'completed'; save()
    except Exception as exc:
        payload.update(status='failed', error=repr(exc)); save(); raise


if __name__ == '__main__': main()
