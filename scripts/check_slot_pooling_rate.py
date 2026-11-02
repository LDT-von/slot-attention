"""Synthetic-only equal-total-rate pooling controls and missing refit completion."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_xor_readout_capacity import network, train_readout, evaluate, frozen_hash
from scripts.compare_crossmodal_shared_bottleneck import paired, explicit_noise
from slotspe_proof.common_private_survival import survival_nll
from slotspe_proof.slot_pooling_diagnostics import code_inputs, code_bayes_logits, RoleBankReadout
from slotspe_proof.xor_readout_diagnostics import configure_readout_training

DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readout_run(seed, condition, readout):
    reference = network(seed)
    params = configure_readout_training(reference)
    initial_frozen = frozen_hash(reference)
    bank = RoleBankReadout(reference, 900000+seed) if readout == 'role_bank' else None
    if bank is not None:
        params = list(bank.parameters())
    forward = reference.readout if bank is None else bank
    expected_count = 1396 if bank is None else 1402
    assert sum(p.numel() for p in params) == expected_count
    optimizer = torch.optim.Adam(params, lr=.003)
    train = paired(310000+seed, 512, 'private_xor')
    batch = torch.Generator().manual_seed(320000+seed)
    random = torch.Generator().manual_seed(330000+seed)
    start = time.perf_counter(); trace = []
    for step in range(500):
        idx = torch.randperm(512, generator=batch)[:32]
        noise = explicit_noise(random, 32)
        g, v = code_inputs(train['shared_factor'][idx], train['private_bits'][idx], noise['private_noise'], condition)
        loss = survival_nll(forward(g, v), train['time'][idx], train['event'][idx])
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite synthetic readout loss')
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step in [0, 99, 299, 499]:
            trace.append({'step': step, 'nll': float(loss.detach())})
    assert frozen_hash(reference) == initial_frozen
    test = paired(360000+seed, 384, 'private_xor')
    random = torch.Generator().manual_seed(370000+seed)
    for n in [192, 128, 384]:
        explicit_noise(random, n)
    scores, bayes, zero = [], [], []
    with torch.no_grad():
        for _ in range(5):
            noise = explicit_noise(random, 384)
            g, v = code_inputs(test['shared_factor'], test['private_bits'], noise['private_noise'], condition)
            scores.append(float(survival_nll(forward(g, v), test['time'], test['event'])))
            bayes.append(float(survival_nll(code_bayes_logits(g, v, condition), test['time'], test['event'])))
            zero.append(float(survival_nll(forward(g, [torch.zeros_like(x) for x in v]), test['time'], test['event'])))
    return {'seed': seed, 'condition': condition, 'readout': readout,
            'parameters': expected_count, 'steps': 500, 'trace': trace,
            'seconds': time.perf_counter()-start, 'frozen_encoder_unchanged': True,
            'expected_nll': sum(scores)/5, 'conditional_code_bayes_nll': sum(bayes)/5,
            'nll_by_noise_draw': scores, 'all_private_zero_nll_delta': sum(zero)/5-sum(scores)/5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Execute synthetic CPU controls only')
    parser.add_argument('--output-root', default='local_evidence/slot_pooling_rate_20261102')
    args = parser.parse_args()
    plan_path = DOC/'slot_pooling_rate_plan_20261102.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if not args.run:
        print(json.dumps(plan, indent=2)); return
    destination = (ROOT/args.output_root).resolve()
    if not destination.is_relative_to(ROOT/'local_evidence') or destination == ROOT/'local_evidence':
        raise ValueError('fresh child of this checkout local_evidence required')
    if destination.exists():
        raise RuntimeError('preserve historical output; choose a fresh output root')
    torch.set_num_threads(2)
    old_path = DOC/'xor_readout_capacity_receipt.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    replay = next(r for r in old['replay_rows'] if r['seed'] == 449)
    state_path = (ROOT/replay['model_state_path']).resolve()
    if not state_path.is_relative_to(ROOT/'local_evidence/synthetic_xor_readout_20261102'):
        raise ValueError('only exact known synthetic replay is permitted')
    assert digest(state_path) == replay['model_state_sha256']
    paths = ['scripts/check_slot_pooling_rate.py', 'slotspe_proof/slot_pooling_diagnostics.py',
             'scripts/check_xor_readout_capacity.py', 'slotspe_proof/xor_readout_diagnostics.py',
             'slotspe_proof/common_private_survival.py', 'slotspe_proof/crossmodal_shared_bottleneck.py',
             'scripts/compare_crossmodal_shared_bottleneck.py']
    payload = {'status': 'running', 'plan': plan, 'plan_sha256': digest(plan_path),
               'historical_partial_receipt_sha256': digest(old_path),
               'source_sha256_before_execution': {p: digest(ROOT/p) for p in paths},
               'rows': [], 'missing_refit': None, 'patient_execution': False, 'server_execution': False}
    destination.mkdir(parents=True)
    output = destination/'receipt.json'
    def save():
        output.write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    save()
    try:
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        assert state['synthetic_only'] is True and state['seed'] == 449
        reference = network(449); reference.load_state_dict(state['state_dict'])
        before = evaluate(reference, 449, 'learned', original_expected=replay['expected_nll'])
        reference, training = train_readout(449, 'learned', reference)
        payload['missing_refit'] = {'seed': 449, 'verified_input_sha256': digest(state_path),
                                   'before': before, **training, **evaluate(reference, 449, 'learned')}
        save(); print(json.dumps({'missing_refit': 'completed', 'nll': payload['missing_refit']['expected_nll']}), flush=True)
        for seed in plan['seeds']:
            for condition in plan['conditions']:
                for readout in plan['readouts']:
                    row = readout_run(seed, condition, readout)
                    payload['rows'].append(row); save()
                    print(json.dumps({k: row[k] for k in ['seed', 'condition', 'readout', 'expected_nll', 'conditional_code_bayes_nll']}), flush=True)
        assert digest(old_path) == payload['historical_partial_receipt_sha256']
        assert all(digest(ROOT/p) == h for p, h in payload['source_sha256_before_execution'].items())
        payload['status'] = 'completed'; save()
    except Exception as exc:
        payload.update(status='failed', error=repr(exc)); save(); raise


if __name__ == '__main__':
    main()
