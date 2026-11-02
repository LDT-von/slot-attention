"""Oracle readout capacity, exact synthetic replay, and fixed-head sensitivity."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_crossmodal_shared_bottleneck import paired, explicit_noise
from scripts.compare_common_private_club import state_hash
from slotspe_proof.crossmodal_shared_bottleneck import CrossmodalSharedBottleneck, shared_private_objective
from slotspe_proof.common_private_survival import survival_nll
from slotspe_proof.xor_readout_diagnostics import oracle_inputs, oracle_forward, oracle_bayes_logits, mixture_survival_logits, configure_readout_training

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def network(seed):
    torch.default_generator.manual_seed(seed)
    return CrossmodalSharedBottleneck(8, 8, width=16, slots=4, common_dim=4, private_dim=4)


def frozen_hash(net):
    h = hashlib.sha256()
    for name, value in net.state_dict().items():
        if not name.startswith(('head.', 'phi.', 'role_offsets.')):
            h.update(name.encode()); h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def equal_tensors(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), 'replay differs from original synthetic cache'
        return 1
    if isinstance(a, list):
        assert len(a) == len(b)
        return sum(equal_tensors(x, y) for x, y in zip(a, b))
    raise TypeError(type(a))


def replay(seed, original, cache):
    net = network(seed); assert state_hash(net) == original['initial_hash']
    optimizer = torch.optim.Adam(net.parameters(), lr=.003)
    train = paired(310000+seed, 512, 'private_xor')
    batch = torch.Generator().manual_seed(320000+seed); noise = torch.Generator().manual_seed(330000+seed)
    trace = {r['step']: r['loss'] for r in original['trace']}
    for step in range(500):
        idx = torch.randperm(512, generator=batch)[:32]
        out = net(*[x[idx] for x in train['xs']], **explicit_noise(noise, 32))
        loss = shared_private_objective(out, train['time'][idx], train['event'][idx], shared_weight=.1, private_weight=.3)['loss']
        if step in trace: assert float(loss.detach()) == trace[step], 'replay loss trace mismatch'
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    net.eval(); path = (ROOT/original['synthetic_cache_path']).resolve()
    assert path.is_relative_to(ROOT/'local_evidence/synthetic_crossmodal_rate_20261102')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original['synthetic_cache_sha256']
    old = torch.load(path, map_location='cpu', weights_only=True); assert old['synthetic_only'] is True
    random = torch.Generator().manual_seed(370000+seed); matches = 0
    with torch.no_grad():
        for i, (offset, n) in enumerate([(340000, 192), (350000, 128), (360000, 384)]):
            pack = paired(offset+seed, n, 'private_xor'); out = net(*pack['xs'], **explicit_noise(random, n))
            values = {'common': out['common'], 'common_mean': out['common_mean'], 'private': out['private'],
                'private_mean': out['private_mean'], 'shared_factor': pack['shared_factor'], 'private_bits': pack['private_bits'], 'risk': pack['risk'],
                'phi_sample': [f(v+e[None]) for f, v, e in zip(net.phi, out['private'], net.role_offsets)],
                'phi_mean': [f(v+e[None]) for f, v, e in zip(net.phi, out['private_mean'], net.role_offsets)]}
            assert set(values) == set(old['frozen'][i])
            matches += sum(equal_tensors(value, old['frozen'][i][key]) for key, value in values.items())
    saved = cache/f'original_replay_{seed}.pt'
    torch.save({'synthetic_only': True, 'state_dict': net.state_dict(), 'seed': seed}, saved)
    return net, {'seed': seed, 'replay_not_new_independent_training': True, 'cache_tensor_equalities': matches,
        'trace_equalities': len(trace), 'model_state_path': str(saved.relative_to(ROOT)),
        'model_state_sha256': hashlib.sha256(saved.read_bytes()).hexdigest()}


def evaluate(net, seed, mode, original_expected=None):
    pack = paired(360000+seed, 384, 'private_xor')
    random = torch.Generator().manual_seed(370000+seed)
    for n in [192, 128, 384]: explicit_noise(random, n)  # exact prior evaluation stream
    perm = torch.randperm(384, generator=torch.Generator().manual_seed(410000+seed))
    mismatch_perm = torch.randperm(384, generator=torch.Generator().manual_seed(420000+seed))
    scores = {}; bayes = []; per_slot = [[[] for _ in range(4)] for _ in range(2)]
    with torch.no_grad():
        for _ in range(5):
            noise = explicit_noise(random, 384)
            if mode == 'learned':
                out = net(*pack['xs'], **noise); g, vs = out['common'], out['private']
                forward = net.readout
            else:
                bits = pack['private_bits'][mismatch_perm] if mode == 'before_phi_mismatched_repeated' else pack['private_bits']
                g, vs = oracle_inputs(pack['shared_factor'], bits, noise['private_noise'], mode)
                forward = lambda cg, vv: oracle_forward(net, cg, vv, mode)
                bayes.append(float(survival_nll(oracle_bayes_logits(g, vs, mode), pack['time'], pack['event'])))
            variants = {'own': (g, vs), 'all_private_prior_N0': (g, noise['private_noise']),
                'all_private_zero': (g, [torch.zeros_like(v) for v in vs]),
                'private_donor_permutation': (g, [v[perm] for v in vs]),
                'common_donor_permutation': (g[perm], vs), 'common_zero': (torch.zeros_like(g), vs)}
            for name, (cg, vv) in variants.items():
                scores.setdefault(name, []).append(float(survival_nll(forward(cg, vv), pack['time'], pack['event'])))
            for j in range(2):
                for k in range(4):
                    replaced = [v.clone() for v in vs]; replaced[j][:, k] = noise['private_noise'][j][:, k]
                    per_slot[j][k].append(float(survival_nll(forward(g, replaced), pack['time'], pack['event'])))
        own = sum(scores['own'])/5
        if original_expected is not None: assert own == original_expected, 'replay expected NLL mismatch'
        full_bayes = float(survival_nll(pack['risk'][:, None]+torch.tensor([-2., -1., 0., 1.]), pack['time'], pack['event']))
        shared_bayes = float(survival_nll(mixture_survival_logits(pack['shared_factor'], torch.full((384,), .5)), pack['time'], pack['event']))
    return {'expected_nll': own, 'conditional_code_bayes_nll': sum(bayes)/5 if bayes else None,
        'full_planted_factor_bayes_nll': full_bayes, 'shared_only_mixture_bayes_nll': shared_bayes,
        'forward_perturbation_nll_deltas': {key: sum(value)/5-own for key, value in scores.items() if key != 'own'},
        'one_slot_prior_nll_deltas': [[sum(v)/5-own for v in group] for group in per_slot],
        'interpretation': 'paired fixed-readout sensitivity, potentially off manifold; not patient causality or unique slot information'}


def train_readout(seed, mode, net=None):
    net = network(seed) if net is None else net
    train = paired(310000+seed, 512, 'private_xor')
    params = configure_readout_training(net, head_only=(mode == 'after_phi_clean'))
    initial_frozen = frozen_hash(net)
    optimizer = torch.optim.Adam(params, lr=.003)
    batch = torch.Generator().manual_seed(320000+seed); random = torch.Generator().manual_seed(330000+seed)
    mismatch = torch.randperm(512, generator=torch.Generator().manual_seed(430000+seed))
    net.eval(); trace = []; start = time.perf_counter()
    for step in range(500):
        idx = torch.randperm(512, generator=batch)[:32]; noise = explicit_noise(random, 32)
        if mode == 'learned':
            with torch.no_grad(): out = net(*[x[idx] for x in train['xs']], **noise)
            logits = net.readout(out['common'], out['private'])
        else:
            source_idx = mismatch[idx] if mode == 'before_phi_mismatched_repeated' else idx
            g, vs = oracle_inputs(train['shared_factor'][idx], train['private_bits'][source_idx], noise['private_noise'], mode)
            logits = oracle_forward(net, g, vs, mode)
        loss = survival_nll(logits, train['time'][idx], train['event'][idx])
        if not torch.isfinite(loss): raise RuntimeError('nonfinite readout loss')
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step in [0, 99, 299, 499]: trace.append({'step': step, 'nll': float(loss.detach())})
    assert frozen_hash(net) == initial_frozen, 'a frozen encoder/posterior changed'
    return net, {'trainable_readout_parameters': sum(p.numel() for p in params), 'trace': trace,
                 'training_seconds': time.perf_counter()-start, 'frozen_parameters_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Print the fixed synthetic plan without training')
    parser.add_argument('--output-root', help='Fresh ignored local_evidence directory; preserves historical receipts')
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan_path = DOC/'xor_readout_capacity_plan_20261102.json'; plan = json.loads(plan_path.read_text())
    if args.plan:
        print(json.dumps(plan, indent=2)); return
    original_path = DOC/'crossmodal_conditional_rate_neural_receipt.json'; original = json.loads(original_path.read_text())
    assert original['status'] == 'completed'
    for path, digest in original['source_sha256_before_execution'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    output = DOC/'xor_readout_capacity_receipt.json'; cache = ROOT/'local_evidence/synthetic_xor_readout_20261102'
    if args.output_root:
        destination = Path(args.output_root).resolve()
        if not destination.is_relative_to(ROOT/'local_evidence') or destination == ROOT/'local_evidence':
            raise ValueError('output root must be a child of this checkout local_evidence')
        if destination.exists(): raise RuntimeError('fresh output directory required')
        output = destination/'receipt.json'; cache = destination/'synthetic_cache'
    if output.exists() or cache.exists(): raise RuntimeError('preserve existing artifacts')
    files = ['scripts/check_xor_readout_capacity.py', 'slotspe_proof/xor_readout_diagnostics.py']
    payload = {'status': 'running', 'plan': plan, 'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'original_receipt_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest(),
        'source_sha256_before_execution': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        'oracle_rows': [], 'replay_rows': [], 'decoder_refit_rows': [], 'patient_execution': False, 'server_execution': False}
    cache.mkdir(parents=True)
    def save(): output.write_bytes((json.dumps(payload, indent=2)+'\n').encode())
    save()
    try:
        for seed in plan['seeds']:
            for mode in plan['oracle_conditions']:
                net, training = train_readout(seed, mode)
                row = {'seed': seed, 'mode': mode, **training, **evaluate(net, seed, mode)}
                payload['oracle_rows'].append(row); save()
                print(json.dumps({'seed': seed, 'mode': mode, 'nll': row['expected_nll'], 'bayes': row['conditional_code_bayes_nll']}), flush=True)
            reference = next(r for r in original['rows'] if (r['task'], r['seed'], r['shared_weight'], r['private_weight']) == ('private_xor', seed, .1, .3))
            net, proof = replay(seed, reference, cache)
            proof.update(evaluate(net, seed, 'learned', original_expected=reference['test_expected_nll_single_draw_average']))
            payload['replay_rows'].append(proof); save()
            net, training = train_readout(seed, 'learned', net)
            row = {'seed': seed, 'mode': 'learned_frozen_encoder_decoder_warmstart_500_extra_steps', **training, **evaluate(net, seed, 'learned')}
            payload['decoder_refit_rows'].append(row); save()
            print(json.dumps({'seed': seed, 'replay_match': True, 'refit_nll': row['expected_nll']}), flush=True)
    except Exception as exc:
        payload.update(status='failed', error=repr(exc)); save(); raise
    payload['status'] = 'completed'; save()


if __name__ == '__main__': main()
