"""Locked 48-run shared-rate x private-budget synthetic ablation."""
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_frozen_content_targets import generated
from scripts.compare_common_private_club import state_hash
from slotspe_proof.crossmodal_shared_bottleneck import CrossmodalSharedBottleneck, shared_private_objective
from slotspe_proof.common_private_survival import survival_nll
from slotspe_proof.common_private_diagnostics import linear_probe
from slotspe_proof.incremental_probe import incremental_probe
from slotspe_proof.slot_increment_probe import slot_increment_probe

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def paired(seed, n, task, shift=False):
    packs = [generated(seed, n, 'separated_informative'), generated(seed+7000000, n, 'separated_informative')]
    rng = torch.Generator().manual_seed(seed+8000000)
    p, q = packs[0]['bits'][:, 0]*2-1, packs[1]['bits'][:, 0]*2-1
    common = packs[0]['common']
    if task == 'shared_only': risk = 2*common
    elif task == 'private_only': risk = 2*p
    elif task == 'private_xor': risk = 3*(p != q).float()-1.5+.3*common
    elif task == 'shared_nuisance_shift':
        risk = 2*p
        common = (-p if shift else p)+.1*torch.randn(n, generator=rng)
    else: raise ValueError(task)
    for pack in packs:
        pack['x'][..., 3] = .3*common[:, None]+.05*torch.randn(n, 12, generator=rng)
    hazards = torch.sigmoid(risk[:, None]+torch.tensor([-2., -1., 0., 1.]))
    hits = torch.rand(n, 4, generator=rng) < hazards
    time_bin = torch.where(hits.any(1), hits.int().argmax(1), 4)
    censor = torch.randint(0, 4, (n,), generator=rng)
    return {'xs': [p['x'] for p in packs], 'shared_factor': common,
        'private_bits': torch.stack([p, q], -1), 'risk': risk,
        'time': torch.minimum(time_bin, censor), 'event': (time_bin <= censor).float()}


def explicit_noise(rng, n):
    return {'shared_noise': [torch.randn(n, 4, generator=rng) for _ in range(2)],
            'private_noise': [torch.randn(n, 4, 4, generator=rng) for _ in range(2)]}


def geometry(groups):
    rows = []
    for x in groups:
        singular = torch.linalg.svdvals(x.double()); total = singular.sum(-1, keepdim=True)
        probability = singular/total.clamp_min(1e-12)
        rank = torch.exp(-(probability*probability.clamp_min(1e-12).log()).sum(-1))
        rank = torch.where(total.squeeze(-1)>1e-12, rank, torch.zeros_like(rank))
        norm = F.normalize(x.double(), dim=-1)
        cosine = norm@norm.transpose(-1, -2)
        pair = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool), diagonal=1)
        rows.append({'uncentered_effective_rank': float(rank.mean()),
                     'pair_cosine': float(cosine[:, pair].mean()),
                     'slot_norm_mean': float(x.norm(dim=-1).mean())})
    return rows


def run(task, seed, shared_weight, private_weight, cache):
    torch.default_generator.manual_seed(seed)
    net = CrossmodalSharedBottleneck(8, 8, width=16, slots=4, common_dim=4, private_dim=4)
    initial = state_hash(net); opt = torch.optim.Adam(net.parameters(), lr=.003)
    train = paired(310000+seed, 512, task)
    batches = torch.Generator().manual_seed(320000+seed); random = torch.Generator().manual_seed(330000+seed)
    grad_max = 0.; grad_sum = 0.; trace = []; start = time.perf_counter()
    for step in range(500):
        idx = torch.randperm(512, generator=batches)[:32]
        out = net(*[x[idx] for x in train['xs']], **explicit_noise(random, 32))
        term = shared_private_objective(out, train['time'][idx], train['event'][idx],
                                       shared_weight=shared_weight, private_weight=private_weight, capacity=1.)
        if not torch.isfinite(term['loss']): raise RuntimeError(f'nonfinite loss at step {step}')
        opt.zero_grad(); term['loss'].backward()
        grad = torch.stack([p.grad.detach().square().sum() for p in net.parameters() if p.grad is not None]).sum().sqrt()
        if not torch.isfinite(grad): raise RuntimeError(f'nonfinite gradient at step {step}')
        grad_max = max(grad_max, float(grad)); grad_sum += float(grad); opt.step()
        if step in [0, 99, 299, 499]:
            trace.append({'step': step, **{k: float(v.detach()) for k, v in term.items()}, 'gradient_norm': float(grad)})
    seconds = time.perf_counter()-start
    net.eval(); packs = [paired(o+seed, n, task, shift=(task=='shared_nuisance_shift' and i==2))
                        for i, (o, n) in enumerate([(340000, 192), (350000, 128), (360000, 384)])]
    frozen = []; eval_rng = torch.Generator().manual_seed(370000+seed)
    with torch.no_grad():
        for pack in packs:
            out = net(*pack['xs'], **explicit_noise(eval_rng, len(pack['risk'])))
            actual = [f(v+e[None]) for f, v, e in zip(net.phi, out['private'], net.role_offsets)]
            means = [f(v+e[None]) for f, v, e in zip(net.phi, out['private_mean'], net.role_offsets)]
            frozen.append({'common': out['common'], 'common_mean': out['common_mean'],
                'private': out['private'], 'private_mean': out['private_mean'], 'phi_sample': actual, 'phi_mean': means,
                'shared_factor': pack['shared_factor'], 'private_bits': pack['private_bits'], 'risk': pack['risk']})
        sample_geometry, mean_geometry = geometry(out['private']), geometry(out['private_mean'])
        deterministic = net(*packs[2]['xs'], sample=False)
        mean_nll = float(survival_nll(deterministic['logits'], packs[2]['time'], packs[2]['event']))
        nll = []; common_rates = []; private_rates = []
        for _ in range(5):
            out = net(*packs[2]['xs'], **explicit_noise(eval_rng, 384))
            nll.append(float(survival_nll(out['logits'], packs[2]['time'], packs[2]['event'])))
            common_rates.append(float(out['shared_rate'].mean()))
            private_rates.append(torch.cat([r.mean(0) for r in out['private_rate_per_slot']]))
    name = f'{task}_{seed}_{int(shared_weight>0)}_{int(private_weight>0)}.pt'
    path = cache/name
    torch.save({'synthetic_only': True, 'frozen': frozen, 'task': task, 'seed': seed,
                'shared_weight': shared_weight, 'private_weight': private_weight}, path)
    shared_factor_probes = {}; private_factor_probes = {}
    for mode in ['common', 'common_mean']:
        shared_factor_probes[mode] = linear_probe(frozen[0][mode], frozen[0]['shared_factor'], frozen[2][mode], frozen[2]['shared_factor'])['test_r2']
        private_factor_probes[mode] = [linear_probe(frozen[0][mode], frozen[0]['private_bits'][:, k],
            frozen[2][mode], frozen[2]['private_bits'][:, k])['test_r2'] for k in range(2)]
    increment = incremental_probe(*[(p['common'], torch.cat([v.mean(1) for v in p['phi_sample']], -1), p['risk']) for p in frozen])
    slot_gain = slot_increment_probe(*[(p['common'], p['phi_mean'], p['risk']) for p in frozen])
    return {'task': task, 'seed': seed, 'shared_weight': shared_weight, 'private_weight': private_weight,
        'initial_hash': initial, 'parameter_count': sum(p.numel() for p in net.parameters()),
        'test_expected_nll_single_draw_average': sum(nll)/5, 'test_posterior_means_nll_diagnostic': mean_nll,
        'test_shared_directional_rate_mean_nats': sum(common_rates)/5,
        'test_private_rate_nats_per_slot': torch.stack(private_rates).mean(0).tolist(),
        'shared_factor_linear_r2': shared_factor_probes, 'private_factor_from_shared_linear_r2': private_factor_probes,
        'private_sample_geometry': sample_geometry, 'private_mean_coordinate_geometry': mean_geometry,
        'actual_pooled_private_risk_increment_probe': increment,
        'mean_coordinate_all_vs_single_probe': slot_gain,
        'gradient_norm_max': grad_max, 'gradient_norm_mean': grad_sum/500, 'trace': trace, 'training_seconds': seconds,
        'synthetic_cache_path': str(path.relative_to(ROOT)), 'synthetic_cache_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    torch.set_num_threads(2)
    plan_path = DOC/'crossmodal_conditional_rate_plan_20261102.json'
    plan = json.loads(plan_path.read_text())
    gate = json.loads((DOC/'crossmodal_conditional_rate_math_receipt.json').read_text())
    assert gate['all_identities_passed']
    output = DOC/'crossmodal_conditional_rate_neural_receipt.json'
    cache = ROOT/'local_evidence/synthetic_crossmodal_rate_20261102'
    if output.exists() or cache.exists(): raise RuntimeError('preserve existing artifacts')
    source = ['slotspe_proof/crossmodal_shared_bottleneck.py', 'slotspe_proof/common_private_survival.py',
        'slotspe_proof/prior_normalized_slots.py', 'slotspe_proof/common_private_diagnostics.py',
        'slotspe_proof/incremental_probe.py', 'slotspe_proof/slot_increment_probe.py',
        'scripts/check_frozen_content_targets.py', 'scripts/compare_crossmodal_shared_bottleneck.py']
    payload = {'status': 'running', 'plan': plan, 'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'source_sha256_before_execution': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source},
        'rows': [], 'patient_execution': False, 'server_execution': False}
    cache.mkdir(parents=True); output.write_bytes((json.dumps(payload, indent=2)+'\n').encode())
    cases = itertools.product(plan['tasks'], plan['seeds'], plan['shared_weight'], plan['private_weight'])
    for task, seed, shared_weight, private_weight in cases:
        try:
            row = run(task, seed, shared_weight, private_weight, cache)
        except Exception as exc:
            payload.update(status='failed', failed_case=[task, seed, shared_weight, private_weight], error=repr(exc))
            output.write_bytes((json.dumps(payload, indent=2)+'\n').encode()); raise
        payload['rows'].append(row); output.write_bytes((json.dumps(payload, indent=2)+'\n').encode())
        print(json.dumps({key: row[key] for key in ['task', 'seed', 'shared_weight', 'private_weight', 'test_expected_nll_single_draw_average']}), flush=True)
    payload['status'] = 'completed'; output.write_bytes((json.dumps(payload, indent=2)+'\n').encode())


if __name__ == '__main__': main()
