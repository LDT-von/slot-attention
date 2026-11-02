"""Locked 30-run synthetic comparison after frozen-target feasibility gate."""
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_frozen_content_targets import generated, fit_targets, training_role_permutation, state_hash as teacher_hash
from scripts.compare_common_private_club import state_hash
from slotspe_proof.content_budget_slots import ContentBudgetSlotSurvival, content_budget_terms
from slotspe_proof.common_private_survival import objective, survival_nll
from slotspe_proof.common_private_diagnostics import linear_probe
from slotspe_proof.incremental_probe import incremental_probe

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def paired(seed, n, condition):
    packs = [generated(seed, n, condition), generated(seed+5000000, n, condition)]
    rng = torch.Generator().manual_seed(seed+6000000)
    packs[1]['x'][..., 3] = .3*packs[0]['common'][:, None] + .05*torch.randn(n, 12, generator=rng)
    weights = torch.tensor([1., .75, -.5, .25])
    risk = .5*packs[0]['common']+(((packs[0]['bits']*2-1)@weights)+((packs[1]['bits']*2-1)@weights.flip(0)))/math.sqrt(8)
    hazards = torch.sigmoid(risk[:, None]+torch.tensor([-2., -1., 0., 1.]))
    hits = torch.rand(n, 4, generator=rng) < hazards
    t = torch.where(hits.any(1), hits.int().argmax(1), 4)
    censor = torch.randint(0, 4, (n,), generator=rng)
    return {'packs': packs, 'xs': [p['x'] for p in packs], 'risk': risk,
            'time': torch.minimum(t, censor), 'event': (t <= censor).float()}


def targets(pack, teachers, permutations, kind):
    result = []
    for p, teacher, perm in zip(pack['packs'], teachers, permutations):
        if kind == 'oracle_content':
            labels = p['bits'][:, perm]
            observed = torch.ones_like(labels, dtype=torch.bool)
        else:
            prediction = teacher(p['x'])
            labels, observed = prediction['labels'], prediction['observed']
            if kind == 'constant_content': labels = torch.zeros_like(labels)
        result.append({'labels': labels.detach(), 'observed': observed})
    return result


def setup(condition, seed):
    train = paired(210000+seed, 512, condition)
    teachers = [fit_targets(p, 220000+seed+j) for j, p in enumerate(train['packs'])]
    permutations = [training_role_permutation(t(p['x'])['token_cluster'], p['token_role']) for t, p in zip(teachers, train['packs'])]
    return train, teachers, permutations


def run(condition, seed, arm, setup_data, cache):
    train, teachers, permutations = setup_data
    frozen_hashes = [teacher_hash(t) for t in teachers]
    train_targets = targets(train, teachers, permutations, arm)
    if arm == 'shuffled_train_content':
        permutation = torch.randperm(512, generator=torch.Generator().manual_seed(230000+seed))
        train_targets = [{key: value[permutation] for key, value in t.items()} for t in train_targets]
    torch.default_generator.manual_seed(seed)
    net = ContentBudgetSlotSurvival(8, 8, width=16, slots=4, common_dim=4, private_dim=4)
    initial = state_hash(net)
    opt = torch.optim.Adam(net.parameters(), lr=.003)
    schedule = torch.Generator().manual_seed(240000+seed)
    noise_rng = torch.Generator().manual_seed(250000+seed)
    anchor = arm != 'budget_only'
    grad_max = 0.; grad_sum = 0.; trace = []; started = time.perf_counter()
    for step in range(500):
        idx = torch.randperm(512, generator=schedule)[:32]
        noise = [torch.randn(32, 4, 4, generator=noise_rng) for _ in range(2)]
        out = net(*[x[idx] for x in train['xs']], noise=noise)
        t = [{key: value[idx] for key, value in target.items()} for target in train_targets]
        term = content_budget_terms(out, t, budget=1.)
        loss = objective(out, train['time'][idx], train['event'][idx], beta=0.) + .3*term['budget_violation'] + anchor*term['content']
        if not torch.isfinite(loss): raise RuntimeError('nonfinite synthetic loss')
        opt.zero_grad(); loss.backward()
        norm = torch.stack([p.grad.detach().square().sum() for p in net.parameters() if p.grad is not None]).sum().sqrt()
        if not torch.isfinite(norm): raise RuntimeError('nonfinite synthetic gradient')
        grad_max = max(grad_max, float(norm)); grad_sum += float(norm)
        opt.step()
        if step in [0, 99, 299, 499]:
            trace.append({'step': step, 'loss': float(loss.detach()), 'content_loss': float(term['content'].detach()),
                          'budget_hinge': float(term['budget_violation'].detach()), 'gradient_norm': float(norm)})
    training_seconds = time.perf_counter()-started
    net.eval(); packs = [paired(o+seed, n, condition) for o, n in [(260000, 192), (270000, 128), (280000, 384)]]
    frozen = []
    with torch.no_grad():
        for pack in packs:
            out = net(*pack['xs'], sample=False)
            pooled = [f(v+e[None]).mean(1) for f, v, e in zip(net.phi, out['private'], net.role_offsets)]
            truth = torch.cat([p['bits'][:, perm] for p, perm in zip(pack['packs'], permutations)], -1)
            frozen.append({'common': out['common'], 'private': out['private'], 'pooled': pooled,
                           'bits': truth, 'risk': pack['risk']})
        deterministic_nll = float(survival_nll(out['logits'], packs[2]['time'], packs[2]['event']))
        noise_rng = torch.Generator().manual_seed(290000+seed)
        nll = []; oracle_ce = []; teacher_ce = []; objective_ce = []
        for _ in range(5):
            noise = [torch.randn(384, 4, 4, generator=noise_rng) for _ in range(2)]
            out = net(*packs[2]['xs'], noise=noise)
            nll.append(float(survival_nll(out['logits'], packs[2]['time'], packs[2]['event'])))
            for kind, store in [('oracle_content', oracle_ce), ('frozen_content', teacher_ce), (arm, objective_ce)]:
                ts = targets(packs[2], teachers, permutations, kind)
                # Shuffled training labels have no matching heldout permutation.
                # This field therefore uses correct frozen heldout content.
                store.append(float(content_budget_terms(out, ts)['content']))
        rate = torch.cat([r.mean(0) for r in out['normalized_kl_per_slot']])
    own = []; off = []
    for j in range(2):
        for k in range(4):
            for target in range(8):
                r = linear_probe(frozen[0]['private'][j][:, k], frozen[0]['bits'][:, target],
                                 frozen[2]['private'][j][:, k], frozen[2]['bits'][:, target])
                (own if target == j*4+k else off).append(r['test_r2'])
    common_probe = linear_probe(frozen[0]['common'], frozen[0]['bits'], frozen[2]['common'], frozen[2]['bits'])
    increment = incremental_probe(*[(p['common'], torch.cat(p['pooled'], -1), p['risk']) for p in frozen])
    filename = cache/f'{condition}_{seed}_{arm}.pt'
    torch.save({'synthetic_only': True, 'frozen': frozen, 'condition': condition, 'seed': seed, 'arm': arm}, filename)
    assert [teacher_hash(t) for t in teachers] == frozen_hashes
    return {'condition': condition, 'seed': seed, 'arm': arm, 'initial_hash': initial,
        'teacher_hashes': frozen_hashes, 'teacher_immutable': True, 'parameter_count': sum(p.numel() for p in net.parameters()),
        'test_expected_nll': sum(nll)/5, 'test_deterministic_nll': deterministic_nll,
        'test_expected_oracle_content_ce': sum(oracle_ce)/5, 'test_expected_frozen_target_ce': sum(teacher_ce)/5,
        'test_expected_arm_target_ce': sum(objective_ce)/5, 'rate_nats_per_slot': rate.tolist(),
        'own_true_bit_linear_r2': sum(own)/len(own), 'other_true_bits_linear_r2': sum(off)/len(off),
        'common_true_bits_linear_r2': common_probe['test_r2'], 'pooled_risk_increment_probe': increment,
        'training_seconds': training_seconds, 'gradient_norm_max': grad_max, 'gradient_norm_mean': grad_sum/500,
        'trace': trace, 'synthetic_cache_path': str(filename.relative_to(ROOT)),
        'synthetic_cache_sha256': hashlib.sha256(filename.read_bytes()).hexdigest()}


def main():
    torch.set_num_threads(2)
    gate = json.loads((DOC/'frozen_target_gate_receipt.json').read_text())
    assert gate['status'] == 'completed' and gate['positive_feasibility_gate_passed']
    plan_path = DOC/'frozen_target_gate_plan_20261102.json'
    plan = json.loads(plan_path.read_text())['planned_neural_followup_if_gate_passes']
    output = DOC/'frozen_target_neural_receipt.json'; cache = ROOT/'local_evidence/synthetic_frozen_targets_20261102'
    if output.exists() or cache.exists(): raise RuntimeError('preserve existing artifacts')
    source = ['scripts/compare_frozen_content_targets.py', 'scripts/check_frozen_content_targets.py',
        'slotspe_proof/frozen_prototype_targets.py', 'slotspe_proof/content_budget_slots.py',
        'slotspe_proof/prior_normalized_slots.py', 'slotspe_proof/common_private_survival.py']
    payload = {'status': 'running', 'plan': plan,
        'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'source_sha256_before_execution': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source},
        'rows': [], 'patient_execution': False, 'server_execution': False,
        'scope': 'prespecified descriptor/content subspaces, synthetic only, no one-hot role metadata'}
    cache.mkdir(parents=True); output.write_bytes((json.dumps(payload, indent=2)+'\n').encode())
    for condition, seed in itertools.product(plan['conditions'], plan['seeds']):
        data = setup(condition, seed)
        for arm in plan['arms']:
            row = run(condition, seed, arm, data, cache); payload['rows'].append(row)
            output.write_bytes((json.dumps(payload, indent=2)+'\n').encode())
            print(json.dumps({key: row[key] for key in ['condition', 'seed', 'arm', 'test_expected_nll', 'own_true_bit_linear_r2']}), flush=True)
    payload['status'] = 'completed'; output.write_bytes((json.dumps(payload, indent=2)+'\n').encode())


if __name__ == '__main__': main()
