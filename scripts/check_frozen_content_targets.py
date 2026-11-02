"""Known-answer synthetic acceptance of train-only frozen prototype targets."""
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slotspe_proof.frozen_prototype_targets import FrozenPrototypeTargets
from slotspe_proof.common_private_diagnostics import linear_probe

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def generated(seed, n, condition):
    rng = torch.Generator().manual_seed(seed)
    factors = torch.randn(n, 4, generator=rng)
    nuisance = torch.randn(n, 4, generator=rng)
    common = torch.randn(n, generator=rng)
    ids = torch.arange(4).repeat_interleave(3)
    centers = torch.tensor([[-2., -2.], [-2., 2.], [2., -2.], [2., 2.]])
    if condition == 'indistinguishable_roles':
        centers.zero_()
    elif condition not in ['separated_informative', 'dominant_common', 'dominant_private_nuisance']:
        raise ValueError(condition)
    x = torch.zeros(n, 12, 8)
    x[..., :2] = centers[ids][None] + .08*torch.randn(n, 12, 2, generator=rng)
    x[..., 2] = factors[:, ids] + .05*torch.randn(n, 12, generator=rng)
    x[..., 3] = common[:, None]*(3. if condition == 'dominant_common' else .3) + .05*torch.randn(n, 12, generator=rng)
    x[..., 4] = nuisance[:, ids]*(3. if condition == 'dominant_private_nuisance' else .2) + .05*torch.randn(n, 12, generator=rng)
    x[..., 5:] = .1*torch.randn(n, 12, 3, generator=rng)
    order = torch.rand(n, 12, generator=rng).argsort(1)
    x = x.gather(1, order[..., None].expand(-1, -1, 8))
    token_role = ids[None].expand(n, -1).gather(1, order)
    risk = ((factors > 0).float()*2-1) @ torch.tensor([1., .75, -.5, .25])
    return {'x': x, 'valid': torch.ones(n, 12, dtype=torch.bool), 'token_role': token_role,
            'bits': (factors > 0).float(), 'nuisance_bits': (nuisance > 0).float(), 'common': common, 'risk': risk}


def fit_targets(pack, seed):
    # Only observed x and mask enter this fit, never planted factors/outcomes.
    return FrozenPrototypeTargets.fit(pack['x'], pack['valid'], clusters=4,
        descriptor_columns=(0, 1), content_columns=(2, 3, 4, 5, 6, 7), seed=seed)


def state_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode()); h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def training_role_permutation(assignment, truth):
    permutations = list(itertools.permutations(range(4)))
    return max(permutations, key=lambda p: float((torch.tensor(p)[assignment] == truth).float().mean()))


def binary_entropy(labels, observed):
    p = (labels.double()*observed).sum(0)/observed.sum(0).clamp_min(1)
    return (-(p*p.clamp_min(1e-12).log()+(1-p)*(1-p).clamp_min(1e-12).log())).tolist()


def evaluate(condition, seed):
    train = generated(180000+seed, 512, condition)
    test = generated(190000+seed, 512, condition)
    teacher = fit_targets(train, seed)
    before = state_hash(teacher)
    a, b = teacher(train['x']), teacher(test['x'])
    perm = training_role_permutation(a['token_cluster'], train['token_role'])
    cluster_accuracy = float((torch.tensor(perm)[b['token_cluster']] == test['token_role']).float().mean())
    role_accuracy = (b['labels'] == test['bits'][:, perm]).float().mean(0)
    nuisance_accuracy = (b['labels'] == test['nuisance_bits'][:, perm]).float().mean(0)
    common_accuracy = (b['labels'] == (test['common'] > 0).float()[:, None]).float().mean(0)
    left, right = teacher(test['x'][:, :6]), teacher(test['x'][:, 6:])
    both = left['observed'] & right['observed']
    agreement = ((left['labels'] == right['labels']) & both).sum()/both.sum()
    pairs = list(itertools.combinations(range(4), 2))
    disagreement = sum(float((b['labels'][:, i] != b['labels'][:, j]).float().mean()) for i, j in pairs)/len(pairs)
    utility = linear_probe(a['labels'], train['risk'], b['labels'], test['risk'])
    assert state_hash(teacher) == before
    return {'condition': condition, 'seed': seed, 'fit_rows': teacher.fit_rows,
        'teacher_hash': before, 'teacher_immutable_after_heldout': True,
        'heldout_cluster_accuracy_train_matched': cluster_accuracy,
        'heldout_informative_bit_accuracy': role_accuracy.tolist(),
        'heldout_nuisance_bit_accuracy': nuisance_accuracy.tolist(),
        'heldout_common_bit_accuracy': common_accuracy.tolist(),
        'heldout_target_entropy_nats': binary_entropy(b['labels'], b['observed']),
        'crossview_agreement_when_both_observed': float(agreement),
        'crossview_both_observed_fraction': float(both.float().mean()),
        'cross_role_target_disagreement': disagreement,
        'target_risk_linear_probe_r2': utility['test_r2'],
        'interpretation': 'train-matched truth used only for diagnosis; entropy/PCA/cluster recovery do not imply biological identification or MI reduction'}


def collision_counterexample():
    # Same unordered observations, different named-role target/outcome.
    first = torch.tensor([-1., 1., 0., 0.]).repeat_interleave(3)
    second = torch.tensor([1., -1., 0., 0.]).repeat_interleave(3)
    assert torch.equal(first.sort().values, second.sort().values)
    return {'same_unordered_bag': True, 'named_role_zero_target': [0, 1],
            'selective_risk': [-2, 2],
            'conclusion': 'No deterministic permutation-invariant function of this bag can recover both named-role labels. It does not imply zero information about every symmetric outcome.'}


def main():
    torch.set_num_threads(2)
    output = DOC/'frozen_target_gate_receipt.json'
    if output.exists(): raise RuntimeError('preserve existing receipt')
    plan_path = DOC/'frozen_target_gate_plan_20261102.json'
    plan = json.loads(plan_path.read_text())
    rows = [evaluate(c, s) for c, s in itertools.product(plan['conditions'], plan['seeds'])]
    positive = [r for r in rows if r['condition'] == 'separated_informative']
    gates = plan['gate']
    passed = all(r['heldout_cluster_accuracy_train_matched'] >= gates['separated_informative_min_heldout_cluster_accuracy']
        and min(r['heldout_informative_bit_accuracy']) >= gates['separated_informative_min_target_bit_accuracy']
        and r['crossview_agreement_when_both_observed'] >= gates['min_target_crossview_agreement_when_both_observed']
        and min(r['heldout_target_entropy_nats']) >= gates['min_target_entropy_nats_each_role'] for r in positive)
    source = ['slotspe_proof/frozen_prototype_targets.py', 'scripts/check_frozen_content_targets.py',
              'docs/research/slotspe_common_private_20261101/frozen_target_gate_plan_20261102.json']
    payload = {'status': 'completed', 'positive_feasibility_gate_passed': passed,
        'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source},
        'rows': rows, 'permutation_collision': collision_counterexample(),
        'prototype_teacher_fits': 12, 'neural_trainings': 0, 'patient_execution': False, 'server_execution': False,
        'not_a_clinical_adapter': True, 'assumptions': 'prespecified descriptor subspace plus dominant within-prototype PCA; no role one-hot input'}
    output.write_bytes((json.dumps(payload, indent=2)+'\n').encode())
    print(json.dumps(payload, indent=2))


if __name__ == '__main__': main()
