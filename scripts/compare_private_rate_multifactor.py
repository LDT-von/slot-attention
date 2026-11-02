"""Locked24-run synthetic private-rate ablation, with pre-training probe gate."""
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
from slotspe_proof.crossmodal_shared_bottleneck import CrossmodalSharedBottleneck, shared_private_objective
from slotspe_proof.common_private_survival import survival_nll
from slotspe_proof.private_rate_fee import private_rate_objective
from slotspe_proof.multifactor_survival_synthetic import generate, noise, standardize, observed_bin_concordance
from slotspe_proof.common_private_diagnostics import linear_probe
from slotspe_proof.incremental_probe import incremental_probe
from slotspe_proof.slot_increment_probe import slot_increment_probe
from scripts.compare_crossmodal_shared_bottleneck import geometry
from scripts.compare_common_private_club import state_hash

DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def data_hash(pack):
    h = hashlib.sha256()
    for value in [*pack['xs'], pack['time'], pack['event']]: h.update(value.numpy().tobytes())
    return h.hexdigest()


def instrument_controls():
    packs = [generate(offset+653, n, 'factorwise') for offset, n in [(840000, 192), (850000, 128), (860000, 384)]]
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
            target = pack['factors']
            if mode == 'shuffled': target = target[torch.randperm(n, generator=torch.Generator().manual_seed(900000+i))]
            parts.append((g, groups, target))
        result[mode] = slot_increment_probe(*parts)
    positive = result['role_placed']['gain_relative_to_fit_mean_baseline']
    duplicate = result['duplicate']['gain_relative_to_fit_mean_baseline']
    shuffled = result['shuffled']['gain_relative_to_fit_mean_baseline']
    passed = positive > .25 and abs(duplicate) < 1e-10 and abs(shuffled) < .05
    return {'passed': passed, 'relative_gains': {'role_placed': positive, 'duplicate': duplicate, 'shuffled': shuffled}, 'details': result}


def gradient_audit(terms, out, weight, parameters):
    losses = {'nll': terms['nll'], 'shared_used': .1*terms['shared_rate'],
              'cross_used': .1*out['cross'], 'private_used': weight*terms['private_fee']}
    vectors = {name: torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True) for name, value in losses.items()}
    norms = {name: math.sqrt(sum(float(g.detach().square().sum()) for g in values if g is not None)) for name, values in vectors.items()}
    numerator = sum(float(a.detach().mul(b.detach()).sum()) for a, b in zip(vectors['nll'], vectors['private_used']) if a is not None and b is not None)
    denominator = norms['nll']*norms['private_used']
    return {'component_gradient_l2': norms, 'nll_private_gradient_cosine': numerator/denominator if denominator > 1e-12 else None}


def core_run(task, seed, arm, cache):
    torch.default_generator.manual_seed(seed)
    model = CrossmodalSharedBottleneck(8, 8, width=16, slots=2, common_dim=4, private_dim=4)
    initial = state_hash(model); optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    generated = [generate(offset+seed, n, task) for offset, n in [(810000, 512), (840000, 192), (850000, 128), (860000, 384)]]
    packs, normalization = standardize(generated[0], generated[1:]); train, held = packs[0], packs[1:]
    batches = torch.Generator().manual_seed(820000+seed); rng = torch.Generator().manual_seed(830000+seed)
    trace = []; curve = []; active_count = torch.zeros(2, 2); any_active = 0; slot_active_sum = 0.; rate_sum = torch.zeros(2, 2)
    grad_max = 0.; grad_sum = 0.; parameters = tuple(model.parameters()); start = time.perf_counter()
    for step in range(500):
        idx = torch.randperm(512, generator=batches)[:32]
        out = model(*[x[idx] for x in train['xs']], **noise(rng, 32))
        terms = private_rate_objective(out, train['time'][idx], train['event'][idx], eta=arm['eta'], private_weight=arm['weight'])
        if not torch.isfinite(terms['loss']): raise RuntimeError(f'nonfinite training loss at {step}')
        component = gradient_audit(terms, out, arm['weight'], parameters) if step in [0, 99, 299, 499] else None
        if arm['name'] == 'flat' and step == 0:
            original = shared_private_objective(out, train['time'][idx], train['event'][idx])
            assert torch.equal(terms['loss'], original['loss'])
        optimizer.zero_grad(); terms['loss'].backward()
        norm = math.sqrt(sum(float(p.grad.detach().square().sum()) for p in parameters if p.grad is not None))
        if not math.isfinite(norm): raise RuntimeError(f'nonfinite training gradient at {step}')
        optimizer.step(); grad_max = max(grad_max, norm); grad_sum += norm
        raw_rates = torch.stack([r.detach().mean(0) for r in out['private_rate_per_slot']])
        active = raw_rates > 1.; active_count += active; any_active += int(active.any()); slot_active_sum += float(active.float().mean()); rate_sum += raw_rates
        entry = {'step': step, **{k: float(v.detach()) for k, v in terms.items()}, 'gradient_l2': norm}
        curve.append(entry)
        if component is not None: trace.append({**entry, **component})
    training_seconds = time.perf_counter()-start
    model.eval(); frozen = []; evaluation_rng = torch.Generator().manual_seed(870000+seed)
    with torch.no_grad():
        for pack in held:
            sampled = model(*pack['xs'], **noise(evaluation_rng, len(pack['shared'])))
            means = model(*pack['xs'], sample=False)
            states = {}
            for mode, output in [('sample', sampled), ('all_mean', means)]:
                states[mode] = {'common': output['common'], 'private': output['private'],
                                'phi': [phi(v+offset[None]) for phi, v, offset in zip(model.phi, output['private'], model.role_offsets)]}
            frozen.append({**states, 'factors': pack['factors'], 'shared': pack['shared'], 'nuisance': pack['nuisance']})
        mean_nll = float(survival_nll(means['logits'], held[2]['time'], held[2]['event']))
        mean_geometry = geometry(means['private']); sample_geometry = geometry(sampled['private']); native_geometry = geometry(means['slots'])
        test = held[2]; donor = torch.randperm(384, generator=torch.Generator().manual_seed(880000+seed))
        scores = {}; concordance = []; rates = []; common_rates = []; single_deltas = torch.zeros(2, 2)
        for _ in range(5):
            eps = noise(evaluation_rng, 384); output = model(*test['xs'], **eps)
            g, private = output['common'], output['private']
            variants = {'own': private, 'private_N0': eps['private_noise'], 'private_donor': [v[donor] for v in private]}
            current = float(survival_nll(output['logits'], test['time'], test['event']))
            for name, values in variants.items(): scores.setdefault(name, []).append(float(survival_nll(model.readout(g, values), test['time'], test['event'])))
            concordance.append(observed_bin_concordance(output['logits'], test['time'], test['event']))
            rates.append(torch.stack([r.mean(0) for r in output['private_rate_per_slot']]))
            common_rates.append(float(output['shared_rate'].mean()))
            for j in range(2):
                for k in range(2):
                    replaced = [v.clone() for v in private]; replaced[j][:, k] = eps['private_noise'][j][:, k]
                    single_deltas[j, k] += float(survival_nll(model.readout(g, replaced), test['time'], test['event']))-current
        own = sum(scores['own'])/5
        latent_nll = float(survival_nll(test['true_logits'], test['time'], test['event']))
    filename = f'{task}_{seed}_{arm["name"]}.pt'; path = cache/filename
    torch.save({'synthetic_only': True, 'frozen': frozen, 'state_dict': model.state_dict(), 'curve': curve,
                'task': task, 'seed': seed, 'arm': arm, 'normalization': normalization}, path)
    row = {'status': 'trained_pending_probes', 'task': task, 'seed': seed, 'arm': arm, 'initial_hash': initial,
           'parameter_count': sum(p.numel() for p in parameters), 'optimizer_steps': 500,
           'input_label_hashes': [data_hash(p) for p in packs], 'train_only_normalization': normalization,
           'expected_single_draw_nll': own, 'single_draw_nll_values': scores['own'],
           'observed_bin_concordance': sum(concordance)/5, 'all_mean_nll_diagnostic': mean_nll,
           'planted_model_nll_privileged': latent_nll, 'test_shared_rate_mean': sum(common_rates)/5,
           'test_private_rate_per_slot': torch.stack(rates).mean(0).tolist(),
           'fixed_readout_nll_deltas': {k: sum(v)/5-own for k, v in scores.items() if k != 'own'},
           'one_slot_N0_nll_deltas': (single_deltas/5).tolist(), 'private_sample_geometry': sample_geometry,
           'private_all_mean_geometry': mean_geometry, 'native_geometry': native_geometry,
           'train_any_budget_active_fraction': any_active/500, 'train_mean_active_slot_fraction': slot_active_sum/500,
           'train_slot_activation_fraction': (active_count/500).tolist(), 'train_mean_rate_per_slot': (rate_sum/500).tolist(),
           'gradient_l2_max': grad_max, 'gradient_l2_mean': grad_sum/500, 'trace': trace,
           'finite_training_steps': 500, 'training_seconds_excluding_probes': training_seconds,
           'synthetic_cache_path': str(path.relative_to(ROOT)), 'synthetic_cache_sha256': sha(path)}
    return row, frozen


def diagnose(frozen):
    result = {}
    for mode in ['sample', 'all_mean']:
        parts = [(r[mode]['common'], r[mode]['phi'], r['factors']) for r in frozen]
        g0, g2 = frozen[0][mode]['common'], frozen[2][mode]['common']
        raw0 = torch.cat([v.flatten(1) for v in frozen[0][mode]['private']], -1)
        raw2 = torch.cat([v.flatten(1) for v in frozen[2][mode]['private']], -1)
        result[mode] = {'all_vs_single_factor_probe': slot_increment_probe(*parts),
                        'raw_private_factor_increment': incremental_probe(*[(r[mode]['common'], torch.cat([v.flatten(1) for v in r[mode]['private']], -1), r['factors']) for r in frozen]),
                        'G_shared_linear': linear_probe(g0, frozen[0]['shared'], g2, frozen[2]['shared']),
                        'G_private_factor_linear': linear_probe(g0, frozen[0]['factors'], g2, frozen[2]['factors']),
                        'private_nuisance_linear': linear_probe(raw0, frozen[0]['nuisance'], raw2, frozen[2]['nuisance'])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output-root', default='local_evidence/private_rate_multifactor_20261102')
    args = parser.parse_args()
    plan_path = DOC/'private_rate_multifactor_plan_20261102.json'; plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if not args.run: print(json.dumps(plan, indent=2)); return
    destination = (ROOT/args.output_root).resolve()
    if not destination.is_relative_to(ROOT/'local_evidence') or destination == ROOT/'local_evidence' or destination.exists():
        raise ValueError('fresh ignored output required')
    torch.set_num_threads(2)
    files = ['scripts/compare_private_rate_multifactor.py', 'slotspe_proof/private_rate_fee.py',
             'slotspe_proof/multifactor_survival_synthetic.py', 'slotspe_proof/crossmodal_shared_bottleneck.py',
             'slotspe_proof/common_private_survival.py', 'slotspe_proof/prior_normalized_slots.py',
             'slotspe_proof/incremental_probe.py', 'slotspe_proof/slot_increment_probe.py',
             'slotspe_proof/common_private_diagnostics.py', 'scripts/compare_common_private_club.py',
             'scripts/compare_crossmodal_shared_bottleneck.py']
    payload = {'status': 'running', 'plan': plan, 'plan_sha256': sha(plan_path),
               'source_sha256_before_execution': {p: sha(ROOT/p) for p in files},
               'instrument_controls': None, 'rows': [], 'patient_execution': False, 'server_execution': False}
    destination.mkdir(parents=True); cache = destination/'synthetic_cache'; cache.mkdir(); output = destination/'receipt.json'
    def save(): output.write_bytes((json.dumps(payload, indent=2)+'\n').encode())
    save()
    try:
        payload['instrument_controls'] = instrument_controls(); save()
        print(json.dumps({'instrument_gate': payload['instrument_controls']['passed'], 'relative_gains': payload['instrument_controls']['relative_gains']}), flush=True)
        if not payload['instrument_controls']['passed']: raise RuntimeError('pre-training probe instrument gate failed')
        for task in plan['tasks']:
            for seed in plan['seeds']:
                initial = None; hashes = None
                for arm in plan['arms']:
                    row, frozen = core_run(task, seed, arm, cache)
                    if initial is None: initial, hashes = row['initial_hash'], row['input_label_hashes']
                    assert row['initial_hash'] == initial and row['input_label_hashes'] == hashes
                    payload['rows'].append(row); save()
                    print(json.dumps({'trained': True, 'task': task, 'seed': seed, 'arm': arm['name'], 'nll': row['expected_single_draw_nll']}), flush=True)
                    row['diagnostics'] = diagnose(frozen); row['status'] = 'completed'; save()
        assert len(payload['rows']) == 24 and all(r['status'] == 'completed' for r in payload['rows'])
        assert all(sha(ROOT/p) == h for p, h in payload['source_sha256_before_execution'].items())
        payload['status'] = 'completed'; save()
    except Exception as exc:
        payload.update(status='failed', error=repr(exc)); save(); raise


if __name__ == '__main__': main()
