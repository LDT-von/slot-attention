"""The KL weight trade-off of the common/private prototype, on generated data.

The semantics receipt shows two facts that must be reconciled:

* at a small KL weight the private slots do carry content that the common
  representation does not carry, and the readout consumes it;
* at a large KL weight the private content collapses onto the common
  representation, so the private branch becomes a redundant copy.

The first semantics run used three KL weights, and the small one was too small
to constrain anything: the KL term contributed less than one tenth of a
percent of the objective. This script sweeps the KL weight on a log scale
between those two regimes and reports, for each weight, both the size of the
KL term in the objective and the amount of private increment that survives.

The metric that matters is the partial probe: how much of the target
variation survives after the common representation has been regressed out,
and how much of that remainder the private branch still explains. A KL weight
that preserves the private increment at no survival cost would be the useful
operating point. A KL weight that only trades survival for compression is
not.

No patient file, no checkpoint and no network run is touched here.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slotspe_proof.common_private_survival import (CommonPrivateSurvival, objective,
                                                  survival_nll)
from slotspe_proof.common_private_diagnostics import (duplication_probe,
                                                      partial_probe,
                                                      prior_substitution_ablation,
                                                      representation_variance)
from scripts.check_common_private_semantics import generated, representations


def train(task, seed, beta, steps, train_n=512, width=16, slots=4,
          common_dim=4, private_dim=4, lr=.003, batch=32):
    torch.manual_seed(seed)
    net = CommonPrivateSurvival(6, 6, width=width, slots=slots,
                                 common_dim=common_dim, private_dim=private_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    pack = generated(1000 + seed, train_n, task)
    for _ in range(steps):
        index = torch.randint(0, train_n, (batch,))
        out = net(pack[0][index], pack[1][index])
        loss = objective(out, pack[2][index], pack[3][index], beta=beta)
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite generated objective')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return net


def measure(task, seed, beta, steps, lam=1e-3):
    net = train(task, seed, beta, steps)
    probe_train_n, probe_test_n = 384, 128
    train_pack = generated(1000 + seed, probe_train_n, task)
    test_pack = generated(5000 + seed, probe_test_n, task)
    out, test_features, factors, time, event = representations(net, test_pack)
    train_features = representations(net, train_pack)[1]

    def join(key):
        return torch.cat([train_features[key], test_features[key]], 0)

    common = join('common')
    private_wsi = join('private_wsi_readout')
    private_omic = join('private_omic_readout')
    factor_matrix = {name: torch.cat([train_pack[4][name], factors[name]], 0)
                     for name in ('shared', 'private_wsi', 'private_omic')}
    train_index = torch.arange(probe_train_n)
    test_index = torch.arange(probe_train_n, probe_train_n + probe_test_n)

    partial = {}
    for factor in ('private_wsi', 'private_omic'):
        for branch_name, branch in (('private_wsi_readout', private_wsi),
                                     ('private_omic_readout', private_omic)):
            partial[f'{branch_name}_on_{factor}'] = partial_probe(
                common[train_index], branch[train_index], factor_matrix[factor][train_index],
                common[test_index], branch[test_index], factor_matrix[factor][test_index],
                lam=lam)

    def nll_of(logits):
        return survival_nll(logits, time, event)

    with torch.no_grad():
        prior_ablation = prior_substitution_ablation(net, out['common'], out['private'],
                                                     time, event, nll_of)

    kl_joint = float(out['kl_joint'].mean())
    latent_dimensions = int(out['latent_dimensions'])
    kl_normalised = kl_joint / latent_dimensions
    nll_val = float(nll_of(out['logits']))
    kl_term = beta * kl_normalised
    return {
        'task': task,
        'seed': seed,
        'beta': beta,
        'steps': steps,
        'survival_nll_validation': nll_val,
        'kl_joint_mean': kl_joint,
        'kl_term_in_objective': kl_term,
        'kl_share_of_objective': (kl_term / (nll_val + kl_term)) if (nll_val + kl_term) > 0 else float('nan'),
        'private_increment_residual_r2': {
            key: value['private_residual_test_r2'] for key, value in partial.items()},
        'common_branch_only_r2_on_private_factor': {
            'private_wsi': partial_probe(
                common[train_index], private_wsi[train_index], factor_matrix['private_wsi'][train_index],
                common[test_index], private_wsi[test_index], factor_matrix['private_wsi'][test_index],
                lam=lam)['common_only_test_r2'],
            'private_omic': partial_probe(
                common[train_index], private_omic[train_index], factor_matrix['private_omic'][train_index],
                common[test_index], private_omic[test_index], factor_matrix['private_omic'][test_index],
                lam=lam)['common_only_test_r2'],
        },
        'private_branch_raw_r2': {
            'private_wsi_on_private_wsi': _raw_r2(private_wsi, factor_matrix['private_wsi'],
                                                  train_index, test_index, lam),
            'private_omic_on_private_omic': _raw_r2(private_omic, factor_matrix['private_omic'],
                                                    train_index, test_index, lam),
        },
        'prior_retains_fraction_of_private_effect': prior_ablation['prior_retains_fraction_of_private_effect'],
        'duplication_from_common': {
            'private_wsi': duplication_probe(
                common[train_index], private_wsi[train_index],
                common[test_index], private_wsi[test_index], lam=lam)['common_to_private_test_r2'],
            'private_omic': duplication_probe(
                common[train_index], private_omic[train_index],
                common[test_index], private_omic[test_index], lam=lam)['common_to_private_test_r2'],
        },
        'representation_variance': {
            'common': representation_variance(common),
            'private_wsi_readout': representation_variance(private_wsi),
            'private_omic_readout': representation_variance(private_omic),
        },
    }


def _raw_r2(features, target, train_index, test_index, lam):
    from slotspe_proof.common_private_diagnostics import linear_probe
    return linear_probe(features[train_index], target[train_index],
                         features[test_index], target[test_index], lam=lam)['test_r2']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='execute the generated-data sweep')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--output', default='docs/research/slotspe_common_private_20261101/kl_tradeoff_receipt.json')
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({'mode': 'plan_only', 'patient_execution': False,
                          'betas': [0., .01, .03, .1, .3, 1., 3.],
                          'tasks': ['shared_and_private', 'private_only'],
                          'seeds': [3, 13],
                          'next': '--run executes generated tensors only'}, indent=2))
        return
    torch.set_num_threads(2)
    betas = (0., .01, .03, .1, .3, 1., 3.)
    results = []
    for task in ('shared_and_private', 'private_only'):
        for seed in (3, 13):
            for beta in betas:
                results.append(measure(task, seed, beta, args.steps))
                print('[run]', task, 'seed', seed, 'beta', beta, 'done', flush=True)
    payload = {
        'schema': 'slotspe_common_private_kl_tradeoff_v1',
        'status': 'completed',
        'patient_execution': False,
        'patient_inference': False,
        'server_execution': False,
        'steps_per_run': args.steps,
        'betas': list(betas),
        'results': results,
        'claim_boundary': ('Generated-data trade-off between the KL term size and the private increment '
                           'that survives it, for one prototype. It locates a usable operating point on '
                           'invented data. It says nothing about patient performance and does not '
                           'establish that the operating point transfers to real features.'),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'completed', 'runs': len(results), 'output': str(destination)}, indent=2))


if __name__ == '__main__':
    main()
