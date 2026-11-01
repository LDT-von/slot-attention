"""CPU-only semantics diagnostics for the common/private slot bottleneck.

The prototype can name a branch "private" without that branch carrying any
information. This script measures, on generated data only, three things:

1. whether the private slots explain target variation that the common
   representation cannot explain (partial probe);
2. whether the survival readout actually consumes the private slots at all
   (swap ablation: exchange or zero a branch and recompute the likelihood);
3. how large the KL term is relative to the survival term, because a KL
   weight whose contribution is numerically negligible cannot support any
   conclusion about compression.

Every measurement is paired with instruments of known answer: an input upper
bound that must recover the factor, a label-shuffled control that must not,
and a task in which the common factor alone drives the outcome, where the
private branch is expected to be unused.

No patient file, no checkpoint, no network run is touched here.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slotspe_proof.common_private_survival import (CommonPrivateSurvival, objective,
                                                  survival_nll)
from slotspe_proof.common_private_diagnostics import (branch_sensitivity,
                                                      duplication_probe,
                                                      linear_probe,
                                                      partial_probe, pearson,
                                                      prior_substitution_ablation,
                                                      probe_suite,
                                                      representation_variance,
                                                      summarise_probe_suite,
                                                      swap_ablation)


def generated(seed, n, task, tokens=12, dim=6, bins=4):
    """Generated patients with known common and modality-private factors.

    The common factor is written into both modalities; each private factor is
    written into exactly one modality, so a probe of the wrong modality cannot
    recover it. The outcome risk is ``c + p + q`` for ``shared_and_private``,
    ``p`` for ``private_only`` and ``c`` for ``common_only``.
    """
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(n, generator=g)
    p = torch.randn(n, generator=g)
    q = torch.randn(n, generator=g)
    wsi = torch.randn(n, tokens, dim, generator=g) * .3
    omic = torch.randn(n, tokens, dim, generator=g) * .3
    wsi[:, :, 0] += c[:, None]
    omic[:, :, 0] += c[:, None]
    wsi[:, :, 1] += p[:, None]
    omic[:, :, 2] += q[:, None]
    if task == 'shared_and_private':
        risk = c + p + q
    elif task == 'private_only':
        risk = p
    elif task == 'common_only':
        risk = c
    else:
        raise ValueError('unknown generated task: ' + str(task))
    hazards = torch.sigmoid(risk[:, None] + torch.tensor([-2., -1., 0., 1.][:bins]))
    hits = torch.rand(n, bins, generator=g) < hazards
    event_time = torch.where(hits.any(1), hits.int().argmax(1), bins)
    censor = torch.randint(0, bins, (n,), generator=g)
    event = (event_time <= censor).float()
    time = torch.minimum(event_time, censor)
    factors = {'shared': c, 'private_wsi': p, 'private_omic': q}
    return wsi, omic, time, event, factors


def train(task, seed, beta, steps, train_n=512, val_n=256, width=16, slots=4,
          common_dim=4, private_dim=4, lr=.003, batch=32):
    torch.manual_seed(seed)
    net = CommonPrivateSurvival(6, 6, width=width, slots=slots,
                                 common_dim=common_dim, private_dim=private_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_pack = generated(1000 + seed, train_n, task)
    for _ in range(steps):
        index = torch.randint(0, train_n, (batch,))
        out = net(train_pack[0][index], train_pack[1][index])
        loss = objective(out, train_pack[2][index], train_pack[3][index], beta=beta)
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite generated objective')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return net


def representations(net, pack):
    """Per-patient branch features of one trained model on one batch."""
    wsi, omic, time, event, factors = pack
    net.eval()
    with torch.no_grad():
        out = net(wsi, omic, sample=False)
    common = out['common'].double()
    private = out['private']
    phi_private = [f(u).mean(1).double() for f, u in zip(net.phi, private)]
    features = {
        'common': common,
        'private_wsi_readout': phi_private[0],
        'private_omic_readout': phi_private[1],
        'private_wsi_flat': private[0].flatten(1).double(),
        'private_omic_flat': private[1].flatten(1).double(),
        'private_all_flat': torch.cat([u.flatten(1) for u in private], 1).double(),
        'slots_wsi_mean': out['slots'][0].mean(1).double(),
        'slots_omic_mean': out['slots'][1].mean(1).double(),
        'readout_input': torch.cat([common] + phi_private, 1).double(),
        'input_wsi_mean': wsi.mean(1).double(),
        'input_omic_mean': omic.mean(1).double(),
    }
    return out, features, factors, time, event


def measure(task, seed, beta, steps, lam=1e-3):
    net = train(task, seed, beta, steps)
    # The probe is fitted on one generated cohort and scored on a disjoint
    # one. Both are concatenated along the patient axis so that a single
    # index pair addresses the two halves without mixing them.
    probe_train_n = 384
    probe_test_n = 128
    train_pack = generated(1000 + seed, probe_train_n, task)
    test_pack = generated(5000 + seed, probe_test_n, task)
    out, test_features, factors, time, event = representations(net, test_pack)
    train_features = representations(net, train_pack)[1]

    def join(key):
        return torch.cat([train_features[key], test_features[key]], 0)

    features = {key: join(key) for key in (
        'common', 'private_wsi_readout', 'private_omic_readout', 'private_wsi_flat',
        'private_omic_flat', 'private_all_flat', 'slots_wsi_mean', 'slots_omic_mean',
        'readout_input', 'input_wsi_mean', 'input_omic_mean')}
    factor_matrix = {name: torch.cat([train_pack[4][name], factors[name]], 0)
                     for name in ('shared', 'private_wsi', 'private_omic')}
    train_index = torch.arange(probe_train_n)
    test_index = torch.arange(probe_train_n, probe_train_n + probe_test_n)

    suite = probe_suite(features, factor_matrix, train_index, test_index, lam=lam)
    input_upper_bound = {
        'shared_from_wsi_input': linear_probe(
            features['input_wsi_mean'][train_index], factor_matrix['shared'][train_index],
            features['input_wsi_mean'][test_index], factor_matrix['shared'][test_index], lam=lam),
        'shared_from_omic_input': linear_probe(
            features['input_omic_mean'][train_index], factor_matrix['shared'][train_index],
            features['input_omic_mean'][test_index], factor_matrix['shared'][test_index], lam=lam),
        'private_wsi_from_wsi_input': linear_probe(
            features['input_wsi_mean'][train_index], factor_matrix['private_wsi'][train_index],
            features['input_wsi_mean'][test_index], factor_matrix['private_wsi'][test_index], lam=lam),
        'private_omic_from_omic_input': linear_probe(
            features['input_omic_mean'][train_index], factor_matrix['private_omic'][train_index],
            features['input_omic_mean'][test_index], factor_matrix['private_omic'][test_index], lam=lam),
    }
    shuffled = {name: value[torch.randperm(value.shape[0], generator=torch.Generator().manual_seed(7))]
                for name, value in factor_matrix.items()}
    shuffled_suite = probe_suite(features, shuffled, train_index, test_index, lam=lam)

    partial = {}
    for factor in ('shared', 'private_wsi', 'private_omic'):
        for branch in ('private_wsi_readout', 'private_omic_readout', 'private_all_flat'):
            partial[f'{branch}_on_{factor}'] = partial_probe(
                features['common'][train_index], features[branch][train_index],
                factor_matrix[factor][train_index],
                features['common'][test_index], features[branch][test_index],
                factor_matrix[factor][test_index], lam=lam)

    duplication = {
        'private_wsi_from_common': duplication_probe(
            features['common'][train_index], features['private_wsi_readout'][train_index],
            features['common'][test_index], features['private_wsi_readout'][test_index], lam=lam),
        'private_omic_from_common': duplication_probe(
            features['common'][train_index], features['private_omic_readout'][train_index],
            features['common'][test_index], features['private_omic_readout'][test_index], lam=lam),
        'private_all_from_common': duplication_probe(
            features['common'][train_index], features['private_all_flat'][train_index],
            features['common'][test_index], features['private_all_flat'][test_index], lam=lam),
    }

    def nll_of(logits):
        return survival_nll(logits, time, event)

    with torch.no_grad():
        ablation = swap_ablation(net, out['common'], out['private'], time, event, nll_of)
        sensitivity = branch_sensitivity(net, out['common'], out['private'],
                                         time, event, nll_of)
        prior_ablation = prior_substitution_ablation(net, out['common'], out['private'],
                                                     time, event, nll_of)

    spread = {name: representation_variance(features[name])
              for name in ('common', 'private_wsi_readout', 'private_omic_readout',
                           'private_all_flat', 'readout_input')}

    kl_joint = float(out['kl_joint'].mean())
    latent_dimensions = int(out['latent_dimensions'])
    kl_normalised = kl_joint / latent_dimensions
    nll_full = ablation['nll']['full']
    train_out = net(train_pack[0][:256], train_pack[1][:256], sample=False)
    nll_train = float(survival_nll(train_out['logits'], train_pack[2][:256], train_pack[3][:256]))
    return {
        'task': task,
        'seed': seed,
        'beta': beta,
        'steps': steps,
        'probe_suite': suite,
        'probe_summary': summarise_probe_suite(suite),
        'input_upper_bound': input_upper_bound,
        'shuffled_control': {branch: {factor: scores['test_r2']
                                      for factor, scores in factors_dict.items()}
                             for branch, factors_dict in shuffled_suite.items()},
        'partial_probe': partial,
        'duplication_probe': duplication,
        'swap_ablation': ablation,
        'branch_sensitivity': sensitivity,
        'prior_substitution': prior_ablation,
        'representation_variance': spread,
        'kl': {
            'kl_joint_mean': kl_joint,
            'latent_dimensions': latent_dimensions,
            'kl_normalised_by_latent_dimension': kl_normalised,
            'kl_term_in_objective': beta * kl_normalised,
            'survival_nll_on_validation': nll_full,
            'survival_nll_on_train_subset': nll_train,
            'kl_term_share_of_total_objective': (
                beta * kl_normalised / (nll_full + beta * kl_normalised)
                if (nll_full + beta * kl_normalised) > 0 else float('nan')),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='execute the generated-data diagnostics')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--output', default='docs/research/slotspe_common_private_20261101/semantics_receipt.json')
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({'mode': 'plan_only', 'patient_execution': False,
                          'tasks': ['shared_and_private', 'private_only', 'common_only'],
                          'betas': [0., .001, 1.], 'seeds': [3, 13],
                          'next': '--run executes generated tensors only'}, indent=2))
        return
    torch.set_num_threads(2)
    results = []
    for task in ('shared_and_private', 'private_only', 'common_only'):
        for seed in (3, 13):
            for beta in (0., .001, 1.):
                results.append(measure(task, seed, beta, args.steps))
                print('[run]', task, 'seed', seed, 'beta', beta, 'done', flush=True)
    payload = {
        'schema': 'slotspe_common_private_semantics_v1',
        'status': 'completed',
        'patient_execution': False,
        'patient_inference': False,
        'server_execution': False,
        'steps_per_run': args.steps,
        'results': results,
        'claim_boundary': ('Generated-data internal behaviour of one prototype. It measures whether the '
                           'private branch carries information beyond the common branch and whether the '
                           'readout consumes it. It does not measure patient performance, does not establish '
                           'that common and private correspond to biological factors, and does not validate '
                           'the architecture on real features.'),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'completed', 'runs': len(results), 'output': str(destination)}, indent=2))


if __name__ == '__main__':
    main()
