"""Synthetic verification of the spread-target arm. No patient data is read.

Checks, in order:

  1. the spread arm's role target is genuinely the per-feature SPREAD of the
     patient's patch bag, and it differs from the archived mean target for a
     bag whose spread carries information the mean does not;
  2. the ENCODER INPUT normalization is IDENTICAL between the 'mean' arm and
     the 'spread' arm, so the two arms differ only in the supervision target;
  3. the spread arm trains, evaluates and writes a checkpoint that reloads
     and reproduces the recorded validation metric;
  4. the archived sources this arm depends on are byte-for-byte unchanged.

This script invents its own patients. It proves integration, not performance.
"""
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from research_models.common_private_synergy.patient_pilot import ROOT, sha, write


def invent_prepared(root, count=40, features=6, seed=4242):
    """Invent a prepared tree with the layout the engine expects.

    The engine resolves `prepared/manifest.json` under the prepared root and
    resolves every patient cache path relative to that same root, so the
    cache lives at <root>/prepared/cache/<id>.pt and the manifest records the
    relative path 'prepared/cache/<id>.pt'.
    """
    prepared = root / 'prepared'
    (prepared / 'cache').mkdir(parents=True)
    rng = torch.Generator().manual_seed(seed)
    manifest = {'omic_sizes': [4, 5], 'patients': {},
                'folds': {'0': {'train': [], 'val': []}}}
    for i in range(count):
        pid = 'INVENTED_%03d' % i
        rel = 'prepared/cache/%s.pt' % pid
        patches = torch.randn(5 + i % 15, features, generator=rng)
        omics = [torch.randn(n, generator=rng) for n in manifest['omic_sizes']]
        torch.save({'patient_id': pid, 'patches': patches, 'omics': omics}, root / rel)
        manifest['patients'][pid] = {'cache_path': rel,
                                    'cache_sha256': sha(root / rel),
                                    'time': float(i + 1),
                                    'censor': float(i % 3 == 0)}
        manifest['folds']['0']['train' if i < count - 8 else 'val'].append(pid)
    write(prepared / 'manifest.json', manifest)
    return manifest


def check_one():
    print('CHECK 1 the role target is the bag SPREAD and differs from the bag MEAN')
    from research_models.common_private_synergy.spread_arm import collect_target_matrices
    from research_models.common_private_synergy.components.spread_target import bag_spread
    from research_models.common_private_synergy.patient_pilot_engine import PreparedStore

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = invent_prepared(root)
        store = PreparedStore(root, manifest)
        ids = manifest['folds']['0']['train']
        mean_targets, mean_matrix, _ = collect_target_matrices(store, ids, 'mean')
        spread_matrix, mean_matrix_again, _ = collect_target_matrices(store, ids, 'spread')

        # The MEAN matrix must be identical in both modes: it is the shared
        # input normalization source.
        if not torch.equal(mean_targets, mean_matrix):
            raise AssertionError('the mean-mode target must be the bag mean')
        if not torch.equal(mean_matrix, mean_matrix_again):
            raise AssertionError('the mean matrix must not depend on the target mode')

        # The SPREAD matrix must equal the per-patient per-feature standard
        # deviation of the real patches.
        for row, pid in enumerate(ids):
            patches, _ = store.load(pid)
            expected = bag_spread(patches)
            if not torch.allclose(spread_matrix[row], expected, atol=0, rtol=0):
                raise AssertionError('the spread target is not the bag standard deviation')
            if not torch.allclose(mean_matrix[row], patches.double().mean(0), atol=1e-12):
                raise AssertionError('the mean target is not the bag mean')

        # The two constructions must NOT coincide: a bag's spread is not
        # recoverable from its mean.
        if torch.allclose(mean_matrix, spread_matrix):
            raise AssertionError('the mean and spread targets coincide; the arm would be a no-op')
    print('  OK the spread target equals the per-patient per-feature standard deviation')
    print('  OK the mean and spread targets differ')

    # The spread statistic must be strictly increasing in the bag spread and
    # must be exactly zero for a single-patch bag.
    collapsed = bag_spread(torch.full((12, 5), 2.0, dtype=torch.float64))
    varied = bag_spread(torch.stack([torch.full((5,), float(i), dtype=torch.float64)
                                     for i in range(12)]))
    single = bag_spread(torch.full((1, 5), 3.0, dtype=torch.float64))
    if not float((varied - collapsed).min()) > 0:
        raise AssertionError('a spread-out bag must have a strictly larger spread')
    if float(single.abs().max()) != 0.0:
        raise AssertionError('a single-real-patch bag must have exactly zero spread')
    if float(collapsed.abs().max()) != 0.0:
        raise AssertionError('a collapsed bag must have exactly zero spread')
    print('  OK a spread-out bag has a strictly larger spread')
    print('  OK a single-real-patch bag has exactly zero spread')
    return True


def check_two():
    print('CHECK 2 the encoder input normalization is identical in both arms')
    from research_models.common_private_synergy.spread_arm import collect_target_matrices
    from research_models.common_private_synergy.spread_arm import SpreadContentAdapter
    from research_models.common_private_synergy.patient_pilot_engine import PreparedStore

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = invent_prepared(root)
        store = PreparedStore(root, manifest)
        ids = manifest['folds']['0']['train']

        adapters = {}
        for mode in ('mean', 'spread'):
            target_matrix, mean_matrix, g_matrix = collect_target_matrices(store, ids, mode)
            input_mean = mean_matrix.mean(0)
            input_sd = mean_matrix.std(0, unbiased=False).clamp_min(1e-4)
            adapters[mode] = SpreadContentAdapter(target_matrix, g_matrix,
                                                  manifest['omic_sizes'], 3,
                                                  input_mean=input_mean, input_sd=input_sd,
                                                  p_target_mode=mode)

        patches, omics = store.load(ids[0])
        p_mean, g_mean = adapters['mean'].token_inputs(patches, omics)
        p_spread, g_spread = adapters['spread'].token_inputs(patches, omics)
        if not torch.equal(p_mean, p_spread):
            raise AssertionError('the encoder p input differs between the two arms')
        if not torch.equal(g_mean, g_spread):
            raise AssertionError('the encoder g input differs between the two arms')

        # And the role TARGETS must differ.
        t_mean = adapters['mean'].p_target(patches)
        t_spread = adapters['spread'].p_target(patches)
        if torch.allclose(t_mean, t_spread):
            raise AssertionError('the role targets coincide between the two arms')
    print('  OK the encoder inputs are bit-for-bit identical')
    print('  OK the role targets differ')
    return True


def check_three():
    print('CHECK 3 the arm trains and reloads on invented patients')
    from research_models.common_private_synergy.spread_arm import run_spread_arm

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = invent_prepared(root)
        c = {'schema': 'full_slot_spread35_v1', 'study': 'blca', 'fold': 0, 'seed': 3,
             'patches': 12, 'batch_size': 4, 'target_dim': 3, 'common_rank': 1, 'width': 8,
             'hidden': 6, 'bins': 4, 'iterations': 2, 'role_epochs': 1, 'task_epochs': 2,
             'patience': 2, 'role_lr': .003, 'task_lr': .001, 'role_weight': 1.,
             'task_mode': 'joint', 'readout_bound': 2., 'torch_threads': 1, 'device': 'cpu',
             'allocation': {'p': [1, 1], 'g': [1, 1]}, 'p_target_mode': 'spread',
             'fixed_task_budget': True, 'read_diagnostics': True, 'ablation_nll': True}
        result = run_spread_arm(root, root / 'run', manifest, c, {}, synthetic=True)
        if result['status'] != 'completed':
            raise AssertionError('the spread arm did not complete')
        if not result['checkpoint_reload_verified']:
            raise AssertionError('the checkpoint reload was not verified')
        if not result['reference_frozen']:
            raise AssertionError('the role reference was not frozen')
        if len(result['epoch_logs']) != c['role_epochs'] + c['task_epochs']:
            raise AssertionError('the epoch budget differs from the configuration')
    print('  OK completed=%s epochs=%d' % (result['status'], len(result['epoch_logs'])))
    return True


def check_four():
    print('CHECK 4 the archived sources this arm depends on are unchanged')
    import json
    receipt = ROOT / 'local_evidence/full_slot_matched35_20261113_01/paired_receipt.json'
    if not receipt.is_file():
        print('  SKIP no archived receipt in this checkout')
        return True
    recorded = json.loads(receipt.read_text(encoding='utf-8'))['source_SHA256']
    checked = 0
    for name, value in recorded.items():
        if not (ROOT / name).is_file():
            raise AssertionError('archived source is missing: ' + name)
        if sha(ROOT / name) != value:
            raise AssertionError('archived source changed: ' + name)
        checked += 1
    print('  OK %d recorded source hashes still match' % checked)
    return True


def main():
    check_one()
    check_two()
    check_three()
    check_four()
    print('ALL SYNTHETIC SPREAD-ARM CHECKS PASSED')


if __name__ == '__main__':
    main()
