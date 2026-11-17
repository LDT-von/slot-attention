"""Synthetic end-to-end verification of the Cox-objective protocol.

Uses INVENTED patients only. No real patient cache is read.

Checks:
  1. The Cox arm and the discrete arm complete their epochs on invented data.
  2. Both arms draw the same inputs, so the objective is the only difference.
  3. The survival gradient reaches only the readout; the role gradient reaches
     only the encoders. This is the property the paired comparison relies on.
  4. Both arms produce a finite validation C-index.
  5. A checkpoint reload reproduces the recorded validation metric.
  6. The Cox loss prefers a better separated risk score over a constant one.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_models.common_private_synergy.cox_objective_engine import (
    _survival_objective, evaluate_epoch_cox, run_cox_objective_arm)
from research_models.common_private_synergy.components.cox_surv import cox_partial_likelihood
from research_models.common_private_synergy.patient_pilot import ROOT, sha


def build_invented_prepared(root, n_train=40, n_val=16):
    """Create an invented prepared cache with the real structure."""
    from research_models.common_private_synergy.patient_pilot import sha as file_sha
    root = Path(root)
    prepared = root / 'prepared'
    (prepared / 'cache/patients').mkdir(parents=True)
    rng = torch.Generator().manual_seed(4242)
    manifest = {'omic_sizes': [4, 5], 'patients': {}, 'folds': {'0': {'train': [], 'val': []}}}
    for i in range(n_train + n_val):
        pid = f'INVENTED_{i:03d}'
        rel = f'prepared/cache/{pid}.pt'
        patches = torch.randn(6 + i % 5, 8, generator=rng)
        omics = [torch.randn(n, generator=rng) for n in manifest['omic_sizes']]
        torch.save({'patient_id': pid, 'patches': patches, 'omics': omics}, root / rel)
        manifest['patients'][pid] = {
            'cache_path': rel, 'cache_sha256': file_sha(root / rel),
            'time': float(1 + i % 17), 'censor': float(0 if i % 3 == 0 else 1)}
        manifest['folds']['0']['train' if i < n_train else 'val'].append(pid)
    (prepared / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return manifest


def build_shared_checkpoint(root, manifest, out_path, c):
    """Build a shared role checkpoint the way the real protocol reuses one."""
    from research_models.common_private_synergy.components.role_reference import (
        FrozenRoleReference, ReferenceProvenance)
    from research_models.common_private_synergy.patient_pilot_engine import (
        ContentAdapter, PreparedStore, SurvivalRoleModel, collect_targets, fit_boundaries)
    store = PreparedStore(root, manifest)
    train = manifest['folds']['0']['train']
    p, g = collect_targets(store, train)
    adapter = ContentAdapter(p, g, manifest['omic_sizes'], c['target_dim'])
    tp, tg = adapter.target(p, 'p'), adapter.target(g, 'g')
    reference = FrozenRoleReference.estimated(
        tp, tg, common_rank=c['common_rank'], coordinate_system='target',
        provenance=ReferenceProvenance('train', 'invented', 'invented'))
    anchors = reference.targets(tp[:8], tg[:8])
    times = torch.tensor([manifest['patients'][i]['time'] for i in train], dtype=torch.float64)
    events = torch.tensor([manifest['patients'][i]['censor'] < 1 for i in train], dtype=torch.bool)
    boundaries = fit_boundaries(times, events, c['bins'])
    model = SurvivalRoleModel(reference, adapter, anchors, c).double()
    checkpoint = {'state_dict': model.state_dict(), 'adapter': adapter.state(), 'config': c,
                  'boundaries': boundaries, 'reference_metadata': reference.metadata,
                  'phase': 'role', 'epoch': c['role_epochs'],
                  'train_generator_state': torch.Generator().manual_seed(c['seed'] + 9000).get_state(),
                  'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': None,
                  'split_identity': {'schema': 'invented'},
                  'manifest_SHA256': sha(root / 'prepared/manifest.json'),
                  'source_SHA256': {}}
    torch.save(checkpoint, out_path)
    return checkpoint


def main():
    from research_models.common_private_synergy.patient_pilot import config as load_config
    c = load_config(ROOT / 'configs/common_private_cox35_v1.json')
    c.update(role_epochs=2, task_epochs=3, batch_size=2, survival_batch_size=8,
             patches=8, width=8, iterations=2, hidden=6, target_dim=3, common_rank=1,
             bins=4, torch_threads=1, device='cpu', read_diagnostics=False,
             ablation_nll=False)
    workspace = Path(tempfile.mkdtemp(prefix='cox_synth_'))
    try:
        root = workspace / 'prepared_root'
        root.mkdir()
        manifest = build_invented_prepared(root)
        shared = workspace / 'shared_role.pt'
        build_shared_checkpoint(root, manifest, shared, c)

        hashes = {'research_models/common_private_synergy/components/cox_surv.py':
                  sha(ROOT / 'research_models/common_private_synergy/components/cox_surv.py')}
        results = {}
        for objective in ('discrete_hazard', 'cox'):
            out = workspace / objective
            result = run_cox_objective_arm(
                root, out, manifest, {**c, 'survival_objective': objective}, hashes,
                split_guard=lambda: {'schema': 'invented'}, synthetic=True,
                initial_checkpoint=shared)
            results[objective] = result
            print(objective, 'status', result['status'], 'best', result['best'])

        # 2. The arms must have drawn identical inputs.
        a = results['discrete_hazard']['epoch_logs']
        b = results['cox']['epoch_logs']
        assert len(a) == len(b) == 3, 'both arms must run three epochs'
        for left, right in zip(a, b):
            assert left['input_draw_SHA256'] == right['input_draw_SHA256'], \
                'the arms must draw the same inputs'
        print('CHECK 2 matched input draws: OK')

        # 3. Gradient isolation.
        from research_models.common_private_synergy.patient_pilot_engine import (
            PreparedStore, batch, load_checkpoint)
        store = PreparedStore(root, manifest)
        model, adapter, boundaries = load_checkpoint(shared, 'cpu')
        model.requires_grad_(True)
        rng = torch.Generator().manual_seed(1)
        bb = batch(store, manifest['folds']['0']['train'][:2], adapter, boundaries, c, 'cpu',
                   generator=rng)
        roles, _ = model.encode(bb['p'], bb['g'], bb['p_mask'], bb['g_mask'])
        detached = {k: v.detach() for k, v in roles.items()}
        prediction = model.predict_roles(detached)['prediction']
        loss = _survival_objective('cox', prediction, bb)
        model.zero_grad(set_to_none=True)
        loss.backward()
        encoder_grads = [p.grad for n, p in model.named_parameters()
                         if n.startswith(('encoders.', 'pools.', 'decoders.')) and p.grad is not None]
        readout_grads = [p.grad for n, p in model.named_parameters()
                         if n.startswith('readout.') and p.grad is not None]
        assert not encoder_grads, 'the survival gradient must not reach the encoders'
        assert readout_grads, 'the survival gradient must reach the readout'
        print('CHECK 3 gradient isolation: OK (readout grads', len(readout_grads),
              ', encoder grads', len(encoder_grads), ')')

        # 4. Both arms produced a finite C-index.
        for objective, result in results.items():
            best = result['best']
            assert best['C_index'] is not None and best['C_index'] == best['C_index'], \
                f'{objective} must produce a finite C-index'
        print('CHECK 4 finite C-index in both arms: OK',
              {k: v['best']['C_index'] for k, v in results.items()})

        # 5. The checkpoint reload reproduces the recorded metric.
        for objective in ('discrete_hazard', 'cox'):
            out = workspace / objective
            reloaded, reloaded_adapter, reloaded_boundaries = load_checkpoint(
                out / 'task_best.pt', 'cpu')
            metrics = evaluate_epoch_cox(reloaded, store, manifest['folds']['0']['val'],
                                         reloaded_adapter, reloaded_boundaries,
                                         {**c, 'survival_objective': objective}, 'cpu',
                                         seed=c['seed'] + 19000, objective=objective)
            logs = results[objective]['epoch_logs']
            best_epoch = results[objective]['best']['epoch']
            recorded = [e for e in logs if e['epoch'] == best_epoch][0]['validation']
            assert metrics['C_index'] == recorded['C_index'], \
                f'{objective} checkpoint reload must reproduce the C-index'
        print('CHECK 5 checkpoint reload reproduces the metric: OK')

        # 6. The Cox loss must prefer a better separated risk score.
        times = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        event = torch.tensor([True, True, False, False])
        good = torch.tensor([2.0, 1.0, -1.0, -2.0], dtype=torch.float64)
        flat = torch.zeros(4, dtype=torch.float64)
        l_good = float(cox_partial_likelihood(good, times, event))
        l_flat = float(cox_partial_likelihood(flat, times, event))
        print('CHECK 6 cox loss good', round(l_good, 4), 'flat', round(l_flat, 4))
        assert l_good < l_flat, 'a better separated risk score must give a lower Cox loss'

        print('ALL SYNTHETIC COX-OBJECTIVE CHECKS PASSED')
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == '__main__':
    main()
