"""Invented patients only: common-start pairing, gradient edges and fixed budget."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from research_models.common_private_synergy.components.losses import discrete_hazard_nll
from research_models.common_private_synergy.components.role_reference import role_content_losses
from research_models.common_private_synergy.full_slot_engine import run_full, task_output
from research_models.common_private_synergy.matched35 import run_matched, validate_protocol
from research_models.common_private_synergy.patient_pilot import ROOT
from research_models.common_private_synergy.patient_pilot_engine import (
    PreparedStore, batch, load_checkpoint,
)


def invented(tmp_path, *, role_epochs=2, task_epochs=3):
    root = tmp_path / 'invented'
    (root / 'prepared').mkdir(parents=True)
    manifest = {'patients': {}, 'omic_sizes': [3, 4],
                'folds': {'0': {'train': [], 'val': []}}}
    rng = torch.Generator().manual_seed(103)
    for index in range(16):
        pid = f'INVENTED_{index}'
        rel = f'prepared/{pid}.pt'
        torch.save({'patient_id': pid, 'patches': torch.randn(5 + index % 5, 4, generator=rng),
                    'omics': [torch.randn(3, generator=rng), torch.randn(4, generator=rng)]}, root / rel)
        manifest['patients'][pid] = {'cache_path': rel, 'time': index + 1.,
                                     'censor': float(index % 3 == 0)}
        manifest['folds']['0']['train' if index < 10 else 'val'].append(pid)
    (root / 'prepared/manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    c = dict(schema='full_slot_matched35_v1', ablation_nll=True, fold=0, seed=3, device='cpu', torch_threads=1, target_dim=2,
             common_rank=1, allocation={'p': [2, 2], 'g': [2, 2]}, width=4,
             iterations=2, hidden=4, bins=4, readout_bound=2., batch_size=2,
             patches=4, role_epochs=role_epochs, task_epochs=task_epochs, role_lr=.003,
             task_lr=.001, role_weight=1., patience=1, task_mode='joint',
             fixed_task_budget=True, read_diagnostics=True)
    return root, manifest, c


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def test_common_start_and_fixed_budget_despite_stale_validation(tmp_path, monkeypatch):
    from research_models.common_private_synergy import full_slot_engine
    root, manifest, c = invented(tmp_path)
    # An improvement at epoch 1 only must not trigger patience=1 stopping.
    monkeypatch.setattr(full_slot_engine, 'better_validation', lambda row, best: best is None)
    out = tmp_path / 'paired'
    paired = run_matched(root, out, manifest, c, {}, synthetic=True)
    assert paired['status'] == 'completed'
    assert paired['total_epochs'] == 8
    assert paired['common_start_verified'] and paired['matched_input_draws_verified']
    assert paired['task_optimizer_reset_verified']
    warmup = torch.load(out / 'shared_role/role_last.pt', weights_only=True)
    arms = [read_json(out / mode / 'receipt.json') for mode in ('joint', 'continued_role_only')]
    for mode, receipt in zip(('joint', 'continued_role_only'), arms):
        assert receipt['best']['epoch'] == 1 and receipt['last']['epoch'] == 3
        assert len(receipt['epoch_logs']) == 3
        assert receipt['optimizer_steps']['role'] == 0
        assert receipt['optimizer_steps']['task'] == 15
        assert receipt['reference_frozen'] and receipt['checkpoint_reload_verified']
        assert receipt['checkpoint_optimizer_and_generator_reload_verified']
        assert receipt['role_parameters_changed_during_task']
        assert receipt['task_optimizer_initial_state_empty']
        assert 'early_stop' not in receipt
        last = torch.load(out / mode / 'task_last.pt', weights_only=True)
        assert any(not torch.equal(v, warmup['state_dict'][k]) for k, v in last['state_dict'].items()
                   if k.startswith('readout.') and isinstance(v, torch.Tensor))
        assert any(not torch.equal(v, warmup['state_dict'][k]) for k, v in last['state_dict'].items()
                   if k.startswith('encoders.') and isinstance(v, torch.Tensor))
        for log in receipt['epoch_logs']:
            assert log['phase'] == 'task'
            assert set(log['train_read_diagnostics']) == {'p_common', 'p_residual', 'g_common', 'g_residual'}
            assert 'ablation_delta_NLL' in log['validation']
            for modality in ('p', 'g'):
                gradients = log['raw_slot_gradient_mean']
                assert any(x > 0 for x in gradients['role'][modality])
                if mode == 'continued_role_only':
                    assert gradients['survival'][modality] == [0.] * 4
                    assert gradients['combined'][modality] == pytest.approx(gradients['role'][modality])
                else:
                    assert any(x > 0 for x in gradients['survival'][modality])
    for key in ('initial_model_tensor_SHA256', 'initial_train_rng_SHA256', 'initial_torch_rng_SHA256',
                'initial_cuda_rng_SHA256', 'task_optimizer_initial_groups', 'task_start_role_parameter_SHA256'):
        assert arms[0][key] == arms[1][key]
    assert [x['input_draw_SHA256'] for x in arms[0]['epoch_logs']] == [x['input_draw_SHA256'] for x in arms[1]['epoch_logs']]
    with pytest.raises(FileExistsError):
        run_matched(root, out, manifest, c, {}, synthetic=True)


def test_two_complete_35_epoch_arms(tmp_path):
    root, manifest, c = invented(tmp_path, role_epochs=1, task_epochs=35)
    c.update(batch_size=10, iterations=1)
    result = run_matched(root, tmp_path/'paired35', manifest, c, {}, synthetic=True)
    assert result['total_epochs'] == 71
    assert result['expected_total_epochs'] == 71
    assert result['shared_role_epochs'] == 1
    for arm in result['arms'].values():
        assert arm['epochs'] == 35 and arm['last']['epoch'] == 35


def test_survival_gradient_detach_preserves_forward_head_and_role_updates(tmp_path):
    root, manifest, c = invented(tmp_path, role_epochs=1)
    warm = tmp_path/'warm'
    run_full(root, warm, manifest, c, {}, synthetic=True, warmup_only=True)
    model, adapter, boundaries = load_checkpoint(warm/'role_last.pt')
    model.requires_grad_(True)
    model.train()
    b = batch(PreparedStore(root, manifest), manifest['folds']['0']['train'][:2],
              adapter, boundaries, c, 'cpu', generator=torch.Generator().manual_seed(91))
    joint = task_output(model, b, 'joint')
    detached = task_output(model, b, 'continued_role_only')
    assert torch.equal(joint['prediction'], detached['prediction'])
    parameters = list(model.readout.parameters())
    encoder_parameters = list(model.encoders.parameters()) + list(model.pools.parameters()) + list(model.decoders.parameters())
    js = discrete_hazard_nll(joint['prediction'], b['completed'], b['event'])
    ds = discrete_hazard_nll(detached['prediction'], b['completed'], b['event'])
    jh = torch.autograd.grad(js, parameters, retain_graph=True, allow_unused=True)
    dh = torch.autograd.grad(ds, parameters, retain_graph=True, allow_unused=True)
    for a, z in zip(jh, dh):
        assert (a is None and z is None) or torch.equal(a, z)
    assert any(x is not None and torch.count_nonzero(x) for x in jh)
    de = torch.autograd.grad(ds, encoder_parameters, retain_graph=True, allow_unused=True)
    assert all(x is None for x in de)
    je = torch.autograd.grad(js, encoder_parameters, retain_graph=True, allow_unused=True)
    assert any(x is not None and torch.count_nonzero(x) for x in je)
    role = sum(role_content_losses(detached['roles'], model.reference.targets(b['p_target'], b['g_target'])).values()) / 4
    rg = torch.autograd.grad(role, encoder_parameters, retain_graph=True, allow_unused=True)
    cg = torch.autograd.grad(ds + c['role_weight']*role, encoder_parameters, allow_unused=True)
    for a, z in zip(rg, cg):
        assert (a is None and z is None) or torch.equal(a, z)


def test_shared_checkpoint_rejects_provenance_and_config_changes(tmp_path):
    root, manifest, c = invented(tmp_path, role_epochs=1)
    warm = tmp_path/'warm'
    run_full(root, warm, manifest, c, {}, synthetic=True, warmup_only=True)
    checkpoint = warm/'role_last.pt'
    with pytest.raises(ValueError, match='protocol differs'):
        run_full(root, tmp_path/'wrong_config', manifest, {**c, 'task_lr': .2}, {},
                 synthetic=True, initial_checkpoint=checkpoint)
    with pytest.raises(ValueError, match='provenance differs'):
        run_full(root, tmp_path/'wrong_identity', manifest, c, {}, synthetic=True,
                 initial_checkpoint=checkpoint, split_guard=lambda: {'schema': 'different'})


def test_source_failure_preserves_failed_pair_receipt(tmp_path):
    root, manifest, c = invented(tmp_path)
    out = tmp_path/'failed_pair'
    with pytest.raises(RuntimeError, match='source changed'):
        run_matched(root, out, manifest, c, {'research_models/common_private_synergy/matched35.py': 'invalid'}, synthetic=True)
    receipt = read_json(out/'paired_receipt.json')
    assert receipt['status'] == 'failed' and receipt['error_type'] == 'RuntimeError'
    assert receipt['arms'] == {}
    assert not (out/'shared_role').exists()


@pytest.mark.parametrize('key,value', [('task_epochs', 34), ('role_epochs', 11), ('fixed_task_budget', False),
                                     ('read_diagnostics', False), ('task_mode', 'frozen'), ('batch_size', True)])
def test_locked_protocol_rejects_mutation(key, value):
    c = read_json(ROOT/'configs/common_private_matched35_v1.json')
    validate_protocol(c)
    c[key] = value
    with pytest.raises(ValueError, match='locked matched35 protocol differs'):
        validate_protocol(c)


def test_default_plan_needs_no_site_packages_or_patient_access(tmp_path):
    result = subprocess.run([sys.executable, '-S', '-B', str(ROOT/'scripts/run_common_private_matched35.py'),
                             '--prepared-root', str(tmp_path/'does_not_exist')],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan['execution'] is False and plan['test_used'] is False
    assert plan['total_epochs'] == 82
    assert plan['config']['task_epochs'] == 35
    assert list(tmp_path.iterdir()) == []


def test_legacy_role_only_keeps_frozen_head_semantics(tmp_path):
    root, manifest, c = invented(tmp_path, role_epochs=1, task_epochs=2)
    c.update(schema='full_slot_research_v1', task_mode='continued_role_only')
    out = tmp_path/'legacy'
    receipt = run_full(root, out, manifest, c, {}, synthetic=True)
    assert receipt['task_gradient_policy'] == 'legacy_role_only_frozen_head'
    before = torch.load(out/'role_last.pt', weights_only=True)['state_dict']
    after = torch.load(out/'task_last.pt', weights_only=True)['state_dict']
    assert all(torch.equal(v, after[k]) for k, v in before.items() if k.startswith('readout.') and isinstance(v, torch.Tensor))
    assert receipt['role_parameters_changed_during_task']
    assert receipt['checkpoint_optimizer_and_generator_reload_verified']
    assert any(v > 0 for v in receipt['epoch_logs'][-1]['raw_slot_gradient_mean']['survival']['p'])
