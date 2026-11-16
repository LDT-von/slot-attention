"""Invented data only: lambda routing, historical gate and fixed35 reuse."""
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from research_models.common_private_synergy.components.losses import discrete_hazard_nll
from research_models.common_private_synergy.components.role_reference import role_content_losses
from research_models.common_private_synergy.full_slot_engine import run_full, task_output
from research_models.common_private_synergy.lambda05 import run_lambda05, validate_protocol
from research_models.common_private_synergy.matched35 import run_matched
from research_models.common_private_synergy.patient_pilot import ROOT, sha
from research_models.common_private_synergy.patient_pilot_engine import PreparedStore, batch, load_checkpoint
from research_models.common_private_synergy.deferred_checks.test_matched35 import invented


def routed_config(endpoint):
    return {**endpoint, 'schema': 'full_slot_lambda05_v1',
            'role_survival_gradient_scale': .5, 'head_survival_gradient_scale': 1.}


def test_lambda_half_reuses_start_draws_and_runs35_new_task_epochs(tmp_path):
    root, manifest, endpoint = invented(tmp_path, role_epochs=1, task_epochs=35)
    endpoint.update(batch_size=10, iterations=1)
    anchors = tmp_path/'anchors'
    run_matched(root, anchors, manifest, endpoint, {}, synthetic=True)
    checkpoint = anchors/'shared_role/role_last.pt'
    before = sha(checkpoint)
    c = routed_config(endpoint)
    result = run_lambda05(root, tmp_path/'half', manifest, c, {}, run_root=anchors, synthetic=True)
    assert result['status'] == 'completed'
    assert result['new_role_epochs'] == 0
    assert result['new_task_epochs'] == result['total_new_epochs'] == 35
    assert result['optimizer_steps'] == {'role': 0, 'task': 35}
    assert result['last']['epoch'] == 35
    assert result['endpoint_start_and_draws_verified']
    assert result['role_survival_gradient_scale'] == .5
    assert result['head_survival_gradient_scale'] == 1.
    assert result['checkpoint_reload_verified'] and result['reference_frozen']
    assert result['role_parameters_changed_during_task']
    assert sha(checkpoint) == before
    assert not (tmp_path/'half/role_last.pt').exists()
    for row in result['epoch_logs']:
        assert row['phase'] == 'task'
        assert row['validation']['ablation_delta_NLL']
        assert row['train_read_diagnostics']
    last = torch.load(tmp_path/'half/task_last.pt', weights_only=True)['state_dict']
    start = torch.load(checkpoint, weights_only=True)['state_dict']
    assert any(not torch.equal(v, last[k]) for k, v in start.items()
               if k.startswith('readout.') and isinstance(v, torch.Tensor))


def test_routing_scales_only_survival_to_roles_not_values_head_or_role_loss(tmp_path):
    root, manifest, c = invented(tmp_path, role_epochs=1, task_epochs=1)
    warm = tmp_path/'warm'
    run_full(root, warm, manifest, c, {}, synthetic=True, warmup_only=True)
    model, adapter, boundaries = load_checkpoint(warm/'role_last.pt')
    model.requires_grad_(True)
    b = batch(PreparedStore(root, manifest), manifest['folds']['0']['train'][:2],
              adapter, boundaries, c, 'cpu', generator=torch.Generator().manual_seed(81))
    results = {scale: task_output(model, b, 'joint', role_scale=scale) for scale in (0., .5, 1.)}
    legacy_detached = task_output(model, b, 'continued_role_only')
    assert torch.equal(results[0.]['prediction'], legacy_detached['prediction'])
    parameters = list(model.encoders.parameters()) + list(model.pools.parameters()) + list(model.decoders.parameters())
    head_parameters = list(model.readout.parameters())
    head_grads, role_grads, role_loss_grads = {}, {}, {}
    for scale, out in results.items():
        assert torch.equal(out['prediction'], results[1.]['prediction'])
        survival = discrete_hazard_nll(out['prediction'], b['completed'], b['event'])
        head_grads[scale] = torch.autograd.grad(survival, head_parameters, retain_graph=True, allow_unused=True)
        role_grads[scale] = torch.autograd.grad(survival, parameters, retain_graph=True, allow_unused=True)
        role_loss = sum(role_content_losses(out['roles'], model.reference.targets(b['p_target'], b['g_target'])).values())/4
        role_loss_grads[scale] = torch.autograd.grad(role_loss, parameters, retain_graph=True, allow_unused=True)
    assert any(x is not None and torch.count_nonzero(x) for x in role_grads[1.])
    assert all(x is None for x in role_grads[0.])
    for half, full in zip(role_grads[.5], role_grads[1.]):
        assert (half is None and full is None) or torch.equal(half, .5*full)
    for scale in (0., .5):
        for actual, expected in zip(head_grads[scale], head_grads[1.]):
            assert (actual is None and expected is None) or torch.equal(actual, expected)
        for actual, expected in zip(role_loss_grads[scale], role_loss_grads[1.]):
            assert (actual is None and expected is None) or torch.equal(actual, expected)


@pytest.mark.parametrize('value', [True, False, float('nan'), float('inf'), -.1, 1.1])
def test_gradient_routing_rejects_invalid_coefficient(tmp_path, value):
    root, manifest, c = invented(tmp_path, role_epochs=1, task_epochs=1)
    warm = tmp_path/'warm'
    run_full(root, warm, manifest, c, {}, synthetic=True, warmup_only=True)
    model, adapter, boundaries = load_checkpoint(warm/'role_last.pt')
    b = batch(PreparedStore(root, manifest), manifest['folds']['0']['train'][:2],
              adapter, boundaries, c, 'cpu', generator=torch.Generator().manual_seed(81))
    with pytest.raises(ValueError, match='scale'):
        task_output(model, b, 'joint', role_scale=value)


@pytest.mark.parametrize('key,value', [('role_survival_gradient_scale', True),
    ('role_survival_gradient_scale', .25), ('head_survival_gradient_scale', .5),
    ('head_survival_gradient_scale', True), ('head_survival_gradient_scale', float('nan')),
    ('task_epochs', 34), ('task_lr', .002), ('fixed_task_budget', False)])
def test_locked_protocol_refuses_unplanned_changes(key, value):
    c = json.loads((ROOT/'configs/common_private_lambda05_v1.json').read_text(encoding='utf-8'))
    validate_protocol(c)
    c[key] = value
    with pytest.raises(ValueError):
        validate_protocol(c)


def test_bad_chain_blocks_before_model_or_checkpoint(tmp_path, monkeypatch):
    c = json.loads((ROOT/'configs/common_private_lambda05_v1.json').read_text(encoding='utf-8'))
    def reject(*args):
        raise ValueError('deliberately invalid historical chain')
    fake = types.ModuleType('scripts.verify_matched35_chain')
    fake.verify_chain = reject
    monkeypatch.setitem(sys.modules, 'scripts.verify_matched35_chain', fake)
    manifest = {'folds': {'0': {'train': list(range(304)), 'val': list(range(304, 380))}}}
    with pytest.raises(ValueError, match='invalid historical chain'):
        run_lambda05(tmp_path/'no_prepared', tmp_path/'out', manifest, c, {},
                     run_root=tmp_path/'no_checkpoint', original_exporter=tmp_path/'no_script',
                     aggregates=tmp_path/'no_aggregate', split_guard=lambda: {})
    assert not (tmp_path/'out').exists()


def test_default_plan_is_stdlib_only_and_does_not_read_missing_inputs(tmp_path):
    result = subprocess.run([sys.executable, '-S', '-B', str(ROOT/'scripts/run_common_private_lambda05.py'),
                             '--run-root', str(tmp_path/'missing'), '--prepared-root', str(tmp_path/'absent')],
                            cwd=tmp_path, check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    assert plan['execution'] is False
    assert plan['total_new_epochs'] == 35 and plan['new_role_epochs'] == 0
    assert plan['role_survival_gradient_scale'] == .5 and plan['head_survival_gradient_scale'] == 1.
    assert not list(tmp_path.iterdir())
