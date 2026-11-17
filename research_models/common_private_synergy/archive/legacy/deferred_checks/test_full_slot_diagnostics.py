import pytest
import torch

from research_models.common_private_synergy.full_slot_diagnostics import (
    ablate_each_slot, better_validation, role_summary, slot_summary,
)


def test_identical_slots_and_identical_reads_are_detected():
    x = torch.tensor([[[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]]])
    attn = torch.tensor([[[.8, .2], [.8, .2]], [[.5, .5], [.5, .5]]])
    result = slot_summary(x, attn)
    assert result['pair_cosine_mean'] == pytest.approx(1.)
    assert result['read_overlap_mean'] == pytest.approx(1.)
    assert result['across_patient_std_rms'] > 0


def test_zero_and_constant_slots_do_not_masquerade_as_diverse():
    result = slot_summary(torch.zeros(4, 2, 3), torch.full((4, 2, 5), .2))
    assert result['zero_slot_fraction'] == 1.
    assert result['pair_cosine_mean'] is None
    assert result['pair_cosine_undefined_count'] == 4
    assert result['centered_patient_effective_rank_by_slot'] == [0., 0.]
    assert result['read_overlap_mean'] == pytest.approx(1.)
    assert role_summary(torch.ones(4, 3))['centered_patient_effective_rank'] == 0.


def test_different_values_can_still_read_identical_content():
    x = torch.tensor([[[1., 0.], [0., 1.]]] * 3)
    result = slot_summary(x, torch.full((3, 2, 4), .25))
    assert result['pair_cosine_mean'] == pytest.approx(0.)
    assert result['read_overlap_mean'] == pytest.approx(1.)
    assert result['across_patient_std_rms'] == 0.


def test_one_dead_slot_and_nonbinary_constant_are_not_hidden():
    x = torch.zeros(13, 2, 3, dtype=torch.float64)
    x[:, 0, 0] = torch.arange(13, dtype=torch.float64)
    result = slot_summary(x, torch.full((13, 2, 4), .25, dtype=torch.float64))
    assert result['zero_fraction_by_slot'][1] == 1.
    assert result['across_patient_std_rms_by_slot'][1] == 0.
    assert result['across_patient_std_rms_by_slot'][0] > 0.
    assert role_summary(torch.full((13, 3), .1, dtype=torch.float64))['centered_patient_effective_rank'] == 0.


def test_one_slot_pair_is_not_applicable_and_bad_attention_rejected():
    result = slot_summary(torch.ones(2, 1, 3), torch.full((2, 1, 4), .25))
    assert result['pair_status'] == 'not_applicable_one_slot'
    assert result['pair_cosine_mean'] is None and result['read_overlap_mean'] is None
    with pytest.raises(ValueError):
        slot_summary(torch.ones(2, 1, 3), torch.ones(2, 1, 4))


def test_best_selection_exact_ties_and_invalid_values():
    best = {'C_index': .6, 'NLL': .8, 'epoch': 3}
    assert better_validation({'C_index': .61, 'NLL': 2., 'epoch': 4}, best)
    assert better_validation({'C_index': .6, 'NLL': .7, 'epoch': 4}, best)
    assert not better_validation({'C_index': .6, 'NLL': .8, 'epoch': 4}, best)
    assert better_validation({'C_index': .6, 'NLL': .8, 'epoch': 2}, best)
    with pytest.raises(ValueError):
        better_validation({'C_index': float('nan'), 'NLL': .8, 'epoch': 1}, None)


def test_ablation_recomputes_pool_and_uses_empty_group_zero():
    class Ref:
        def project(self, decoded, modality, role):
            return decoded
    class Pool:
        def __call__(self, x):
            return x.mean(1)
    class Model:
        reference = Ref()
        pools = {m + '_' + r: Pool() for m in ('p', 'g') for r in ('common', 'residual')}
        decoders = {k: lambda x: x for k in pools}
        def predict_roles(self, roles):
            return {'prediction': sum(roles.values())}
    model = Model()
    parts = {m: {'common_slots': torch.tensor([[[1.], [3.]]]),
                 'private_slots': torch.tensor([[[2.]]])} for m in ('p', 'g')}
    roles = {m + '_' + r: torch.tensor([[2.]]) for m in ('p', 'g') for r in ('common', 'residual')}
    out = {'parts': parts, 'roles': roles, 'prediction': model.predict_roles(roles)['prediction']}
    change = ablate_each_slot(model, out)
    assert change['p_common'] == pytest.approx([1., 1.])
    assert change['p_residual'] == pytest.approx([2.])


def test_full_epoch_reuses_validation_bags_and_supports_multislot(tmp_path):
    from research_models.common_private_synergy.components.role_reference import FrozenRoleReference, ReferenceProvenance
    from research_models.common_private_synergy.full_slot_epoch import evaluate_epoch
    from research_models.common_private_synergy.patient_pilot_engine import (
        ContentAdapter, PreparedStore, SurvivalRoleModel, collect_targets, fit_boundaries,
    )
    rng = torch.Generator().manual_seed(17)
    manifest = {'patients': {}}
    ids = []
    for index in range(12):
        pid = str(index)
        ids.append(pid)
        torch.save({'patient_id': pid, 'patches': torch.randn(8, 4, generator=rng),
                    'omics': [torch.randn(3, generator=rng), torch.randn(4, generator=rng)]}, tmp_path / (pid + '.pt'))
        manifest['patients'][pid] = {'cache_path': pid + '.pt', 'time': index + 1., 'censor': float(index % 3 == 0)}
    store = PreparedStore(tmp_path, manifest)
    p, g = collect_targets(store, ids[:8])
    adapter = ContentAdapter(p, g, [3, 4], 2)
    tp, tg = adapter.target(p, 'p'), adapter.target(g, 'g')
    ref = FrozenRoleReference.estimated(tp, tg, common_rank=1, coordinate_system='target',
        provenance=ReferenceProvenance('train', 'invented', 'invented'))
    config = {'allocation': {'p': [2, 2], 'g': [2, 2]}, 'width': 4, 'iterations': 2,
              'hidden': 4, 'bins': 4, 'readout_bound': 2., 'batch_size': 2, 'patches': 4}
    model = SurvivalRoleModel(ref, adapter, ref.targets(tp, tg), config).double()
    times = torch.tensor([manifest['patients'][i]['time'] for i in ids[:8]], dtype=torch.float64)
    events = torch.tensor([manifest['patients'][i]['censor'] < 1 for i in ids[:8]])
    boundaries = fit_boundaries(times, events, 4)
    first = evaluate_epoch(model, store, ids[8:], adapter, boundaries, config, 'cpu', seed=99, survival=True)
    torch.manual_seed(889)
    second = evaluate_epoch(model, store, ids[8:], adapter, boundaries, config, 'cpu', seed=99, survival=True)
    assert first == second
    assert first['C_index_status'] == 'computed'
    assert first['slots']['p_common']['slots'] == 2
    assert len(first['ablation_mean_abs_logit_change']['p_common']) == 2
    role_only = evaluate_epoch(model, store, ids[8:], adapter, boundaries, config, 'cpu', seed=99, survival=False)
    assert 'C_index' not in role_only and not role_only['ablation_mean_abs_logit_change']


@pytest.mark.parametrize('mode', ['frozen', 'joint'])
@pytest.mark.parametrize('starved', [False, True])
def test_full_training_checkpoint_and_gradient_paths(tmp_path, mode, monkeypatch, starved):
    import json
    from research_models.common_private_synergy import full_slot_engine
    from research_models.common_private_synergy.components.losses import discrete_hazard_nll
    from research_models.common_private_synergy.components.role_reference import role_content_losses
    from research_models.common_private_synergy.patient_pilot_engine import PreparedStore, batch, load_checkpoint, model_forward
    if starved:
        from research_models.common_private_synergy.components import slots as slot_components
        original_read = slot_components.masked_slot_read
        def starved_read(logits, real_mask):
            bias = logits.new_zeros(1, logits.shape[1], 1)
            bias[:, -1] = -1000
            return original_read(logits + bias, real_mask)
        monkeypatch.setattr(slot_components, 'masked_slot_read', starved_read)
    original_save = full_slot_engine._save_checkpoint
    def capture(path, checkpoint):
        original_save(path, checkpoint)
        if path.name == 'task_last.pt' and checkpoint['epoch'] in (1, 2):
            torch.save(checkpoint, tmp_path / ('capture' + str(checkpoint['epoch']) + '.pt'))
    monkeypatch.setattr(full_slot_engine, '_save_checkpoint', capture)
    root = tmp_path / 'input'
    (root / 'prepared').mkdir(parents=True)
    manifest = {'patients': {}, 'omic_sizes': [3, 4], 'folds': {'0': {'train': [], 'val': []}}}
    rng = torch.Generator().manual_seed(103)
    for index in range(16):
        pid = str(index)
        rel = 'prepared/' + pid + '.pt'
        torch.save({'patient_id': pid, 'patches': torch.randn(8, 4, generator=rng),
                    'omics': [torch.randn(3, generator=rng), torch.randn(4, generator=rng)]}, root / rel)
        manifest['patients'][pid] = {'cache_path': rel, 'time': index + 1., 'censor': float(index % 3 == 0)}
        manifest['folds']['0']['train' if index < 10 else 'val'].append(pid)
    (root / 'prepared/manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    c = {'fold': 0, 'seed': 3, 'device': 'cpu', 'torch_threads': 1, 'target_dim': 2,
         'common_rank': 1, 'allocation': {'p': [2, 2], 'g': [2, 2]}, 'width': 4,
         'iterations': 2, 'hidden': 4, 'bins': 4, 'readout_bound': 2., 'batch_size': 2,
         'patches': 4, 'role_epochs': 2, 'task_epochs': 3, 'role_lr': .003,
         'task_lr': .001, 'role_weight': 1., 'patience': 3, 'task_mode': mode}
    result = full_slot_engine.run_full(root, tmp_path / 'output', manifest, c, {}, synthetic=True)
    assert result['status'] == 'completed' and result['checkpoint_reload_verified']
    assert len(result['epoch_logs']) == 5
    assert result['best']['epoch'] in (1, 2, 3)
    assert result['last']['epoch'] == 3
    assert result['read_normalization']['policy'] == 'masked_add_eps_l1_v2'
    if starved:
        assert result['epoch_logs'][0]['train_raw_reading_mass']['p']['below_threshold_count'] > 0
        assert result['epoch_logs'][0]['validation']['raw_reading_mass']['g_residual']['below_threshold_count'] > 0
    saved_role = torch.load(tmp_path / 'output/role_last.pt', weights_only=True)
    saved_last = torch.load(tmp_path / 'output/task_last.pt', weights_only=True)
    assert saved_last['optimizer_state'] and saved_last['train_generator_state'] is not None
    assert saved_last['selection_state']['best'] == result['best']
    reconstructed, _, _ = load_checkpoint(tmp_path / 'output/task_last.pt')
    if mode == 'joint':
        reconstructed.requires_grad_(True)
        parameters = reconstructed.parameters()
    else:
        reconstructed.readout.requires_grad_(True)
        parameters = reconstructed.readout.parameters()
    restored_optimizer = torch.optim.Adam(parameters, lr=c['task_lr'])
    restored_optimizer.load_state_dict(saved_last['optimizer_state'])
    restored_rng = torch.Generator()
    restored_rng.set_state(saved_last['train_generator_state'])
    assert restored_optimizer.state_dict()['param_groups'] == saved_last['optimizer_state']['param_groups']
    assert torch.equal(restored_rng.get_state(), saved_last['train_generator_state'])
    before = saved_role['state_dict']['encoders.p.common_queries']
    after = saved_last['state_dict']['encoders.p.common_queries']
    assert torch.equal(before, after) == (mode == 'frozen')
    grad = result['epoch_logs'][-1]['raw_slot_gradient_mean']['combined']['p']
    assert (grad is None) == (mode == 'frozen')
    survival_grad = result['epoch_logs'][-1]['raw_slot_gradient_mean']['survival']['p']
    assert (survival_grad is None) == (mode == 'frozen')
    if mode == 'joint':
        assert any(value > 0 for value in survival_grad)
    # Reconstruct the exact second task epoch from the first checkpoint. This
    # checks optimizer and sampling state, not merely model inference reload.
    state1 = torch.load(tmp_path / 'capture1.pt', weights_only=True)
    state2 = torch.load(tmp_path / 'capture2.pt', weights_only=True)
    resumed, adapter, boundaries = load_checkpoint(tmp_path / 'capture1.pt')
    if mode == 'joint':
        resumed.requires_grad_(True)
        parameters = resumed.parameters()
    else:
        resumed.readout.requires_grad_(True)
        parameters = resumed.readout.parameters()
    optimizer = torch.optim.Adam(parameters, lr=c['task_lr'])
    optimizer.load_state_dict(state1['optimizer_state'])
    generator = torch.Generator()
    generator.set_state(state1['train_generator_state'])
    torch.set_rng_state(state1['torch_rng_state'])
    train = manifest['folds']['0']['train']
    order = torch.randperm(len(train), generator=generator).tolist()
    store = PreparedStore(root, manifest)
    resumed.train()
    for start in range(0, len(train), c['batch_size']):
        ids = [train[i] for i in order[start:start + c['batch_size']]]
        b = batch(store, ids, adapter, boundaries, c, 'cpu', generator=generator)
        output = model_forward(resumed, b)
        loss = discrete_hazard_nll(output['prediction'], b['completed'], b['event'])
        if mode == 'joint':
            targets = resumed.reference.targets(b['p_target'], b['g_target'])
            loss = loss + c['role_weight'] * sum(role_content_losses(output['roles'], targets).values()) / 4
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    for key, value in resumed.state_dict().items():
        expected = state2['state_dict'][key]
        assert (torch.equal(value, expected) if isinstance(value, torch.Tensor) else value == expected), key
    assert torch.equal(generator.get_state(), state2['train_generator_state'])
