"""Invented tensors only: observation invariance and signed deletion diagnostics."""
import copy

import pytest
import torch

from research_models.common_private_synergy.components.contracts import ArchitectureConfig
from research_models.common_private_synergy.components.losses import discrete_hazard_nll
from research_models.common_private_synergy.components.slots import (
    READ_EPS, RoleSlotEncoder, masked_slot_read, observe_slot_read,
)
from research_models.common_private_synergy.full_slot_diagnostics import (
    ablate_each_slot, read_diagnostics_summary,
)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_stable_log_mass_and_uniform_fraction_in_underflow(dtype):
    logits = torch.zeros(2, 4, 7, dtype=dtype)
    logits[:, -1] = -1000
    mask = torch.tensor([[True] * 3 + [False] * 4, [True] * 7])
    q, k = torch.ones(2, 4, 3, dtype=dtype), torch.ones(2, 7, 3, dtype=dtype)
    attention, mass = masked_slot_read(logits, mask)
    d = observe_slot_read(logits, q, k, mask, mass)
    assert torch.equal(mass[:, -1], torch.zeros(2, dtype=dtype))
    assert torch.isfinite(d['log_pre_epsilon_mass']).all()
    expected = -1000 - torch.log(torch.tensor(3., dtype=torch.float64)) + mask.sum(-1).double().log()
    torch.testing.assert_close(d['log_pre_epsilon_mass'][:, -1], expected)
    assert torch.equal(d['epsilon_uniform_fraction'][:, -1], torch.ones(2, dtype=torch.float64))
    assert torch.equal(d['logit_gap_real_mean'][:, -1], torch.full((2,), -1000., dtype=torch.float64))
    torch.testing.assert_close(attention[:, -1], mask.to(dtype) / mask.sum(-1, keepdim=True))


def test_log_domain_agrees_with_direct_mass_and_ignores_padding():
    rng = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 4, 8, dtype=torch.float64, generator=rng)
    q = torch.randn(2, 4, 3, dtype=torch.float64, generator=rng)
    k = torch.randn(2, 8, 3, dtype=torch.float64, generator=rng)
    mask = torch.tensor([[True] * 3 + [False] * 5, [True] * 6 + [False] * 2])
    _, mass = masked_slot_read(logits, mask)
    first = observe_slot_read(logits, q, k, mask, mass)
    torch.testing.assert_close(first['log_pre_epsilon_mass'], mass.log(), atol=1e-14, rtol=1e-14)
    logits = logits.masked_fill(~mask[:, None, :], 1e10)
    k = k.masked_fill(~mask[:, :, None], 1e10)
    _, mass2 = masked_slot_read(logits, mask)
    second = observe_slot_read(logits, q, k, mask, mass2)
    for name in first:
        torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)
    for b in range(2):
        for s in range(4):
            others = [t for t in range(4) if t != s]
            gap = logits[b, s, mask[b]] - logits[b, others][:, mask[b]].max(0).values
            torch.testing.assert_close(first['logit_gap_real_mean'][b, s], gap.mean(), atol=1e-14, rtol=1e-14)
            assert first['logit_gap_real_min'][b, s] == gap.min()


def test_observation_preserves_forward_gradients_and_rng():
    torch.set_num_threads(1)
    torch.manual_seed(10)
    config = ArchitectureConfig(3, 3, width=4, slots=4, common_slots=2, iterations=3)
    plain = RoleSlotEncoder(3, config).double()
    observed = copy.deepcopy(plain)
    observed.observe_read_diagnostics = True
    x = torch.randn(2, 6, 3, dtype=torch.float64)
    mask = torch.tensor([[True] * 3 + [False] * 3, [True] * 6])
    rng_before = torch.get_rng_state().clone()
    a, b = plain(x, mask), observed(x, mask)
    assert torch.equal(torch.get_rng_state(), rng_before)
    for name in ('slots', 'common_slots', 'private_slots', 'read_attention', 'raw_read_mass'):
        assert torch.equal(a[name], b[name])
    assert 'read_diagnostics' not in a
    assert all(v.shape == (2, 3, 4) and not v.requires_grad for v in b['read_diagnostics'].values())
    a['slots'].square().sum().backward()
    b['slots'].square().sum().backward()
    for (name, p), (other, q) in zip(plain.named_parameters(), observed.named_parameters()):
        assert name == other and torch.equal(p.grad, q.grad), name


def test_summary_retains_iteration_slot_axes_but_no_patient_rows():
    raw = torch.tensor([[[0., 1.], [1., 0.]], [[0., 0.], [1., 1.]], [[1., 1.], [1., 0.]]])
    result = read_diagnostics_summary({'raw_mass': raw, 'query_norm': raw + 1})
    assert result['bag_observations'] == 3 and result['observation_count'] == 12
    assert result['axes'] == ['iteration', 'slot']
    assert result['below_threshold_count_by_iteration_slot'] == [[2, 1], [0, 2]]
    assert result['metrics']['query_norm']['mean_by_iteration_slot'] == (raw + 1).double().mean(0).tolist()
    with pytest.raises(ValueError):
        read_diagnostics_summary({'raw_mass': raw, 'other': torch.ones(2, 2, 2)})
    with pytest.raises(ValueError):
        read_diagnostics_summary({'raw_mass': torch.full((2, 2, 2), float('nan'))})


def test_signed_deleted_nll_matches_manual_repool_and_singleton_zero():
    class Reference:
        def project(self, value, modality, role):
            return value
    class Model:
        reference = Reference()
        pools = {m + '_' + r: lambda x: x.mean(1) for m in ('p', 'g') for r in ('common', 'residual')}
        decoders = {key: lambda x: x for key in pools}
        def predict_roles(self, roles):
            return {'prediction': sum(roles.values())}
    model = Model()
    parts = {m: {'common_slots': torch.tensor([[[1.], [3.]]], dtype=torch.float64),
                 'private_slots': torch.tensor([[[2.]]], dtype=torch.float64)} for m in ('p', 'g')}
    roles = {m + '_' + r: torch.tensor([[2.]], dtype=torch.float64) for m in ('p', 'g') for r in ('common', 'residual')}
    out = {'parts': parts, 'roles': roles, 'prediction': model.predict_roles(roles)['prediction']}
    completed, event = torch.tensor([0]), torch.tensor([True])
    old = ablate_each_slot(model, out)
    result = ablate_each_slot(model, out, completed, event)
    assert old == result['mean_abs_logit_change']
    full = discrete_hazard_nll(out['prediction'], completed, event)
    for key, expected_logits in [('p_common', [9., 7.]), ('p_residual', [6.])]:
        for slot, logit in enumerate(expected_logits):
            nll = float(discrete_hazard_nll(torch.tensor([[logit]], dtype=torch.float64), completed, event))
            assert result['NLL'][key][slot] == pytest.approx(nll)
            assert result['delta_NLL'][key][slot] == pytest.approx(nll - float(full))
    assert result['delta_NLL']['p_common'][0] < 0 < result['delta_NLL']['p_common'][1]
    with pytest.raises(ValueError):
        ablate_each_slot(model, out, completed)


def test_epoch_diagnostic_aggregates_and_nll_weight_uneven_batches(tmp_path):
    from research_models.common_private_synergy.components.role_reference import FrozenRoleReference, ReferenceProvenance
    from research_models.common_private_synergy.full_slot_epoch import evaluate_epoch
    from research_models.common_private_synergy.patient_pilot_engine import (
        ContentAdapter, PreparedStore, SurvivalRoleModel, collect_targets, fit_boundaries, batch, model_forward,
    )
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(17)
    manifest = {'patients': {}}
    ids = [str(i) for i in range(13)]
    for index, pid in enumerate(ids):
        torch.save({'patient_id': pid, 'patches': torch.randn(8, 4, generator=rng),
                    'omics': [torch.randn(3, generator=rng), torch.randn(4, generator=rng)]}, tmp_path / (pid + '.pt'))
        manifest['patients'][pid] = {'cache_path': pid + '.pt', 'time': index + 1., 'censor': float(index % 3 == 0)}
    store = PreparedStore(tmp_path, manifest)
    p, g = collect_targets(store, ids[:8])
    adapter = ContentAdapter(p, g, [3, 4], 2)
    tp, tg = adapter.target(p, 'p'), adapter.target(g, 'g')
    ref = FrozenRoleReference.estimated(tp, tg, common_rank=1, coordinate_system='target',
        provenance=ReferenceProvenance('train', 'invented', 'invented'))
    c = {'allocation': {'p': [2, 2], 'g': [2, 2]}, 'width': 4, 'iterations': 2,
         'hidden': 4, 'bins': 3, 'readout_bound': 2., 'batch_size': 2, 'patches': 4, 'read_diagnostics': True}
    model = SurvivalRoleModel(ref, adapter, ref.targets(tp, tg), c).double()
    times = torch.tensor([manifest['patients'][i]['time'] for i in ids[:8]], dtype=torch.float64)
    events = torch.tensor([manifest['patients'][i]['censor'] < 1 for i in ids[:8]])
    boundaries = fit_boundaries(times, events, 3)
    result = evaluate_epoch(model, store, ids[8:], adapter, boundaries, c, 'cpu', seed=99, survival=True)
    assert result['read_diagnostics']['p_common']['bag_observations'] == 5
    assert result['read_diagnostics']['p_common']['observation_count'] == 20
    manual_nll = {key: [] for key in result['ablation_NLL']}
    generator = torch.Generator().manual_seed(99)
    for start in range(8, len(ids), 2):
        group = ids[start:start + 2]
        b = batch(store, group, adapter, boundaries, c, 'cpu', generator=generator)
        with torch.no_grad():
            out = model_forward(model, b)
            a = ablate_each_slot(model, out, b['completed'], b['event'])
        for key in manual_nll:
            manual_nll[key].append(torch.tensor(a['NLL'][key], dtype=torch.float64) * len(group))
    for key in manual_nll:
        expected = torch.stack(manual_nll[key]).sum(0) / 5
        assert result['ablation_NLL'][key] == pytest.approx(expected.tolist(), abs=1e-12)
        assert result['ablation_delta_NLL'][key] == pytest.approx((expected - result['NLL']).tolist(), abs=1e-12)
