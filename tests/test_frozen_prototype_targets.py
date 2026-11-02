import pytest
import torch
from scripts.check_frozen_content_targets import generated, fit_targets, state_hash
from slotspe_proof.frozen_prototype_targets import FrozenPrototypeTargets


@pytest.fixture(autouse=True)
def cpu():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]): yield
    torch.set_num_threads(old)


def test_frozen_teacher_has_no_gradient_or_heldout_state_update():
    train = generated(12, 64, 'separated_informative')
    teacher = fit_targets(train, 13)
    before = state_hash(teacher)
    test = generated(14, 16, 'dominant_common')['x'].requires_grad_()
    result = teacher(test)
    assert not result['scores'].requires_grad and not result['labels'].requires_grad
    assert state_hash(teacher) == before and teacher.fit_rows == 64
    assert list(teacher.parameters()) == []


def test_token_order_and_masked_nan_padding_do_not_change_targets():
    pack = generated(15, 64, 'separated_informative'); teacher = fit_targets(pack, 16)
    a = teacher(pack['x']); b = teacher(pack['x'].flip(1))
    padded = torch.cat([pack['x'], torch.full((64, 3, 8), float('nan'))], 1)
    mask = torch.cat([pack['valid'], torch.zeros(64, 3, dtype=torch.bool)], 1)
    c = teacher(padded, mask)
    assert torch.equal(a['labels'], b['labels']) and torch.equal(a['labels'], c['labels'])
    assert torch.equal(a['observed'], c['observed'])
    assert torch.allclose(a['scores'], b['scores'], atol=1e-10)


def test_fitting_deterministic_and_preserves_global_torch_rng():
    pack = generated(17, 64, 'separated_informative')
    before = torch.random.get_rng_state().clone()
    a = fit_targets(pack, 18); b = fit_targets(pack, 18)
    assert state_hash(a) == state_hash(b)
    assert torch.equal(before, torch.random.get_rng_state())


def test_missing_prototype_is_unknown_without_patient_exclusion():
    pack = generated(19, 64, 'separated_informative'); teacher = fit_targets(pack, 20)
    a = teacher(pack['x']); mask = a['token_cluster'] == a['token_cluster'][:, :1]
    b = teacher(pack['x'], mask)
    assert torch.equal(b['observed'].sum(1), torch.ones(64, dtype=torch.long))


def test_degenerate_descriptors_cannot_be_forced_into_four_prototypes():
    pack = generated(21, 64, 'separated_informative'); pack['x'][..., :2] = 0
    with pytest.raises(ValueError, match='distinct'): fit_targets(pack, 22)


def test_constant_content_cannot_masquerade_as_balanced_pca_target():
    pack = generated(23, 64, 'separated_informative'); pack['x'][..., 2:] = 0
    with pytest.raises(ValueError, match='constant'): fit_targets(pack, 24)


def test_fit_weights_each_example_equally_when_one_bag_is_duplicated():
    pack = generated(25, 64, 'separated_informative')
    a = fit_targets(pack, 26)
    x = torch.zeros(64, 24, 8); valid = torch.zeros(64, 24, dtype=torch.bool)
    x[:, :12] = pack['x']; valid[:, :12] = True
    x[0] = pack['x'][0].repeat_interleave(2, dim=0); valid[0] = True
    b = FrozenPrototypeTargets.fit(x, valid, clusters=4, descriptor_columns=(0, 1),
        content_columns=(2, 3, 4, 5, 6, 7), seed=26)
    for name, value in a.state_dict().items():
        assert torch.allclose(value, b.state_dict()[name], atol=1e-9, rtol=0), name
