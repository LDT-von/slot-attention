import torch
import pytest

from scripts.check_xor_readout_capacity import network
from scripts.compare_common_private_club import state_hash
from slotspe_proof.controlled_slot_readout import ControlledSlotReadout, controlled_pool, hadamard
from slotspe_proof.slot_pooling_diagnostics import RoleBankReadout


def test_fixed_signed_projection_retains_all_linear_role_coordinates():
    values = torch.arange(32).float().reshape(2, 4, 4)
    encoded = values.repeat(1, 1, 4)
    z = controlled_pool(encoded, torch.arange(4), 'signed', 'mean').reshape(2, 4, 4)
    reconstructed = torch.einsum('kr,brd->bkd', hadamard(4), z)
    assert torch.equal(values, reconstructed)
    assert torch.equal(hadamard(8)@hadamard(8).T, 8*torch.eye(8))
    with pytest.raises(ValueError): hadamard(3)


def test_pooling_controls_differ_only_by_role_and_scale():
    value = torch.randn(7, 4, 16)
    ids = torch.arange(4); perm = torch.tensor([3, 0, 1, 2])
    for role in ['blind', 'signed']:
        own = controlled_pool(value, ids, role, 'mean')
        assert torch.allclose(4*own, controlled_pool(value, ids, role, 'sum'))
        assert torch.allclose(own, controlled_pool(value[:, perm], ids[perm], role, 'mean'), atol=2e-7)
    assert not torch.allclose(controlled_pool(value, ids, 'signed', 'mean'), controlled_pool(value[:, perm], ids, 'signed', 'mean'))


def test_factorial_models_share_every_weight_within_phi_shape():
    reference = network(419)
    g = torch.randn(6, 4); v = [torch.randn(6, 4, 4) for _ in range(2)]
    expected = reference.readout(g, v)
    for shape, count in [('full', 1396), ('block', 1402)]:
        hashes = []
        for role in ['blind', 'signed']:
            for scale in ['mean', 'sum']:
                before = torch.get_rng_state().clone()
                model = ControlledSlotReadout(reference, phi_shape=shape, role_access=role, scale=scale, seed=900419)
                assert torch.equal(before, torch.get_rng_state())
                assert sum(p.numel() for p in model.parameters()) == count
                hashes.append(state_hash(model))
                if (shape, role, scale) == ('full', 'blind', 'mean'):
                    assert torch.equal(model(g, v), expected)
        assert len(set(hashes)) == 1


def test_all_slot_gradients_and_paired_role_permutation():
    model = ControlledSlotReadout(network(431), phi_shape='full', role_access='signed', scale='mean', seed=900431)
    v = [torch.randn(5, 4, 4, requires_grad=True) for _ in range(2)]
    g = torch.zeros(5, 4); ids = torch.tensor([2, 3, 0, 1])
    own = model(g, v)
    assert torch.allclose(own, model(g, [x[:, ids] for x in v], [ids]*2), atol=1e-7)
    own.square().sum().backward()
    assert all(bool((x.grad.abs().sum((0, 2)) > 0).all()) for x in v)


def test_actual_phi_can_exactly_preserve_raw_slot_values_in_signed_pool():
    # GELU(x)-GELU(-x)=x: both parameterizations contain this identity map.
    for shape in ['full', 'block']:
        model = ControlledSlotReadout(network(419), phi_shape=shape, role_access='signed', scale='mean', seed=900419).double()
        phi = model.phi[0]
        with torch.no_grad():
            for layer in [phi[0], phi[2]]:
                layer.weight.zero_(); layer.bias.zero_()
            for j in range(4):
                phi[0].weight[j, j] = 1
                phi[0].weight[j+4, j] = -1
            for out in range(phi[2].out_features):
                phi[2].weight[out, out % 4] = 1
                phi[2].weight[out, out % 4+4] = -1
        value = torch.randn(7, 4, 4, dtype=torch.double)
        encoded = phi(value)
        if shape == 'block': encoded = encoded.repeat(1, 1, 4)
        pooled = controlled_pool(encoded, torch.arange(4), 'signed', 'mean').reshape(7, 4, 4)
        recovered = torch.einsum('kr,brd->bkd', hadamard(4, dtype=torch.double), pooled)
        assert torch.allclose(recovered, value, atol=1e-12, rtol=1e-12)


def test_block_signed_and_role_bank_have_same_function_class():
    reference = network(419)
    bank = RoleBankReadout(reference, 900419).double()
    signed = ControlledSlotReadout(reference, phi_shape='block', role_access='signed', scale='mean', seed=900419).double()
    # z_signed=(H.T/4 tensor I4) z_bank, so compensate the first head layer.
    inverse = torch.kron(hadamard(4, dtype=torch.double), torch.eye(4, dtype=torch.double))
    with torch.no_grad():
        for offset in [4, 20]:
            signed.head[0].weight[:, offset:offset+16].copy_(bank.head[0].weight[:, offset:offset+16]@inverse)
    g = torch.randn(5, 4, dtype=torch.double)
    values = [torch.randn(5, 4, 4, dtype=torch.double, requires_grad=True) for _ in range(2)]
    a, b = bank(g, values), signed(g, values)
    assert torch.allclose(a, b, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad(a.square().sum(), values, retain_graph=True)
    gb = torch.autograd.grad(b.square().sum(), values)
    assert all(torch.allclose(x, y, atol=1e-12, rtol=1e-12) for x, y in zip(ga, gb))


def test_coordinatewise_adam_need_not_preserve_orthogonal_reparameterization():
    rotation = torch.tensor([[1., -1.], [1., 1.]], dtype=torch.double)/(2**.5)
    target = torch.tensor([1., 2.], dtype=torch.double)
    def update(rotated, adam):
        value = torch.nn.Parameter(torch.zeros(2, dtype=torch.double))
        optimizer = torch.optim.Adam([value], lr=.1, eps=1e-12) if adam else torch.optim.SGD([value], lr=.1)
        destination = rotation@target if rotated else target
        (.5*(value-destination).square().sum()).backward(); optimizer.step()
        return rotation.T@value.detach() if rotated else value.detach()
    assert torch.allclose(update(False, False), update(True, False), atol=1e-12)
    assert not torch.allclose(update(False, True), update(True, True), atol=1e-3)
