"""Synthetic contracts for the V3 relation and matched-capacity residuals."""

import pytest
import torch
from torch import nn

from slotspe_proof.slot_relations import SlotRelationResidual


def _inputs(batch=3, slots=4, output_dim=5, private_dim=3, shared_dim=2):
    generator = torch.Generator().manual_seed(541)
    return (
        torch.randn(batch, slots, output_dim, generator=generator),
        torch.randn(batch, shared_dim, generator=generator),
        torch.randn(batch, slots, private_dim, generator=generator),
    )


@pytest.mark.parametrize("mode", ["relation", "capacity"])
def test_initial_residual_is_exact_identity(mode):
    module = SlotRelationResidual(3, 2, 5, mode=mode)
    latent, shared, private = _inputs()
    assert torch.equal(module(latent, shared, private), latent)
    assert torch.count_nonzero(module.g[0].weight) > 0
    assert torch.count_nonzero(module.g[-1].weight) == 0
    assert torch.count_nonzero(module.g[-1].bias) == 0


@pytest.mark.parametrize("mode", ["relation", "capacity"])
def test_zero_last_layer_has_live_gradient_path(mode):
    module = SlotRelationResidual(3, 2, 5, mode=mode)
    latent, shared, private = _inputs()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.2)
    module(latent, shared, private).square().mean().backward()
    assert module.g[-1].weight.grad.abs().sum() > 0
    assert torch.count_nonzero(module.g[0].weight.grad) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    module(latent, shared, private).square().mean().backward()
    assert module.g[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())


@pytest.mark.parametrize("mode", ["relation", "capacity"])
def test_permutation_equivariance_with_nonzero_residual(mode):
    module = SlotRelationResidual(3, 2, 5, mode=mode).double()
    with torch.no_grad():
        module.g[-1].weight.fill_(0.07)
        module.g[-1].bias.fill_(0.02)
    latent, shared, private = (value.double() for value in _inputs())
    permutation = torch.tensor([2, 0, 3, 1])
    expected = module(latent, shared, private)[:, permutation]
    actual = module(latent[:, permutation], shared, private[:, permutation])
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ["relation", "capacity"])
def test_disabled_is_same_tensor_and_preserves_rng_and_gradients(mode):
    module = SlotRelationResidual(3, 2, 5, mode=mode)
    latent, shared, private = _inputs()
    latent.requires_grad_()
    shared.requires_grad_()
    private.requires_grad_()
    before = torch.random.get_rng_state().clone()
    result = module(latent, shared, private, enabled=False)
    assert result is latent
    assert torch.equal(before, torch.random.get_rng_state())
    result.sum().backward()
    assert torch.equal(latent.grad, torch.ones_like(latent))
    assert shared.grad is None and private.grad is None
    assert all(parameter.grad is None for parameter in module.parameters())
    # The disabled branch must not inspect or evaluate the unused branches.
    assert module(latent, None, None, enabled=False) is latent


def test_constructor_isolates_cpu_rng_and_never_calls_global_manual_seed(monkeypatch):
    before = torch.random.get_rng_state().clone()

    def reject_global_seed(*args, **kwargs):
        raise AssertionError("global manual_seed would reseed CUDA")

    monkeypatch.setattr(torch, "manual_seed", reject_global_seed)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", reject_global_seed)
    first = SlotRelationResidual(seed=901)
    assert torch.equal(before, torch.random.get_rng_state())
    second = SlotRelationResidual(seed=901)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    other = SlotRelationResidual(seed=902)
    assert not torch.equal(first.g[0].weight, other.g[0].weight)


def test_native_parameter_counts_and_declared_architectures():
    relation = SlotRelationResidual()
    capacity = SlotRelationResidual(mode="capacity")
    assert relation.g[0].in_features == 640
    assert relation.g[0].out_features == 128
    assert capacity.g[0].in_features == 256
    assert capacity.g[0].out_features == 224
    assert isinstance(relation.g[1], nn.ReLU)
    assert sum(p.numel() for p in relation.parameters()) == 115072
    assert sum(p.numel() for p in capacity.parameters()) == 115168
    assert (115168 - 115072) / 115072 < 0.001


@pytest.mark.parametrize("mode", ["relation", "capacity"])
def test_native_shape(mode):
    module = SlotRelationResidual(mode=mode)
    generator = torch.Generator().manual_seed(74)
    latent = torch.randn(2, 8, 256, generator=generator)
    shared = torch.randn(2, 64, generator=generator)
    private = torch.randn(2, 8, 192, generator=generator)
    assert module(latent, shared, private).shape == latent.shape


@pytest.mark.parametrize(
    "kwargs",
    [
        {"private_dim": 0}, {"private_dim": True}, {"shared_dim": -1},
        {"output_dim": 1.5}, {"mode": "unknown"}, {"seed": -1}, {"seed": True},
    ],
)
def test_invalid_constructor(kwargs):
    with pytest.raises(ValueError):
        SlotRelationResidual(**kwargs)


@pytest.mark.parametrize(
    "case",
    ["rank", "latent_dim", "batch", "slots", "shared_dim", "private_dim", "dtype", "integer", "nan", "inf", "non_tensor", "empty", "one_slot", "enabled"],
)
def test_invalid_inputs(case):
    module = SlotRelationResidual(3, 2, 5)
    latent, shared, private = _inputs()
    enabled = True
    if case == "rank":
        latent = latent[0]
    elif case == "latent_dim":
        latent = latent[..., :4]
    elif case == "batch":
        shared = shared[:2]
    elif case == "slots":
        private = private[:, :3]
    elif case == "shared_dim":
        shared = shared[:, :1]
    elif case == "private_dim":
        private = private[..., :2]
    elif case == "dtype":
        private = private.double()
    elif case == "integer":
        private = private.long()
    elif case == "nan":
        shared[0, 0] = float("nan")
    elif case == "inf":
        latent[0, 0, 0] = float("inf")
    elif case == "non_tensor":
        private = private.tolist()
    elif case == "empty":
        latent, shared, private = latent[:0], shared[:0], private[:0]
    elif case == "one_slot":
        latent, private = latent[:, :1], private[:, :1]
    elif case == "enabled":
        enabled = 0
    with pytest.raises((ValueError, TypeError)):
        module(latent, shared, private, enabled=enabled)


def _product_module(mode):
    module = SlotRelationResidual(1, 1, 1, mode=mode).double()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
        # Positive product region avoids a ReLU kink in the finite difference.
        module.g[0].weight[0, 2 if mode == "relation" else 0] = 1
        module.g[-1].weight[0, 0] = 1
    return module


def test_controlled_cross_slot_mixed_finite_difference_is_nonzero():
    relation, capacity = (_product_module(mode) for mode in ("relation", "capacity"))
    latent = torch.zeros(1, 2, 1, dtype=torch.double)
    shared = torch.zeros(1, 1, dtype=torch.double)
    private = torch.tensor([[[2.0], [3.0]]], dtype=torch.double)
    step = 1e-3

    def mixed(module):
        values = []
        for first, second in ((1, 1), (1, 0), (0, 1), (0, 0)):
            changed = private.clone()
            changed[0, 0, 0] += first * step
            changed[0, 1, 0] += second * step
            values.append(module(latent, shared, changed)[0, 0, 0].item())
        return (values[0] - values[1] - values[2] + values[3]) / step**2

    assert mixed(relation) == pytest.approx(1.0, abs=1e-8)
    assert mixed(capacity) == pytest.approx(0.0, abs=1e-8)
    # For two slots, a self-message would add Ui squared and violate this value.
    assert relation(latent, shared, private)[0, 0, 0].item() == pytest.approx(6.0)


def test_finite_inputs_that_overflow_do_not_silently_return_nonfinite_outputs():
    module = _product_module("relation").float()
    latent = torch.zeros(1, 2, 1)
    shared = torch.zeros(1, 1)
    private = torch.full((1, 2, 1), 1e30)
    with pytest.raises(FloatingPointError, match="nonfinite output"):
        module(latent, shared, private)


def test_relation_branch_can_fit_the_complete_xor_truth_table():
    module = SlotRelationResidual(1, 1, 1, mode="relation", seed=901)
    bits = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    private = bits[..., None]
    latent = private.clone()
    shared = torch.zeros(4, 1)
    target = (-bits[:, 0] * bits[:, 1])[:, None, None].expand_as(latent)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.03)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        output = module(latent, shared, private)
        loss = (output - target).square().mean()
        loss.backward()
        optimizer.step()
        if loss.item() < 1e-5:
            break
    fitted = module(latent, shared, private)
    assert (fitted - target).square().mean().item() < 1e-3
    assert torch.equal(fitted.sign(), target)
    # This demonstrates representability only, not superiority to a baseline.
