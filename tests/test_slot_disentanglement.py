"""Generated inputs only: algebraic and numerical checks for V2."""

import itertools

import pytest
import torch

from slotspe_proof.slot_disentanglement import (
    SharedPrivateSlots,
    content_set_loss,
    private_summary,
    separation_terms,
    summary_hsic,
    target_variance_floor,
)


def sample(*shape, seed=17, requires_grad=False):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=torch.float64).requires_grad_(requires_grad)


def test_adapter_shapes_determinism_and_caller_rng_preservation():
    before = torch.default_generator.get_state().clone()
    first = SharedPrivateSlots()
    assert torch.equal(before, torch.default_generator.get_state())
    second = SharedPrivateSlots()
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.equal(before, torch.default_generator.get_state())
    latent, record = first(sample(3, 8, 256).float())
    assert latent.shape == (3, 8, 256)
    assert set(record) == {"shared", "private", "content"}
    assert record["shared"].shape == (3, 64)
    assert record["private"].shape == (3, 8, 192)
    assert record["content"].shape == (3, 8, 32)


def test_adapter_does_not_seed_cuda(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("adapter initialization must not touch CUDA RNG")
    monkeypatch.setattr(torch, "manual_seed", fail)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", fail)
    SharedPrivateSlots(dimension=12, shared_dim=3, private_dim=9, target_dim=4)


def test_adapter_is_permutation_equivariant_and_content_is_private_only():
    model = SharedPrivateSlots(dimension=12, shared_dim=3, private_dim=9, target_dim=4).double()
    slots = sample(3, 8, 12, requires_grad=True)
    permutation = torch.tensor([5, 0, 7, 2, 1, 4, 6, 3])
    latent, record = model(slots)
    permuted, other = model(slots[:, permutation])
    torch.testing.assert_close(permuted, latent[:, permutation])
    torch.testing.assert_close(other["shared"], record["shared"])
    torch.testing.assert_close(other["private"], record["private"][:, permutation])
    torch.testing.assert_close(other["content"], record["content"][:, permutation])
    with torch.no_grad():
        model.shared_encoder.weight.add_(10)
        model.shared_encoder.bias.add_(5)
    _, altered = model(slots)
    torch.testing.assert_close(altered["content"], record["content"], rtol=0, atol=0)


def test_adapter_task_and_content_gradients_reach_both_encoders():
    model = SharedPrivateSlots(dimension=12, shared_dim=3, private_dim=9, target_dim=4).double()
    slots = sample(4, 8, 12, requires_grad=True)
    latent, record = model(slots)
    target = sample(4, 5, 4, seed=51, requires_grad=True)
    loss = latent.square().mean() + content_set_loss(record["content"], target)
    loss.backward()
    assert target.grad is None
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert slots.grad is not None and torch.isfinite(slots.grad).all()


@pytest.mark.parametrize("targets", [1, 3, 8])
def test_constant_set_bound_matches_target_variance_identity(targets):
    target = sample(2, targets, 4)
    constant = sample(2, 1, 4, seed=31)
    pred = constant.expand(-1, 8, -1).clone()
    floor = target_variance_floor(target)
    expected = floor + (constant.squeeze(1) - target.mean(dim=1)).square().mean(dim=-1)
    torch.testing.assert_close(content_set_loss(pred, target), expected.mean())
    assert content_set_loss(pred, target) >= floor.mean() - 1e-14
    mean_pred = target.mean(dim=1, keepdim=True).expand(-1, 8, -1)
    torch.testing.assert_close(content_set_loss(mean_pred, target), floor.mean())


def test_nonconstant_content_can_strictly_beat_constant_floor_and_is_set_invariant():
    target = sample(2, 5, 4, requires_grad=True)
    pred = torch.cat((target.detach().clone(), sample(2, 3, 4, seed=12)), dim=1).requires_grad_()
    loss = content_set_loss(pred, target)
    assert loss == 0
    assert torch.all(target_variance_floor(target) > 0)
    assert not target_variance_floor(target).requires_grad
    permuted = content_set_loss(pred[:, [4, 1, 6, 0, 7, 3, 5, 2]], target[:, [3, 0, 4, 2, 1]])
    torch.testing.assert_close(loss, permuted)
    loss.backward()
    assert target.grad is None
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_hungarian_rectangular_loss_matches_exhaustive_injections():
    pred = sample(2, 4, 3)
    target = sample(2, 2, 3, seed=33)
    exhaustive = []
    for batch in range(2):
        possibilities = [
            (pred[batch, list(indices)] - target[batch]).square().mean()
            for indices in itertools.permutations(range(4), 2)
        ]
        exhaustive.append(torch.stack(possibilities).min())
    torch.testing.assert_close(content_set_loss(pred, target), torch.stack(exhaustive).mean())


def test_private_summary_formula_and_slot_permutation_invariance():
    private = sample(5, 8, 7)
    expected = torch.cat((private.mean(1), private.square().mean(1)), dim=-1)
    torch.testing.assert_close(private_summary(private), expected)
    torch.testing.assert_close(private_summary(private.flip(1)), expected)


def test_hsic_matches_explicit_biased_trace_and_patient_permutation():
    shared = sample(8, 4)
    summary = sample(8, 11, seed=23)
    def gram(value):
        centered = value - value.mean(0)
        normalized = centered / (centered.square().mean(0).sqrt() * value.shape[1] ** 0.5)
        return (-0.5 * (normalized[:, None] - normalized[None, :]).square().sum(-1)).exp()
    center = torch.eye(8, dtype=torch.float64) - torch.ones(8, 8, dtype=torch.float64) / 8
    expected = torch.trace(gram(shared) @ center @ gram(summary) @ center) / 64
    actual = summary_hsic(shared, summary)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)
    assert actual >= 0
    assert actual.dtype == torch.float64
    permutation = torch.tensor([3, 7, 0, 2, 4, 1, 6, 5])
    torch.testing.assert_close(summary_hsic(shared[permutation], summary[permutation]), actual)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_hsic_resists_private_rescaling_in_value_and_scale_gradient(scale):
    shared = sample(8, 5, requires_grad=True)
    private = sample(8, 8, 7, seed=29, requires_grad=True)
    multiplier = torch.tensor(scale, dtype=torch.float64, requires_grad=True)
    value = summary_hsic(shared, private_summary(multiplier * private))
    reference = summary_hsic(shared, private_summary(private))
    torch.testing.assert_close(value, reference, atol=1e-14, rtol=1e-12)
    gradient_shared, gradient_private, gradient_scale = torch.autograd.grad(
        value, (shared, private, multiplier)
    )
    assert torch.isfinite(gradient_shared).all() and gradient_shared.abs().sum() > 0
    assert torch.isfinite(gradient_private).all() and gradient_private.abs().sum() > 0
    assert torch.isfinite(gradient_scale)
    assert abs(gradient_scale.item() * scale) < 1e-12
    assert abs((gradient_private * private).sum().item()) < 1e-12


def test_hsic_standardization_has_correct_autograd():
    shared = sample(5, 2, requires_grad=True)
    summary = sample(5, 3, seed=41, requires_grad=True)
    assert torch.autograd.gradcheck(summary_hsic, (shared, summary), eps=1e-6, atol=1e-5)


def test_separation_terms_match_formula_and_retain_finite_gradients():
    shared = (0.1 * sample(8, 4)).requires_grad_()
    private = sample(8, 8, 7, seed=66, requires_grad=True)
    record = separation_terms(shared, private)
    assert set(record) == {"hsic", "variance", "separation"}
    expected_variance = (1 - (shared.var(0, unbiased=False) + 1e-4).sqrt()).relu().mean()
    torch.testing.assert_close(record["variance"], expected_variance)
    torch.testing.assert_close(record["separation"], 0.1 * record["hsic"] + 0.1 * record["variance"])
    record["separation"].backward()
    assert torch.isfinite(shared.grad).all() and shared.grad.abs().sum() > 0
    assert torch.isfinite(private.grad).all() and private.grad.abs().sum() > 0


def test_constant_shared_variance_penalty_has_no_escape_gradient():
    shared = torch.zeros(8, 4, dtype=torch.float64, requires_grad=True)
    penalty = (1 - (shared.var(0, unbiased=False) + 1e-4).sqrt()).relu().mean()
    torch.testing.assert_close(penalty, torch.tensor(0.99, dtype=torch.float64))
    penalty.backward()
    assert torch.count_nonzero(shared.grad) == 0
    with pytest.raises(ValueError, match="scale degeneracy"):
        separation_terms(shared, sample(8, 8, 7))


@pytest.mark.parametrize("which", ["shared", "summary", "single_column", "small_scale"])
def test_hsic_rejects_constant_or_nearly_constant_columns(which):
    shared, summary = sample(8, 4), sample(8, 7, seed=9)
    if which == "shared":
        shared.zero_()
    elif which == "summary":
        summary.fill_(1)
    elif which == "single_column":
        summary[:, 0] = 0
    else:
        summary *= 1e-12
    with pytest.raises(ValueError, match="scale degeneracy"):
        summary_hsic(shared, summary)


def test_hsic_rejects_single_patient_and_shape_mismatch():
    with pytest.raises(ValueError, match="at least two"):
        summary_hsic(sample(1, 4), sample(1, 7))
    with pytest.raises(ValueError, match="same patient count"):
        summary_hsic(sample(3, 4), sample(4, 7))
    with pytest.raises(ValueError, match="same patient count"):
        separation_terms(sample(3, 4), sample(4, 8, 7))


@pytest.mark.parametrize("function,args", [
    (private_summary, (torch.ones(3, 8, 4, dtype=torch.long),)),
    (private_summary, (sample(3, 4),)),
    (target_variance_floor, (sample(3, 0, 4),)),
    (content_set_loss, (sample(2, 3, 4), sample(2, 4, 4))),
    (content_set_loss, (sample(2, 3, 4), sample(3, 2, 4))),
    (content_set_loss, (sample(2, 3, 4), sample(2, 2, 5))),
    (summary_hsic, (sample(3, 4).to(torch.complex128), sample(3, 7))),
])
def test_invalid_tensor_contracts_are_rejected(function, args):
    with pytest.raises((ValueError, TypeError)):
        function(*args)


@pytest.mark.parametrize("function,args,bad_index", [
    (private_summary, (sample(3, 8, 4),), 0),
    (target_variance_floor, (sample(3, 8, 4),), 0),
    (content_set_loss, (sample(3, 8, 4), sample(3, 5, 4)), 0),
    (content_set_loss, (sample(3, 8, 4), sample(3, 5, 4)), 1),
    (summary_hsic, (sample(3, 4), sample(3, 7)), 1),
    (separation_terms, (sample(3, 4), sample(3, 8, 7)), 0),
])
def test_nonfinite_inputs_are_rejected(function, args, bad_index):
    args[bad_index].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        function(*args)


def test_adapter_rejects_bad_dimensions_dtype_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="positive integer"):
        SharedPrivateSlots(shared_dim=0)
    with pytest.raises(ValueError, match="seed"):
        SharedPrivateSlots(seed=True)
    model = SharedPrivateSlots(dimension=12, shared_dim=3, private_dim=9, target_dim=4)
    with pytest.raises(ValueError, match="feature dimension"):
        model(sample(3, 8, 13).float())
    with pytest.raises(ValueError, match="parameter device and dtype"):
        model(sample(3, 8, 12))
    invalid = sample(3, 8, 12).float()
    invalid[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        model(invalid)
