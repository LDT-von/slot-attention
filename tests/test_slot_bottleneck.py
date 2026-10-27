"""Analytic contracts and invented-input native integration, no patient I/O."""
from pathlib import Path

import pytest
import torch
from torch import nn

from slotspe_proof.model import build_model, make_args
from slotspe_proof.slot_bottleneck import GaussianSlotBottleneck, SlotVIBHost, diagonal_gaussian_kl


@pytest.fixture(autouse=True)
def isolated_runtime():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(threads)


def test_analytic_kl_matches_distribution_library_and_reductions():
    g = torch.Generator().manual_seed(19)
    mean = torch.randn(3, 8, 7, generator=g, dtype=torch.float64)
    logvar = torch.randn(3, 8, 7, generator=g, dtype=torch.float64)
    actual = diagonal_gaussian_kl(mean, logvar)
    q = torch.distributions.Normal(mean, (.5 * logvar).exp())
    expected = torch.distributions.kl_divergence(q, torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))).sum((-2, -1))
    torch.testing.assert_close(actual, expected)
    assert torch.equal(diagonal_gaussian_kl(torch.zeros_like(mean), torch.zeros_like(mean)), torch.zeros(3, dtype=mean.dtype))
    assert bool((actual > 0).all())


def test_kl_and_reparameterization_pass_double_precision_gradcheck():
    g = torch.Generator().manual_seed(310)
    mean = torch.randn(2, 3, 2, generator=g, dtype=torch.float64, requires_grad=True)
    logvar = torch.randn(2, 3, 2, generator=g, dtype=torch.float64, requires_grad=True)
    noise = torch.randn(2, 3, 2, generator=g, dtype=torch.float64)
    assert torch.autograd.gradcheck(diagonal_gaussian_kl, (mean, logvar))
    assert torch.autograd.gradcheck(lambda m, l: m + (.5*l).exp()*noise, (mean, logvar))


def test_shared_encoder_equivariant_only_with_permuted_noise():
    module = GaussianSlotBottleneck(5).double()
    g = torch.Generator().manual_seed(11)
    slots = torch.randn(2, 8, 5, generator=g, dtype=torch.float64)
    noise = torch.randn(2, 8, 5, generator=g, dtype=torch.float64)
    permutation = [5, 0, 7, 3, 4, 2, 6, 1]
    first, first_kl = module(slots, noise=noise)
    second, second_kl = module(slots[:, permutation], noise=noise[:, permutation])
    torch.testing.assert_close(first[:, permutation], second)
    torch.testing.assert_close(first_kl, second_kl)


def test_initialization_and_sampling_leave_global_rng_untouched():
    state = torch.get_rng_state().clone()
    module = GaussianSlotBottleneck(4)
    assert torch.equal(state, torch.get_rng_state())
    slots = torch.ones(2, 8, 4)
    g = torch.Generator().manual_seed(3)
    first, _ = module(slots, generator=g)
    assert torch.equal(state, torch.get_rng_state())
    second, _ = module(slots, generator=torch.Generator().manual_seed(3))
    assert torch.equal(first, second)
    assert not torch.equal(first[:, 0], first[:, 1])
    with pytest.raises(ValueError, match="isolated"):
        module(slots)


def test_sampling_and_kl_use_identical_clamped_log_variance():
    module = GaussianSlotBottleneck(2).double()
    with torch.no_grad():
        module.logvar_bias.copy_(torch.tensor([-100., 100.]))
    slots = torch.zeros(1, 2, 2, dtype=torch.float64)
    latent, kl = module(slots, noise=torch.ones_like(slots))
    effective = torch.tensor([-12., 8.], dtype=slots.dtype).expand_as(slots)
    torch.testing.assert_close(latent, (.5 * effective).exp())
    torch.testing.assert_close(kl, diagonal_gaussian_kl(slots, effective))


@pytest.mark.parametrize("mean,logvar", [
    (torch.zeros(2, 3), torch.zeros(2, 3)),
    (torch.zeros(2, 3, 1), torch.zeros(2, 3, 2)),
    (torch.full((1, 1, 1), float("nan")), torch.zeros(1, 1, 1)),
    (torch.zeros(1, 1, 1), torch.full((1, 1, 1), 1000.)),
])
def test_invalid_posterior_fails(mean, logvar):
    with pytest.raises(ValueError):
        diagonal_gaussian_kl(mean, logvar)


class RecordingHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.slot_attention_wsi = nn.Identity()
        self.slot_attention_omic = nn.Identity()
        self.parameter = nn.Parameter(torch.ones(()))
        self.fail = False
        self.extra_call = False

    def forward(self, *, slots, omic_missing):
        wsi = self.slot_attention_wsi(slots)
        if self.fail:
            raise RuntimeError("injected failure")
        omic = self.slot_attention_omic(slots)
        reconstruction = self.slot_attention_omic(slots + 10.)
        if self.extra_call:
            self.slot_attention_wsi(slots)
        return wsi, omic, reconstruction


def test_only_prediction_calls_replaced_kl_normalization_and_no_hook_leaks():
    native = RecordingHost()
    wrapped = SlotVIBHost(native, 4)
    x = torch.ones(2, 8, 4)
    with pytest.raises(RuntimeError, match="unavailable"):
        wrapped.kl_loss()
    wsi, omic, recon = wrapped(slots=x, omic_missing=False, vib_seed=10)
    assert not torch.equal(wsi, x) and not torch.equal(omic, x)
    assert not torch.equal(wsi, omic)
    assert torch.equal(recon, x + 10.)
    expected = diagonal_gaussian_kl(x, torch.full_like(x, -6.)) * 2
    torch.testing.assert_close(wrapped.kl_loss(reduction="none"), expected)
    torch.testing.assert_close(wrapped.kl_loss(), expected.mean() / 64)
    assert not native.slot_attention_omic._forward_hooks
    native.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        wrapped(slots=x, omic_missing=False, vib_seed=10)
    assert not native.slot_attention_wsi._forward_hooks
    with pytest.raises(RuntimeError, match="unavailable"):
        wrapped.kl_loss()
    native.fail = False
    assert wrapped(slots=x, omic_missing=False, vib_seed=10)[0].shape == x.shape
    native.extra_call = True
    with pytest.raises(RuntimeError, match="contract"):
        wrapped(slots=x, omic_missing=False, vib_seed=10)
    assert not native.slot_attention_omic._forward_hooks
    with pytest.raises(RuntimeError, match="unavailable"):
        wrapped.kl_loss()


def native_inputs():
    g = torch.Generator().manual_seed(71)
    return {"x_wsi": torch.randn(2, 9, 8, generator=g),
            "x_omic1": torch.randn(2, 3, generator=g),
            "x_omic2": torch.randn(2, 5, generator=g),
            "y": torch.tensor([0, 2]), "c": torch.tensor([0., 1.]), "omic_missing": False}


def native_model():
    args = make_args([3, 5], encoding_dim=8)
    args.wsi_projection_dim, args.slot_iters = 16, 2
    return build_model(Path(__file__).resolve().parents[1] / "third_party/SlotSPE", args, "original")


def test_off_exact_native_outputs_losses_gradients_rng_and_optimizer_step():
    native = native_model()
    wrapped = SlotVIBHost(native, 16, mode="off")
    assert list(wrapped.bottlenecks.parameters()) == []
    initial = {name: value.clone() for name, value in native.state_dict().items()}
    input_batch = native_inputs()
    seed = torch.get_rng_state().clone()

    def step(model):
        torch.set_rng_state(seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        logits, aux = model(**input_batch)
        total = native.loss_fn(logits, y=input_batch["y"], c=input_batch["c"], t=None)/2 + aux
        total.backward()
        gradients = {name: p.grad.clone() if p.grad is not None else None for name, p in native.named_parameters()}
        optimizer.step()
        state = {name: value.clone() for name, value in native.state_dict().items()}
        return logits.detach(), total.detach(), gradients, state, torch.get_rng_state().clone()

    first = step(native)
    native.load_state_dict(initial)
    native.zero_grad(set_to_none=True)
    second = step(wrapped)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert torch.equal(first[4], second[4])
    for group in (2, 3):
        for name, value in first[group].items():
            assert (value is None and second[group][name] is None) or torch.equal(value, second[group][name])
    assert wrapped.kl_loss().item() == 0


def test_native_train_and_eval_connected_reproducible_and_not_mean_eval():
    native = native_model()
    wrapped = SlotVIBHost(native, 16)
    x = native_inputs()
    logits, auxiliary = wrapped(**x, vib_seed=3)
    objective = native.loss_fn(logits, y=x["y"], c=x["c"], t=None)/2 + auxiliary + .001*wrapped.kl_loss()
    objective.backward()
    for module in wrapped.bottlenecks.values():
        assert module.mean_weight.grad is not None and module.logvar_bias.grad is not None
        assert bool(torch.isfinite(module.mean_weight.grad).all())
        assert module.mean_weight.grad.abs().sum() > 0
    assert native.slot_attention_wsi.to_k.weight.grad.abs().sum() > 0
    wrapped.eval()
    state = torch.get_rng_state().clone()
    with torch.no_grad():
        first = wrapped(**x, vib_seed=11)[0]
        torch.set_rng_state(state)
        second = wrapped(**x, vib_seed=11)[0]
        torch.set_rng_state(state)
        third = wrapped(**x, vib_seed=12)[0]
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_main_prediction_loss_alone_reaches_both_bottlenecks_and_native_slots():
    native = native_model()
    wrapped = SlotVIBHost(native, 16)
    x = native_inputs()
    logits, _ = wrapped(**x, vib_seed=3307)
    # Exclude native auxiliary and KL: neither may hide a broken task path.
    main = native.loss_fn(logits, y=x["y"], c=x["c"], t=None) / 2
    main.backward()
    for name, module in wrapped.bottlenecks.items():
        for parameter in (module.mean_weight, module.mean_bias, module.logvar_weight, module.logvar_bias):
            assert parameter.grad is not None
            assert bool(torch.isfinite(parameter.grad).all())
            assert parameter.grad.abs().sum() > 0
        upstream = getattr(native, "slot_attention_" + name).to_v.weight.grad
        assert upstream is not None and bool(torch.isfinite(upstream).all()) and upstream.abs().sum() > 0


def test_zero_kl_can_remove_all_input_information():
    module = GaussianSlotBottleneck(2)
    with torch.no_grad():
        module.mean_weight.zero_()
        module.logvar_bias.zero_()
    noise = torch.ones(1, 8, 2)
    a, ka = module(torch.zeros_like(noise), noise=noise)
    b, kb = module(torch.ones_like(noise)*100, noise=noise)
    assert torch.equal(a, b) and ka.item() == kb.item() == 0


def test_conditional_independence_does_not_make_aggregate_slots_independent():
    # Ideal A,e1,e2 iid N(0,1): Z1=A+e1,Z2=A+e2 have rho=1/2.
    covariance = torch.tensor([[2., 1.], [1., 2.]], dtype=torch.float64)
    rho = covariance[0, 1]/torch.sqrt(covariance[0, 0]*covariance[1, 1])
    mutual_information = -.5*torch.log1p(-rho.square())
    assert rho.item() == .5 and mutual_information.item() > 0


def test_orthogonal_copies_and_xor_defeat_naive_redundancy_claims():
    value = torch.tensor([-2., -1., 1., 2.])
    a = torch.stack((value, torch.zeros_like(value)), dim=-1)
    b = torch.stack((torch.zeros_like(value), value), dim=-1)
    assert torch.equal((a*b).sum(-1), torch.zeros_like(value))
    assert torch.equal(a[:, 0], b[:, 1])  # same recoverable information
    bits = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    target = bits[:, 0] ^ bits[:, 1]
    for coordinate in (0, 1):
        for bit in (0, 1):
            assert target[bits[:, coordinate] == bit].float().mean() == .5
    assert torch.equal(target, bits.sum(-1) % 2)
