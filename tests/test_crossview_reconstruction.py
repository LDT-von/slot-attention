"""Invented-input checks only; no patient data or optimizer steps."""
import copy
from pathlib import Path

import pytest
import torch

from slotspe_proof.crossview_reconstruction import CrossViewV2
from slotspe_proof.plugin_engine import _build
from slotspe_proof.redundancy_content import ContentReference


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def deterministic_test_environment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    # Individual tests use a CUDA-specific fork if they run on CUDA.
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(threads)


def generated_cases(n=4, *, length=12):
    gen = torch.Generator(device="cpu").manual_seed(70013)
    return [
        {"x_wsi": torch.randn(1, length, 8, generator=gen),
         "x_omic1": torch.randn(1, 3, generator=gen),
         "x_omic2": torch.randn(1, 5, generator=gen),
         "omic_missing": False,
         "y": torch.tensor([i % 4]), "c": torch.tensor([float(i % 2)])}
        for i in range(n)
    ]


def make_host(preset="V2_content"):
    config = {"source_root": str(REPO / "third_party/SlotSPE"), "projection_dim": 16,
              "slot_iters": 2, "device": "cpu", "native_alpha": .5,
              "native_reconstruction_weight": .01}
    return _build(config, {"omic_sizes": [3, 5], "encoding_dim": 8}, preset)


def reference_state():
    return ContentReference([3, 5], 8, target_dim=5, clusters=4).fit(generated_cases()).state_dict()


def make_wrapper(preset="V2_content", *, reference=None):
    host = make_host(preset)
    state = reference if reference is not None else reference_state()
    before = torch.get_rng_state().clone()
    wrapper = CrossViewV2(host, state, temperature=.7, hidden_dim=12, decoder_seed=77)
    assert torch.equal(before, torch.get_rng_state())
    return wrapper


def split(seed, count):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ix = torch.randperm(count, generator=gen)
    return ix[:count // 2], ix[count // 2:]


def clone_case(case):
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in case.items()}


@pytest.mark.parametrize("preset", ["V2_content", "V2_separate"])
def test_off_matches_host_logits_aux_rng_and_full_bag(preset):
    wrapper = make_wrapper(preset)
    wrapper.train()
    case = generated_cases(1)[0]
    saved = case["x_wsi"].clone()
    state = torch.get_rng_state().clone()
    original = wrapper.host(plugin_seed=17, **case)
    post = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    actual = wrapper(plugin_seed=17, mode="off", run_aux=False, **case)
    assert torch.equal(original[0], actual[0]) and torch.equal(original[1], actual[1])
    assert torch.equal(torch.get_rng_state(), post)
    assert torch.equal(case["x_wsi"], saved)
    assert wrapper.crossview_loss.item() == 0 and wrapper.last_crossview["skipped"]
    assert not wrapper.backbone.slot_attention_wsi._forward_hooks
    assert not wrapper.backbone.slot_attention_omic._forward_hooks


def test_compute_and_train_same_main_forward_same_split_and_global_rng():
    wrapper = make_wrapper()
    wrapper.train()
    case = generated_cases(1)[0]
    state = torch.get_rng_state().clone()
    main = wrapper.host(plugin_seed=9, **case)
    post = torch.get_rng_state().clone()
    states = []
    parameter_names = tuple(name for name, _ in wrapper.named_parameters())
    for mode in ("compute", "train"):
        torch.set_rng_state(state)
        got = wrapper(plugin_seed=9, real_count=10, aux_seed=123, mode=mode, run_aux=True, **case)
        assert torch.equal(got[0], main[0]) and torch.equal(got[1], main[1])
        assert torch.equal(torch.get_rng_state(), post)
        assert tuple(name for name, _ in wrapper.named_parameters()) == parameter_names
        assert wrapper.last_crossview["a_count"] == 5 and wrapper.last_crossview["b_count"] == 5
        states.append((wrapper.crossview_loss.item(), wrapper.crossview_raw_loss.item(), dict(wrapper.last_crossview)))
    assert states[0][0] == 0 and states[1][0] >= 0
    assert states[0][1] == states[1][1] and states[1][1] == states[1][0]
    assert {k:v for k,v in states[0][2].items() if k != "mode"} == {
        k:v for k,v in states[1][2].items() if k != "mode"}


def test_b_content_cannot_change_prediction_from_same_a_slots():
    wrapper = make_wrapper()
    wrapper.train()
    case = generated_cases(1)[0]
    a, b = split(53, 10)
    changed = clone_case(case)
    changed["x_wsi"][:, b] += 15
    assert torch.equal(changed["x_wsi"][:, a], case["x_wsi"][:, a])
    assert not torch.allclose(wrapper._histogram(changed["x_wsi"][:, b]), wrapper._histogram(case["x_wsi"][:, b]))
    captured = []
    handle = wrapper.decoder.register_forward_hook(
        lambda _m, _i, output: captured.append(tuple(part.detach().clone() for part in output)))
    try:
        state = torch.get_rng_state().clone()
        wrapper(real_count=10, aux_seed=53, mode="train", run_aux=True, **case)
        torch.set_rng_state(state)
        wrapper(real_count=10, aux_seed=53, mode="train", run_aux=True, **changed)
    finally:
        handle.remove()
    assert len(captured) == 2 and all(torch.equal(left, right) for left,right in zip(*captured))
    assert not any(isinstance(value, torch.Tensor) for value in wrapper.last_crossview.values())


def test_aux_gradient_uses_only_real_a_and_reaches_v2_encoder():
    wrapper = make_wrapper()
    wrapper.train()
    case = generated_cases(1)[0]
    case["x_wsi"].requires_grad_(True)
    a, b = split(34, 10)
    wrapper(real_count=10, aux_seed=34, mode="train", run_aux=True, **case)
    wrapper.crossview_loss.backward()
    grad = case["x_wsi"].grad
    assert grad is not None and grad[:, a].abs().sum() > 0
    assert torch.count_nonzero(grad[:, b]) == 0
    assert torch.count_nonzero(grad[:, 10:]) == 0
    assert wrapper.host.parts["wsi"].private_encoder.weight.grad.abs().sum() > 0
    assert wrapper.backbone.wsi_mlp[0].weight.grad.abs().sum() > 0
    assert all(not buf.requires_grad for name, buf in wrapper.named_buffers() if name.startswith("reference_"))
    assert wrapper.decoder.moment_readout.weight.grad.abs().sum() > 0


def test_fixed_moments_preserve_weighted_feature_sum():
    wrapper = make_wrapper()
    view = generated_cases(1)[0]["x_wsi"][:, :5]
    mass, moments = wrapper._statistics(view)
    with torch.no_grad():
        projected = ((view - wrapper.reference_mean.float()) / wrapper.reference_std.float()) @ wrapper.reference_projection.float()
    assert torch.allclose(mass.sum(-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(moments.sum(dim=1), projected.mean(dim=1), atol=1e-5, rtol=1e-5)
    assert not mass.requires_grad and not moments.requires_grad


def test_compute_control_runs_zero_gradient_auxiliary_backward_graph():
    wrapper = make_wrapper()
    wrapper.train()
    case = generated_cases(1)[0]
    wrapper(real_count=10, aux_seed=34, mode="compute", run_aux=True, **case)
    assert wrapper.crossview_loss.item() == 0 and wrapper.crossview_loss.requires_grad
    wrapper.crossview_loss.backward()
    for parameter in (wrapper.host.parts["wsi"].private_encoder.weight,
                      wrapper.backbone.wsi_mlp[0].weight,
                      wrapper.decoder.readout.weight,
                      wrapper.decoder.moment_readout.weight):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def test_padding_excluded_and_short_cases_keep_native_forward():
    wrapper = make_wrapper()
    wrapper.train()
    case = generated_cases(1)[0]
    changed = clone_case(case)
    changed["x_wsi"][:, 8:] = 1234
    state = torch.get_rng_state().clone()
    wrapper(real_count=8, aux_seed=4, mode="train", run_aux=True, **case)
    loss = wrapper.crossview_loss.detach().clone()
    torch.set_rng_state(state)
    wrapper(real_count=8, aux_seed=4, mode="train", run_aux=True, **changed)
    assert torch.equal(loss, wrapper.crossview_loss.detach())
    for count in (0, 1):
        torch.set_rng_state(state)
        baseline = wrapper.host(plugin_seed=3, **case)
        rng_after = torch.get_rng_state().clone()
        torch.set_rng_state(state)
        result = wrapper(plugin_seed=3, real_count=count, aux_seed=4, mode="train", run_aux=True, **case)
        assert torch.equal(baseline[0], result[0]) and torch.equal(baseline[1], result[1])
        assert torch.equal(torch.get_rng_state(), rng_after)
        assert wrapper.crossview_loss.item() == 0 and wrapper.last_crossview["skipped"]


def test_decoder_is_slot_permutation_invariant_and_each_slot_has_gradient():
    wrapper = make_wrapper()
    gen = torch.Generator().manual_seed(981)
    slots = torch.randn(1, 8, wrapper.host.parts["wsi"].private_dim, generator=gen, requires_grad=True)
    pred = wrapper.decoder(slots)
    order = torch.tensor([6, 1, 4, 2, 7, 0, 3, 5])
    assert torch.allclose(pred, wrapper.decoder(slots[:, order]), atol=1e-7, rtol=1e-6)
    pred[0, 0].backward()
    assert slots.grad.shape == slots.shape
    assert (slots.grad.abs().sum(dim=-1) > 0).all()


def test_reference_cloned_frozen_and_checkpoint_requires_same_reference():
    state = reference_state()
    intact = copy.deepcopy(state)
    wrapper = make_wrapper(reference=state)
    case = generated_cases(1)[0]
    target = wrapper._histogram(case["x_wsi"][:, :4]).clone()
    for key in ("means", "stds", "projections"):
        state[key][0].add_(12)
    state["static_targets"]["wsi"].add_(12)
    assert torch.equal(target, wrapper._histogram(case["x_wsi"][:, :4]))
    same = make_wrapper(reference=intact)
    same.load_state_dict(wrapper.state_dict())
    assert torch.equal(target, same._histogram(case["x_wsi"][:, :4]))
    other = copy.deepcopy(intact)
    other["means"][0] += 1
    mismatched = make_wrapper(reference=other)
    with pytest.raises(RuntimeError, match="checkpoint/reference"):
        mismatched.load_state_dict(wrapper.state_dict())


@pytest.mark.parametrize("bad", [-1, 13, True, 1.5, None])
def test_invalid_real_count_fails_before_native_forward(bad):
    wrapper = make_wrapper()
    case = generated_cases(1)[0]
    state = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="real_count"):
        wrapper(real_count=bad, aux_seed=1, mode="train", run_aux=True, **case)
    assert torch.equal(state, torch.get_rng_state())


def test_invalid_reference_seed_and_modes_fail_closed():
    state = reference_state()
    bad = copy.deepcopy(state)
    bad["stds"][0][0] = 0
    with pytest.raises(ValueError, match="scale"):
        make_wrapper(reference=bad)
    wrapper = make_wrapper(reference=state)
    case = generated_cases(1)[0]
    with pytest.raises(ValueError, match="aux_seed"):
        wrapper(real_count=10, aux_seed=None, mode="train", run_aux=True, **case)
    with pytest.raises(ValueError, match="run_aux"):
        wrapper(real_count=10, aux_seed=3, mode="train", run_aux=False, **case)
    with pytest.raises(ValueError, match="one patient's"):
        case["x_wsi"] = case["x_wsi"].expand(2, -1, -1)
        wrapper(real_count=10, aux_seed=3, mode="train", run_aux=True, **case)


def test_float64_module_and_case():
    wrapper = make_wrapper().double()
    case = generated_cases(1)[0]
    for name in ("x_wsi", "x_omic1", "x_omic2", "c"):
        case[name] = case[name].double()
    logits, aux = wrapper(real_count=10, aux_seed=2, mode="train", run_aux=True, **case)
    assert logits.dtype == torch.float64 and aux.dtype == torch.float64
    assert wrapper.crossview_loss.dtype == torch.float64
    assert torch.isfinite(wrapper.crossview_loss)


def test_native_size_256_dim_8_slots_10_iterations_generated_backward():
    config = {"source_root": str(REPO / "third_party/SlotSPE"), "projection_dim": 256,
              "slot_iters": 10, "device": "cpu", "native_alpha": .5,
              "native_reconstruction_weight": .01}
    host = _build(config, {"omic_sizes": [3, 5], "encoding_dim": 8}, "V2_separate")
    cases = generated_cases(3, length=16)
    reference = ContentReference([3, 5], 8, target_dim=32, clusters=8).fit(cases)
    wrapper = CrossViewV2(host, reference.state_dict(), decoder_seed=17)
    wrapper.train()
    logits, native_aux = wrapper(real_count=16, aux_seed=19, mode="train", run_aux=True,
                                 plugin_seed=23, **cases[0])
    assert logits.shape == (1, 4) and torch.isfinite(native_aux)
    assert host.last_records["wsi"]["private"].shape == (1, 8, 192)
    assert wrapper.last_crossview["a_count"] == wrapper.last_crossview["b_count"] == 8
    assert torch.isfinite(wrapper.crossview_loss)
    wrapper.crossview_loss.backward()
    assert wrapper.decoder.readout.weight.grad is not None
    assert torch.isfinite(wrapper.decoder.readout.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_auxiliary_preserves_cpu_and_current_device_rng():
    wrapper = make_wrapper().cuda()
    case = generated_cases(1)[0]
    for key, value in case.items():
        if isinstance(value, torch.Tensor):
            case[key] = value.cuda()
    device = case["x_wsi"].device.index
    with torch.random.fork_rng(devices=[device]):
        cpu_before = torch.get_rng_state().clone()
        gpu_before = torch.cuda.get_rng_state(device).clone()
        baseline = wrapper.host(plugin_seed=13, **case)
        cpu_after = torch.get_rng_state().clone()
        gpu_after = torch.cuda.get_rng_state(device).clone()
        torch.set_rng_state(cpu_before)
        torch.cuda.set_rng_state(gpu_before, device)
        got = wrapper(plugin_seed=13, real_count=10, aux_seed=61, mode="train", run_aux=True, **case)
        assert torch.equal(baseline[0], got[0]) and torch.equal(baseline[1], got[1])
        assert torch.equal(torch.get_rng_state(), cpu_after)
        assert torch.equal(torch.cuda.get_rng_state(device), gpu_after)
        assert torch.isfinite(wrapper.crossview_loss)
