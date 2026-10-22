"""Full original SlotSPE allocation controls on invented tensors only.

These checks establish adapter fidelity, finite derivatives, and RNG pairing.
They do not read patients or checkpoints, update weights, or establish benefit.
"""
from pathlib import Path

import pytest
import torch

from survot_rank.research.slotspe_proof.model import build_model, make_args
from survot_rank.research.slotspe_proof.native_allocation import (
    NativeAllocation, VARIANTS,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VARIANTS = {
    "S0": {"wsi": "original", "omic": "original"},
    "S1": {"wsi": "sinkhorn", "omic": "sinkhorn"},
    "S2": {"wsi": "mesh_uniform", "omic": "original"},
    "S3": {"wsi": "original", "omic": "mesh_uniform"},
    "S4": {"wsi": "mesh_uniform", "omic": "mesh_uniform"},
}


@pytest.fixture(autouse=True)
def isolated_torch_runtime():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        yield
    torch.set_num_threads(previous_threads)


def make_fixture(training=True):
    torch.manual_seed(9103)
    args = make_args([4, 4], encoding_dim=8)
    args.wsi_projection_dim = 16
    args.slot_num_wsi = args.slot_num_omics = 4
    args.slot_iters = 2
    model = build_model(ROOT / "third_party/SlotSPE", args, "original")
    model.train(training)
    batch = {
        "x_wsi": torch.randn(2, 7, 8),
        "x_omic1": torch.randn(2, 4),
        "x_omic2": torch.randn(2, 4),
        "omic_missing": False,
        "y": torch.tensor([0, 2]),
        "c": torch.tensor([0, 1]),
    }
    return model, batch


def main_loss(model, logits, batch):
    return model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None) / 2


def gradient_snapshot(model):
    return {name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()}


def test_native_variant_table_targets_only_two_original_sa_modules():
    assert VARIANTS == EXPECTED_VARIANTS


@pytest.mark.parametrize("training", [False, True])
def test_s0_replays_native_logits_auxiliary_and_all_parameter_gradients(training):
    model, batch = make_fixture(training)
    torch.manual_seed(77)
    expected_logits, expected_aux = model(**batch)
    (main_loss(model, expected_logits, batch) + expected_aux).backward()
    expected_gradients = gradient_snapshot(model)
    model.zero_grad(set_to_none=True)

    torch.manual_seed(77)
    with NativeAllocation(model, variant="S0", noise_seed=123, capture=True):
        actual_logits, actual_aux = model(**batch)
        (main_loss(model, actual_logits, batch) + actual_aux).backward()

    torch.testing.assert_close(actual_logits, expected_logits, atol=1e-6, rtol=1e-6)
    if training:
        torch.testing.assert_close(actual_aux, expected_aux, atol=1e-6, rtol=1e-6)
    else:
        assert actual_aux == expected_aux == 0.0
    actual_gradients = gradient_snapshot(model)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name, expected in expected_gradients.items():
        actual = actual_gradients[name]
        assert (actual is None) == (expected is None), name
        if expected is not None:
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5,
                                       msg=lambda message: f"{name}: {message}")


@pytest.mark.parametrize("variant", EXPECTED_VARIANTS)
@pytest.mark.parametrize("training", [False, True])
def test_five_arms_keep_finite_original_loss_and_sa_main_gradient(variant, training):
    model, batch = make_fixture(training)
    with NativeAllocation(model, variant=variant, noise_seed=123, capture=True) as adapter:
        logits, auxiliary = model(**batch)
        loss = main_loss(model, logits, batch)
        assert torch.isfinite(logits).all() and torch.isfinite(loss)
        assert torch.isfinite(torch.as_tensor(auxiliary))
        gradients = torch.autograd.grad(
            loss, (model.slot_attention_wsi.to_v.weight,
                   model.slot_attention_omic.to_v.weight), retain_graph=True)
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
        selector_parameters = tuple(p for name, p in model.named_parameters()
                                    if "pred_keep_slot" in name)
        selector_gradients = torch.autograd.grad(
            loss, selector_parameters, allow_unused=True, retain_graph=True)
        # Assignment changes must not silently repair the original boolean
        # selector mask. Its direct connection to the main loss stays absent.
        assert selector_parameters and all(g is None for g in selector_gradients)
        (loss + auxiliary).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters()
                   if p.grad is not None)

        expected_calls = {"wsi": 1, "omic": 2 if training else 1}
        assert len(adapter.records) == sum(expected_calls.values())
        for modality, count in expected_calls.items():
            records = [r for r in adapter.records if r["modality"] == modality]
            assert [r["call_index"] for r in records] == list(range(count))
            assert all(r["mode"] == EXPECTED_VARIANTS[variant][modality] for r in records)
            for record in records:
                assert record["final"].shape == (2, 4, 16)
                assert record["final"].device.type == "cpu"
                assert not record["final"].requires_grad
                assert torch.isfinite(record["final"]).all()
                expected_tokens = (7 if modality == "wsi" or
                                   record["call_index"] == 1 else 2)
                assert record["source_input_count"] == expected_tokens
                assert len(record["attentions"]) == 2
                for attention in record["attentions"]:
                    assert attention.shape == (2, 8, 4, expected_tokens)
                    assert attention.device.type == "cpu" and not attention.requires_grad
                    assert torch.isfinite(attention).all()
                    torch.testing.assert_close(attention.sum(-1),
                                               torch.ones(2, 8, 4), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("variant", EXPECTED_VARIANTS)
def test_extra_mesh_noise_does_not_shift_native_global_rng(variant):
    model, batch = make_fixture(False)
    torch.manual_seed(2027)
    with torch.no_grad(), NativeAllocation(model, variant="S0", noise_seed=123):
        model(**batch)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(2027)
    with torch.no_grad(), NativeAllocation(model, variant=variant, noise_seed=123):
        model(**batch)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_context_preserves_parameters_state_and_restores_forward_on_exception():
    model, batch = make_fixture(False)
    original_methods = (model.slot_attention_wsi.forward,
                        model.slot_attention_omic.forward)
    parameter_ids = {name: id(value) for name, value in model.named_parameters()}
    untouched_methods = {
        name: getattr(model, name).forward
        for name in ("slot_decoder_wsi", "slot_decoder_omic", "cross_attention",
                     "self_attention_wsi", "self_attention_omic",
                     "reconstruction_head_omic", "reconstruction_head_joint",
                     "reconstruction_head_wsi", "to_logits")
    }
    original_loss = model.loss_fn
    state = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(RuntimeError, match="invented interruption"):
        with NativeAllocation(model, variant="S4", noise_seed=123):
            with torch.no_grad():
                model(**batch)
            assert {name: id(value) for name, value in model.named_parameters()} == parameter_ids
            assert model.state_dict().keys() == state.keys()
            assert model.loss_fn is original_loss
            assert all(getattr(model, name).forward == method
                       for name, method in untouched_methods.items())
            raise RuntimeError("invented interruption")
    assert (model.slot_attention_wsi.forward,
            model.slot_attention_omic.forward) == original_methods
    assert model.state_dict().keys() == state.keys()
    for name, expected in state.items():
        assert torch.equal(model.state_dict()[name], expected), name


def test_reset_noise_seed_and_global_rng_repeat_mesh_full_forward():
    model, batch = make_fixture(False)
    with torch.no_grad(), NativeAllocation(model, variant="S4", noise_seed=123,
                                          capture=True) as adapter:
        torch.manual_seed(47)
        first, _ = model(**batch)
        first_records = list(adapter.records)
        adapter.set_noise_seed(123)
        torch.manual_seed(47)
        second, _ = model(**batch)
        last_records = adapter.records[-2:]
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert [(r["modality"], r["call_index"]) for r in last_records] == [
        ("wsi", 0), ("omic", 0)]
    for a, b in zip(first_records, last_records):
        assert torch.equal(a["final"], b["final"])
        assert all(torch.equal(x, y) for x, y in zip(a["attentions"], b["attentions"]))
