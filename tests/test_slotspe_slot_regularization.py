"""Direct slot-diversity contracts, exclusively on invented tensors.

These checks establish objective and hook behavior, not patient performance.
"""
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from survot_rank.research.slotspe_proof.model import build_model, make_args
from survot_rank.research.slotspe_proof.slot_regularization import (
    QueryFreeSlotDecoder,
    SlotCapture,
    matched_set_mse,
    slot_gram_loss,
    slot_metrics,
    slotbert_loss,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_torch_runtime():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        yield
    torch.set_num_threads(previous_threads)


def test_gram_distinguishes_orthogonal_duplicate_and_antipodal_slots():
    orthogonal = torch.eye(3).unsqueeze(0)
    duplicate = torch.tensor([[[1., 0.], [2., 0.]]])
    antipodal = torch.tensor([[[1., 0.], [-2., 0.]]])
    assert slot_gram_loss(orthogonal).item() == pytest.approx(0.)
    assert slot_gram_loss(duplicate).item() == pytest.approx(1.)
    assert slot_gram_loss(antipodal).item() == pytest.approx(1.)
    # Merely making two copies point in opposite directions is not diversity.
    anti_metrics = slot_metrics(antipodal)
    assert anti_metrics["cosine"] == pytest.approx(-1.)
    assert anti_metrics["absolute_cosine"] == pytest.approx(1.)
    assert anti_metrics["rms_cosine"] == pytest.approx(1.)
    assert anti_metrics["effective_rank"] == pytest.approx(1.)


def test_slotbert_uniform_target_has_known_orthogonal_minimum():
    orthogonal = torch.eye(4).unsqueeze(0)
    assert slotbert_loss(orthogonal).item() == pytest.approx(math.log(4))
    duplicate = torch.tensor([[[1., 0.], [2., 0.]]])
    antipodal = torch.tensor([[[1., 0.], [-2., 0.]]])
    # For two equal directions the row logits differ by 1 / temperature.
    expected = math.log(2 * math.cosh(1.))
    assert slotbert_loss(duplicate).item() == pytest.approx(expected)
    assert slotbert_loss(antipodal).item() == pytest.approx(expected)
    assert slotbert_loss(duplicate) > slotbert_loss(torch.eye(2).unsqueeze(0))


@pytest.mark.parametrize("objective", [slot_gram_loss, slotbert_loss])
def test_diversity_objectives_are_slot_permutation_and_scale_invariant(objective):
    slots = torch.tensor([[[1., 0., .3], [.2, 1., .4], [1., .7, -.2]]],
                         requires_grad=True)
    value = objective(slots)
    reordered = slots[:, [2, 0, 1]] * torch.tensor([[[2.], [.3], [7.]]])
    torch.testing.assert_close(objective(reordered), value)
    value.backward()
    assert slots.grad is not None
    assert torch.isfinite(slots.grad).all()
    assert slots.grad.abs().sum() > 0


@pytest.mark.parametrize("slots", [
    torch.ones(1, 1, 3),
    torch.zeros(1, 2, 3),
    torch.full((1, 2, 3), float("nan")),
    torch.ones(2, 3),
    torch.ones(1, 2, 3, dtype=torch.int64),
])
def test_invalid_or_zero_slots_do_not_produce_diversity_claim(slots):
    for operation in (slot_gram_loss, slotbert_loss, slot_metrics):
        with pytest.raises(ValueError):
            operation(slots)


@pytest.mark.parametrize("temperature", [0., -1., float("nan"), float("inf")])
def test_slotbert_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        slotbert_loss(torch.eye(2).unsqueeze(0), temperature)


def test_metrics_report_raw_slot_count_norm_and_angular_rank():
    slots = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    metrics = slot_metrics(slots)
    assert metrics["slot_count"] == 4
    assert metrics["case_count"] == 2
    assert metrics["effective_rank"] == pytest.approx(4.)
    assert metrics["min_norm"] == pytest.approx(1.)
    assert metrics["cosine"] == pytest.approx(0.)
    assert metrics["max_cosine"] == pytest.approx(0.)
    assert metrics["fraction_cosine_above_09"] == pytest.approx(0.)


def test_set_matching_preserves_prediction_gradient_and_detaches_targets():
    pred = torch.tensor([[[0., 0.], [3., 0.]]], requires_grad=True)
    target = torch.tensor([[[2., 0.], [0., 0.]]], requires_grad=True)
    loss = matched_set_mse(pred, target)
    assert loss.item() == pytest.approx(.25)
    loss.backward()
    torch.testing.assert_close(pred.grad,
                               torch.tensor([[[0., 0.], [.5, 0.]]]))
    assert target.grad is None


def test_set_matching_is_invariant_to_both_set_orders_and_reduces_per_case():
    pred = torch.tensor([[[0., 1.], [4., 1.]], [[0., 2.], [4., 2.]]])
    target = torch.tensor([[[4., 0.], [0., 0.]], [[4., 0.], [0., 0.]]])
    values = matched_set_mse(pred, target, reduction="none")
    torch.testing.assert_close(values, torch.tensor([.5, 2.]))
    torch.testing.assert_close(matched_set_mse(pred[:, [1, 0]], target),
                               values.mean())
    torch.testing.assert_close(matched_set_mse(pred, target[:, [1, 0]]),
                               values.mean())


@pytest.mark.parametrize("pred,target,reduction", [
    (torch.ones(1, 2, 3), torch.ones(1, 3, 3), "mean"),
    (torch.ones(1, 2, 3), torch.ones(1, 2, 3), "sum"),
    (torch.ones(0, 2, 3), torch.ones(0, 2, 3), "mean"),
    (torch.ones(1, 2, 3), torch.full((1, 2, 3), float("inf")), "none"),
])
def test_set_matching_rejects_invalid_contracts(pred, target, reduction):
    with pytest.raises(ValueError):
        matched_set_mse(pred, target, reduction)


def test_query_free_decoder_is_shared_equivariant_and_has_no_identity_signal():
    decoder = QueryFreeSlotDecoder(3, 5)
    slots = torch.randn(2, 4, 3, requires_grad=True)
    outputs = decoder(slots)
    torch.testing.assert_close(decoder(slots[:, [2, 0, 3, 1]]),
                               outputs[:, [2, 0, 3, 1]])
    identical = slots[:, :1].expand(-1, 4, -1)
    decoded_identical = decoder(identical)
    torch.testing.assert_close(decoded_identical,
                               decoded_identical[:, :1].expand(-1, 4, -1))
    outputs.square().mean().backward()
    assert slots.grad is not None and slots.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in decoder.parameters())


class _Slots(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, values):
        return values * self.scale


class _FakeHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.slot_attention_wsi = _Slots(2.)
        self.slot_attention_omic = _Slots(3.)

    def forward(self, values):
        wsi = self.slot_attention_wsi(values)
        omic = self.slot_attention_omic(values + 1.)
        if self.training:
            self.slot_attention_omic(values + 7.)
        return wsi + omic


def test_capture_preserves_graph_and_first_prediction_call_with_reset():
    model = _FakeHost().train()
    values = torch.ones(1, 2, 3, requires_grad=True)
    with SlotCapture(model) as capture:
        output = model(values)
        assert len(capture.slots["wsi"]) == 1
        assert len(capture.slots["omic"]) == 2
        torch.testing.assert_close(capture.slots["omic"][0],
                                   torch.full_like(values, 6.))
        torch.testing.assert_close(capture.slots["omic"][1],
                                   torch.full_like(values, 24.))
        torch.testing.assert_close(output, torch.full_like(values, 8.))
        capture.slots["omic"][0].sum().backward()
        assert model.slot_attention_omic.scale.grad.item() == pytest.approx(12.)
        assert values.grad is not None
        model(values.detach() * 2.)
        assert len(capture.slots["wsi"]) == 1
        assert len(capture.slots["omic"]) == 2
    assert not any(capture.slots.values())
    assert not model._forward_pre_hooks
    assert not model.slot_attention_wsi._forward_hooks
    assert not model.slot_attention_omic._forward_hooks


def test_capture_cleans_hooks_after_body_exception_and_can_be_reused():
    model = _FakeHost().eval()
    capture = SlotCapture(model)
    with pytest.raises(RuntimeError, match="deliberate"):
        with capture:
            model(torch.ones(1, 2, 3))
            raise RuntimeError("deliberate")
    assert not any(capture.slots.values())
    assert not model._forward_pre_hooks
    assert not model.slot_attention_wsi._forward_hooks
    assert not model.slot_attention_omic._forward_hooks
    with capture:
        model(torch.ones(1, 2, 3))
        assert len(capture.slots["omic"]) == 1


def test_capture_cleans_partial_installation_on_missing_module():
    model = _FakeHost()
    del model.slot_attention_omic
    capture = SlotCapture(model)
    with pytest.raises(AttributeError):
        with capture:
            pass
    assert not model._forward_pre_hooks
    assert not model.slot_attention_wsi._forward_hooks
    assert not capture._handles


def test_capture_preserves_existing_hooks_and_rejects_double_entry():
    model = _FakeHost()
    existing = model.slot_attention_wsi.register_forward_hook(
        lambda module, args, output: None)
    capture = SlotCapture(model)
    with capture:
        with pytest.raises(RuntimeError, match="already active"):
            capture.__enter__()
        assert len(model.slot_attention_wsi._forward_hooks) == 2
    assert len(model.slot_attention_wsi._forward_hooks) == 1
    existing.remove()


def test_native_hook_is_observational_and_direct_losses_reach_both_modalities():
    torch.manual_seed(9103)
    args = make_args([4, 4], encoding_dim=8)
    args.wsi_projection_dim = 16
    args.slot_num_wsi = args.slot_num_omics = 4
    args.slot_iters = 3
    model = build_model(ROOT / "third_party/SlotSPE", args, "original").train()
    batch = {
        "x_wsi": torch.randn(2, 7, 8),
        "x_omic1": torch.randn(2, 4),
        "x_omic2": torch.randn(2, 4),
        "omic_missing": False,
        "y": torch.tensor([0, 2]),
        "c": torch.tensor([0, 1]),
    }
    torch.manual_seed(190)
    expected_logits, expected_aux = model(**batch)
    torch.manual_seed(190)
    with SlotCapture(model) as capture:
        logits, aux = model(**batch)
        torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
        torch.testing.assert_close(aux, expected_aux, rtol=0, atol=0)
        assert len(capture.slots["wsi"]) == 1
        assert len(capture.slots["omic"]) == 2
        loss = sum(slot_gram_loss(capture.slots[name][0]) +
                   slotbert_loss(capture.slots[name][0])
                   for name in ("wsi", "omic"))
        loss.backward()
        for module in (model.slot_attention_wsi, model.slot_attention_omic):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            assert gradients
            assert all(torch.isfinite(gradient).all() for gradient in gradients)
            assert sum(gradient.abs().sum() for gradient in gradients) > 0
