"""Synthetic-only tests for the in-memory observed forward contract."""

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slotspe_proof.roi_trace import capture_verified, row_manifest_sha256


class SplitHeads(torch.nn.Module):
    def forward(self, x):
        b, n, d = x.shape
        return x.reshape(b, n, 2, d // 2).transpose(1, 2)


class MergeHeads(torch.nn.Module):
    def forward(self, x):
        b, h, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, h * d)


class SyntheticSlot(torch.nn.Module):
    def __init__(self, *, alter_when_traced=False):
        super().__init__()
        self.dim = 8
        self.iters = 2
        self.eps = 1e-8
        self.scale = 8 ** -0.5
        self.norm_input = torch.nn.LayerNorm(8)
        self.norm_slots = torch.nn.LayerNorm(8)
        self.norm_pre_ff = torch.nn.LayerNorm(8)
        self.to_k = torch.nn.Linear(8, 8)
        self.to_q = torch.nn.Linear(8, 8)
        self.to_v = torch.nn.Linear(8, 8)
        self.split_heads = SplitHeads()
        self.merge_heads = MergeHeads()
        self.combine_heads = torch.nn.Linear(8, 8)
        self.gru = torch.nn.GRUCell(8, 8)
        self.mlp = torch.nn.Linear(8, 8)
        self.alter_when_traced = alter_when_traced

    def forward(self, inputs):
        b = inputs.shape[0]
        slots = torch.randn(b, 3, 8, device=inputs.device)
        normalized = self.norm_input(inputs)
        keys = self.split_heads(self.to_k(normalized))
        values = self.split_heads(self.to_v(normalized))
        for _ in range(self.iters):
            previous = slots
            queries = self.split_heads(self.to_q(self.norm_slots(slots)))
            dots = torch.einsum("b h s d, b h n d -> b h s n", queries, keys)
            attention = torch.nn.functional.normalize((dots * self.scale).softmax(dim=-2) + self.eps, p=1, dim=-1)
            updates = torch.einsum("b h n d, b h s n -> b h s d", values, attention)
            updates = self.combine_heads(self.merge_heads(updates))
            slots = self.gru(updates.reshape(-1, 8), previous.reshape(-1, 8)).reshape(b, 3, 8)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        if self.alter_when_traced and self.to_q._forward_hooks:
            slots = slots + 1
        return slots


class SyntheticDecoder(torch.nn.Module):
    def forward(self, slots):
        scores = slots.mean(-1)
        index = (scores + torch.rand_like(scores)).argmax(-1, keepdim=True)
        keep = torch.zeros_like(scores).scatter_(1, index, 1.0)
        gate = keep * torch.softmax(scores, dim=-1)
        return (gate.unsqueeze(-1) * slots).sum(1), gate, keep


class SyntheticModel(torch.nn.Module):
    def __init__(self, *, alter_when_traced=False, dict_output=False, mutate=False):
        super().__init__()
        self.slot_attention_wsi = SyntheticSlot(alter_when_traced=alter_when_traced)
        self.slot_attention_omic = SyntheticSlot()
        self.slot_decoder_wsi = SyntheticDecoder()
        self.slot_decoder_omic = SyntheticDecoder()
        self.dict_output = dict_output
        self.mutate = mutate

    def forward(self, *, x_wsi, x_omic, omic_missing):
        if self.mutate:
            x_wsi.add_(1)
        wsi = self.slot_attention_wsi(x_wsi)
        omic = self.slot_attention_omic(x_omic)
        first = self.slot_decoder_wsi(wsi)[0]
        second = self.slot_decoder_omic(omic)[0]
        logits = (first + second)[:, :2]
        if self.dict_output:
            return {"logits": logits, "aux": (0.0, ["synthetic"])}
        return logits, 0.0


def inputs():
    x = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8.],
                       [8., 7., 6., 5., 4., 3., 2., 1.],
                       [0., 0., 0., 0., 0., 0., 0., 0.]]])
    batch = {"x_wsi": x, "x_omic": torch.ones(1, 2, 8), "omic_missing": False}
    rows = [
        {"token_index": 0, "is_real": True, "slide_id": "fictional-slide-1", "feature_row": 4, "box": [0, 0, 10, 10]},
        {"token_index": 1, "is_real": True, "slide_id": "fictional-slide-2", "feature_row": 0},
        {"token_index": 2, "is_real": False, "slide_id": None, "feature_row": None},
    ]
    return batch, rows, ["fictional-pathway-a", "fictional-pathway-b"]


def capture(model=None):
    batch, rows, names = inputs()
    return capture_verified(model or SyntheticModel().eval(), batch, row_manifest=rows, pathway_names=names)


def test_trace_is_exact_per_head_and_does_not_advance_outer_rng():
    torch.manual_seed(817)
    model = SyntheticModel().eval()
    before = torch.get_rng_state().clone()
    result = capture(model)
    assert torch.equal(before, torch.get_rng_state())
    assert result["logits"].shape == (1, 2)
    assert result["modalities"]["wsi"]["last_attention_by_head"].shape == (2, 3, 3)
    assert result["modalities"]["omic"]["last_attention_by_head"].shape == (2, 3, 2)
    assert result["modalities"]["wsi"]["attention_by_iteration"].shape == (2, 2, 3, 3)
    assert result["modalities"]["wsi"]["slot_states"].shape == (3, 3, 8)
    assert len(result["modalities"]["wsi"]["selected_indices"]) == 1
    assert result["row_manifest"][2]["is_real"] is False
    assert result["checks"]["entire_forward_output_exact"] is True
    assert result["checks"]["row_manifest_provenance"] == "caller_supplied_internal_consistency_only"


def test_hash_includes_order_and_additional_fields():
    _, rows, _ = inputs()
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    assert row_manifest_sha256(rows) == hashlib.sha256(canonical).hexdigest()
    assert row_manifest_sha256(rows) != row_manifest_sha256(list(reversed(rows)))
    assert capture()["row_manifest_sha256"] == row_manifest_sha256(rows)


def test_input_hashes_bind_actual_wsi_and_omic_tensors():
    batch, rows, names = inputs()
    model = SyntheticModel().eval()
    first = capture_verified(model, batch, row_manifest=rows, pathway_names=names)["checks"]
    batch["x_wsi"][0, 0, 0] += 1
    second = capture_verified(model, batch, row_manifest=rows, pathway_names=names)["checks"]
    assert second["wsi_input_sha256"] != first["wsi_input_sha256"]
    assert second["omic_input_sha256"] == first["omic_input_sha256"]
    batch["x_omic"][0, 0, 0] += 1
    third = capture_verified(model, batch, row_manifest=rows, pathway_names=names)["checks"]
    assert third["wsi_input_sha256"] == second["wsi_input_sha256"]
    assert third["omic_input_sha256"] != second["omic_input_sha256"]


def test_dict_output_is_checked_and_returned():
    result = capture(SyntheticModel(dict_output=True).eval())
    assert torch.equal(result["native_output"]["logits"], result["logits"])
    assert result["native_output"]["aux"][1] == ["synthetic"]


def test_forward_which_changes_when_traced_fails_closed():
    with pytest.raises(RuntimeError, match="differs"):
        capture(SyntheticModel(alter_when_traced=True).eval())


def test_wrong_hook_attention_fails_replay_even_if_forward_is_unchanged(monkeypatch):
    from slotspe_proof import problem_trace

    def wrong_attention(module, queries, keys):
        shape = (*queries.shape[:-1], keys.shape[-2])
        return torch.full(shape, 1.0 / keys.shape[-2], device=queries.device, dtype=queries.dtype)

    monkeypatch.setattr(problem_trace, "_attention", wrong_attention)
    with pytest.raises(RuntimeError, match="replay"):
        capture()


def test_forward_which_mutates_input_fails_without_modifying_caller_input():
    batch, rows, names = inputs()
    old = batch["x_wsi"].clone()
    with pytest.raises(RuntimeError, match="forward input"):
        capture_verified(SyntheticModel(mutate=True).eval(), batch, row_manifest=rows, pathway_names=names)
    assert torch.equal(batch["x_wsi"], old)


def test_training_mode_is_rejected_without_changing_model():
    model = SyntheticModel()
    with pytest.raises(ValueError, match="eval mode"):
        capture(model)
    assert model.training


def test_original_slotspe_generated_inputs_are_exactly_observed():
    from slotspe_proof.model import load_source, make_args

    source = load_source(Path(__file__).resolve().parents[1] / "third_party" / "SlotSPE")
    args = make_args([2, 3], encoding_dim=8)
    args.wsi_projection_dim = 32
    args.slot_iters = 2
    torch.manual_seed(411)
    model = source.SlotSPE(args).eval()
    batch, rows, _ = inputs()
    batch.pop("x_omic")
    batch["x_omic1"] = torch.tensor([[0.3, 0.7]])
    batch["x_omic2"] = torch.tensor([[0.5, 0.2, 0.8]])
    before = torch.get_rng_state().clone()
    result = capture_verified(model, batch, row_manifest=rows, pathway_names=["synthetic-a", "synthetic-b"],
                              source_root=Path(__file__).resolve().parents[1] / "third_party" / "SlotSPE")
    assert torch.equal(before, torch.get_rng_state())
    assert result["logits"].shape == (1, 4)
    assert result["modalities"]["wsi"]["last_attention_by_head"].shape == (8, 8, 3)
    assert result["modalities"]["omic"]["last_attention_by_head"].shape == (8, 8, 2)
    assert len(result["modalities"]["wsi"]["selected_indices"]) == 2
    assert len(result["modalities"]["omic"]["selected_indices"]) == 2
    assert all(result["native_purity"]["checks"].values())


def test_non_native_model_rejected_by_source_bound_capture():
    batch, rows, names = inputs()
    with pytest.raises(RuntimeError, match="Original-only audit"):
        capture_verified(SyntheticModel().eval(), batch, row_manifest=rows, pathway_names=names,
                         source_root=Path(__file__).resolve().parents[1] / "third_party" / "SlotSPE")


@pytest.mark.parametrize("corruption", ["order", "duplicate", "padding", "invalid_names", "missing_omic", "mismatch_tokens", "nonfinite"])
def test_invalid_provenance_and_inputs_fail_closed(corruption):
    batch, rows, names = inputs()
    if corruption == "order":
        rows[0]["token_index"] = 1
    elif corruption == "duplicate":
        rows[1]["slide_id"] = rows[0]["slide_id"]
        rows[1]["feature_row"] = rows[0]["feature_row"]
    elif corruption == "padding":
        batch["x_wsi"][0, 2, 0] = 1
    elif corruption == "invalid_names":
        names = ["same", "same"]
    elif corruption == "missing_omic":
        batch["omic_missing"] = True
    elif corruption == "mismatch_tokens":
        names = ["only-one-name"]
    elif corruption == "nonfinite":
        batch["x_wsi"][0, 0, 0] = float("nan")
    with pytest.raises((ValueError, RuntimeError)):
        capture_verified(SyntheticModel().eval(), batch, row_manifest=rows, pathway_names=names)
