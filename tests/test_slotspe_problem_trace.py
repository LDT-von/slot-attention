"""Synthetic-only checks for native tracing and deterministic replay."""

import pytest

torch = pytest.importorskip("torch")

from survot_rank.research.slotspe_proof.problem_trace import (
    NativeSlotTrace,
    replay_rng,
    replay_slots,
    rng_state,
)


class SplitHeads(torch.nn.Module):
    def forward(self, value):
        batch, items, dimension = value.shape
        return value.reshape(batch, items, 2, dimension // 2).transpose(1, 2)


class MergeHeads(torch.nn.Module):
    def forward(self, value):
        batch, heads, items, dimension = value.shape
        return value.transpose(1, 2).reshape(batch, items, heads * dimension)


class SyntheticSlot(torch.nn.Module):
    """Minimal local fixture, independent of real SlotSPE and patient inputs."""

    def __init__(self, iterations=3):
        super().__init__()
        self.dim = 8
        self.iters = iterations
        self.eps = 1e-8
        self.scale = self.dim ** -0.5
        self.norm_input = torch.nn.LayerNorm(self.dim)
        self.norm_slots = torch.nn.LayerNorm(self.dim)
        self.norm_pre_ff = torch.nn.LayerNorm(self.dim)
        self.to_k = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.to_v = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.to_q = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.split_heads = SplitHeads()
        self.merge_heads = MergeHeads()
        self.combine_heads = torch.nn.Linear(self.dim, self.dim)
        self.gru = torch.nn.GRUCell(self.dim, self.dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.dim, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, self.dim),
        )
        self.native_states = []
        self.native_attentions = []

    def forward(self, inputs):
        batch = inputs.shape[0]
        slots = torch.randn(batch, 3, self.dim, device=inputs.device)
        self.native_states = [slots.detach().clone()]
        self.native_attentions = []
        normalized = self.norm_input(inputs)
        keys = self.split_heads(self.to_k(normalized))
        values = self.split_heads(self.to_v(normalized))
        for _ in range(self.iters):
            previous = slots
            normalized_slots = self.norm_slots(slots)
            queries = self.split_heads(self.to_q(normalized_slots))
            dots = torch.einsum("b h i d, b h j d -> b h i j", queries, keys)
            attention = (dots * self.scale).softmax(dim=-2)
            attention = torch.nn.functional.normalize(
                attention + self.eps, p=1, dim=-1
            )
            updates = torch.einsum(
                "b h j d, b h i j -> b h i d", values, attention
            )
            updates = self.combine_heads(self.merge_heads(updates))
            slots = self.gru(
                updates.reshape(-1, self.dim), previous.reshape(-1, self.dim)
            ).reshape(batch, 3, self.dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
            self.native_attentions.append(attention.detach().clone())
            self.native_states.append(slots.detach().clone())
        return slots


class SyntheticModel(torch.nn.Module):
    def __init__(self, iterations=3):
        super().__init__()
        self.slot_attention_wsi = SyntheticSlot(iterations)
        self.slot_attention_omic = SyntheticSlot(iterations)

    def forward(self, inputs):
        return self.slot_attention_wsi(inputs), self.slot_attention_omic(inputs)


def test_trace_preserves_native_outputs_rng_and_iteration_states():
    torch.manual_seed(6)
    model = SyntheticModel()
    inputs = torch.randn(2, 5, 8, requires_grad=True)
    entry = rng_state()
    with replay_rng(entry):
        expected = model(inputs)
        expected_after = rng_state()
    with replay_rng(entry), NativeSlotTrace(model) as trace:
        actual = model(inputs)
        actual_after = rng_state()
    assert torch.equal(expected_after["cpu"], actual_after["cpu"])
    for modality, output, target in zip(("wsi", "omic"), actual, expected):
        assert torch.equal(output, target)
        record = trace.records[modality]
        native = getattr(model, "slot_attention_" + modality)
        assert len(record["states"]) == native.iters + 1
        assert len(record["attentions"]) == native.iters
        for captured, observed in zip(record["states"], native.native_states):
            assert torch.equal(captured, observed)
            assert not captured.requires_grad
        for captured, observed in zip(
            record["attentions"], native.native_attentions
        ):
            assert torch.equal(captured, observed)
            assert captured.shape == (2, 2, 3, 5)
            assert torch.allclose(captured.sum(dim=-1), torch.ones(2, 2, 3))
            assert not captured.requires_grad
        assert torch.equal(record["inputs"], inputs)
        assert torch.equal(record["final"], output)


def test_replay_is_exact_retains_gradients_and_draws_no_randomness():
    torch.manual_seed(11)
    model = SyntheticModel()
    inputs = torch.randn(2, 5, 8, requires_grad=True)
    with NativeSlotTrace(model) as trace:
        model(inputs)
    entry = rng_state()
    replay = replay_slots(
        model.slot_attention_wsi, inputs, trace.records["wsi"]["initial"]
    )
    assert torch.equal(entry["cpu"], rng_state()["cpu"])
    assert torch.equal(replay["final"], trace.records["wsi"]["final"])
    for replayed, captured in zip(replay["states"], trace.records["wsi"]["states"]):
        assert torch.equal(replayed, captured)
    for replayed, captured in zip(
        replay["attentions"], trace.records["wsi"]["attentions"]
    ):
        assert torch.equal(replayed, captured)
    replay["final"].square().sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum() > 0


def test_replay_inside_live_trace_does_not_add_or_change_native_records():
    model = SyntheticModel()
    inputs = torch.randn(2, 5, 8)
    with NativeSlotTrace(model) as trace:
        model(inputs)
        record = trace.records["wsi"]
        replay = replay_slots(model.slot_attention_wsi, inputs, record["initial"])
        assert torch.equal(replay["final"], record["final"])
        assert len(record["states"]) == model.slot_attention_wsi.iters + 1
        assert len(record["attentions"]) == model.slot_attention_wsi.iters
        assert len(trace.calls["wsi"]) == 1


def test_repeated_calls_keep_first_record_and_reset_clears_all_calls():
    model = SyntheticModel()
    inputs = torch.ones(2, 5, 8)
    with NativeSlotTrace(model) as trace:
        model(inputs)
        first = trace.records["wsi"]
        model(inputs * 2)
        assert trace.records["wsi"] is first
        assert len(trace.calls["wsi"]) == len(trace.calls["omic"]) == 2
        assert torch.equal(trace.calls["wsi"][1]["inputs"], inputs * 2)
        trace.reset()
        assert trace.records == {}
        assert trace.calls == {"wsi": [], "omic": []}
        model(inputs * 3)
        assert len(trace.calls["wsi"]) == 1
        assert torch.equal(trace.records["wsi"]["inputs"], inputs * 3)
    before = len(trace.calls["wsi"])
    model(inputs)
    assert len(trace.calls["wsi"]) == before


def test_entry_rng_replays_native_initialization_and_restores_outer_rng():
    model = SyntheticModel()
    inputs = torch.zeros(2, 5, 8)
    with NativeSlotTrace(model) as trace:
        model(inputs)
    outer = rng_state()
    with replay_rng(trace.records["omic"]["rng_entry"]):
        repeated = model.slot_attention_omic(inputs)
        assert torch.equal(repeated, trace.records["omic"]["final"])
    assert torch.equal(outer["cpu"], rng_state()["cpu"])


def test_zero_iterations_trace_and_replay_are_well_defined():
    model = SyntheticModel(iterations=0)
    inputs = torch.ones(2, 5, 8)
    with NativeSlotTrace(model) as trace:
        model(inputs)
    record = trace.records["wsi"]
    assert len(record["states"]) == 1
    assert record["attentions"] == []
    replay = replay_slots(model.slot_attention_wsi, inputs, record["initial"])
    assert replay["final"] is record["initial"]
    assert torch.equal(replay["final"], record["final"])


def test_replay_rejects_invalid_iteration_count():
    module = SyntheticSlot()
    with pytest.raises(ValueError, match="non-negative integer"):
        replay_slots(module, torch.zeros(2, 5, 8), torch.zeros(2, 3, 8), -1)
