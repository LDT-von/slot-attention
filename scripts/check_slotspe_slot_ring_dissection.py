"""Synthetic checks for the read-only slot collapse ring dissection.

These validate the ablation SEMANTICS on small CPU tensors.  They prove nothing
about patients and are explicitly not patient-performance evidence.  They do not
load any prepared patient tensor, and importing this module never imports torch
at module scope.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CHECKS = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


@check("plan_requires_none_reference_arm")
def _requires_none():
    from survot_rank.research.slotspe_proof.slot_ring_dissection import plan
    try:
        plan({"dissection_rings": ["r1_slot_axis_softmax"]})
    except ValueError as exc:
        assert "none" in str(exc), exc
        return {"rejected": str(exc)}
    raise AssertionError("a plan without the reference arm must be rejected")


@check("plan_rejects_unknown_ring")
def _unknown_ring():
    from survot_rank.research.slotspe_proof.slot_ring_dissection import plan
    try:
        plan({"dissection_rings": ["none", "r9_not_a_ring"]})
    except ValueError as exc:
        assert "r9_not_a_ring" in str(exc), exc
        return {"rejected": str(exc)}
    raise AssertionError("an unknown ring must be rejected")


@check("plan_rejects_duplicate_rings")
def _duplicate():
    from survot_rank.research.slotspe_proof.slot_ring_dissection import plan
    try:
        plan({"dissection_rings": ["none", "none"]})
    except ValueError as exc:
        return {"rejected": str(exc)}
    raise AssertionError("duplicate rings must be rejected")


@check("plan_rejects_bad_patient_index")
def _bad_index():
    from survot_rank.research.slotspe_proof.slot_ring_dissection import plan
    for bad in ([-1], ["0"], [1.5]):
        try:
            plan({"dissection_rings": ["none"], "dissection_patients": bad})
        except ValueError:
            continue
        raise AssertionError(f"patient index {bad!r} must be rejected")
    return {"rejected": "negative, string and float indices"}


@check("plan_declares_zero_training")
def _zero_training():
    from survot_rank.research.slotspe_proof.slot_ring_dissection import plan
    value = plan({"dissection_rings": ["none"]})
    assert value["training_runs"] == 0
    assert value["optimizer_steps"] == 0
    assert value["parameters_updated"] == 0
    assert value["default_action"] == "plan_only"
    return {k: value[k] for k in ("training_runs", "optimizer_steps", "parameters_updated")}


def _reference_module(torch):
    """The untouched native class, loaded from the preserved source."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "orig_slot_attention_ref", "/tmp/orig_slot_attention_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MultiHeadSlotAttention


@check("none_arm_matches_native_forward_bitwise")
def _none_matches_native(torch):
    from survot_rank.research.slotspe_proof.slot_ring_dissection import _dissect_forward
    Native = _reference_module(torch)
    torch.manual_seed(0)
    native = Native(num_slots=4, dim=8, heads=2, dim_head=4, iters=3)
    inputs = torch.randn(2, 7, 8)
    torch.manual_seed(1234)
    with torch.no_grad():
        expected = native(inputs)
        torch.manual_seed(1234)
        observed = _dissect_forward(native, inputs, "none")
    error = float((expected - observed).abs().max())
    assert error == 0.0, f"the 'none' arm must be bitwise identical, got {error}"
    return {"max_abs_difference": error}


@check("each_ring_changes_the_forward")
def _rings_change(torch):
    from survot_rank.research.slotspe_proof.slot_ring_dissection import _dissect_forward
    Native = _reference_module(torch)
    results = {}
    for ring in ("none", "r1_slot_axis_softmax", "r2_l1_normalize", "r3_gru_residual", "r4_shared_mu"):
        torch.manual_seed(7)
        module = Native(num_slots=4, dim=8, heads=2, dim_head=4, iters=3)
        inputs = torch.randn(2, 7, 8)
        torch.manual_seed(99)
        with torch.no_grad():
            slots = _dissect_forward(module, inputs, ring)
        assert bool(torch.isfinite(slots).all()), f"{ring} produced nonfinite slots"
        results[ring] = [round(float(v), 5) for v in slots.flatten()[:3]]
    distinct = {tuple(v) for v in results.values()}
    assert len(distinct) == 5, f"every ring must differ from the others: {results}"
    return results


@check("r4_does_not_add_or_change_parameters")
def _r4_parameters(torch):
    from survot_rank.research.slotspe_proof.slot_ring_dissection import _dissect_forward
    Native = _reference_module(torch)
    torch.manual_seed(7)
    module = Native(num_slots=4, dim=8, heads=2, dim_head=4, iters=3)
    before = {k: v.detach().clone() for k, v in module.state_dict().items()}
    names_before = set(module.state_dict())
    inputs = torch.randn(2, 7, 8)
    torch.manual_seed(99)
    _dissect_forward(module, inputs, "r4_shared_mu")
    names_after = set(module.state_dict())
    assert names_before == names_after, "R4 must not add or remove parameters"
    for key, value in module.state_dict().items():
        assert torch.equal(value, before[key]), f"R4 must not modify {key}"
    return {"parameters": len(names_after), "unchanged": True}


@check("no_ring_updates_gradients")
def _no_gradients(torch):
    from survot_rank.research.slotspe_proof.slot_ring_dissection import _dissect_forward
    Native = _reference_module(torch)
    for ring in ("none", "r1_slot_axis_softmax", "r2_l1_normalize", "r3_gru_residual", "r4_shared_mu"):
        torch.manual_seed(7)
        module = Native(num_slots=4, dim=8, heads=2, dim_head=4, iters=3)
        inputs = torch.randn(2, 7, 8)
        torch.manual_seed(99)
        with torch.no_grad():
            _dissect_forward(module, inputs, ring)
        dirty = [n for n, p in module.named_parameters() if p.grad is not None]
        assert not dirty, f"{ring} populated .grad on {dirty}"
    return {"gradients_populated": 0}


@check("attention_mass_semantics")
def _attention_mass(torch):
    """Confirm which axis each ring actually changes.

    The attention layout is (B, heads, slots, tokens), so dim -2 is the slot
    axis and dim -1 is the token axis.  The native L1 normalize runs over
    dim -1, i.e. it pins each slot's TOTAL MASS across tokens to 1 -- it does
    NOT pin the cross-slot mass at a token.  That distinction matters for
    reading the ring definitions honestly.
    """
    from survot_rank.research.slotspe_proof.slot_ring_dissection import _dissect_forward
    Native = _reference_module(torch)

    def capture(ring):
        torch.manual_seed(7)
        module = Native(num_slots=4, dim=8, heads=2, dim_head=4, iters=3)
        inputs = torch.randn(2, 7, 8)
        seen = []

        class Sink:
            def record_attention(self, a):
                seen.append(a.detach().clone())

            def record(self, s):
                pass

        torch.manual_seed(99)
        with torch.no_grad():
            _dissect_forward(module, inputs, ring, Sink())
        return seen[0].double()

    native = capture("none")
    r1 = capture("r1_slot_axis_softmax")
    r2 = capture("r2_l1_normalize")

    # Native: L1 over the token axis => each slot carries unit total mass.
    assert abs(float(native.sum(-1).mean()) - 1.0) < 1e-6, float(native.sum(-1).mean())
    # Native softmax runs over the SLOT axis (dim -2): slots compete with each
    # other, so the cross-slot mass AT A TOKEN is unconstrained.  It is not
    # 1/num_slots (that would mean a slot-axis L1 normalize); here each slot's
    # unit mass spreads over num_tokens, so the mean cross-slot sum is
    # num_slots / num_tokens = 4/7 for this shape.
    slots_axis, tokens_axis = native.shape[-2], native.shape[-1]
    expected_cross = slots_axis / tokens_axis
    assert abs(float(native.sum(-2).mean()) - expected_cross) < 1e-3, (
        float(native.sum(-2).mean()), expected_cross
    )
    # R1 moves that softmax to the token axis; unit per-slot token mass now
    # comes from the softmax itself, and the pattern must actually change.
    assert abs(float(r1.sum(-1).mean()) - 1.0) < 1e-5, float(r1.sum(-1).mean())
    assert not torch.allclose(r1, native), "R1 must change the attention pattern"
    # R2 removes the renormalization entirely, so total mass is no longer 1.
    assert abs(float(r2.sum(-1).mean()) - 1.0) > 1e-6, float(r2.sum(-1).mean())
    return {
        "layout": "[batch, heads, slots, tokens]",
        "native_per_slot_token_mass": float(native.sum(-1).mean()),
        "native_cross_slot_mass_at_token": float(native.sum(-2).mean()),
        "r1_per_slot_token_mass": float(r1.sum(-1).mean()),
        "r2_per_slot_token_mass": float(r2.sum(-1).mean()),
        "note": "L1 normalize pins per-slot token mass, not cross-slot mass at a token",
    }


@check("collector_records_one_state_per_iteration_plus_initial")
def _collector_counts(torch):
    from survot_rank.research.slotspe_proof.slot_ring_dissection import (
        _dissect_forward, IterationCollector,
    )
    Native = _reference_module(torch)
    torch.manual_seed(7)
    module = Native(num_slots=4, dim=8, heads=2, dim_head=4, iters=3)
    inputs = torch.randn(2, 7, 8)
    collector = IterationCollector()
    torch.manual_seed(99)
    _dissect_forward(module, inputs, "none", collector)
    assert len(collector.slot_states) == module.iters + 1, len(collector.slot_states)
    assert len(collector.attentions) == module.iters, len(collector.attentions)
    return {"slot_states": len(collector.slot_states), "attentions": len(collector.attentions)}


@check("cross_slot_divergence_responds_to_r1")
def _divergence_response(torch):
    """R1 must measurably change cross-slot reading divergence."""
    from survot_rank.research.slotspe_proof.slot_ring_dissection import (
        _dissect_forward, update_divergence,
    )
    Native = _reference_module(torch)

    def measure(ring, seed=5):
        torch.manual_seed(seed)
        module = Native(num_slots=6, dim=16, heads=2, dim_head=8, iters=4)
        inputs = torch.randn(3, 32, 16)
        seen = []

        class Sink:
            def record_attention(self, a):
                seen.append(a.detach().clone())

            def record(self, s):
                pass

        torch.manual_seed(31)
        _dissect_forward(module, inputs, ring, Sink())
        return torch.stack([update_divergence(torch, a).mean() for a in seen])

    native = measure("none")
    r1 = measure("r1_slot_axis_softmax")
    assert abs(float(native[-1] - r1[-1])) > 1e-6, "R1 must change final-iteration divergence"
    return {
        "native_final": round(float(native[-1]), 6),
        "r1_final": round(float(r1[-1]), 6),
    }


@check("pairwise_cosine_is_symmetric_and_bounded")
def _cosine_bounds(torch):
    from survot_rank.research.slotspe_proof.slot_ring_dissection import pairwise_cosine
    torch.manual_seed(3)
    slots = torch.randn(2, 5, 9)
    cosine = pairwise_cosine(torch, slots)
    assert bool(((cosine >= -1.0 - 1e-9) & (cosine <= 1.0 + 1e-9)).all())
    identical = pairwise_cosine(torch, torch.ones(1, 3, 4))
    assert torch.allclose(identical.double(), torch.ones(1).double(), atol=1e-9)
    return {"random": [round(float(v), 5) for v in cosine], "identical_slots": 1.0}


def main():
    import torch  # checks are CPU-only; importing here keeps plan() torch-free
    torch.set_num_threads(1)
    results, failures = [], 0
    for name, fn in CHECKS:
        try:
            # Plan-only checks take no torch argument; numeric checks do.
            value = fn() if fn.__code__.co_argcount == 0 else fn(torch)
            results.append({"check": name, "status": "pass", "detail": value})
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - report every failure
            failures += 1
            results.append({"check": name, "status": "fail", "error": repr(exc)})
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    print("NOTE: synthetic mechanism checks only. No patient evidence, no performance claim.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())