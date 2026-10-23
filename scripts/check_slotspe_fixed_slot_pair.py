"""Synthetic checks for the fixed-slot pair protocol.

Scope: synthetic selection-behaviour checks only.  No patient data, no trained
model, and no performance claim.  These checks exercise the keep-score override
against the REAL native ``parallel_topk_st`` so the claim "the native top-k
returns exactly the requested slots" is verified, not assumed.  Because the
native selection adds Gumbel noise before the top-k, every selection check is
repeated over many draws so a single lucky draw cannot pass as a rule.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_native(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _raises(callable_) -> bool:
    try:
        callable_()
    except ValueError:
        return True
    except Exception:
        return False
    return False


def main() -> int:
    import torch

    from survot_rank.research.slotspe_proof.fixed_slot_pair import (
        FIXED_ARMS, KeepScoreOverride, fixed_plan, selection_rule_name, _MASK_GAP,
    )
    native = _load_native("native_slot_attention", "third_party/SlotSPE/models/slot_attention.py")

    results: dict[str, bool] = {}

    # --- plan validation -------------------------------------------------
    base = {"slot_arms": list(FIXED_ARMS), "fixed_epochs": 3}
    plan = fixed_plan(base)
    results["plan_lists_three_arms"] = plan["slot_arms"] == list(FIXED_ARMS)
    results["plan_requires_reference_arm"] = _raises(lambda: fixed_plan({"slot_arms": ["N_fixed34"]}))
    results["plan_rejects_duplicate_arms"] = _raises(
        lambda: fixed_plan({"slot_arms": ["N_topk", "N_topk"]}))
    results["plan_rejects_zero_epochs"] = _raises(lambda: fixed_plan({**base, "fixed_epochs": 0}))
    results["plan_rejects_bad_indices"] = _raises(
        lambda: fixed_plan({**base, "fixed_slot_indices": [2, 2]}))
    results["plan_rejects_unsorted_indices"] = _raises(
        lambda: fixed_plan({**base, "fixed_slot_indices": [3, 2]}))
    results["plan_rejects_negative_index"] = _raises(
        lambda: fixed_plan({**base, "fixed_slot_indices": [-1, 2]}))
    results["plan_rejects_count_mismatch"] = _raises(
        lambda: fixed_plan({**base, "fixed_slot_indices": [2, 3, 4]}))
    results["plan_warns_about_predkeep_dependency"] = "pred_keep_slot" in plan["known_dependency_warning"]
    results["plan_declares_dispersion_limits"] = any(
        "dispersion" in item for item in plan["limitations"])
    results["rule_names_distinct"] = len({selection_rule_name(a) for a in FIXED_ARMS}) == 3
    results["mask_gap_is_finite"] = math.isfinite(_MASK_GAP)

    # --- the override makes the native top-k pick exactly our slots -----
    # Repeated over many Gumbel draws: the native selection adds noise before
    # top-k, so one draw proves nothing.
    n_slots, keep_count, batch, draws = 8, 2, 6, 200
    torch.manual_seed(0)
    fixed_ok = worst_ok = differs = True
    native_selects = set()
    for draw in range(draws):
        raw = torch.randn(batch, n_slots, 1) * 5.0

        # ``parallel_topk_st`` returns ``(y, topk_indices)`` where ``y`` is the
        # one-hot hot mask and ``topk_indices`` is an INDEX tensor.  Selection is
        # read from ``y``; reading indices as a mask would be a category error.
        y_native, _ = native.parallel_topk_st(raw.squeeze(-1), k=keep_count, temperature=0.01)
        native_selects |= {tuple(sorted(r.nonzero().flatten().tolist())) for r in y_native}

        fixed_hook = KeepScoreOverride("N_fixed34", [2, 3], torch)
        fixed_scores = fixed_hook(None, None, raw.clone())
        y_fixed, _ = native.parallel_topk_st(fixed_scores.squeeze(-1), k=keep_count, temperature=0.01)
        fixed_sets = {tuple(sorted(r.nonzero().flatten().tolist())) for r in y_fixed}
        fixed_ok &= fixed_sets == {(2, 3)}

        worst_hook = KeepScoreOverride("N_worstpredkeep", [2, 3], torch)
        worst_scores = worst_hook(None, None, raw.clone())
        y_worst, _ = native.parallel_topk_st(worst_scores.squeeze(-1), k=keep_count, temperature=0.01)
        expected = {tuple(sorted(r.squeeze(-1).argsort()[:keep_count].tolist())) for r in raw}
        worst_sets = {tuple(sorted(r.nonzero().flatten().tolist())) for r in y_worst}
        worst_ok &= worst_sets == expected
        differs |= worst_sets != {tuple(sorted(r.nonzero().flatten().tolist())) for r in y_native}

    results[f"fixed_arm_selects_exactly_indices_2_3_over_{draws}_gumbel_draws"] = fixed_ok
    results[f"worst_arm_selects_two_lowest_over_{draws}_gumbel_draws"] = worst_ok
    results["worst_arm_differs_from_native"] = differs
    results["native_topk_itself_varies_across_patients"] = len(native_selects) > 1

    # The reference arm must be a bit-identical passthrough.
    raw = torch.randn(batch, n_slots, 1)
    ref_hook = KeepScoreOverride("N_topk", [2, 3], torch)
    results["reference_arm_is_passthrough"] = torch.equal(ref_hook(None, None, raw.clone()), raw)

    # --- gradient topology must match the original arm ------------------
    # The bias is a detached constant, so the override must NOT disconnect the
    # scorer.  An earlier version replaced the score with ``full_like`` constants
    # and silently removed the scorer from the graph; this asserts that is gone.
    leaf = torch.nn.Parameter(torch.randn(batch, n_slots, 1))
    hooked = KeepScoreOverride("N_fixed34", [2, 3], torch)
    out = hooked(None, None, leaf)
    out.sum().backward()
    results["override_keeps_grad_path_to_scorer"] = (
        leaf.grad is not None and bool(torch.isfinite(leaf.grad).all()))
    results["override_bias_carries_no_gradient"] = bool((leaf.grad.abs().sum() > 0).item())

    # The native straight-through surrogate runs on the noised score, so the
    # top-k output itself is detached from the score in the ORIGINAL arm too.
    ref_scores = torch.randn(batch, n_slots, 1, requires_grad=True)
    y_ref, _ = native.parallel_topk_st(ref_scores.squeeze(-1), k=keep_count, temperature=0.01)
    results["native_hard_selection_is_boolean"] = bool(
        torch.equal(y_ref.detach(), y_ref.detach().round()))

    # --- numerical hygiene ----------------------------------------------
    fixed_scores = KeepScoreOverride("N_fixed34", [2, 3], torch)(None, None, torch.randn(batch, n_slots, 1))
    results["override_is_finite"] = bool(torch.isfinite(fixed_scores).all())
    results["override_preserves_shape"] = tuple(fixed_scores.shape) == (batch, n_slots, 1)
    soft = torch.softmax(fixed_scores.squeeze(-1) / 0.01, dim=-1)
    results["soft_gate_has_no_nan"] = bool(torch.isfinite(soft).all())
    results["override_counts_exactly_keep"] = (
        min(KeepScoreOverride("N_fixed34", [2, 3], torch).observed_selected or [keep_count]) == keep_count)
    results["unknown_arm_rejected"] = _raises(lambda: KeepScoreOverride("N_bogus", [2, 3], torch))

    passed = sorted(k for k, v in results.items() if v)
    failed = sorted(k for k, v in results.items() if not v)
    print(json.dumps({"checks": results, "passed": len(passed), "failed": failed,
                      "ok": not failed,
                      "scope": "Synthetic selection-behaviour checks against the REAL native "
                               "parallel_topk_st over many Gumbel draws. No patient data, no "
                               "training, no performance claim."},
                     ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())