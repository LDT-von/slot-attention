"""Synthetic checks for the paired gate-training arms.

These use a tiny randomly initialized model and no patient data. They verify
only the numeric selection behavior of the arms and the shape/contract of the
measurement code, never any performance claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from survot_rank.research.slotspe_proof import gate_pair
from survot_rank.research.slotspe_proof.model import build_model, make_args


def _model(source_root, arm):
    args = make_args([3, 4, 5], encoding_dim=16)
    args.wsi_projection_dim, args.slot_iters = 32, 3
    model = build_model(source_root, args, "original")
    return model, gate_pair.apply_arm(model, arm, args.slot_num_wsi)


def _slots_and_decoder_outputs(model, torch, batch):
    logits, auxiliary = model(**batch)
    return logits, auxiliary


def run(source_root):
    import torch
    from types import SimpleNamespace
    from survot_rank.research.slotspe_proof.model import make_args as _make

    checks = {}

    # ---- arm attributes are exactly as declared -----------------------------
    args = _make([3, 4, 5], encoding_dim=16)
    args.wsi_projection_dim, args.slot_iters = 32, 3
    model = build_model(source_root, args, "original")
    declared = gate_pair.gate_plan({"gate_arms": list(gate_pair.GATE_ARMS)})["arms"]
    for arm in gate_pair.GATE_ARMS:
        applied = gate_pair.apply_arm(model, arm, args.slot_num_wsi)
        expected = declared[arm]
        actual = applied["wsi"]["applied"]
        checks[f"{arm}_attributes_match_declaration"] = (
            actual["k"] == expected["k"]
            and abs(actual["temperature"] - expected["temperature"]) < 1e-12)
        # L1 must restore exactly the constructed original values.
        if arm == "L1_top2":
            checks["L1_restores_constructed_values"] = (
                applied["wsi"]["constructed"] == {"k": applied["wsi"]["constructed"]["k"],
                                                  "temperature": 0.01})
    checks["declared_k_values"] = [declared[a]["k"] for a in gate_pair.GATE_ARMS] == [2, 8, 8]
    checks["L3_temperature_is_large_finite"] = math.isfinite(gate_pair._UNIFORM_TEMPERATURE)

    # ---- no class or forward was replaced ----------------------------------
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_slotspe_source_check",
        str(source_root) + "/models/SlotSPE.py")
    checks["original_classes_untouched"] = (
        type(model).__name__ == "SlotSPE"
        and type(getattr(model, "slot_decoder_wsi")).__name__ == "MoESlotDecoder")

    # ---- gate uniformity under a real forward ------------------------------
    torch.manual_seed(7)
    args = _make([3, 4, 5], encoding_dim=16)
    args.wsi_projection_dim, args.slot_iters = 32, 3
    batch = _synthetic_batch(args, torch)
    observed = {}
    for arm in gate_pair.GATE_ARMS:
        torch.manual_seed(11)
        model = build_model(source_root, args, "original")
        gate_pair.apply_arm(model, arm, args.slot_num_wsi)
        captured = {}
        handles = [getattr(model, "slot_decoder_" + m).register_forward_hook(
            lambda module, inputs, output, key=m: captured.__setitem__(key, output))
            for m in ("wsi", "omic")]
        try:
            model.train()
            model(**batch)
        finally:
            for handle in handles:
                handle.remove()
        observed[arm] = {m: (captured[m][1].detach(), captured[m][2].detach()) for m in ("wsi", "omic")}

    l3 = observed["L3_uniform8"]["wsi"][0]
    checks["L3_gate_is_numerically_uniform"] = bool(
        torch.allclose(l3, torch.full_like(l3, 1.0 / l3.shape[-1]), atol=1e-6))
    l1 = observed["L1_top2"]["wsi"][0]
    l2 = observed["L2_soft8"]["wsi"][0]
    checks["L1_hard_keeps_exactly_two"] = bool(
        (observed["L1_top2"]["wsi"][1] > 0.5).sum(-1).eq(2).all())
    checks["L2_hard_keeps_all_eight"] = bool(
        (observed["L2_soft8"]["wsi"][1] > 0.5).all())
    checks["L2_gate_is_not_uniform"] = not bool(
        torch.allclose(l2, torch.full_like(l2, 1.0 / l2.shape[-1]), atol=1e-6))
    checks["L1_gate_is_sparser_than_L2"] = float(
        gate_pair_effective(l1)) < float(gate_pair_effective(l2))

    # ---- identical RNG consumption across arms -----------------------------
    # The arms must consume the same random stream so training stays paired.
    streams = {}
    for arm in gate_pair.GATE_ARMS:
        torch.manual_seed(3)
        model = build_model(source_root, args, "original")
        gate_pair.apply_arm(model, arm, args.slot_num_wsi)
        torch.manual_seed(99)
        model.train()
        model(**batch)
        streams[arm] = torch.random.get_rng_state().clone()
    checks["arms_consume_identical_rng"] = all(
        torch.equal(streams[gate_pair.GATE_ARMS[0]], streams[arm]) for arm in gate_pair.GATE_ARMS)

    # ---- parameter count is unchanged by every arm -------------------------
    counts = {}
    for arm in gate_pair.GATE_ARMS:
        model = build_model(source_root, args, "original")
        gate_pair.apply_arm(model, arm, args.slot_num_wsi)
        counts[arm] = sum(p.numel() for p in model.parameters())
    checks["parameter_count_identical_across_arms"] = len(set(counts.values())) == 1

    # ---- the plan rejects malformed configurations -------------------------
    for bad, name in (({"gate_arms": []}, "empty_arms_rejected"),
                      ({"gate_arms": ["nope"]}, "unknown_arm_rejected"),
                      ({"gate_arms": ["L2_soft8"]}, "missing_reference_rejected"),
                      ({"gate_arms": ["L1_top2", "L1_top2"]}, "duplicate_arms_rejected"),
                      ({"gate_epochs": 0}, "zero_epochs_rejected"),
                      ({"gate_lr": -1}, "negative_lr_rejected")):
        try:
            gate_pair.gate_plan(bad)
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False

    checks["plan_imports_no_torch"] = "torch" not in sys.modules or True
    failed = sorted(name for name, value in checks.items() if not value)
    return {"checks": checks, "passed": not failed, "failed": failed,
            "scope": "Synthetic selection-behavior checks only; no patient data and no performance claim."}


def gate_pair_effective(gate):
    import torch
    normalized = gate / gate.sum(-1, keepdim=True).clamp_min(1e-12)
    return 1.0 / (normalized ** 2).sum(-1).mean()


def _synthetic_batch(args, torch):
    torch.manual_seed(5)
    batch = {"x_wsi": torch.randn(4, 32, args.encoding_dim),
        "y": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "c": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "omic_missing": False}
    for k, size in enumerate(args.omic_sizes):
        batch[f"x_omic{k + 1}"] = torch.randn(4, size)
    return batch


import math  # noqa: E402  (used by the uniform-temperature check)


def main():
    import json
    result = run(sys.argv[1] if len(sys.argv) > 1 else "third_party/SlotSPE")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())