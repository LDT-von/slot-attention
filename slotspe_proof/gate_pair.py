"""Opt-in paired gate training: does the selector causally shape slot specialization?

This is a separate experiment from every frozen-checkpoint diagnostic in this
package.  It trains three arms that differ ONLY in the numeric selection
attributes of the two native ``MoESlotDecoder`` modules:

``L1_top2``
    The unchanged original arm: ``k=2``, temperature 0.01, Gumbel-top-k sampling
    during training.  This is the reference behaviour.
``L2_soft8``
    ``k=8`` so every candidate is hard-selected, while the LEARNED soft weights
    are retained.  This removes only the hard sparsity of the selector.
``L3_uniform8``
    ``k=8`` plus an exactly uniform gate.  ``temperature`` is set to a large
    finite value so the native ``softmax(scores / temperature)`` becomes
    numerically uniform without replacing any class or forward.  The learned
    scores then no longer influence the weighted output.

No class, forward, or original source file is modified.  Because
``parallel_topk_st`` draws its Gumbel noise once per call independently of
``k``, all three arms consume identical random streams, and every step reseeds
torch/numpy/python RNG identically before the arm runs.

Importing this module never imports torch, loads a model, or reads patient data.
Planning validates only the declared factors.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

GATE_ARMS = ("L1_top2", "L2_soft8", "L3_uniform8")

# A large finite temperature makes softmax(scores / temperature) numerically
# uniform over the candidates.  It is finite so the native division stays well
# defined and no attribute becomes inf.
_UNIFORM_TEMPERATURE = 1e6


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def gate_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the declared factors without importing torch or a model."""
    arms = list(config.get("gate_arms", GATE_ARMS))
    if not arms or any(arm not in GATE_ARMS for arm in arms):
        raise ValueError(f"gate_arms must be a nonempty subset of {list(GATE_ARMS)}")
    if "L1_top2" not in arms:
        raise ValueError("L1_top2 is the reference arm and must be included")
    if len(set(arms)) != len(arms):
        raise ValueError("gate_arms must be unique")
    device = config.get("device", "cuda:0")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a nonempty string")
    epochs = _integer(config.get("gate_epochs", 3), "gate_epochs", minimum=1)
    return {
        "schema_version": 1,
        "phase": "gate_pair_train",
        "gate_arms": arms,
        "fold": _integer(config.get("fold", 0), "fold"),
        "seed": _integer(config.get("seed", 3), "seed"),
        "device": device,
        "gate_epochs": epochs,
        "gate_lr": _positive_float(config.get("gate_lr", 5e-4), "gate_lr"),
        "gate_batch_size": _integer(config.get("gate_batch_size", 8), "gate_batch_size", minimum=1),
        "gate_eval_repeats": _integer(config.get("gate_eval_repeats", 5), "gate_eval_repeats", minimum=1),
        "initialization": "same_new_initial_state",
        "training_budget": "equal_epochs_updates_batch_order_and_step_seeds",
        "arm_difference": "numeric selection attributes of the two native slot decoders only",
        "checkpoint_rule": "final_epoch_only_no_best_selection",
        "optimizer": {"name": "Adam", "lr": _positive_float(config.get("gate_lr", 5e-4), "gate_lr")},
        "scheduler": {"name": "CosineAnnealingLR", "t_max": epochs, "eta_min": 1e-6},
        "primary_readout": "trained_slot_redundancy_not_validation_cindex",
        "reference_arm_recorded_val_cindex": config.get("reference_arm_recorded_val_cindex"),
        "reference_arm_recorded_val_cindex_per_repeat": config.get("reference_arm_recorded_val_cindex_per_repeat"),
        "reference_tolerance": _positive_float(config.get("reference_tolerance", 1e-6), "reference_tolerance"),
        "arms": {arm: {
            "k": 2 if arm == "L1_top2" else 8,
            "temperature": 0.01 if arm != "L3_uniform8" else _UNIFORM_TEMPERATURE,
            "hard_selected_slots": 2 if arm == "L1_top2" else 8,
            "learned_scores_influence_weighted_output": arm != "L3_uniform8",
        } for arm in arms},
        "limitations": [
            "single fold: the recorded cross-fold dispersion exceeds any candidate effect",
            "short epoch budget: the final epoch is not a converged model",
            "validation cohort is small: C-index is recorded but is not the primary readout",
            "L3 makes the learned scores inert rather than removing the parameters",
        ],
        "interpretation": (
            "Paired causal contrast of the gate mechanism at one fixed budget on one fold. "
            "A short budget does not establish training adequacy, and one fold cannot "
            "separate a gate effect from the known cross-fold dispersion."
        ),
    }


def _read(path: Any) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: Any, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Any) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _epoch_orders(ids: list[str], epochs: int, seed: int) -> list[list[str]]:
    """One deterministic batch order per epoch, shared by every arm.

    This reproduces the recorded harness order exactly
    (``torch.randperm`` under ``seed*1000+epoch`` over training indices) rather
    than inventing a new shuffle.  Identical across arms by construction, and it
    lets the reference arm act as a fidelity anchor against the recorded
    fold/seed baseline.
    """
    import torch

    if not ids or len(set(ids)) != len(ids):
        raise ValueError("training IDs must be nonempty and unique")
    orders = []
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(len(ids), generator=generator).tolist()
        orders.append([ids[index] for index in order])
    return orders


def apply_arm(model: Any, arm: str, slot_count: int) -> dict[str, Any]:
    """Set ONLY native numeric attributes; never replace a class or forward.

    ``k`` is the native hard-selection count and ``temperature`` is the native
    soft-gate divisor.  Both are plain Python attributes of the original
    module, so overriding them cannot alter any other behavior, and L1 restores
    exactly the constructed original values.
    """
    if arm not in GATE_ARMS:
        raise ValueError(f"Unknown gate arm: {arm}")
    if slot_count < 2:
        raise ValueError("slot_count must be at least two")
    k = 2 if arm == "L1_top2" else slot_count
    temperature = _UNIFORM_TEMPERATURE if arm == "L3_uniform8" else 0.01
    recipe = {}
    for modality in ("wsi", "omic"):
        name = "slot_decoder_" + modality
        decoder = getattr(model, name, None)
        if decoder is None:
            raise ValueError(f"model has no native {name} module")
        if not hasattr(decoder, "k") or isinstance(decoder.k, bool) or not isinstance(decoder.k, int):
            raise ValueError(f"{name}.k is not a native integer selection attribute")
        if not isinstance(decoder.temperature, float):
            raise ValueError(f"{name}.temperature is not a native float attribute")
        constructed = (decoder.k, decoder.temperature)
        decoder.k = k
        decoder.temperature = temperature
        recipe[modality] = {"module": name, "constructed": {"k": constructed[0], "temperature": constructed[1]},
            "applied": {"k": decoder.k, "temperature": decoder.temperature},
            "attribute_names_unchanged": True}
    return recipe


def _restored_kwargs(original: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(original, Mapping):
        return copy.deepcopy(dict(original))
    return copy.deepcopy(vars(original))


def run(config, *, source_root, input_root, output_dir, device_override=None):
    """Execute ONLY under an explicit ``--run`` CLI with authorization."""
    plan = gate_plan(config)
    plan["device"] = device_override or plan["device"]

    import numpy as np
    import torch
    from sksurv.metrics import concordance_index_censored

    from .model import build_model
    from .patient import _batch, _rng
    from .problem_trace import NativeSlotTrace

    device = torch.device(plan["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device unavailable; no automatic CPU fallback")
    torch.set_num_threads(int(config.get("torch_threads", 2)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to((Path(__file__).resolve().parents[1] / "local_evidence").resolve()):
        raise ValueError("gate training outputs must be inside ignored local_evidence")
    if output_dir.exists():
        raise FileExistsError("gate training requires a fresh output directory")

    root = Path(input_root).resolve()
    manifest = _read(root / "prepared/manifest.json")
    fold = manifest["folds"][str(plan["fold"])]
    train_ids, val_ids = list(fold["train"]), list(fold["val"])
    if not train_ids or not val_ids or set(train_ids) & set(val_ids):
        raise ValueError("prepared train/validation IDs must be disjoint and nonempty")
    orders = _epoch_orders(train_ids, plan["gate_epochs"], plan["seed"])

    source = str(Path(source_root).resolve())
    args = SimpleNamespace(**_restored_kwargs(config["model_args"]))
    # omic_sizes/encoding_dim describe the prepared cohort and are taken from the
    # manifest, then checked against the declared recipe when one is supplied.
    declared_omic = config["model_args"].get("omic_sizes")
    if declared_omic is not None and list(declared_omic) != list(manifest["omic_sizes"]):
        raise ValueError("declared omic_sizes disagree with the prepared manifest")
    args.omic_sizes = list(manifest["omic_sizes"])
    if config["model_args"].get("encoding_dim") != manifest["encoding_dim"]:
        raise ValueError("declared encoding_dim disagrees with the prepared manifest")
    args.encoding_dim = manifest["encoding_dim"]
    for required in ("wsi_projection_dim", "slot_iters", "temperature", "topk_ratio",
                     "top_k_method", "bag_loss", "alpha_surv", "lambda_recon_loss",
                     "rna_format", "n_classes"):
        if required not in vars(args):
            raise ValueError(f"declared model args must include {required}")
    if args.rna_format != "Pathways" or args.bag_loss != "nll_surv":
        raise ValueError("this protocol requires the native Pathways + nll_surv recipe")
    slot_count = int(args.slot_num_wsi)
    if int(args.slot_num_omics) != slot_count:
        raise ValueError("this protocol requires equal WSI and omic slot counts")

    output_dir.mkdir(parents=True)
    receipt = {"status": "started", "phase": "gate_pair_train", "plan": plan,
        "prepared_input_root": str(root), "prepared_manifest_sha256": sha(root / "prepared/manifest.json"),
        "runner_sha256": sha(__file__), "source_root": source,
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "started_utc": datetime.now(timezone.utc).isoformat(), "patient_outputs_private": True}

    # One initial state shared by every arm, drawn once under the declared seed.
    _rng(plan["seed"], torch, np)
    reference = build_model(source, args, "original")
    shared_initial = {name: value.detach().cpu().clone() for name, value in reference.state_dict().items()}
    parameter_count = sum(p.numel() for p in reference.parameters())
    initial_digest = hashlib.sha256(
        b"".join(shared_initial[name].numpy().tobytes() for name in sorted(shared_initial))).hexdigest()
    del reference
    receipt["shared_initial_state_sha256"] = initial_digest
    receipt["parameter_count"] = parameter_count

    _write_json(output_dir / "recipe.json", {**recipe_fields(plan, root, fold, orders, initial_digest, parameter_count),
        "arm_difference": plan["arm_difference"]})
    _write_json(output_dir / "execution_receipt.json", receipt)

    records = []
    try:
        for arm in plan["gate_arms"]:
            arm_root = output_dir / arm
            arm_root.mkdir()
            _rng(plan["seed"], torch, np)
            model = build_model(source, args, "original").to(device)
            if sum(p.numel() for p in model.parameters()) != parameter_count:
                raise RuntimeError("gate arm changed the model parameter count")
            model.load_state_dict(shared_initial, strict=True)
            applied = apply_arm(model, arm, slot_count)
            _write_json(arm_root / "arm_recipe.json", applied)
            _write_json(arm_root / "curve.json", [])

            optimizer = torch.optim.Adam(model.parameters(), lr=plan["gate_lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=plan["gate_epochs"], eta_min=1e-6)

            curve, steps = [], 0
            for epoch, order in enumerate(orders):
                model.train()
                totals = np.zeros(3)
                seen = 0
                for batch_index, start in enumerate(range(0, len(order), plan["gate_batch_size"])):
                    ids = order[start:start + plan["gate_batch_size"]]
                    # Identical per-step seed for every arm: any difference in the
                    # resulting trajectory is attributable to the gate attributes.
                    _rng(plan["seed"] * 100000 + epoch * 1000 + batch_index, torch, np)
                    batch = _batch(root, manifest, fold, ids, device, torch)
                    optimizer.zero_grad(set_to_none=True)
                    logits, auxiliary = model(**batch)
                    main = model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None) / len(ids)
                    total = main + auxiliary
                    if not bool(torch.isfinite(total)):
                        raise RuntimeError(f"nonfinite loss in arm={arm} epoch={epoch + 1}")
                    total.backward()
                    optimizer.step()
                    totals += np.asarray([float(main.detach()), float(auxiliary.detach()), float(total.detach())]) * len(ids)
                    seen += len(ids)
                    steps += 1
                validation = evaluate(model, root, manifest, fold, val_ids, plan["gate_batch_size"],
                                      device, plan["seed"] * 10000 + epoch, torch, np, concordance_index_censored)
                curve.append({"epoch": epoch + 1,
                    "train_main_nll": float(totals[0] / seen),
                    "train_aux_loss": float(totals[1] / seen),
                    "train_total_loss": float(totals[2] / seen),
                    "val_nll_exploratory": validation["nll"],
                    "val_cindex_exploratory": validation["cindex"],
                    "val_cindex_status": validation["cindex_status"]})
                _write_json(arm_root / "curve.json", curve)
                print(f"arm={arm} epoch={epoch + 1}/{plan['gate_epochs']} "
                      f"loss={curve[-1]['train_total_loss']:.5f} val_C={validation['cindex']}", flush=True)
                scheduler.step()

            checkpoint = arm_root / "last.pt"
            torch.save({"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "arm": arm, "fold": plan["fold"], "seed": plan["seed"], "optimizer_steps": steps,
                "selection_attributes": applied}, checkpoint)

            redundancy = measure_redundancy(model, root, manifest, fold, val_ids,
                                            plan["gate_batch_size"], device, plan["seed"], torch, np,
                                            slot_count)
            _write_json(arm_root / "trained_redundancy.json", redundancy)

            selector = selector_gradient(model, root, manifest, fold, train_ids[:plan["gate_batch_size"]],
                                         device, torch, np)
            _write_json(arm_root / "selector_gradient.json", selector)

            repeats = [evaluate(model, root, manifest, fold, val_ids, plan["gate_batch_size"], device,
                                plan["seed"] * 1000000 + 80000 + repeat, torch, np, concordance_index_censored)
                       for repeat in range(plan["gate_eval_repeats"])]
            _write_json(arm_root / "eval_repeats.json", repeats)
            scores = [r["cindex"] for r in repeats if r["cindex"] is not None]
            records.append({"arm": arm, "optimizer_steps": steps, "checkpoint_sha256": sha(checkpoint),
                "shared_initial_state_sha256": initial_digest, "selection_attributes": applied,
                "final_epoch_curve": curve[-1] if curve else None,
                "val_cindex_per_repeat": scores,
                "val_cindex_mean_native_eval": float(np.mean(scores)) if scores else None,
                "val_cindex_std_native_eval": float(np.std(scores)) if len(scores) > 1 else 0.0 if scores else None,
                "redundancy": redundancy["summary"], "selector_gradient": selector["parameters"],
                "interpretation": plan["interpretation"]})
            del model, optimizer, scheduler
            if device.type == "cuda":
                torch.cuda.empty_cache()

        contrasts = build_contrasts(records)
        result = {"schema": "slotspe_gate_pair_v1", "status": "completed", "plan": plan,
            "counts": fold["counts"], "arms": records, "contrasts": contrasts,
            "primary_readout": "trained_slot_redundancy",
            "cindex_role": "recorded for continuity only; not the primary readout and not a performance claim",
            "harm_or_correction_efficacy_proved": False,
            "fidelity_anchor": fidelity_anchor(records, plan),
            "new_optimizer_steps": sum(r["optimizer_steps"] for r in records)}
        _write_json(output_dir / "summary.json", result)
        receipt.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
                       summary_sha256=sha(output_dir / "summary.json"))
        _write_json(output_dir / "execution_receipt.json", receipt)
        return result
    except BaseException as exc:
        receipt.update(status="failed_retained", error=type(exc).__name__ + ": " + str(exc),
                       finished_utc=datetime.now(timezone.utc).isoformat())
        _write_json(output_dir / "execution_receipt.json", receipt)
        raise


def recipe_fields(plan, root, fold, orders, initial_digest, parameter_count):
    return {
        "plan": plan,
        "prepared_input_root": str(root),
        "prepared_manifest_sha256": sha(Path(root) / "prepared/manifest.json"),
        "fold_binning": fold["binning"],
        "fold_binning_source": "unchanged prepared manifest; no bin fitting in this phase",
        "epoch_sample_orders": orders,
        "shared_initial_state_sha256": initial_digest,
        "parameter_count": parameter_count,
        "loss": "native loss_fn(main logits)/batch_size + native train auxiliary scalar",
        "step_seed_rule": "seed*100000 + epoch*1000 + batch_index, identical across arms",
        "per_step_rng_reset": "python/numpy/torch/cuda RNG reseeded identically before every arm step",
        "gumbel_noise_draw_count_per_decoder_call": "one rand_like, independent of k; arms consume identical random streams",
        "original_source_files_modified": False,
        "validation": "native eval only; final epoch; no checkpoint selection",
        "known_cross_fold_dispersion": {"validation_cindex_std_across_folds": 0.06178,
            "validation_cindex_std_across_seeds": 0.06059,
            "source": "docs/research/slotspe_gpu_trisurv_20261022 five-fold record"},
    }


def fidelity_anchor(records, plan):
    """Independent check that the reference arm reproduces the recorded run.

    L1 restates the unchanged original decoder, so at the same fold/seed/budget
    it must reproduce the already-recorded baseline.  A mismatch would mean the
    runner itself changed the protocol and every other arm number is then
    uninterpretable, so this is recorded explicitly rather than assumed.
    """
    reference = next((r for r in records if r["arm"] == "L1_top2"), None)
    observed = reference["val_cindex_mean_native_eval"] if reference else None
    expected = plan.get("reference_arm_recorded_val_cindex")
    per_repeat_expected = plan.get("reference_arm_recorded_val_cindex_per_repeat")
    per_repeat_observed = reference.get("val_cindex_per_repeat") if reference else None
    detail = None
    if per_repeat_expected and per_repeat_observed:
        # Exact per-repeat comparison is far stronger than comparing means: it
        # catches an off-by-one in eval seeds or a silently changed protocol.
        detail = {"expected": list(per_repeat_expected)[:len(per_repeat_observed)],
            "observed": list(per_repeat_observed)[:len(per_repeat_expected)],
            "max_abs_delta": max(abs(float(a) - float(b))
                                 for a, b in zip(per_repeat_expected, per_repeat_observed)
                                 if a is not None and b is not None)}
    if observed is None or expected is None:
        return {"status": "not_available",
                "observed": observed, "expected": expected,
                "note": "No recorded baseline supplied in the config; reference arm reported as-is."}
    delta = float(observed) - float(expected)
    reproduced = abs(delta) <= plan.get("reference_tolerance", 1e-6)
    if detail is not None and detail["max_abs_delta"] > plan.get("reference_tolerance", 1e-6):
        reproduced = False
    return {"status": "reproduced" if reproduced else "mismatch",
            "observed": float(observed), "expected": float(expected), "delta": delta,
            "tolerance": plan.get("reference_tolerance", 1e-6),
            "per_repeat": detail,
            "note": "L1 is the unchanged original arm; reproducing the recorded baseline "
                    "is required before any contrast in this run can be attributed to the gate."}


def evaluate(model, root, manifest, fold, ids, batch_size, device, seed, torch, np, concordance):
    from .patient import _batch, _rng
    model.eval()
    _rng(seed, torch, np)
    predictions, loss = [], 0.0
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            selected = ids[start:start + batch_size]
            batch = _batch(root, manifest, fold, selected, device, torch)
            logits, _ = model(**batch)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("nonfinite evaluation logits")
            loss += float(model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None))
            predictions.append(logits.detach().cpu())
    logits = torch.cat(predictions)
    # Stable risk: negative cumulative survival. This must stay identical to the
    # recorded harness (patient.py::_evaluate); an earlier cumulative-sum form
    # here produced different numbers and is not the original definition.
    risks = -(1 - logits.sigmoid()).cumprod(-1).sum(-1)
    events = np.asarray([manifest["patients"][pid]["censor"] < 1 for pid in ids], dtype=bool)
    times = np.asarray([manifest["patients"][pid]["time"] for pid in ids], dtype=float)
    try:
        score = float(concordance(events, times, risks.numpy())[0])
        score = score if math.isfinite(score) else None
        status = "computed" if score is not None else "undefined_nonfinite"
    except ValueError as exc:
        score, status = None, "undefined: " + str(exc)
    return {"nll": loss / len(ids), "cindex": score, "cindex_status": status,
            "risks": risks.tolist(), "eval_seed": seed}


def selector_gradient(model, root, manifest, fold, ids, device, torch, np):
    """Whether the trained selector still receives a gradient from each path."""
    from .patient import _batch, _rng
    model.train()
    _rng(0, torch, np)
    batch = _batch(root, manifest, fold, ids, device, torch)
    logits, auxiliary = model(**batch)
    main = model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None) / len(ids)
    named = [(name, p) for name, p in model.named_parameters() if "pred_keep_slot" in name and p.requires_grad]
    model.eval()
    if not named:
        return {"status": "not_observed_no_matching_selector_parameters", "parameters": {}}
    parameters = [p for _, p in named]
    grads = torch.autograd.grad(main, parameters, retain_graph=True, allow_unused=True)
    aux_grads = torch.autograd.grad(auxiliary, parameters, allow_unused=True)

    def describe(gradient):
        if gradient is None:
            return {"connected": False, "nonzero": False, "norm": None}
        return {"connected": True, "nonzero": bool(torch.count_nonzero(gradient)),
                "norm": float(gradient.norm()), "finite": bool(torch.isfinite(gradient).all())}

    return {"parameters": {name: {"main": describe(g), "aux": describe(a)}
                           for (name, _), g, a in zip(named, grads, aux_grads)},
            "expected_by_construction": {
                "L1_top2": "main path disconnected (hard_keep is boolean); auxiliary path connected",
                "L2_soft8": "main path disconnected (hard_keep is boolean); auxiliary path connected",
                "L3_uniform8": "main path disconnected because scores are numerically inert; auxiliary path may remain connected"},
            "scope": "Connectivity of one trained arm at one fixed budget; not an efficacy claim"}


def measure_redundancy(model, root, manifest, fold, ids, batch_size, device, seed, torch, np, slot_count=8):
    """Trained-slot redundancy under one shared fixed noise draw for every arm.

    Reuses the historical definitions so numbers stay comparable with the
    retained-slot audit: pairwise cosine over all candidate pairs, reading
    overlap as mean per-head ``sum_token min(p_i, p_j)``, and centered effective
    rank from the singular spectrum.
    """
    from .patient import _batch, _rng
    from .problem_trace import NativeSlotTrace
    model.eval()
    trace = NativeSlotTrace(model)
    captured = {}
    handles = [getattr(model, "slot_decoder_" + m).register_forward_hook(
        lambda module, inputs, output, key=m: captured.__setitem__(key, output)) for m in ("wsi", "omic")]
    per_patient = {"wsi": [], "omic": []}
    gate_effective = {"wsi": [], "omic": []}
    selected_count = {"wsi": [], "omic": []}
    try:
        with torch.no_grad():
            for index, start in enumerate(range(0, len(ids), batch_size)):
                selected = ids[start:start + batch_size]
                # One fixed entry seed for every arm and batch: the slot draws
                # and the Gumbel noise are then identical across arms, so any
                # difference in the measured slots comes from trained weights.
                _rng(seed * 1000 + index, torch, np)
                trace.reset()
                captured.clear()
                batch = _batch(root, manifest, fold, selected, device, torch)
                model(**batch)
                for modality in ("wsi", "omic"):
                    record = trace.records[modality]
                    slots = record["final"].double().cpu().numpy()
                    attention = record["attentions"][-1].double().cpu().numpy()
                    _, gate, keep = captured[modality]
                    per_patient[modality].append(pair_statistics(slots, attention))
                    # Effective slot count must be PER PATIENT and therefore
                    # lie in [1, num_slots]. An earlier version normalised by the
                    # batch-wide gate sum and reduced over every element, so it
                    # also counted batch size and could exceed the slot count
                    # (L3 reported 61.6 with only 8 slots). Normalise each row
                    # by its own sum and reduce within the row.
                    row_sum = gate.sum(dim=-1, keepdim=True)
                    if bool((row_sum > 0).all()):
                        row_normalized = (gate / row_sum).double().cpu().numpy()
                        per_patient_gate_effective = 1.0 / np.sum(row_normalized ** 2, axis=-1)
                        gate_effective[modality].extend(float(v) for v in per_patient_gate_effective)
                    # ``keep`` is the native 0/1 tensor, so a reduction over it is
                    # integral; torch >=2.10 rejects ``.mean()`` on Long, hence
                    # the explicit float cast before averaging.
                    selected_count[modality].append(float((keep > 0.5).sum(-1).double().mean()))
    finally:
        for handle in handles:
            handle.remove()
        trace.close()

    summary = {}
    for modality in ("wsi", "omic"):
        rows = per_patient[modality]
        summary[modality] = {
            "patients": len(rows),
            "all_pair_cosine_mean": _mean(rows, "all_pair_cosine"),
            "all_pair_cosine_min": _min(rows, "all_pair_cosine"),
            "pairs_cosine_above_0_95_fraction": _mean(rows, "pairs_above_0_95"),
            "all_pair_reading_overlap_mean": _mean(rows, "all_pair_reading_overlap"),
            "centered_energy_fraction_mean": _mean(rows, "centered_energy_fraction"),
            "centered_effective_rank_mean": _mean(rows, "centered_effective_rank"),
            "gate_weight_effective_slots_mean": _float_mean(gate_effective[modality]),
            "mean_hard_selected_slots": _float_mean(selected_count[modality]),
        }
    violations = _range_check(summary, slot_count)
    if violations:
        raise RuntimeError(f"Impossible redundancy summary (bound violations): {violations}")
    return {"schema": "slotspe_trained_redundancy_v1", "seed": seed,
        "definitions": "pairwise cosine over all candidate pairs; per-head reading overlap sum_token min(p_i,p_j); centered effective rank; gate effective slots = 1/sum(w_i^2) with w normalised per patient, so it lies in [1, num_slots]",
        "comparison_basis": "one shared fixed noise draw per batch, identical across arms",
        "slot_count": slot_count,
        "summary": summary,
        "note": "Per-patient arrays stay in ignored local_evidence; only aggregates are reported."}


def pair_statistics(slots, attention):
    """Per-patient slot statistics matching the historical redundancy audit.

    Returns one value PER PATIENT (axis 0 preserved).  Cosine and reading
    overlap are averaged over candidate pairs and heads, so an earlier version
    that reduced over the batch produced a scalar here and broke the
    aggregate/contrast stage downstream.
    """
    import numpy as np
    count = slots.shape[1]
    norms = np.linalg.norm(slots, axis=-1, keepdims=True)
    if not bool((norms > 0).all()):
        raise ValueError("Zero slot norm makes cosine undefined")
    units = slots / norms
    cosine = np.clip(units @ np.swapaxes(units, -1, -2), -1, 1)
    pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
    if not pairs:
        raise ValueError("Redundancy requires at least two candidate slots")
    values = np.asarray([cosine[..., i, j] for i, j in pairs], dtype=float)  # [pairs, batch]
    rows = attention[..., :, :]  # [batch, heads, slots, tokens]
    rows = rows / rows.sum(-1, keepdims=True)
    overlaps = np.asarray([np.minimum(rows[..., i, :], rows[..., j, :]).sum(-1).mean(-1)
                           for i, j in pairs], dtype=float)  # [pairs, batch]
    centered = slots - slots.mean(-2, keepdims=True)
    spectrum = np.linalg.svd(centered, compute_uv=False) ** 2
    total = spectrum.sum(-1)
    safe = np.where(total[..., None] > 0, total[..., None], 1.0)
    probabilities = np.where(total[..., None] > 0, spectrum / safe, 0.0)
    effective_rank = np.exp(-(np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0)).sum(-1))
    energy = (centered ** 2).sum((-1, -2)) / (slots ** 2).sum((-1, -2))
    return {"all_pair_cosine": values.mean(axis=0).tolist(),          # [batch]
            "all_pair_cosine_min": values.min(axis=0).tolist(),       # [batch]
            "pairs_above_0_95": (values > 0.95).mean(axis=0).tolist(),
            "all_pair_reading_overlap": overlaps.mean(axis=0).tolist(),
            "centered_energy_fraction": energy.tolist(),             # [batch]
            "centered_effective_rank": effective_rank.tolist()}       # [batch]


def _mean(rows, key):
    import numpy as np
    values = [value for row in rows for value in row[key]]
    return float(np.mean(values)) if values else None


def _min(rows, key):
    import numpy as np
    values = [value for row in rows for value in row[key]]
    return float(np.min(values)) if values else None


def _float_mean(values):
    import numpy as np
    return float(np.mean(values)) if values else None


def _range_check(summary, slot_count, rtol=1e-5):
    """Effective slot count is bounded by the slot count; flag violations.

    ``rtol`` absorbs float32 accumulation over many patients.  A uniform gate
    legitimately equals exactly ``slot_count``, so a bare ``<=`` would reject
    the correct L1/L2 value.
    """
    problems = {}
    for modality, rows in summary.items():
        for key, ceiling in (("gate_weight_effective_slots_mean", slot_count),
                             ("mean_hard_selected_slots", slot_count)):
            value = rows.get(key)
            if value is not None and not (0.0 < value <= ceiling * (1.0 + rtol)):
                problems[f"{modality}.{key}"] = {"value": value, "must_not_exceed": ceiling}
    return problems


def build_contrasts(records):
    """Paired contrasts between consecutive gate levels, reported as differences."""
    import numpy as np
    by_arm = {record["arm"]: record for record in records}
    order = [arm for arm in GATE_ARMS if arm in by_arm]
    contrasts = []
    for before, after in zip(order, order[1:]):
        first, second = by_arm[before], by_arm[after]
        row = {"comparison": f"{after}_minus_{before}",
            "hard_selected_slots": [first["selection_attributes"]["wsi"]["applied"]["k"],
                                    second["selection_attributes"]["wsi"]["applied"]["k"]],
            "learned_scores_still_influence_weights": [
                before != "L3_uniform8", after != "L3_uniform8"],
            "question": {"L2_soft8_minus_L1_top2": "does removing hard sparsity alone change learned slots?",
                         "L3_uniform8_minus_L2_soft8": "does removing the learned gate weights alone change learned slots?"}.get(
                             f"{after}_minus_{before}", "consecutive gate level contrast")}
        for modality in ("wsi", "omic"):
            for metric, key in (("all_pair_cosine_mean", "all_pair_cosine"),
                                ("all_pair_reading_overlap_mean", "all_pair_reading_overlap"),
                                ("centered_effective_rank_mean", "centered_effective_rank"),
                                ("centered_energy_fraction_mean", "centered_energy_fraction"),
                                ("gate_weight_effective_slots_mean", "gate_weight_effective_slots")):
                a = first["redundancy"].get(modality, {}).get(metric)
                b = second["redundancy"].get(modality, {}).get(metric)
                if a is not None and b is not None:
                    row.setdefault("redundancy_difference", {})[f"{modality}_{metric}"] = b - a
        a = first.get("val_cindex_mean_native_eval")
        b = second.get("val_cindex_mean_native_eval")
        if a is not None and b is not None:
            row["val_cindex_difference"] = b - a
        row["comparison_against_known_dispersion"] = (
            "The recorded cross-fold and cross-seed C-index dispersion is larger than this "
            "difference; a single-fold C-index contrast is not evidence of a performance change.")
        contrasts.append(row)
    return contrasts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, help="Prepared batch root containing prepared/manifest.json")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run", action="store_true", help="Explicitly execute paired gate training")
    args = parser.parse_args(argv)
    config = _read(args.config)
    config.setdefault("model_args", {})
    if not args.run:
        print(json.dumps({"action": "plan_only", "training_steps": 0, "gate_arms": list(GATE_ARMS),
            "arms": gate_plan(config)["arms"],
            "requires_authorization": "Only --run trains; plan does not read patient tensors."},
            ensure_ascii=False, indent=2))
        return 0
    if args.input_root is None or args.output_dir is None:
        parser.error("--run requires --input-root and a fresh --output-dir inside local_evidence")
    source = args.source_root or (Path(__file__).resolve().parents[1] / "third_party/SlotSPE")
    result = run(config, source_root=source, input_root=args.input_root,
                 output_dir=args.output_dir, device_override=args.device)
    print(json.dumps({"status": result["status"],
        "arms": {r["arm"]: {"redundancy_wsi_cos": r["redundancy"]["wsi"]["all_pair_cosine_mean"],
                            "redundancy_omic_cos": r["redundancy"]["omic"]["all_pair_cosine_mean"],
                            "val_cindex": r["val_cindex_mean_native_eval"]} for r in result["arms"]},
        "contrasts": result["contrasts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())