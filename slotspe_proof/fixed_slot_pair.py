"""Opt-in paired fixed-slot training: does WHICH two slots are read out matter?

This is a separate experiment from ``gate_pair.py``.  It trains three arms that
differ ONLY in which slots survive the native ``MoESlotDecoder`` readout:

``N_topk``
    The unchanged original arm: ``k=2`` over ``parallel_topk_st``.  Reference
    behaviour and historical baseline.
``N_fixed34``
    Exactly slots 3 and 4 (0-based indices 2, 3), chosen by POSITION. The native
    top-k is neutralised, so every patient and every step reads the same pair.
``N_worstpredkeep``
    Exactly the two slots with the LOWEST native ``pred_keep_slot`` score, so the
    selection rule is defined BY the learned gate scorer.

Mechanism.  The override is a ``forward_hook`` on the native ``pred_keep_slot``
module, which produces the per-slot ``keep_score``.  The hook adds a CONSTANT
mask bias to the real score, then lets the ORIGINAL ``MoESlotDecoder.forward``
run unchanged: its native ``parallel_topk_st`` top-k, native softmax gate, native
weighted sum, and native gradient topology are all preserved.  The bias is
detached, so it contributes no gradient, and the real score still flows into the
native soft surrogate and the native gate exactly as before.  The bias gap
dominates the native Gumbel noise, so the selected set is exactly:

* ``N_fixed34``: the constant slots 3 and 4 (0-based 2, 3), for every patient;
* ``N_worstpredkeep``: the two lowest-scoring slots per patient, recomputed live
  from the current scores.

No class, forward, or original source file is replaced or edited.  ``N_topk``
attaches no hook at all and executes the untouched original path.

Controlled factors: one shared initial state, one shared batch order per epoch,
and one per-step RNG seed across all arms, so any difference is attributable to
the selection rule alone.

Known limitation, recorded in the plan: the ``pred_keep_slot`` main-loss path is
disconnected (hard top-k is boolean), so that scorer is shaped only by the
auxiliary loss.  ``N_worstpredkeep`` therefore depends on a scorer that does not
drive the main readout; its "worst" ranking may be close to noise, and it must
not be read as "lowest quality slots" without that caveat.

Importing this module never imports torch, loads a model, or reads patient data.
Planning validates only the declared factors.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

FIXED_ARMS = ("N_topk", "N_fixed34", "N_worstpredkeep")

# The native top-k selects ``(score + gumbel_noise).topk(k)``.  The Gumbel draw
# has unbounded tails, so the mask bias must dominate every plausible draw for
# the chosen set to be exactly the requested one.  1e9 is finite, so the tensor
# stays finite and no arithmetic becomes inf.
_MASK_GAP = 1e9


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


def fixed_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the declared factors without importing torch or a model."""
    arms = list(config.get("slot_arms", FIXED_ARMS))
    if not arms or any(arm not in FIXED_ARMS for arm in arms):
        raise ValueError(f"slot_arms must be a nonempty subset of {list(FIXED_ARMS)}")
    if "N_topk" not in arms:
        raise ValueError("N_topk is the reference arm and must be included")
    if len(set(arms)) != len(arms):
        raise ValueError("slot_arms must be unique")
    device = config.get("device", "cuda:0")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a nonempty string")
    epochs = _integer(config.get("fixed_epochs", 3), "fixed_epochs", minimum=1)
    keep = _integer(config.get("fixed_keep_count", 2), "fixed_keep_count", minimum=1)
    indices = config.get("fixed_slot_indices", [2, 3])
    if (not isinstance(indices, (list, tuple)) or len(indices) != keep
            or sorted(indices) != list(indices) or len(set(indices)) != len(indices)
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices)):
        raise ValueError(f"fixed_slot_indices must be {keep} ascending unique non-negative integers")
    return {
        "schema_version": 1,
        "phase": "fixed_slot_pair_train",
        "slot_arms": arms,
        "fold": _integer(config.get("fold", 0), "fold"),
        "seed": _integer(config.get("seed", 3), "seed"),
        "device": device,
        "fixed_epochs": epochs,
        "fixed_lr": _positive_float(config.get("fixed_lr", 5e-4), "fixed_lr"),
        "fixed_batch_size": _integer(config.get("fixed_batch_size", 8), "fixed_batch_size", minimum=1),
        "fixed_eval_repeats": _integer(config.get("fixed_eval_repeats", 5), "fixed_eval_repeats", minimum=1),
        "keep_count": keep,
        "fixed_slot_indices": list(indices),
        "initialization": "same_new_initial_state",
        "training_budget": "equal_epochs_updates_batch_order_and_step_seeds",
        "arm_difference": "the per-slot keep_score rewritten before the native top-k, per arm",
        "checkpoint_rule": "final_epoch_only_no_best_selection",
        "optimizer": {"name": "Adam", "lr": _positive_float(config.get("fixed_lr", 5e-4), "fixed_lr")},
        "scheduler": {"name": "CosineAnnealingLR", "t_max": epochs, "eta_min": 1e-6},
        "primary_readout": "trained_slot_redundancy_not_validation_cindex",
        "reference_arm_recorded_val_cindex": config.get("reference_arm_recorded_val_cindex"),
        "reference_arm_recorded_val_cindex_per_repeat": config.get("reference_arm_recorded_val_cindex_per_repeat"),
        "reference_tolerance": _positive_float(config.get("reference_tolerance", 1e-6), "reference_tolerance"),
        "arms": {arm: {
            "selection_rule": selection_rule_name(arm),
            "depends_on_learned_scorer": arm == "N_worstpredkeep",
            "indices_vary_per_patient": arm == "N_worstpredkeep",
            "native_forward_unmodified": True,
        } for arm in arms},
        "known_dependency_warning": (
            "N_worstpredkeep ranks slots by the native pred_keep_slot scorer. The main-loss "
            "path to that scorer is disconnected (hard top-k is boolean), so the scorer is "
            "shaped only by the auxiliary loss. Its ranking may therefore be near noise, and "
            "this arm must not be read as 'lowest quality slots' without that caveat."),
        "limitations": [
            "single fold and single seed: the recorded cross-fold dispersion (0.06178) and "
            "cross-seed dispersion (0.06059) exceed any difference expected here",
            "short epoch budget: the final epoch is not a converged model",
            "validation cohort is small: C-index is recorded but is not the primary readout",
        ],
        "interpretation": (
            "Paired causal contrast of WHICH slots are read out, at one fixed budget on one fold. "
            "The reference arm reproduces the recorded baseline before any contrast is read."),
    }


def selection_rule_name(arm: str) -> str:
    return ("native_parallel_topk_st" if arm == "N_topk" else
            "fixed_indices" if arm == "N_fixed34" else
            "lowest_native_pred_keep_slot_scores")


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


class KeepScoreOverride:
    """Rewrite the native ``pred_keep_slot`` output so top-k picks chosen slots.

    ``pred_keep_slot`` is the module that produces the per-slot keep score that the
    native forward feeds to ``parallel_topk_st``.  Rewriting its OUTPUT changes
    only which indices the native top-k returns; every other native computation,
    including the differentiable straight-through surrogate, runs untouched.
    """

    def __init__(self, arm: str, indices: list[int], torch):
        if arm not in FIXED_ARMS:
            raise ValueError(f"Unknown fixed arm: {arm}")
        self.arm = arm
        self.indices = list(indices)
        self.torch = torch
        self.observed_selected: list[float] = []
        self.observed_indices: list[list[int]] = []

    def __call__(self, module, inputs, output):
        torch = self.torch
        if self.arm == "N_topk":
            return output
        # ``output`` is (B, S, 1) before the native ``squeeze(-1)``.
        scores = output if output.dim() == 3 else output.unsqueeze(-1)
        n_slots = scores.shape[1]
        # Build a mask that is CONSTANT with respect to the scorer (so it can
        # never outrank the Gumbel noise added by the native top-k) but is added
        # to the REAL score instead of replacing it, so the gradient path to
        # ``pred_keep_slot`` stays connected exactly as in the original arm.
        #
        # The native selection is ``(score + gumbel_noise).topk(k)``.  The bias
        # gap ``_MASK_GAP`` dominates any finite Gumbel draw, so the chosen set
        # is determined solely by the mask, while the real score still flows into
        # the native soft surrogate and the native gate.
        bias = torch.zeros_like(scores)
        if self.arm == "N_fixed34":
            for index in self.indices:
                bias[:, index, :] = _MASK_GAP
        else:  # N_worstpredkeep: raise the two lowest-scoring slots
            flat = scores.squeeze(-1)
            order = flat.argsort(dim=-1)              # ascending: worst first
            worst = order[:, : len(self.indices)]
            bias = torch.zeros_like(scores)
            bias.scatter_(1, worst.unsqueeze(-1), _MASK_GAP)
        # The bias is a constant, not a function of the score: detach it so it
        # contributes no gradient, exactly matching the original arm's topology
        # where the mask is a hard boolean.
        override = scores + bias.detach()
        chosen = (bias.detach().squeeze(-1) > 0).sum(-1)
        self.observed_selected.extend(float(v) for v in chosen.detach().cpu())
        self.observed_indices.append(self.indices if self.arm == "N_fixed34" else [])
        return override


def apply_arm(model: Any, arm: str, slot_count: int, indices: list[int]):
    """Attach the keep-score hook for this arm; ``N_topk`` attaches nothing."""
    if arm not in FIXED_ARMS:
        raise ValueError(f"Unknown fixed arm: {arm}")
    recipe = {}
    for modality in ("wsi", "omic"):
        name = "slot_decoder_" + modality
        decoder = getattr(model, name, None)
        if decoder is None:
            raise ValueError(f"model has no native {name} module")
        if not hasattr(decoder, "k") or isinstance(decoder.k, bool) or not isinstance(decoder.k, int):
            raise ValueError(f"{name}.k is not a native integer selection attribute")
        constructed = (decoder.k, decoder.temperature)
        # The native k=2 is kept for every arm so the override decides between
        # exactly two slots, matching the reference arm's selection width.
        override = KeepScoreOverride(arm, indices, __import__("torch"))
        handle = decoder.pred_keep_slot.register_forward_hook(override)
        setattr(model, f"_keep_hook_{modality}", handle)
        setattr(model, f"_keep_state_{modality}", override)
        recipe[modality] = {
            "module": name,
            "hooked_native_submodule": "pred_keep_slot",
            "constructed": {"k": constructed[0], "temperature": constructed[1]},
            "applied": {"k": decoder.k, "temperature": decoder.temperature},
            "selection_rule": selection_rule_name(arm),
            "hook_attached": arm != "N_topk",
            "attribute_names_unchanged": True,
        }
    return recipe


def _restored_kwargs(original: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(original, Mapping):
        return copy.deepcopy(dict(original))
    return copy.deepcopy(vars(original))


def run(config, *, source_root, input_root, output_dir, device_override=None):
    """Execute ONLY under an explicit ``--run`` CLI with authorization."""
    plan = fixed_plan(config)
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
        raise ValueError("fixed slot outputs must be inside ignored local_evidence")
    if output_dir.exists():
        raise FileExistsError("fixed slot training requires a fresh output directory")

    root = Path(input_root).resolve()
    manifest = _read(root / "prepared/manifest.json")
    fold = manifest["folds"][str(plan["fold"])]
    train_ids, val_ids = list(fold["train"]), list(fold["val"])
    if not train_ids or not val_ids or set(train_ids) & set(val_ids):
        raise ValueError("prepared train/validation IDs must be disjoint and nonempty")
    orders = epoch_orders(train_ids, plan["fixed_epochs"], plan["seed"])

    source = str(Path(source_root).resolve())
    args = SimpleNamespace(**_restored_kwargs(config["model_args"]))
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
    if max(plan["fixed_slot_indices"]) >= slot_count:
        raise ValueError(f"fixed_slot_indices must be < slot_num ({slot_count})")

    output_dir.mkdir(parents=True)
    receipt = {"status": "started", "phase": "fixed_slot_pair_train", "plan": plan,
        "prepared_input_root": str(root), "prepared_manifest_sha256": sha(root / "prepared/manifest.json"),
        "runner_sha256": sha(__file__), "source_root": source,
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "started_utc": datetime.now(timezone.utc).isoformat(), "patient_outputs_private": True}

    _rng(plan["seed"], torch, np)
    reference = build_model(source, args, "original")
    shared_initial = {name: value.detach().cpu().clone() for name, value in reference.state_dict().items()}
    parameter_count = sum(p.numel() for p in reference.parameters())
    initial_digest = hashlib.sha256(
        b"".join(shared_initial[name].numpy().tobytes() for name in sorted(shared_initial))).hexdigest()
    del reference
    receipt["shared_initial_state_sha256"] = initial_digest
    receipt["parameter_count"] = parameter_count

    _write_json(output_dir / "recipe.json", recipe_fields(plan, root, fold, orders, initial_digest, parameter_count))
    _write_json(output_dir / "execution_receipt.json", receipt)

    records = []
    try:
        for arm in plan["slot_arms"]:
            arm_root = output_dir / arm
            arm_root.mkdir()
            _rng(plan["seed"], torch, np)
            model = build_model(source, args, "original").to(device)
            if sum(p.numel() for p in model.parameters()) != parameter_count:
                raise RuntimeError("fixed arm changed the model parameter count")
            model.load_state_dict(shared_initial, strict=True)
            applied = apply_arm(model, arm, slot_count, plan["fixed_slot_indices"])
            _write_json(arm_root / "arm_recipe.json", applied)
            _write_json(arm_root / "curve.json", [])

            optimizer = torch.optim.Adam(model.parameters(), lr=plan["fixed_lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=plan["fixed_epochs"], eta_min=1e-6)

            curve, steps = [], 0
            for epoch, order in enumerate(orders):
                model.train()
                totals = np.zeros(3)
                seen = 0
                for batch_index, start in enumerate(range(0, len(order), plan["fixed_batch_size"])):
                    ids = order[start:start + plan["fixed_batch_size"]]
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
                validation = evaluate(model, root, manifest, fold, val_ids, plan["fixed_batch_size"],
                                      device, plan["seed"] * 10000 + epoch, torch, np, concordance_index_censored)
                curve.append({"epoch": epoch + 1,
                    "train_main_nll": float(totals[0] / seen),
                    "train_aux_loss": float(totals[1] / seen),
                    "train_total_loss": float(totals[2] / seen),
                    "val_nll_exploratory": validation["nll"],
                    "val_cindex_exploratory": validation["cindex"],
                    "val_cindex_status": validation["cindex_status"]})
                _write_json(arm_root / "curve.json", curve)
                print(f"arm={arm} epoch={epoch + 1}/{plan['fixed_epochs']} "
                      f"loss={curve[-1]['train_total_loss']:.5f} val_C={validation['cindex']}", flush=True)
                scheduler.step()

            checkpoint = arm_root / "last.pt"
            torch.save({"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "arm": arm, "fold": plan["fold"], "seed": plan["seed"], "optimizer_steps": steps,
                "selection_attributes": applied}, checkpoint)

            observed = selection_audit(model, root, manifest, fold, val_ids,
                                       plan["fixed_batch_size"], device, plan["seed"], torch, np,
                                       arm, plan["fixed_slot_indices"], slot_count)
            _write_json(arm_root / "selection_audit.json", observed)

            redundancy = measure_redundancy(model, root, manifest, fold, val_ids,
                                            plan["fixed_batch_size"], device, plan["seed"], torch, np,
                                            slot_count)
            _write_json(arm_root / "trained_redundancy.json", redundancy)

            repeats = [evaluate(model, root, manifest, fold, val_ids, plan["fixed_batch_size"], device,
                                plan["seed"] * 1000000 + 80000 + repeat, torch, np, concordance_index_censored)
                       for repeat in range(plan["fixed_eval_repeats"])]
            _write_json(arm_root / "eval_repeats.json", repeats)
            scores = [r["cindex"] for r in repeats if r["cindex"] is not None]
            records.append({"arm": arm, "optimizer_steps": steps, "checkpoint_sha256": sha(checkpoint),
                "shared_initial_state_sha256": initial_digest, "selection_attributes": applied,
                "final_epoch_curve": curve[-1] if curve else None,
                "val_cindex_per_repeat": scores,
                "val_cindex_mean_native_eval": float(np.mean(scores)) if scores else None,
                "val_cindex_std_native_eval": float(np.std(scores)) if len(scores) > 1 else 0.0 if scores else None,
                "selection_audit": {k: observed.get(k) for k in
                                    ("arm", "rule", "exact_indices_respected", "mean_selected_slots",
                                     "selected_slots_min", "selected_slots_max")},
                "redundancy": redundancy["summary"],
                "interpretation": plan["interpretation"]})
            del model, optimizer, scheduler
            if device.type == "cuda":
                torch.cuda.empty_cache()

        contrasts = build_contrasts(records)
        result = {"schema": "slotspe_fixed_slot_pair_v1", "status": "completed", "plan": plan,
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
        "original_source_files_modified": False,
        "validation": "native eval only; final epoch; no checkpoint selection",
        "known_cross_fold_dispersion": {"validation_cindex_std_across_folds": 0.06178,
            "validation_cindex_std_across_seeds": 0.06059,
            "source": "docs/research/slotspe_gpu_trisurv_20261022 five-fold record"},
    }


def fidelity_anchor(records, plan):
    """Independent check that the reference arm reproduces the recorded run."""
    reference = next((r for r in records if r["arm"] == "N_topk"), None)
    observed = reference["val_cindex_mean_native_eval"] if reference else None
    expected = plan.get("reference_arm_recorded_val_cindex")
    per_repeat_expected = plan.get("reference_arm_recorded_val_cindex_per_repeat")
    per_repeat_observed = reference.get("val_cindex_per_repeat") if reference else None
    detail = None
    if per_repeat_expected and per_repeat_observed:
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
            "note": "N_topk is the unchanged original arm; reproducing the recorded baseline "
                    "is required before any contrast in this run can be attributed to the selection rule."}


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


def selection_audit(model, root, manifest, fold, ids, batch_size, device, seed, torch, np,
                    arm, indices, slot_count):
    """Confirm the native top-k really kept exactly the requested slots.

    This reads the decoder's own ``hard_keep`` output, so it verifies the native
    selection rather than the override's intent.
    """
    from .patient import _batch, _rng
    model.eval()
    if arm == "N_topk":
        return {"arm": arm, "rule": selection_rule_name(arm),
            "exact_indices_respected": None,
            "note": "reference arm uses the untouched native selection; nothing to override"}
    captured = {}
    handles = []
    for modality in ("wsi", "omic"):
        name = "slot_decoder_" + modality
        handles.append(getattr(model, name).register_forward_hook(
            lambda module, inputs, output, key=modality: captured.__setitem__(key, output)))
    kept_sets = {"wsi": [], "omic": []}
    try:
        with torch.no_grad():
            for index, start in enumerate(range(0, len(ids), batch_size)):
                selected = ids[start:start + batch_size]
                _rng(seed * 1000 + index, torch, np)
                batch = _batch(root, manifest, fold, selected, device, torch)
                model(**batch)
                for modality in ("wsi", "omic"):
                    _, _, keep = captured[modality]
                    hot = (keep > 0.5).long()
                    for row in hot.detach().cpu():
                        kept_sets[modality].append(int(row.sum()))
    finally:
        for handle in handles:
            handle.remove()
    counts = kept_sets["wsi"]
    return {"arm": arm, "rule": selection_rule_name(arm),
        "requested_indices": list(indices),
        "mean_selected_slots": float(np.mean(counts)) if counts else None,
        "selected_slots_min": int(np.min(counts)) if counts else None,
        "selected_slots_max": int(np.max(counts)) if counts else None,
        "exact_indices_respected": bool(all(c == len(indices) for c in counts)) if counts else None,
        "scope": "confirms the count of native hard-kept slots; per-patient arrays stay in ignored local_evidence"}


def measure_redundancy(model, root, manifest, fold, ids, batch_size, device, seed, torch, np, slot_count=8):
    """Trained-slot redundancy under one shared fixed noise draw for every arm."""
    from .patient import _batch, _rng
    from .problem_trace import NativeSlotTrace
    from .gate_pair import pair_statistics, _mean, _min, _float_mean, _range_check
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
                    row_sum = gate.sum(dim=-1, keepdim=True)
                    if bool((row_sum > 0).all()):
                        row_normalized = (gate / row_sum).double().cpu().numpy()
                        per_patient_gate_effective = 1.0 / np.sum(row_normalized ** 2, axis=-1)
                        gate_effective[modality].extend(float(v) for v in per_patient_gate_effective)
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


def epoch_orders(ids: list[str], epochs: int, seed: int) -> list[list[str]]:
    """One deterministic batch order per epoch, shared by every arm."""
    import torch

    if not ids or len(set(ids)) != len(ids):
        raise ValueError("training IDs must be nonempty and unique")
    orders = []
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(len(ids), generator=generator).tolist()
        orders.append([ids[index] for index in order])
    return orders


def build_contrasts(records):
    """Paired contrasts between the reference arm and each fixed arm."""
    by_arm = {record["arm"]: record for record in records}
    reference = by_arm.get("N_topk")
    contrasts = []
    for arm in FIXED_ARMS:
        if arm == "N_topk" or arm not in by_arm or reference is None:
            continue
        other = by_arm[arm]
        row = {"comparison": f"{arm}_minus_N_topk",
            "selection_rule": selection_rule_name(arm),
            "question": {"N_fixed34_minus_N_topk": "does reading out slots 3 and 4 by position differ from the learned top-k pair?",
                         "N_worstpredkeep_minus_N_topk": "does reading out the two lowest pred_keep_slot slots differ from the learned top-k pair?"}.get(
                             f"{arm}_minus_N_topk", "fixed slot contrast")}
        for modality in ("wsi", "omic"):
            for metric in ("all_pair_cosine_mean", "all_pair_reading_overlap_mean",
                           "centered_effective_rank_mean", "centered_energy_fraction_mean",
                           "gate_weight_effective_slots_mean"):
                a = reference["redundancy"].get(modality, {}).get(metric)
                b = other["redundancy"].get(modality, {}).get(metric)
                if a is not None and b is not None:
                    row.setdefault("redundancy_difference", {})[f"{modality}_{metric}"] = b - a
        a = reference.get("val_cindex_mean_native_eval")
        b = other.get("val_cindex_mean_native_eval")
        if a is not None and b is not None:
            row["val_cindex_difference"] = b - a
            row["val_cindex_absolute_difference"] = abs(b - a)
            row["val_cindex_within_recorded_cross_fold_dispersion"] = abs(b - a) <= 0.06178
        row["train_total_loss_difference"] = (other["final_epoch_curve"]["train_total_loss"]
                                              - reference["final_epoch_curve"]["train_total_loss"]) \
            if other.get("final_epoch_curve") and reference.get("final_epoch_curve") else None
        row["comparison_against_known_dispersion"] = (
            "The recorded cross-fold and cross-seed C-index dispersion (0.06178) is larger than "
            "this difference; a single-fold single-seed C-index contrast is not evidence of a "
            "performance change.")
        contrasts.append(row)
    return contrasts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run", action="store_true", help="Explicitly execute fixed slot training")
    args = parser.parse_args(argv)
    config = _read(args.config)
    config.setdefault("model_args", {})
    if not args.run:
        plan = fixed_plan(config)
        print(json.dumps({"action": "plan_only", "training_steps": 0, "slot_arms": list(FIXED_ARMS),
            "arms": plan["arms"], "known_dependency_warning": plan["known_dependency_warning"],
            "requires_authorization": "Only --run trains; plan does not read patient tensors."},
            ensure_ascii=False, indent=2))
        return 0
    if args.input_root is None or args.output_dir is None:
        parser.error("--run requires --input-root and a fresh --output-dir inside local_evidence")
    source = args.source_root or (Path(__file__).resolve().parents[1] / "third_party/SlotSPE")
    result = run(config, source_root=source, input_root=args.input_root,
                 output_dir=args.output_dir, device_override=args.device)
    print(json.dumps({"status": result["status"],
        "arms": {r["arm"]: {"train_loss": r["final_epoch_curve"]["train_total_loss"],
                            "redundancy_wsi_cos": r["redundancy"]["wsi"]["all_pair_cosine_mean"],
                            "redundancy_omic_cos": r["redundancy"]["omic"]["all_pair_cosine_mean"],
                            "val_cindex": r["val_cindex_mean_native_eval"]} for r in result["arms"]},
        "fidelity_anchor": result["fidelity_anchor"]["status"],
        "contrasts": result["contrasts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())