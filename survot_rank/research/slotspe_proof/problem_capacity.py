"""Opt-in matched capacity training; importing this module never imports torch.

This is a deliberately separate experiment from frozen-checkpoint diagnostics.
The caller must provide an explicitly authorized, private execution destination.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Mapping


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def capacity_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the factors without importing a model or touching patient data."""
    counts = config.get("capacity_counts", [4, 8, 16])
    if not isinstance(counts, (list, tuple)) or len(counts) < 2:
        raise ValueError("capacity_counts must contain at least two slot counts")
    counts = [_integer(k, "capacity count", minimum=2) for k in counts]
    if len(set(counts)) != len(counts):
        raise ValueError("capacity_counts must be unique")
    if config.get("capacity_modality", "wsi") != "wsi":
        raise ValueError("this protocol varies WSI capacity only")
    lr = config.get("capacity_lr", 0.0005)
    if isinstance(lr, bool) or not isinstance(lr, (int, float)):
        raise ValueError("capacity_lr must be a finite positive number")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("capacity_lr must be a finite positive number")
    device = config.get("device", "cuda:0")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a nonempty string")
    return {
        "schema_version": 1,
        "phase": "capacity_train",
        "seed": _integer(config.get("seed", 3), "seed"),
        "fold": _integer(config.get("fold", 0), "fold"),
        "device": device,
        "capacity_counts": counts,
        "capacity_modality": "wsi",
        "capacity_epochs": _integer(config.get("capacity_epochs", 3), "capacity_epochs", minimum=1),
        "capacity_lr": float(lr),
        "capacity_batch_size": _integer(config.get("capacity_batch_size", 8), "capacity_batch_size", minimum=1),
        "initialization": "same_new_initial_state",
        "training_budget": "equal_epochs_updates_and_token_budget",
        "selection": {"wsi_k": 2, "omic_k": 2, "omic_num_slots": 8},
        "checkpoint_rule": "final_epoch_only_no_best_selection",
        "optimizer": {"name": "Adam", "lr": float(lr)},
        "scheduler": {"name": "none"},
        "factor_change": "native WSI slot count; native decoder numeric k fixed at 2 in both modalities",
        "evidence_scope": "matched_capacity_smoke_training; no automatic problem or efficacy inference",
        "limitations": [
            "short budget does not establish training adequacy",
            "equal update counts do not imply equal compute costs",
            "patient-adaptive capacity requires an independent complexity-by-capacity analysis",
            "equal step seeds do not imply identical internal random draws at different slot counts",
        ],
    }


def validate_matched_state_contract(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    """Reject architecture/initial-state mismatches before any patient forward.

    Contracts include sorted state keys, per-key shape/dtype/value digest and the
    total number of trainable parameter elements. This helper is tensor-free so
    the fail-closed rule can be tested without model or patient execution.
    """
    if reference.get("parameter_numel") != candidate.get("parameter_numel"):
        raise ValueError("capacity arms have unequal trainable parameter counts")
    ref_state = reference.get("state")
    cand_state = candidate.get("state")
    if not isinstance(ref_state, Mapping) or not isinstance(cand_state, Mapping):
        raise ValueError("state contracts must contain a state mapping")
    if not ref_state or set(ref_state) != set(cand_state):
        raise ValueError("capacity arms have incompatible state keys")
    for name in sorted(ref_state):
        for field in ("shape", "dtype", "sha256"):
            if field not in ref_state[name] or field not in cand_state[name]:
                raise ValueError(f"incomplete state contract: {name}.{field}")
            if ref_state[name][field] != cand_state[name][field]:
                raise ValueError(f"capacity arms differ in initial state: {name}.{field}")


def _state_contract(model: Any, torch: Any) -> dict[str, Any]:
    state = {}
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        state[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return {
        "parameter_numel": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state": state,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_destination(output_root: Any) -> Path:
    path = Path(output_root).resolve()
    if not any(part.lower() == "local_evidence" for part in path.parts):
        raise ValueError("capacity outputs must be inside ignored local_evidence")
    if path.exists():
        raise FileExistsError("capacity training requires a fresh private output directory")
    return path


def _epoch_orders(ids: list[str], epochs: int, seed: int) -> list[list[str]]:
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("training IDs must be nonempty and unique")
    orders = []
    for epoch in range(epochs):
        order = list(ids)
        random.Random(seed + epoch).shuffle(order)
        orders.append(order)
    return orders


def _step_seed(seed: int, phase: str, epoch: int, batch_index: int) -> int:
    value = f"slotspe-capacity-v1:{seed}:{phase}:{epoch}:{batch_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def _args_dict(args: Any) -> dict[str, Any]:
    if isinstance(args, Mapping):
        return copy.deepcopy(dict(args))
    return copy.deepcopy(vars(args))


def _decoder_recipe(model: Any, *, k: int = 2) -> dict[str, Any]:
    """Set only native numeric k attributes; never replace a class/forward."""
    result = {}
    for modality, names in (
        ("wsi", ("slot_decoder_wsi",)),
        ("omic", ("slot_decoder_omic", "slot_decoder_omics")),
    ):
        matches = [name for name in names if hasattr(model, name)]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one native {modality} slot decoder")
        name = matches[0]
        decoder = getattr(model, name)
        if not hasattr(decoder, "k") or isinstance(decoder.k, bool) or not isinstance(decoder.k, int):
            raise ValueError(f"{name}.k is not a native integer selection attribute")
        previous = decoder.k
        decoder.k = k
        result[modality] = {"module": name, "attribute": "k", "constructed_value": previous, "actual_value": k}
    return result


def _native_losses(model: Any, batch: Mapping[str, Any], size: int, torch: Any):
    logits, auxiliary = model(**batch)
    main = model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None) / size
    if not torch.is_tensor(auxiliary) or auxiliary.ndim != 0:
        raise ValueError("native train auxiliary loss must be a scalar tensor")
    total = main + auxiliary
    for name, value in (("main", main), ("auxiliary", auxiliary), ("total", total)):
        if value.ndim != 0 or not bool(torch.isfinite(value.detach()).item()):
            raise ValueError(f"nonfinite or nonscalar native {name} loss")
    return main, auxiliary, total


def run_capacity(config, *, source_root, input_root, manifest, model_args, output_root):
    """Execute only when an outer CLI has authorized --run --phase capacity_train.

    The function never loads a previously trained checkpoint for a capacity arm.
    Native auxiliary loss and the original main-loss batch scaling are retained.
    """
    plan = capacity_plan(config)
    destination = _private_destination(output_root)
    fold = manifest.get("folds", {}).get(str(plan["fold"]))
    if not isinstance(fold, Mapping):
        raise ValueError("prepared manifest does not contain the requested fold")
    for field in ("train", "val", "labels", "binning"):
        if field not in fold:
            raise ValueError(f"prepared fold is missing {field}")
    train_ids, val_ids = list(fold["train"]), list(fold["val"])
    if not val_ids or len(set(val_ids)) != len(val_ids) or set(train_ids) & set(val_ids):
        raise ValueError("prepared train/validation IDs must be unique, nonempty and disjoint")
    orders = _epoch_orders(train_ids, plan["capacity_epochs"], plan["seed"])
    base_args = _args_dict(model_args)
    if int(base_args.get("slot_num_omics", 8)) != 8:
        raise ValueError("this capacity protocol requires the unchanged 8-slot omic arm")
    # Deliberately lazy: plan validation and import never load torch or a model.
    import numpy as np
    import torch

    from .model import build_model
    from .patient import _batch, _rng

    device = torch.device(plan["device"])
    prepared_root = Path(input_root).resolve()
    source = str(Path(source_root).resolve())
    source_args = dict(base_args, slot_num_wsi=8, slot_num_omics=8, topk_ratio=0.25)
    _rng(plan["seed"], torch, np)
    reference = build_model(source, SimpleNamespace(**source_args), "original")
    _decoder_recipe(reference)
    shared_initial = {name: value.detach().cpu().clone() for name, value in reference.state_dict().items()}
    reference_contract = _state_contract(reference, torch)
    del reference

    # Reject incompatible capacities before any patient forward or training.
    arms = []
    for count in plan["capacity_counts"]:
        args = dict(base_args, slot_num_wsi=count, slot_num_omics=8, topk_ratio=2.0 / count)
        _rng(plan["seed"], torch, np)
        model = build_model(source, SimpleNamespace(**args), "original")
        decoder = _decoder_recipe(model)
        model.load_state_dict(shared_initial, strict=True)
        contract = _state_contract(model, torch)
        validate_matched_state_contract(reference_contract, contract)
        arms.append((count, args, decoder, model))

    destination.mkdir(parents=True, exist_ok=False)
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    recipe = {
        "plan": plan,
        "source_root": source,
        "prepared_input_root": str(prepared_root),
        "prepared_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "fold_binning": fold["binning"],
        "fold_binning_source": "unchanged prepared manifest; no bin fitting in this phase",
        "initial_state_contract": reference_contract,
        "epoch_sample_orders": orders,
        "loss": "native loss_fn(main logits)/batch_size + native train auxiliary scalar",
        "step_seed_rule": "SHA256(slotspe-capacity-v1:seed:phase:epoch:batch_index), first 4 bytes",
        "original_forward_and_classes_unchanged": True,
        "selection_numeric_attribute_override_declared": True,
        "validation": "native eval main loss only; no best checkpoint selection or C-index inference",
    }
    _write_json(destination / "recipe.json", recipe)
    torch.save({"state_dict": shared_initial}, destination / "shared_initial.pt")
    records = []
    try:
        for count, args, decoder, model in arms:
            arm_root = destination / f"wsi_K{count}"
            arm_root.mkdir()
            _write_json(arm_root / "args.json", {"model_args": args, "decoder_k": decoder})
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=plan["capacity_lr"])
            curve = []
            for epoch, order in enumerate(orders):
                model.train()
                totals = {"main": 0.0, "auxiliary": 0.0, "total": 0.0}
                seen = steps = 0
                for start in range(0, len(order), plan["capacity_batch_size"]):
                    ids = order[start:start + plan["capacity_batch_size"]]
                    _rng(_step_seed(plan["seed"], "train", epoch, steps), torch, np)
                    batch = _batch(prepared_root, manifest, fold, ids, device, torch)
                    optimizer.zero_grad(set_to_none=True)
                    main, auxiliary, total = _native_losses(model, batch, len(ids), torch)
                    total.backward()
                    optimizer.step()
                    for name, value in (("main", main), ("auxiliary", auxiliary), ("total", total)):
                        totals[name] += float(value.detach().cpu()) * len(ids)
                    seen += len(ids)
                    steps += 1
                model.eval()
                val_total, val_seen = 0.0, 0
                with torch.no_grad():
                    for index, start in enumerate(range(0, len(val_ids), plan["capacity_batch_size"])):
                        ids = val_ids[start:start + plan["capacity_batch_size"]]
                        _rng(_step_seed(plan["seed"], "val", epoch, index), torch, np)
                        batch = _batch(prepared_root, manifest, fold, ids, device, torch)
                        logits, _ = model(**batch)
                        loss = model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None) / len(ids)
                        if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
                            raise ValueError("nonfinite or nonscalar native validation loss")
                        val_total += float(loss.detach().cpu()) * len(ids)
                        val_seen += len(ids)
                curve.append({
                    "epoch": epoch + 1,
                    "train_samples": seen,
                    "train_steps": steps,
                    "train_mean_main": totals["main"] / seen,
                    "train_mean_auxiliary": totals["auxiliary"] / seen,
                    "train_mean_total": totals["total"] / seen,
                    "validation_samples": val_seen,
                    "validation_main": val_total / val_seen,
                })
                _write_json(arm_root / "curve.json", curve)
            checkpoint = arm_root / "last.pt"
            torch.save({"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()}}, checkpoint)
            records.append({
                "wsi_num_slots": count,
                "omic_num_slots": 8,
                "wsi_selected_k": 2,
                "omic_selected_k": 2,
                "parameter_numel": reference_contract["parameter_numel"],
                "epochs": plan["capacity_epochs"],
                "updates": sum(point["train_steps"] for point in curve),
                "checkpoint": str(checkpoint.relative_to(destination)),
                "checkpoint_sha256": _file_sha256(checkpoint),
                "args": str((arm_root / "args.json").relative_to(destination)),
                "curve": str((arm_root / "curve.json").relative_to(destination)),
            })
            model.to("cpu")
            del optimizer
        result = {
            "status": "matched_capacity_training_completed",
            "plan": plan,
            "arms": records,
            "problem_exists": None,
            "efficacy": None,
            "next_evidence": "independent content complexity and per-capacity coverage/omission interaction",
        }
        _write_json(destination / "capacity_training.json", result)
        files = [path for path in destination.rglob("*") if path.is_file()]
        _write_json(destination / "artifact_manifest.json", [
            {"path": str(path.relative_to(destination)), "sha256": _file_sha256(path), "bytes": path.stat().st_size}
            for path in sorted(files)
        ])
        return result
    except Exception as error:
        _write_json(destination / "failed.json", {"status": "failed", "error_type": type(error).__name__, "message": str(error)})
        raise
