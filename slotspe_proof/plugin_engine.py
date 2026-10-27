"""Matched plugin training, entered only by explicit run/generated commands.

Original source and protected queues are immutable. Patient artifacts stay in
ignored local_evidence. Generated execution is labelled and uses invented data.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from . import patient
from .model import build_model, make_args
from .plugin_host import SlotPluginHost
from .plugin_queue import PRESETS, STAGES, REPO, normalize_config, output_path
from .redundancy_content import ContentReference
from .redundancy_engine import state_hash, _cpu_state
from .slot_disentanglement import target_variance_floor, content_set_loss
from .slot_regularization import slot_metrics


def _now():
    return datetime.now(timezone.utc).isoformat()


def _dump(path, value):
    patient._dump(Path(path), value)


def schedule_seed(phase, index, position):
    phases = {"train": 0, "select": 1, "repeat": 2}
    if phase not in phases or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 10**6
                                  for x in (index, position)):
        raise ValueError("invalid random schedule coordinate")
    return 3 * 10**12 + phases[phase] * 10**11 + index * 10**6 + position


def logical_groups(order, size=8):
    if size < 2 or len(order) < 2 or len(set(order)) != len(order):
        raise ValueError("need distinct patients and logical batch size at least two")
    groups = [order[start:start + size] for start in range(0, len(order), size)]
    if len(groups[-1]) == 1 and len(groups) > 1:
        groups[-2].extend(groups.pop())
    return groups


def best_candidate(score, previous=0.):
    if score is None or not math.isfinite(score) or not 0 <= score <= 1:
        raise RuntimeError("undefined/nonfinite validation C-index; no checkpoint fallback")
    return score >= previous


def _build(config, manifest, preset, initial=None):
    args = make_args(manifest["omic_sizes"], encoding_dim=manifest["encoding_dim"])
    if (args.alpha_surv != config.get("native_alpha", .5) or
            args.lambda_recon_loss != config.get("native_reconstruction_weight", .01)):
        raise RuntimeError("effective native loss defaults differ from the locked protocol")
    args.wsi_projection_dim, args.slot_iters = config["projection_dim"], config["slot_iters"]
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(3)
        native = build_model(config["source_root"], args, "original")
    if initial is not None:
        native.load_state_dict(initial, strict=True)
    host = SlotPluginHost(native, preset, dimension=config["projection_dim"])
    return host.to(torch.device(config["device"]))


def _finite_parameters(host):
    if any(not bool(torch.isfinite(p).all()) for p in host.parameters()):
        raise RuntimeError("nonfinite model parameters; no clipping or silent recovery")


def logical_update(host, cases, targets, ids, seeds, optimizer):
    if not cases or len(set(ids)) != len(ids) or len(ids) != len(cases) or len(seeds) != len(cases):
        raise ValueError("one distinct patient and matched seed per case required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("native seeds must be unique within the logical batch")
    if any(case["x_wsi"].ndim != 3 or case["x_wsi"].shape[0] != 1 or
           case["y"].shape != (1,) or case["c"].shape != (1,) for case in cases):
        raise ValueError("one native forward and record per patient required")
    optimizer.zero_grad(set_to_none=True)
    host.train()
    main, auxiliary, records = [], [], []
    for case, seed in zip(cases, seeds):
        patient._rng(seed, torch, np)
        logits, aux = host(**case, plugin_seed=seed + 4 * 10**11)
        loss = host.backbone.loss_fn(logits, y=case["y"], c=case["c"], t=None)
        main.append(loss)
        auxiliary.append(torch.as_tensor(aux, device=logits.device))
        records.append(host.last_records)
    main, auxiliary = torch.stack(main).mean(), torch.stack(auxiliary).mean()
    terms = host.added_losses(records, targets)
    total = main + auxiliary + terms["added"]
    if not bool(torch.isfinite(total)):
        raise RuntimeError("nonfinite plugin objective")
    total.backward()
    gradients = [p.grad for p in host.parameters() if p.grad is not None]
    if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise RuntimeError("missing/nonfinite plugin gradients")
    norm = torch.stack([g.detach().norm() for g in gradients]).norm()
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("nonfinite total gradient norm")
    optimizer.step()
    _finite_parameters(host)
    return {"main_nll": float(main.detach()), "native_auxiliary": float(auxiliary.detach()),
            "total_loss": float(total.detach()), "gradient_l2": float(norm),
            **{name: float(value.detach()) for name, value in terms.items()},
            "patients": len(cases), "optimizer_steps": 1, "root_forwards": len(cases)}


def _case(root, manifest, pid, device, *, training, seed=0, reference=None):
    item = torch.load(Path(root) / manifest["patients"][pid]["cache_path"], map_location="cpu", weights_only=True)
    if item["patient_id"] != pid:
        raise RuntimeError("prepared cache identity mismatch")
    patches = item["patches"]
    if training:
        bag, selected = patient._sample_bag(patches, manifest["patches"], np, torch,
                                            generator_seed=seed + 6 * 10**11)
        real = bag[:len(selected)]
    else:
        bag, real = patches, patches
    fold = manifest["folds"]["0"]
    case = {"x_wsi": bag.unsqueeze(0).to(device), "omic_missing": False,
            "y": torch.tensor([fold["labels"][pid]], dtype=torch.long, device=device),
            "c": torch.tensor([manifest["patients"][pid]["censor"]], dtype=torch.float32, device=device)}
    for index, genes in enumerate(item["omics"], 1):
        case[f"x_omic{index}"] = genes.unsqueeze(0).to(device)
    targets = None
    if reference is not None:
        target_input = {"x_wsi": real.unsqueeze(0), **{f"x_omic{i+1}": g.unsqueeze(0) for i, g in enumerate(item["omics"])}}
        targets = {name: value.to(device) for name, value in reference.targets(target_input).items()}
    return case, targets


def _geometry(records):
    result = {}
    for modality in ("wsi", "omic"):
        result[modality] = {space: slot_metrics(records[modality][space]) for space in ("native", "mean", "latent")}
    return result


def _evaluate(host, root, manifest, config, phase, index, reference=None, raw_path=None):
    from sksurv.metrics import concordance_index_censored
    host.eval()
    ids = manifest["folds"]["0"]["val"]
    rows, raw = [], []
    with torch.no_grad():
        for position, pid in enumerate(ids):
            seed = schedule_seed(phase, index, position)
            case, targets = _case(root, manifest, pid, config["device"], training=False,
                                   reference=reference if host.parts else None)
            patient._rng(seed, torch, np)
            logits, _ = host(**case, plugin_seed=seed + 4 * 10**11)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("nonfinite validation prediction")
            risk = -(1 - logits.sigmoid()).cumprod(-1).sum(-1)
            row = {"patient_id": pid, "risk": float(risk[0]), "logits": logits[0].cpu().tolist(),
                   "nll": float(host.backbone.loss_fn(logits, y=case["y"], c=case["c"], t=None)),
                   "native_seed": seed, "plugin_seed": seed + 4 * 10**11,
                   "real_patches": case["x_wsi"].shape[1], "geometry": _geometry(host.last_records)}
            if host.parts:
                row["content"] = {name: {"mse": float(content_set_loss(host.last_records[name]["content"], targets[name].unsqueeze(0))),
                                  "constant_floor": float(target_variance_floor(targets[name].unsqueeze(0))[0])}
                                  for name in ("wsi", "omic")}
            rows.append(row)
            if raw_path is not None:
                raw.append({"patient_id": pid, "logits": logits.cpu(),
                            "slots": {name: {key: value.detach().cpu() for key, value in host.last_records[name].items()}
                                      for name in ("wsi", "omic")}})
    events = np.asarray([manifest["patients"][pid]["censor"] < 1 for pid in ids])
    times = np.asarray([manifest["patients"][pid]["time"] for pid in ids])
    score = float(concordance_index_censored(events, times, np.asarray([row["risk"] for row in rows]))[0])
    best_candidate(score)  # undefined values never become a score or a checkpoint
    if raw_path is not None:
        torch.save(raw, raw_path)
    return {"cindex": score, "nll": float(np.mean([row["nll"] for row in rows])), "rows": rows}


def _checkpoint(host, optimizer, scheduler, protocol, epoch, steps):
    return {"state_dict": _cpu_state(host), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "protocol": protocol, "epoch": epoch,
            "optimizer_steps": steps, "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "rng_protocol": "per-case random schedules plus isolated plugin initialization; seed coordinates saved in curve"}


def _run_arm(config, root, manifest, preset, initial, reference, verify_live=lambda: None):
    out = Path(root) / "arms" / preset
    if out.exists():
        raise RuntimeError("arm output already exists; preserve partial/completed evidence")
    out.mkdir(parents=True)
    try:
        return _train_arm(config, root, manifest, preset, initial, reference, verify_live)
    except BaseException as exc:
        if not (out / "failure.json").exists():
            _dump(out / "failure.json", {"status": "failed", "phase": "initialization",
                                        "type": type(exc).__name__, "message": str(exc)})
        raise


def _train_arm(config, root, manifest, preset, initial, reference, verify_live):
    out = Path(root) / "arms" / preset
    started = time.perf_counter()
    host = _build(config, manifest, preset, initial)
    native_initial = state_hash(_cpu_state(host.backbone))
    if native_initial != state_hash(initial):
        raise RuntimeError("native initialization differs across plugin arms")
    optimizer = torch.optim.Adam(host.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
    protocol = {"preset": preset, "config": config, "initial_core_hash": native_initial,
                "initial_plugin_hash": state_hash({k: v for k, v in host.state_dict().items() if not k.startswith("core.backbone.")}),
                "parameter_count": sum(p.numel() for p in host.parameters()),
                "native_parameter_count": sum(p.numel() for p in host.backbone.parameters()),
                "checkpoint_policy": "initial0_ge_tie_latest_one_selection_draw_per_epoch_then_20_repeats",
                "validation": "B1_all_real_patches_no_padding_no_subsampling",
                "initialization": "shared_native_state_and_paired_plugin_state_no_trained_parent_warmstart",
                "scope": "single_fold_exploratory_not_generalization", "started_utc": _now()}
    _dump(out / "protocol.json", protocol)
    train_ids = manifest["folds"]["0"]["train"]
    positions = {pid: position for position, pid in enumerate(train_ids)}
    curve, steps, best, best_epoch = [], 0, 0., None
    if config["device"].startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(config["device"]))
    try:
        for epoch in range(config["epochs"]):
            verify_live()
            order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(3000 + epoch)).tolist()
            batches = []
            for group in logical_groups(order, config["batch_size"]):
                ids = [train_ids[position] for position in group]
                seeds = [schedule_seed("train", epoch, positions[pid]) for pid in ids]
                loaded = [_case(root, manifest, pid, config["device"], training=True, seed=seed,
                          reference=reference if host.parts else None) for pid, seed in zip(ids, seeds)]
                result = logical_update(host, [item[0] for item in loaded], [item[1] for item in loaded], ids, seeds, optimizer)
                steps += 1
                batches.append({"patient_ids": ids, "native_seeds": seeds, **result})
            selection = _evaluate(host, root, manifest, config, "select", epoch, reference)
            scheduler.step()
            checkpoint = _checkpoint(host, optimizer, scheduler, protocol, epoch + 1, steps)
            torch.save(checkpoint, out / "last.pt")
            if best_candidate(selection["cindex"], best):
                best, best_epoch = selection["cindex"], epoch + 1
                torch.save(checkpoint, out / "best.pt")
                _dump(out / "best_selection_predictions.json", selection)
            curve.append({"epoch": epoch + 1, "batches": batches, "selection_cindex": selection["cindex"],
                          "selection_nll": selection["nll"], "best_epoch": best_epoch, "optimizer_steps": steps})
            _dump(out / "curve.json", curve)
            print(f"preset={preset} epoch={epoch+1}/{config['epochs']} steps={steps} val_C={selection['cindex']:.6f}", flush=True)
        if best_epoch is None:
            raise RuntimeError("no valid best-validation checkpoint")
        host.load_state_dict(torch.load(out / "best.pt", map_location="cpu", weights_only=True)["state_dict"], strict=True)
        evaluations = []
        for repeat in range(config["eval_repeats"]):
            result = _evaluate(host, root, manifest, config, "repeat", repeat, reference,
                               raw_path=out / f"raw_repeat_{repeat:02d}.pt")
            evaluations.append(result)
            _dump(out / "eval_repeats.json", evaluations)
        scores = [item["cindex"] for item in evaluations]
        summary = {"status": "completed", "preset": preset, "best_epoch": best_epoch,
                   "best_selection_cindex": best, "epochs": config["epochs"], "patches": config["patches"],
                   "optimizer_steps": steps, "train_patients": len(train_ids),
                   "validation_patients": len(manifest["folds"]["0"]["val"]),
                   "repeat_cindex_mean": float(np.mean(scores)), "repeat_cindex_std_population": float(np.std(scores)),
                   "independent_runs": 1, "random_repeats_are_not_independent_patients": True,
                   "best_sha256": patient._sha(out / "best.pt"), "last_sha256": patient._sha(out / "last.pt"),
                   "elapsed_seconds": time.perf_counter() - started,
                   "cuda_peak_bytes": torch.cuda.max_memory_allocated(config["device"]) if config["device"].startswith("cuda") else None,
                   "claim_boundary": "single_fold_single_seed_exploratory_no_patient_efficacy_established"}
        verify_live()
        _dump(out / "summary.json", summary)
        return summary
    except BaseException as exc:
        _dump(out / "failure.json", {"status": "failed", "type": type(exc).__name__, "message": str(exc),
                                    "optimizer_steps_completed": steps, "elapsed_seconds": time.perf_counter()-started})
        raise


def implementation_hashes():
    paths = [*sorted((REPO / "slotspe_proof").glob("*.py")), REPO / "scripts/run_slotspe_plugins.py",
             REPO / "configs/slotspe_plugins_fold0_seed3.json"]
    paths += [path for path in sorted((REPO / "third_party/SlotSPE").rglob("*"))
              if path.is_file() and "__pycache__" not in path.parts]
    return {path.relative_to(REPO).as_posix(): patient._sha(path) for path in paths}


def _artifact_hashes(root, stage):
    return {path.relative_to(root).as_posix(): patient._sha(path) for preset in STAGES[stage]
            for path in sorted((Path(root) / "arms" / preset).rglob("*")) if path.is_file()}


def execute(config, output_root, stage="v1"):
    config = normalize_config(config, require_inputs=True)
    root = output_path(output_root)
    manifest_path = root / "plugin_queue_manifest.json"
    existed = manifest_path.exists()
    try:
        return _execute(config, root, stage)
    except BaseException as exc:
        if not existed and manifest_path.exists():
            queue = patient._read(manifest_path)
            if "initial_sha256" not in queue:
                queue["preparation_failure"] = {"status": "failed", "type": type(exc).__name__,
                                                  "message": str(exc), "failed_utc": _now()}
                _dump(manifest_path, queue)
        raise


def _execute(config, output_root, stage="v1"):
    config = normalize_config(config, require_inputs=True)
    root = output_path(output_root)
    if stage not in STAGES:
        raise ValueError("unknown plugin stage")
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    torch.set_num_threads(config["torch_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    code = implementation_hashes()
    inputs = {str(path): patient._sha(path) for path in patient._csv_paths(config)}
    identity = {"config": config, "implementation": code, "metadata_sha256": inputs,
                "runtime": {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
                            "numpy": np.__version__, "platform": platform.platform(),
                            "libraries": {name: importlib.metadata.version(name) for name in
                                ("scipy", "scikit-survival", "scikit-learn", "pandas", "h5py", "einops")}}}
    manifest_path = root / "plugin_queue_manifest.json"
    if manifest_path.exists():
        queue = patient._read(manifest_path)
        if queue["identity"] != identity:
            raise RuntimeError("code/config/metadata/runtime changed; preserve output and start a fresh queue")
        if not {"prepared_manifest_sha256", "initial_sha256"}.issubset(queue):
            raise RuntimeError("prior preparation incomplete; preserve it and choose a fresh queue")
    else:
        if stage != "v1" or root.exists() and any(root.iterdir()):
            raise RuntimeError("start stage v1 in a fresh ignored output directory")
        root.mkdir(parents=True, exist_ok=True)
        queue = {"schema": "slotspe_plugin_queue_v1", "identity": identity, "stages": {}, "started_utc": _now()}
        _dump(manifest_path, queue)
        patient.prepare(config, root)
        prepared = patient._manifest(config, root, verify_cache=True)
        reference = ContentReference(prepared["omic_sizes"], prepared["encoding_dim"])

        def train_inputs():
            for pid in prepared["folds"]["0"]["train"]:
                item = torch.load(root / prepared["patients"][pid]["cache_path"], weights_only=True)
                yield {"x_wsi": item["patches"].unsqueeze(0),
                       **{f"x_omic{i+1}": g.unsqueeze(0) for i, g in enumerate(item["omics"])}}

        reference.fit(train_inputs())
        initial_host = _build(config, prepared, "B0_native")
        torch.save({"native": _cpu_state(initial_host.backbone), "reference": reference.state_dict()}, root / "initial.pt")
        queue["prepared_manifest_sha256"] = patient._sha(root / "prepared/manifest.json")
        queue["initial_sha256"] = patient._sha(root / "initial.pt")
        _dump(manifest_path, queue)
        del initial_host
    for prior in list(STAGES)[:list(STAGES).index(stage)]:
        record = queue["stages"].get(prior, {})
        if record.get("status") != "completed" or record.get("artifacts") != _artifact_hashes(root, prior):
            raise RuntimeError("previous stage missing, incomplete or changed: " + prior)
    if stage in queue["stages"]:
        raise RuntimeError("stage already started; partial/completed stages are not overwritten")
    if (patient._sha(root / "prepared/manifest.json") != queue["prepared_manifest_sha256"] or
            patient._sha(root / "initial.pt") != queue["initial_sha256"]):
        raise RuntimeError("preparation or shared initialization changed")
    prepared = patient._manifest(config, root, verify_cache=True)
    initial = torch.load(root / "initial.pt", map_location="cpu", weights_only=True)
    reference = ContentReference(prepared["omic_sizes"], prepared["encoding_dim"])
    reference.load_state_dict(initial["reference"])

    def verify_live():
        if implementation_hashes() != code or any(patient._sha(Path(p)) != h for p, h in inputs.items()):
            raise RuntimeError("live implementation/metadata drift")

    queue["stages"][stage] = {"status": "running", "started_utc": _now()}
    _dump(manifest_path, queue)
    try:
        summaries = [_run_arm(config, root, prepared, preset, initial["native"], reference, verify_live)
                     for preset in STAGES[stage]]
        patient._manifest(config, root, verify_cache=True)
        queue["stages"][stage] = {"status": "completed", "summaries": summaries,
                                   "artifacts": _artifact_hashes(root, stage), "completed_utc": _now()}
        _dump(manifest_path, queue)
        _dump(root / f"summary_{stage}.json", {"stage": stage, "arms": summaries,
                "claim_boundary": "paired_single_fold_exploratory_not_generalization"})
        return {"status": "completed", "stage": stage, "presets": list(STAGES[stage]), "output_root": str(root)}
    except BaseException as exc:
        queue["stages"][stage] = {"status": "failed", "type": type(exc).__name__, "message": str(exc)}
        _dump(manifest_path, queue)
        raise


def generated_smoke(output_root):
    """Nine native-size one-update conditions on eight invented cases only."""
    root = output_path(output_root)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("preserve generated receipts; choose a fresh output directory")
    root.mkdir(parents=True, exist_ok=True)
    try:
        return _generated_smoke_run(root)
    except BaseException as exc:
        if not (root / "failure.json").exists():
            _dump(root / "failure.json", {"status": "failed", "phase": "initialization",
                                         "type": type(exc).__name__, "message": str(exc)})
        raise


def _generated_smoke_run(root):
    torch.set_num_threads(2)
    code = implementation_hashes()
    generator = torch.Generator().manual_seed(7773)
    cases = [{"x_wsi": torch.randn(1, 4096, 8, generator=generator), "omic_missing": False,
              "x_omic1": torch.randn(1, 3, generator=generator), "x_omic2": torch.randn(1, 5, generator=generator),
              "y": torch.tensor([i % 4]), "c": torch.tensor([float(i % 2)])} for i in range(8)]
    reference = ContentReference([3, 5], 8).fit(cases)
    targets = [reference.targets(case) for case in cases]
    config = {"source_root": str(REPO / "third_party/SlotSPE"), "projection_dim": 256, "slot_iters": 10, "device": "cpu"}
    manifest = {"omic_sizes": [3, 5], "encoding_dim": 8}
    initial = _cpu_state(_build(config, manifest, "B0_native").backbone)
    rows = []
    try:
        for preset in PRESETS:
            started = time.perf_counter()
            host = _build(config, manifest, preset, initial)
            optimizer = torch.optim.Adam(host.parameters(), lr=.0005)
            result = logical_update(host, cases, targets, [f"generated_{i}" for i in range(8)],
                                     [schedule_seed("train", 0, i) for i in range(8)], optimizer)
            changed = sum(not torch.equal(value, initial[name]) for name, value in _cpu_state(host.backbone).items())
            if changed == 0:
                raise RuntimeError("generated native core did not update")
            row = {"preset": preset, **result, "native_tensors_updated": changed, "seconds": time.perf_counter()-started}
            rows.append(row)
            _dump(root / "generated_curve.json", rows)
            print(f"generated preset={preset} steps=1 loss={result['total_loss']:.6f}", flush=True)
            del host, optimizer
        if implementation_hashes() != code:
            raise RuntimeError("source changed during generated checks")
        summary = {"status": "passed", "scope": "invented_features_only_not_UNI2h_not_patients",
                   "native_shape": {"D": 256, "K": 8, "iterations": 10, "patches": 4096, "logical_batch": 8},
                   "generated_encoding_dim": 8, "generated_pathways": [3, 5], "conditions": rows,
                   "patient_runs": 0, "claim_boundary": "execution_and_gradient_checks_only_not_patient_efficacy"}
        _dump(root / "generated_summary.json", summary)
        return {"status": "passed", "conditions": len(rows), "patient_runs": 0, "output_root": str(root)}
    except BaseException as exc:
        _dump(root / "failure.json", {"status": "failed", "type": type(exc).__name__, "message": str(exc)})
        raise
