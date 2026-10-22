"""Portable paired patient experiments for the SlotSPE implementation audit.

Importing this module uses the standard library only.  ``prepare`` explicitly
reads the configured server data and creates one immutable, sampled cache per
patient; ``run_pair`` explicitly trains two arms.  No model, patient-data path,
training run, or result from the earlier local audit is imported here.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
from datetime import datetime, timezone


SCHEMA_VERSION = 1
ARMS = ("original", "readout_fix")
_DEFAULTS = {
    "study": "blca", "label_col": "survival_months_dss", "signature": "combine",
    "folds": [0, 1, 2, 3, 4], "seeds": [3, 17, 29], "epochs": 30,
    "patches": 4096, "batch_size": 8, "eval_repeats": 20,
    "encoding_dim": None, "device": "cuda:0", "lr": 5e-4,
    "experiment": "reproduction",
}
_CENSORS = {"survival_months_os": "censorship_os", "survival_months_pfi": "censorship_pfi",
            "survival_months_dss": "censorship_dss", "survival_months_dfs": "censorship_dfs"}


def _config(config):
    result = {**_DEFAULTS, **dict(config)}
    for name in ("source_root", "metadata_root", "features_root"):
        value = Path(result[name]).expanduser()
        if not value.is_absolute():
            raise ValueError(f"{name} must be an absolute server path")
        result[name] = str(value.resolve())
    for name in ("epochs", "patches", "batch_size", "eval_repeats"):
        value = result[name]
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
        result[name] = int(value)
    for name in ("folds", "seeds"):
        values = list(result[name])
        if not values or any(isinstance(v, bool) or int(v) != v or int(v) < 0 for v in values):
            raise ValueError(f"{name} must be a nonempty list of nonnegative integers")
        result[name] = [int(v) for v in values]
        if len(set(result[name])) != len(result[name]):
            raise ValueError(f"{name} contains duplicates")
    if result["encoding_dim"] is not None:
        if int(result["encoding_dim"]) != result["encoding_dim"] or int(result["encoding_dim"]) < 1:
            raise ValueError("encoding_dim must be a positive integer or null")
        result["encoding_dim"] = int(result["encoding_dim"])
    if result["label_col"] not in _CENSORS:
        raise ValueError("Unsupported survival endpoint")
    if not math.isfinite(float(result["lr"])) or float(result["lr"]) <= 0:
        raise ValueError("lr must be positive and finite")
    result["lr"] = float(result["lr"])
    return result


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _source_hashes(config):
    root = Path(config["source_root"])
    files = []
    for directory in ("models", "utils", "dataset"):
        files.extend(p for p in (root / directory).rglob("*.py") if "__pycache__" not in p.parts)
    if not (root / "models/SlotSPE.py").is_file() or not (root / "dataset/dataset_survival.py").is_file():
        raise FileNotFoundError("Configured source_root must contain models/SlotSPE.py and dataset/dataset_survival.py")
    return {p.relative_to(root).as_posix(): _sha(p) for p in sorted(files)}


def _csv_paths(config):
    root = Path(config["metadata_root"])
    study = config["study"]
    signature = "signatures.csv" if config["signature"] == "six" else (
        "combine_signatures.csv" if config["signature"] == "all" else f"{config['signature']}_signatures.csv")
    result = [root / "clinical/all" / f"{study}.csv",
              root / "raw_rna_data_inter" / f"{study}_rna_inter.csv",
              root / "signatures" / signature]
    result.extend(root / "splits/5fold" / study / f"fold_{fold}.csv" for fold in config["folds"])
    for path in result:
        if not path.is_file():
            raise FileNotFoundError(f"Required metadata absent (RNA is never substituted): {path}")
    return result


def _preparation_config(config):
    return {name: config[name] for name in ("source_root", "metadata_root", "features_root", "study",
            "label_col", "signature", "folds", "patches", "encoding_dim")}


def _h5_feature_array(handle):
    """Read supported UNI2-h layouts without dropping a real patch axis."""
    values = handle["features"]
    if values.ndim == 2:
        return values[...]
    if values.ndim == 3 and values.shape[0] == 1:
        return values[0]
    raise ValueError("HDF5 features must have shape [N,D] or [1,N,D]")


def _dataset_module(config):
    """Load exactly the requested dataset file, independent of sys.path."""
    path = Path(config["source_root"]) / "dataset/dataset_survival.py"
    name = "_slotspe_proof_dataset_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load requested dataset module: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError("Dataset source provenance mismatch")
    return module


def _manifest(config, output_root, verify_cache=False):
    root = Path(output_root)
    path = root / "prepared/manifest.json"
    manifest = _read(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Unsupported prepared manifest schema")
    if manifest["preparation_config"] != _preparation_config(config):
        raise RuntimeError("Preparation configuration changed; use a fresh batch output directory")
    if manifest["source_hashes"] != _source_hashes(config):
        raise RuntimeError("SlotSPE source changed after preparation; use a fresh batch")
    current_csv = {str(p): _sha(p) for p in _csv_paths(config)}
    if manifest["input_csv_sha256"] != current_csv:
        raise RuntimeError("Clinical/RNA/signature/split CSV changed after preparation")
    if verify_cache:
        for patient in manifest["patients"].values():
            cache = root / patient["cache_path"]
            if not cache.is_file() or _sha(cache) != patient["cache_sha256"]:
                raise RuntimeError(f"Prepared input cache changed: {cache}")
    return manifest


def prepare(config, output_root):
    """Cache real paired inputs once, then fit separate train-only bins per fold.

    Missing clinical endpoints, RNA columns, slides, nonfinite inputs, and bags
    shorter than ``patches`` are explicitly excluded as complete cases.  Missing
    RNA CSV is a hard error.  No missing feature is replaced by a zero vector.
    An existing complete cache can be verified and reused; a partial preparation
    is deliberately not overwritten.
    """
    config = _config(config)
    root = Path(output_root)
    prepared = root / "prepared"
    if (prepared / "manifest.json").exists():
        return _manifest(config, root, verify_cache=True)
    if prepared.exists() and any(prepared.iterdir()):
        raise RuntimeError("Partial preparation exists; retain it and select a fresh output batch")
    # Dependency imports belong to explicit execution, never queue planning.
    import numpy as np
    import pandas as pd
    import torch

    csv_hashes = {str(p): _sha(p) for p in _csv_paths(config)}
    source_hashes = _source_hashes(config)
    module = _dataset_module(config)
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "cache/patients").mkdir(parents=True)
    start = time.perf_counter()
    censor = _CENSORS[config["label_col"]]

    class TrainBinningFactory(module.SurvivalDatasetFactory):
        def _setup_clinical_data(self):
            # The original factory bins before splitting.  This paired proof
            # replaces ONLY that preparation, shared by both model arms.
            path = Path(self.data_path) / "clinical/all" / f"{self.study}.csv"
            clinical = pd.read_csv(path)
            columns = ["case id", self.label_col, self.censorship_var, "wsi"]
            clinical = clinical[columns].dropna().reset_index(drop=True)
            clinical["case id"] = clinical["case id"].astype(str)
            if clinical["case id"].duplicated().any():
                raise ValueError("Duplicate clinical patient IDs")
            clinical.insert(2, "label", -1)
            self.clinical_df = clinical
            self.bins = None

    with (prepared / "factory_prepare.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        factory = TrainBinningFactory(config["study"], config["metadata_root"], "Pathways", config["label_col"],
                    signature=config["signature"], n_bins=4, num_patches=config["patches"])
    if not factory.omic_sizes or any(int(n) < 1 for n in factory.omic_sizes):
        raise ValueError("No nonempty real RNA pathways overlap the signature")
    if factory.gene_data_df.columns.duplicated().any() or factory.gene_data_df.index.duplicated().any():
        raise ValueError("RNA contains duplicate patient columns or gene indices")
    clinical_ids = set(factory.clinical_df["case id"])
    patients, exclusions, folds = {}, {}, {}
    encoding_dim = config["encoding_dim"]

    def cache_patient(row, dataset):
        nonlocal encoding_dim
        pid = str(row["case id"])
        if pid in patients:
            return None
        if pid in exclusions:
            return exclusions[pid]
        if pid not in factory.gene_data_df.columns:
            return "missing_rna_patient_column"
        event_time, event_censor = float(row[config["label_col"]]), float(row[censor])
        if not math.isfinite(event_time) or event_time < 0 or event_censor not in (0.0, 1.0):
            return "invalid_survival_endpoint"
        slide_names = [s.strip() for s in str(row["wsi"]).split(",") if s.strip()]
        if not slide_names:
            return "missing_slide_identifiers"
        # Match the requested original loader's feature filename convention.
        paths = [Path(config["features_root"]) / (s.rstrip(".svs") + ".pt") for s in slide_names]
        if not all(path.is_file() for path in paths):
            # Fall back to UNI2-h .h5 if no .pt files exist (both layouts use
            # the same slide-name stem; only the extension differs).
            h5_paths = [Path(config["features_root"]) / (s.rstrip(".svs") + ".h5") for s in slide_names]
            if all(p.is_file() for p in h5_paths):
                paths = h5_paths
            else:
                return "at_least_one_missing_slide_feature"
        chunks, slide_info = [], []
        is_h5 = paths[0].suffix == ".h5"
        for path in paths:
            before = path.stat()
            try:
                if is_h5:
                    import h5py  # late import keeps the .pt path lean
                    with h5py.File(path, "r") as hf:
                        raw = _h5_feature_array(hf)
                    tensor = torch.from_numpy(raw)
                else:
                    tensor = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            except (RuntimeError, EOFError):
                # Legacy torch zip format or legacy h5 container; retry the
                # matching reader without mmap/weights_only.
                if is_h5:
                    with h5py.File(path, "r") as hf:
                        raw = _h5_feature_array(hf)
                    tensor = torch.from_numpy(raw)
                else:
                    tensor = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
                raise ValueError(f"Feature file must contain a nonempty [patch,channel] tensor: {path}")
            if encoding_dim is None:
                encoding_dim = int(tensor.shape[1])
            if tensor.shape[1] != encoding_dim:
                raise ValueError(f"Encoding width mismatch in {path}: {tensor.shape[1]} != {encoding_dim}")
            chunks.append(tensor)
            slide_info.append({"path": str(path.resolve()), "patch_count": int(tensor.shape[0]),
                "channels": int(tensor.shape[1]), "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns})
        sizes = [len(t) for t in chunks]
        total = sum(sizes)
        if total < config["patches"]:
            return "fewer_real_patches_than_configured_no_padding"
        sample_seed = int(hashlib.sha256((pid + "_slotspe_paired_proof_v1").encode()).hexdigest()[:8], 16)
        indices = np.sort(np.random.default_rng(sample_seed).choice(total, config["patches"], replace=False))
        cumulative = np.cumsum([0] + sizes)
        bag = torch.empty((config["patches"], encoding_dim), dtype=torch.float32)
        for index, tensor in enumerate(chunks):
            selected = (indices >= cumulative[index]) & (indices < cumulative[index + 1])
            bag[selected] = tensor[indices[selected] - cumulative[index]].float()
        omics = dataset.load_genes(pid)
        if not isinstance(omics, (list, tuple)) or len(omics) != len(factory.omic_sizes):
            raise ValueError("Original Pathways loader returned an unexpected RNA layout")
        if any(g.ndim != 1 or len(g) != int(size) for g, size in zip(omics, factory.omic_sizes)):
            raise ValueError("Original RNA pathway width changed")
        if not torch.isfinite(bag).all() or not all(torch.isfinite(g).all() for g in omics):
            return "nonfinite_selected_inputs"
        for path, info in zip(paths, slide_info):
            after = path.stat()
            if after.st_size != info["size_bytes"] or after.st_mtime_ns != info["mtime_ns"]:
                raise RuntimeError(f"Feature file changed during cache preparation: {path}")
        name = hashlib.sha256(pid.encode()).hexdigest() + ".pt"
        cache = prepared / "cache/patients" / name
        torch.save({"patient_id": pid, "wsi": bag, "omics": [g.detach().cpu().float() for g in omics]}, cache)
        patients[pid] = {"patient_id": pid, "time": event_time, "censor": event_censor,
            "cache_path": cache.relative_to(root).as_posix(), "cache_sha256": _sha(cache),
            "full_patch_count": total, "sample_seed": sample_seed,
            "patch_indices_concatenated_slide_order": indices.tolist(), "slides": slide_info,
            "selected_patch_tensor_sha256": hashlib.sha256(bag.numpy().tobytes()).hexdigest()}
        return None

    for fold in config["folds"]:
        split_path = Path(config["metadata_root"]) / "splits/5fold" / config["study"] / f"fold_{fold}.csv"
        split = pd.read_csv(split_path)
        split_ids = {}
        for kind in ("train", "val"):
            ids = split[kind].dropna().astype(str).tolist()
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate IDs in fold {fold} {kind}")
            split_ids[kind] = ids
        if set(split_ids["train"]) & set(split_ids["val"]):
            raise ValueError(f"Patient overlap in fold {fold}")
        fold_excluded, retained = [], {}
        for kind in ("train", "val"):
            for pid in split_ids[kind]:
                if pid not in clinical_ids:
                    fold_excluded.append({"patient_id": pid, "split": kind,
                                          "reason": "missing_clinical_endpoint_or_wsi_string"})
            dataset = module.SurvivalDataset(factory, config["features_root"], kind, fold, encoding_dim or 1)
            retained[kind] = []
            for _, row in dataset.label_df.iterrows():
                pid = str(row["case id"])
                reason = cache_patient(row, dataset)
                if reason:
                    exclusions[pid] = reason
                    fold_excluded.append({"patient_id": pid, "split": kind, "reason": reason})
                else:
                    retained[kind].append(pid)
            if not retained[kind]:
                raise ValueError(f"No complete-case {kind} patients remain in fold {fold}")
        event_times = [patients[pid]["time"] for pid in retained["train"] if patients[pid]["censor"] < 1]
        if len(event_times) < 4:
            raise ValueError(f"Fold {fold} has fewer than four uncensored training patients")
        _, edges = pd.qcut(event_times, q=4, labels=False, retbins=True)
        edges[0], edges[-1] = -np.inf, np.inf
        labels = {pid: int(np.searchsorted(edges[1:-1], patients[pid]["time"], side="right"))
                  for kind in ("train", "val") for pid in retained[kind]}
        folds[str(fold)] = {"fold": fold, "split_csv": str(split_path), "train": retained["train"],
            "val": retained["val"], "labels": labels, "excluded": fold_excluded,
            "original_split_counts": {kind: len(ids) for kind, ids in split_ids.items()},
            "counts": {"train": len(retained["train"]), "val": len(retained["val"]),
                       "excluded": len(fold_excluded), "overlap": 0},
            "binning": {"fit_subset": "complete_case_train_uncensored_only", "q": 4,
                "edges": ["-inf", *[float(x) for x in edges[1:-1]], "inf"], "right": False,
                "same_labels_both_arms": True, "validation_outcomes_used_for_training_bins": False}}
        print(f"Prepared fold={fold} train={len(retained['train'])} val={len(retained['val'])} excluded={len(fold_excluded)}", flush=True)
    if source_hashes != _source_hashes(config) or csv_hashes != {str(p): _sha(p) for p in _csv_paths(config)}:
        raise RuntimeError("Source or metadata changed during preparation")
    manifest = {"schema_version": SCHEMA_VERSION, "prepared_utc": _now(),
        "preparation_config": _preparation_config(config), "source_root": config["source_root"],
        "source_hashes": source_hashes, "input_csv_sha256": csv_hashes,
        "dataset_module_path": str(Path(config["source_root"]) / "dataset/dataset_survival.py"),
        "omic_sizes": [int(n) for n in factory.omic_sizes], "encoding_dim": encoding_dim,
        "patches": config["patches"], "folds": folds, "patients": patients,
        "counts": {"unique_complete_case_patients": len(patients), "per_fold": {f: v["counts"] for f, v in folds.items()}},
        "exclusion_policy": "Complete observed clinical/RNA/all slides, finite selected inputs, at least configured real patches; no zero substitution or padding",
        "patch_policy": "Fixed per-patient seeded sampling without replacement, shared by arms, folds, epochs, and evaluations",
        "feature_provenance": "Original feature file path/shape/size/mtime retained; cached tensors content-hashed; full original feature files are not content-hashed",
        "rna_transform": "Requested original Pathways loader, no added normalization",
        "scope": "Paired mechanism proof; final-epoch exploratory validation; no OT mechanism or performance gain asserted",
        "preparation_seconds": time.perf_counter() - start}
    _dump(prepared / "manifest.json", manifest)
    return manifest


def _rng(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch(root, manifest, fold_data, ids, device, torch):
    cache = [torch.load(Path(root) / manifest["patients"][pid]["cache_path"],
                        map_location="cpu", weights_only=True) for pid in ids]
    if any(item["patient_id"] != pid for item, pid in zip(cache, ids)):
        raise RuntimeError("Prepared cache patient identity mismatch")
    batch = {"x_wsi": torch.stack([item["wsi"] for item in cache]).to(device),
        "y": torch.tensor([fold_data["labels"][pid] for pid in ids], dtype=torch.long, device=device),
        "c": torch.tensor([manifest["patients"][pid]["censor"] for pid in ids], dtype=torch.float32, device=device),
        "omic_missing": False}
    for k in range(len(manifest["omic_sizes"])):
        batch[f"x_omic{k + 1}"] = torch.stack([item["omics"][k] for item in cache]).to(device)
    return batch


def _evaluate(model, root, manifest, fold_data, ids, config, device, seed, torch, np, concordance):
    model.eval()
    _rng(seed, torch, np)
    predictions, loss = [], 0.0
    with torch.no_grad():
        for start in range(0, len(ids), config["batch_size"]):
            selected = ids[start:start + config["batch_size"]]
            batch = _batch(root, manifest, fold_data, selected, device, torch)
            logits, _ = model(**batch)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Nonfinite evaluation logits")
            loss += float(model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None))
            predictions.append(logits.detach().cpu())
    logits = torch.cat(predictions)
    risks = -(1 - logits.sigmoid()).cumprod(-1).sum(-1)
    events = np.asarray([manifest["patients"][pid]["censor"] < 1 for pid in ids], dtype=bool)
    times = np.asarray([manifest["patients"][pid]["time"] for pid in ids], dtype=float)
    try:
        score = float(concordance(events, times, risks.numpy())[0])
        score = score if math.isfinite(score) else None
        metric_status = "computed" if score is not None else "undefined_nonfinite"
    except ValueError as exc:
        score, metric_status = None, "undefined: " + str(exc)
    return {"nll": loss / len(ids), "cindex": score, "cindex_status": metric_status,
            "risks": risks.tolist(), "logits": logits.tolist(), "eval_seed": seed}


def _selector_gradients(model, root, manifest, fold_data, ids, device, seed, torch, np):
    model.train()
    _rng(seed, torch, np)
    batch = _batch(root, manifest, fold_data, ids, device, torch)
    logits, auxiliary = model(**batch)
    main = model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None) / len(ids)
    named = [(name, parameter) for name, parameter in model.named_parameters() if "pred_keep_slot" in name and parameter.requires_grad]
    if not named:
        return {"probe_seed": seed, "patient_ids": ids, "status": "not_observed_no_matching_selector_parameters", "parameters": {}}
    parameters = [parameter for _, parameter in named]
    main_grads = torch.autograd.grad(main, parameters, retain_graph=True, allow_unused=True) if main.requires_grad else [None] * len(parameters)
    auxiliary_grads = (torch.autograd.grad(auxiliary, parameters, allow_unused=True)
        if isinstance(auxiliary, torch.Tensor) and auxiliary.requires_grad else [None] * len(parameters))

    def describe(gradient):
        if gradient is None:
            return {"kind": "None", "connected": False, "norm": None, "finite": None, "nonzero_elements": 0}
        return {"kind": "nonzero" if bool(torch.count_nonzero(gradient)) else "zero",
                "connected": True, "norm": float(gradient.norm()), "finite": bool(torch.isfinite(gradient).all()),
                "nonzero_elements": int(torch.count_nonzero(gradient))}

    result = {"probe_seed": seed, "patient_ids": list(ids), "status": "measured",
        "main_nll": float(main.detach()), "auxiliary_loss": float(auxiliary.detach()) if isinstance(auxiliary, torch.Tensor) else float(auxiliary),
        "parameters": {name: {"main": describe(g), "aux": describe(a)} for (name, _), g, a in zip(named, main_grads, auxiliary_grads)}}
    model.eval()
    return result


def _write_predictions(path, manifest, fold_data, ids, repeats):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        names = ["patient_id", "time", "censor", "label"] + [f"risk_repeat_{i:02d}" for i in range(len(repeats))]
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for index, pid in enumerate(ids):
            patient = manifest["patients"][pid]
            row = {"patient_id": pid, "time": patient["time"], "censor": patient["censor"], "label": fold_data["labels"][pid]}
            row.update({f"risk_repeat_{repeat:02d}": value["risks"][index] for repeat, value in enumerate(repeats)})
            writer.writerow(row)


def _repeat_summary(repeats, np):
    risks = np.asarray([r["risks"] for r in repeats], dtype=float)
    scores = np.asarray([r["cindex"] for r in repeats if r["cindex"] is not None], dtype=float)
    # Exact identical repeats must report exactly zero, including roundoff in
    # calculating the mean of twenty identical floating point values.
    risk_std = (risks - risks[:1]).std(0)
    score_std = float((scores - scores[0]).std()) if len(scores) else None
    return {"repeats": len(repeats), "defined_cindex_repeats": len(scores),
        "val_cindex_mean_exploratory": float(scores.mean()) if len(scores) else None,
        "val_cindex_std_population": score_std,
        "val_cindex_min_exploratory": float(scores.min()) if len(scores) else None,
        "val_cindex_max_exploratory": float(scores.max()) if len(scores) else None,
        "patient_risk_std_mean": float(risk_std.mean()), "patient_risk_std_max": float(risk_std.max()),
        "patient_risk_range_max": float(np.ptp(risks, axis=0).max()),
        "evaluation_randomness_observed": bool(np.any(np.ptp(risks, axis=0) > 0)),
        "std_computation": "Population std after subtracting first repeat (translation invariant)"}


def run_pair(config, output_root, fold: int, seed: int):
    """Train original then readout_fix, and evaluate the original weights again.

    Both trained arms share initial state, cases, bins, sampled inputs, batch
    order, step seeds, optimizer/schedule and full configured epoch budget.  A
    nonempty pair directory is never resumed or overwritten by this function.
    """
    config = _config(config)
    from .queue import training_arms
    arms = training_arms(config)
    if fold not in config["folds"] or seed not in config["seeds"]:
        raise ValueError("Pair fold/seed is outside the declared queue")
    root = Path(output_root)
    pair_root = root / "pairs" / f"fold_{fold}" / f"seed_{seed}"
    if pair_root.exists() and any(pair_root.iterdir()):
        raise RuntimeError(f"Refusing to overwrite or resume a prior pair attempt: {pair_root}")
    manifest = _manifest(config, root, verify_cache=True)
    import numpy as np
    import torch
    from sksurv.metrics import concordance_index_censored
    from .model import build_model, make_args

    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable; no CPU fallback or epoch reduction")
    torch.set_num_threads(int(config.get("torch_threads", 2)))
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    fold_data = manifest["folds"][str(fold)]
    train_ids, val_ids = fold_data["train"], fold_data["val"]
    pair_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    protocol = {"schema_version": SCHEMA_VERSION, "fold": fold, "seed": seed,
        "epochs": config["epochs"], "batch_size": config["batch_size"], "patches": config["patches"],
        "lr": config["lr"], "weight_decay": 0.0, "optimizer": "Adam_original_adam_branch_lr_only",
        "scheduler": "CosineAnnealingLR_Tmax_full_epochs_eta_min_1e-6", "arms": arms,
        "experiment": config["experiment"],
        "checkpoint_selection": "fixed_final_epoch_no_validation_selection", "eval_repeats": config["eval_repeats"],
        "counts": fold_data["counts"], "binning": fold_data["binning"],
        "same_patient_inputs_bins_initialization_batch_order_and_step_seeds": True,
        "source_hashes_before": _source_hashes(config), "manifest_sha256": _sha(root / "prepared/manifest.json"),
        "runner_sha256": _sha(__file__), "model_controls_sha256": _sha(Path(__file__).with_name("model.py")),
        "torch": str(torch.__version__), "python": sys.executable, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "started_utc": _now(),
        "differences_from_original_benchmark_protocol": [
            "Complete-case paired cohort; missing inputs excluded and recorded",
            "Train-only uncensored quantile bins shared by both arms",
            "Fixed sampled real patches for training and validation; no zero padding",
            "Final-epoch checkpoints and repeated validation predictions; no checkpoint selection",
            "This queue follows its declared epochs/batch/patch budget, which may differ from the original benchmark"],
        "scope": "Engineering mechanism proof; original cross branch remains dense; no OT addition or established performance gain"}
    _dump(pair_root / "protocol.json", protocol)
    args = make_args(manifest["omic_sizes"], encoding_dim=manifest["encoding_dim"])
    _rng(seed, torch, np)
    original = build_model(config["source_root"], args, "original")
    initial = {name: value.detach().cpu().clone() for name, value in original.state_dict().items()}
    parameter_count = sum(p.numel() for p in original.parameters())
    del original
    torch.save(initial, pair_root / "shared_initial_state.pt")
    initial_sha = _sha(pair_root / "shared_initial_state.pt")
    summaries = []
    try:
        if config["experiment"] == "attribution_abc":
            from .attribution import matched_bc_check
            ids = train_ids[:config["batch_size"]]
            check = matched_bc_check(config["source_root"], args, initial,
                _batch(root, manifest, fold_data, ids, device, torch), seed + 1000, device)
            check.update(patient_ids=ids, input_scope="prepared_training_patient_inputs_shared_initial_weights")
            _dump(pair_root / "matched_state_bc_checks.json", check)
            structural = [value for key, value in check["checks"].items() if key != "C_main_nonzero_each_modality"]
            if not all(structural):
                raise RuntimeError("B/C matched-state contract failed; report retained")
        for arm in arms:
            out = pair_root / arm
            out.mkdir()
            arm_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _rng(seed, torch, np)
            model = build_model(config["source_root"], args, arm).to(device)
            if config["experiment"] == "original_audit":
                from .native_audit import model_purity
                model_purity(model, config["source_root"])
            model.load_state_dict(initial, strict=True)
            if sum(p.numel() for p in model.parameters()) != parameter_count:
                raise RuntimeError("Control changed model parameter count")
            if any(not torch.equal(value.detach().cpu(), initial[name]) for name, value in model.state_dict().items()):
                raise RuntimeError("Paired initial states differ")
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
            curve, steps = [], 0
            for epoch in range(config["epochs"]):
                model.train()
                order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(seed * 1000 + epoch)).tolist()
                losses, grad_norms, seen = np.zeros(3), [], 0
                epoch_start = time.perf_counter()
                for batch_index, batch_start in enumerate(range(0, len(order), config["batch_size"])):
                    ids = [train_ids[index] for index in order[batch_start:batch_start + config["batch_size"]]]
                    _rng(seed * 100000 + epoch * 1000 + batch_index, torch, np)
                    batch = _batch(root, manifest, fold_data, ids, device, torch)
                    optimizer.zero_grad(set_to_none=True)
                    logits, auxiliary = model(**batch)
                    main = model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None) / len(ids)
                    total = main + auxiliary
                    if not bool(torch.isfinite(total)):
                        raise RuntimeError(f"Nonfinite loss at arm={arm} epoch={epoch + 1} batch={batch_index}")
                    total.backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    # Avoid a separate norm kernel/CPU launch for each of the
                    # many pathway encoder parameters when the runtime supports
                    # the multi-tensor operation. No clipping is applied.
                    foreach_norm = getattr(torch, "_foreach_norm", None)
                    norms = foreach_norm(gradients, 2) if foreach_norm is not None else [gradient.detach().norm() for gradient in gradients]
                    grad_norm = torch.stack(norms).norm()
                    if not bool(torch.isfinite(grad_norm)):
                        raise RuntimeError(f"Nonfinite gradient at arm={arm} epoch={epoch + 1} batch={batch_index}")
                    grad_norms.append(float(grad_norm))
                    optimizer.step()
                    losses += np.asarray([float(main.detach()), float(auxiliary.detach()) if isinstance(auxiliary, torch.Tensor) else float(auxiliary), float(total.detach())]) * len(ids)
                    seen += len(ids)
                    steps += 1
                validation = _evaluate(model, root, manifest, fold_data, val_ids, config, device,
                    seed * 10000 + epoch, torch, np, concordance_index_censored)
                entry = {"epoch": epoch + 1, "train_main_nll": float(losses[0] / seen),
                    "train_aux_loss": float(losses[1] / seen), "train_total_loss": float(losses[2] / seen),
                    "grad_l2_mean": float(np.mean(grad_norms)), "grad_l2_max": float(np.max(grad_norms)),
                    "val_nll_exploratory": validation["nll"], "val_cindex_exploratory": validation["cindex"],
                    "val_cindex_status": validation["cindex_status"], "optimizer_steps": steps,
                    "seconds": time.perf_counter() - epoch_start}
                curve.append(entry)
                _dump(out / "curve.json", curve)
                print(f"fold={fold} seed={seed} arm={arm} epoch={epoch + 1}/{config['epochs']} loss={entry['train_total_loss']:.5f} val_C={validation['cindex']}", flush=True)
                scheduler.step()
            checkpoint = {"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "fold": fold, "seed": seed, "arm": arm, "epochs": config["epochs"], "optimizer_steps": steps,
                "omic_sizes": manifest["omic_sizes"], "encoding_dim": manifest["encoding_dim"],
                "protocol": protocol, "checkpoint_selection": "fixed_final_epoch"}
            torch.save(checkpoint, out / "last.pt")
            observer = None
            if config["experiment"] == "original_audit":
                from .native_audit import GateObserver
                observer = GateObserver(model, val_ids)
            repeats = []
            try:
                for repeat in range(config["eval_repeats"]):
                    eval_seed = seed * 1000000 + 80000 + repeat
                    if observer is not None:
                        observer.begin(repeat, eval_seed)
                    repeats.append(_evaluate(model, root, manifest, fold_data, val_ids, config, device,
                        eval_seed, torch, np, concordance_index_censored))
            finally:
                if observer is not None:
                    observer.close()
            if observer is not None:
                _dump(out / "natural_gate_observations.json", observer.result(_sha(out / "last.pt")))
            _dump(out / "eval_repeats.json", repeats)
            _write_predictions(out / "validation_predictions.csv", manifest, fold_data, val_ids, repeats)
            gradient = _selector_gradients(model, root, manifest, fold_data, train_ids[:config["batch_size"]],
                device, seed * 1000000 + 90000, torch, np)
            _dump(out / "trained_selector_gradients.json", gradient)
            from .interventions import slot_intervention
            interventions = []
            for batch_start in range(0, len(val_ids), config["batch_size"]):
                ids = val_ids[batch_start:batch_start + config["batch_size"]]
                probe_seed = seed * 1000000 + 91000 + batch_start
                _rng(probe_seed, torch, np)
                interventions.append(slot_intervention(model,
                    _batch(root, manifest, fold_data, ids, device, torch), ids,
                    probe_seed, _sha(out / "last.pt")))
            _dump(out / "patient_slot_intervention.json", {
                "status": "measured", "arm": arm, "split": "validation",
                "input_scope": "prepared_patient_slots_final_checkpoint_not_invented_slots",
                "checkpoint_sha256": _sha(out / "last.pt"), "patient_ids": val_ids,
                "batches": interventions, "cross_held_fixed": True,
                "max_self_pool_change": max(row["max_self_pool_change"] for row in interventions),
                "self_interference": "observed" if any(row["self_interference"] == "observed" for row in interventions) else "not_observed"})
            if config["experiment"] == "original_audit":
                from .native_audit import run_final_audit
                _dump(out / "native_model_audit.json", run_final_audit(model, root, manifest, fold_data,
                    config, device, seed, repeats, _sha(out / "last.pt")))
            parameter_results = list(gradient["parameters"].values())
            summary = {"fold": fold, "seed": seed, "arm": arm, "epochs": config["epochs"], "optimizer_steps": steps,
                "checkpoint_sha256": _sha(out / "last.pt"), "parameter_count": parameter_count,
                "same_initial_state_sha256": initial_sha, "selector_parameter_count": len(parameter_results),
                "main_gradient_none_parameter_count": sum(not p["main"]["connected"] for p in parameter_results),
                "main_gradient_connected_parameter_count": sum(p["main"]["connected"] for p in parameter_results),
                "main_gradient_nonzero_parameter_count": sum(p["main"]["nonzero_elements"] > 0 for p in parameter_results),
                "main_gradient_zero_parameter_count": sum(p["main"]["connected"] and p["main"]["nonzero_elements"] == 0 for p in parameter_results),
                "main_gradient_finite": all(p["main"]["finite"] is not False for p in parameter_results),
                "disconnected_selector_status": "observed" if parameter_results and any(not p["main"]["connected"] for p in parameter_results) else "not_observed",
                "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "peak_memory_scope": "training_and_final_diagnostics" if device.type == "cuda" else "cpu_not_measured",
                "elapsed_seconds": time.perf_counter() - arm_start, **_repeat_summary(repeats, np)}
            _dump(out / "summary.json", summary)
            summaries.append(summary)
            del model, optimizer, scheduler, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
        # This evaluation-only control reads exactly the trained original
        # state, changes no parameters and performs zero optimizer steps.
        eval_summary = None
        if config["experiment"] == "reproduction":
            same_out = pair_root / "same_weights_eval_fix"
            same_out.mkdir()
            same_start = time.perf_counter()
            original_checkpoint = pair_root / "original/last.pt"
            digest = _sha(original_checkpoint)
            if digest != summaries[0]["checkpoint_sha256"]:
                raise RuntimeError("Original checkpoint changed before same-weights control")
            saved = torch.load(original_checkpoint, map_location="cpu", weights_only=True)
            model = build_model(config["source_root"], args, "eval_fix").to(device)
            model.load_state_dict(saved["state_dict"], strict=True)
            if any(not torch.equal(value.detach().cpu(), saved["state_dict"][name]) for name, value in model.state_dict().items()):
                raise RuntimeError("Original/eval_fix state dictionaries are not exactly equal")
            repeats = [_evaluate(model, root, manifest, fold_data, val_ids, config, device,
                       seed * 1000000 + 80000 + repeat, torch, np, concordance_index_censored)
                       for repeat in range(config["eval_repeats"])]
            _dump(same_out / "eval_repeats.json", repeats)
            _write_predictions(same_out / "validation_predictions.csv", manifest, fold_data, val_ids, repeats)
            eval_summary = {"fold": fold, "seed": seed, "arm": "same_weights_eval_fix",
                "trained_checkpoint_arm": "original", "training_epochs": saved["epochs"], "new_optimizer_steps": 0,
                "checkpoint_path": original_checkpoint.relative_to(root).as_posix(), "checkpoint_sha256": digest,
                "state_dict_exact_equality_before_evaluation": True,
                "change": "Deterministic slot initialization and stable evaluation selection; original readout retained",
                "readout": "Original dense cross and original mean-of-all self outputs retained",
                "main_selector_gradient_test_performed": False, "elapsed_seconds": time.perf_counter() - same_start,
                **_repeat_summary(repeats, np)}
            _dump(same_out / "summary.json", eval_summary)
            del model, saved
            if device.type == "cuda":
                torch.cuda.empty_cache()
        protocol["source_hashes_after"] = _source_hashes(config)
        protocol["source_unchanged"] = protocol["source_hashes_before"] == protocol["source_hashes_after"]
        if not protocol["source_unchanged"]:
            raise RuntimeError("Source changed during patient pair")
        if _sha(root / "prepared/manifest.json") != protocol["manifest_sha256"]:
            raise RuntimeError("Prepared manifest changed during patient pair")
        protocol["completed_utc"] = _now()
        protocol["elapsed_training_eval_seconds"] = time.perf_counter() - start
        _dump(pair_root / "protocol.json", protocol)
        result = {"schema_version": SCHEMA_VERSION, "fold": fold, "seed": seed, "status": "completed",
            "counts": fold_data["counts"], "epochs": config["epochs"], "source_unchanged": True,
            "scope": "Paired mechanism evidence; exploratory validation is not a generalization or OT claim",
            "arms": summaries, "same_weights_eval_fix": eval_summary,
            "checks": ({"original_disconnection": summaries[0]["disconnected_selector_status"],
                "evaluation_randomness": "observed" if summaries[0]["evaluation_randomness_observed"] else "not_observed",
                "self_interference": _read(pair_root / "original/patient_slot_intervention.json")["self_interference"],
                "padding_sensitivity": _read(pair_root / "original/native_model_audit.json")["padding"]["sensitivity"],
                "natural_gate_near_zero": _read(pair_root / "original/natural_gate_observations.json")["near_zero_gate"]}
                if config["experiment"] == "original_audit" else {
                "original_disconnection": summaries[0]["disconnected_selector_status"],
                "repair_all_selectors_connected": bool(summaries[1]["selector_parameter_count"] and summaries[1]["main_gradient_none_parameter_count"] == 0),
                "repair_all_selector_gradients_nonzero": bool(summaries[1]["selector_parameter_count"] and summaries[1]["main_gradient_nonzero_parameter_count"] == summaries[1]["selector_parameter_count"]),
                "eval_fix_exact_repeat_invariance": eval_summary["patient_risk_range_max"] == 0} if eval_summary else {
                "A_main_selector_disconnected": summaries[0]["main_gradient_connected_parameter_count"] == 0,
                "B_main_selector_disconnected": summaries[1]["main_gradient_connected_parameter_count"] == 0,
                "C_main_selector_connected": summaries[2]["main_gradient_none_parameter_count"] == 0,
                "all_arms_eval_repeat_invariance": all(row["patient_risk_range_max"] == 0 for row in summaries)}),
            "experiment": config["experiment"],
            "elapsed_seconds": time.perf_counter() - start}
        if config["experiment"] == "attribution_abc":
            result["contrasts"] = []
            for before, after in (("A", "B"), ("B", "C")):
                first = _read(pair_root / before / "eval_repeats.json")[0]
                second = _read(pair_root / after / "eval_repeats.json")[0]
                difference = np.asarray(second["risks"]) - np.asarray(first["risks"])
                result["contrasts"].append({"comparison": after + "_minus_" + before,
                    "question": "self_selected_pooling" if before == "A" else "fused_loss_selector_gradient",
                    "val_nll_difference": second["nll"] - first["nll"],
                    "val_cindex_difference": second["cindex"] - first["cindex"] if first["cindex"] is not None and second["cindex"] is not None else None,
                    "patient_ids": val_ids, "risk_differences": difference.tolist(),
                    "prediction_change": "observed" if bool(np.any(difference != 0)) else "not_observed",
                    "interpretation": "Matched training comparison at fixed final epoch; a short run is not a generalization or causal biology conclusion"})
        _dump(pair_root / "summary.json", result)
        return result
    except Exception as exc:
        _dump(pair_root / "failure.json", {"fold": fold, "seed": seed, "failed_utc": _now(),
            "exception": type(exc).__name__, "message": str(exc), "completed_arms": summaries,
            "scope": "Failed attempt retained; no mechanism or performance conclusion"})
        raise


def summarize(config, output_root):
    """Collect completed pairs without loading data tensors or training libraries."""
    config = _config(config)
    from .queue import training_arms
    arms = training_arms(config)
    root = Path(output_root)
    manifest = _manifest(config, root, verify_cache=False)
    complete, missing = [], []
    for fold in config["folds"]:
        for seed in config["seeds"]:
            path = root / "pairs" / f"fold_{fold}" / f"seed_{seed}" / "summary.json"
            if not path.exists():
                missing.append({"fold": fold, "seed": seed})
                continue
            value = _read(path)
            if value.get("status") != "completed" or value.get("fold") != fold or value.get("seed") != seed:
                raise RuntimeError(f"Invalid completed-pair summary: {path}")
            if value.get("epochs") != config["epochs"]:
                raise RuntimeError(f"Completed pair epoch budget differs: {path}")
            folder = path.parent
            protocol = _read(folder / "protocol.json")
            if protocol["arms"] != arms or [arm["arm"] for arm in value["arms"]] != arms:
                raise RuntimeError(f"Completed training arms differ: {path}")
            for key in ("epochs", "batch_size", "patches", "lr", "eval_repeats"):
                if protocol[key] != config[key]:
                    raise RuntimeError(f"Completed pair configuration differs ({key}): {path}")
            if protocol["runner_sha256"] != _sha(__file__) or protocol["model_controls_sha256"] != _sha(Path(__file__).with_name("model.py")):
                raise RuntimeError(f"Proof implementation changed after patient pair: {path}")
            if protocol["manifest_sha256"] != _sha(root / "prepared/manifest.json") or protocol["source_hashes_after"] != _source_hashes(config):
                raise RuntimeError(f"Prepared data/source changed after patient pair: {path}")
            shared = _sha(folder / "shared_initial_state.pt")
            expected_steps = math.ceil(len(manifest["folds"][str(fold)]["train"]) / config["batch_size"]) * config["epochs"]
            val_ids = manifest["folds"][str(fold)]["val"]
            for arm in value["arms"]:
                arm_root = folder / arm["arm"]
                if _sha(arm_root / "last.pt") != arm["checkpoint_sha256"] or arm["same_initial_state_sha256"] != shared:
                    raise RuntimeError(f"Checkpoint/initial-state provenance mismatch: {arm_root}")
                curve = _read(arm_root / "curve.json")
                repeats = _read(arm_root / "eval_repeats.json")
                if len(curve) != config["epochs"] or arm["optimizer_steps"] != expected_steps or len(repeats) != config["eval_repeats"]:
                    raise RuntimeError(f"Incomplete training/evaluation budget: {arm_root}")
                if any(len(r["risks"]) != len(val_ids) for r in repeats):
                    raise RuntimeError(f"Prediction/patient count mismatch: {arm_root}")
                with (arm_root / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
                    predictions = list(csv.DictReader(stream))
                if [r["patient_id"] for r in predictions] != val_ids:
                    raise RuntimeError(f"Prediction patient identities/order differ: {arm_root}")
                for index, row in enumerate(predictions):
                    if any(float(row[f"risk_repeat_{j:02d}"]) != r["risks"][index] for j, r in enumerate(repeats)):
                        raise RuntimeError(f"CSV/JSON patient risks differ: {arm_root}")
            if config["experiment"] == "reproduction" and (value["same_weights_eval_fix"]["checkpoint_sha256"] != value["arms"][0]["checkpoint_sha256"] or value["same_weights_eval_fix"]["new_optimizer_steps"] != 0):
                raise RuntimeError(f"Evaluation-only control did not reuse the original weights: {path}")
            complete.append(value)
    result = {"schema_version": SCHEMA_VERSION, "generated_utc": _now(),
        "status": "completed" if not missing else "incomplete", "expected_pairs": len(config["folds"]) * len(config["seeds"]),
        "completed_pairs": len(complete), "missing_pairs": missing, "epochs": config["epochs"],
        "counts": manifest["counts"], "results": complete,
        "scope": "Mechanism proof on paired folds/seeds; validation metrics are exploratory; no OT necessity or generalization improvement established",
        "experiment": config["experiment"],
        "checks": ({key + "_observed_pairs": sum(x["checks"][key] == "observed" for x in complete)
            for key in ("original_disconnection", "evaluation_randomness", "self_interference", "padding_sensitivity", "natural_gate_near_zero")}
            if config["experiment"] == "original_audit" else {"original_disconnection_observed_pairs": sum(x["checks"]["original_disconnection"] == "observed" for x in complete),
            "original_disconnection_not_observed_pairs": sum(x["checks"]["original_disconnection"] == "not_observed" for x in complete),
            "repair_all_selectors_connected_pairs": sum(x["checks"]["repair_all_selectors_connected"] for x in complete),
            "repair_all_selector_gradients_nonzero_pairs": sum(x["checks"]["repair_all_selector_gradients_nonzero"] for x in complete),
            "eval_fix_exact_repeat_invariance_pairs": sum(x["checks"]["eval_fix_exact_repeat_invariance"] for x in complete)} if "original" in arms else {
                key + "_pairs": sum(x["checks"][key] for x in complete) for key in (
                    "A_main_selector_disconnected", "B_main_selector_disconnected", "C_main_selector_connected", "all_arms_eval_repeat_invariance")})}
    result["validation_metrics_by_arm"] = {}
    for arm in arms:
        rows = [row for pair in complete for row in pair["arms"] if row["arm"] == arm]
        scores = [row["val_cindex_mean_exploratory"] for row in rows if row["val_cindex_mean_exploratory"] is not None]
        result["validation_metrics_by_arm"][arm] = {"completed_checkpoints": len(rows),
            "mean_validation_cindex_exploratory": sum(scores) / len(scores) if scores else None,
            "defined_cindex_pairs": len(scores),
            "warning": "Cross-fold/seed mean is descriptive; pairs share patients and are not independent evidence units"}
    _dump(root / "patient_summary.json", result)
    return result
