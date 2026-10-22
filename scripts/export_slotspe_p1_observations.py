"""Export a P1-format native slot observation root from a local trained pair.

The historical Windows P1 root (E:\\.review\\slotspe_p1_20261020) was never
migrated to this host, so no P1-format observations exist here. This script
re-derives an equivalent export from the current local trained checkpoint and
the current prepared cache. It is a NEW local observation, not the historical
one, and it is labelled with its own input_scope.

The original executable source is read only. No training, no optimizer, no
model repair, no source patch, and no new slot generation. The native
observer only reads values that the native forward already computed.

Patient features, weights, predictions, identifiers and per-patient arrays
stay inside the ignored local_evidence directory and are never staged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def plan(run_root, fold, seed):
    run_root = Path(run_root).resolve()
    pair = run_root / "pairs" / f"fold_{fold}" / f"seed_{seed}" / "original"
    names = ["prepared/manifest.json", f"pairs/fold_{fold}/seed_{seed}/original/last.pt",
             f"pairs/fold_{fold}/seed_{seed}/original/eval_repeats.json",
             f"pairs/fold_{fold}/seed_{seed}/original/summary.json"]
    missing = [name for name in names if not (run_root / name).is_file()]
    return {"schema": "slotspe_p1_export_plan_v1", "run_root": str(run_root), "fold": fold, "seed": seed,
            "ready": not missing, "missing": missing, "training_runs": 0, "optimizer_steps": 0,
            "new_slot_generation": 0, "default_action": "plan_only",
            "scope": "Re-derive a P1-format native slot observation root from the current local trained checkpoint"}


def export(run_root, output, fold=0, seed=3, device="cuda:0", batch_size=8):
    import numpy as np
    import torch
    from torch.torch_version import TorchVersion
    from survot_rank.research.slotspe_proof.model import build_model, make_args
    from survot_rank.research.slotspe_proof.native_audit import model_purity, _state_digest
    from survot_rank.research.slotspe_proof.p1_diagnostics import NativeSlotObserver, array_statistics

    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    if not output.is_relative_to((REPO / "local_evidence").resolve()):
        raise ValueError("Output must be a fresh directory inside ignored local_evidence")
    if output.exists():
        raise ValueError("A fresh output directory is required")
    pair_root = run_root / "pairs" / f"fold_{fold}" / f"seed_{seed}"
    pair = pair_root / "original"
    manifest_path = run_root / "prepared" / "manifest.json"
    checkpoint = pair / "last.pt"
    manifest = read(manifest_path)
    summary = read(pair / "summary.json")
    repeats = read(pair / "eval_repeats.json")
    if not repeats or len(repeats) != summary["repeats"]:
        raise ValueError("Historical repeat budget differs from the recorded summary")
    if summary["arm"] != "original" or summary["fold"] != fold or summary["seed"] != seed:
        raise ValueError("Pair identity differs from the requested fold/seed")
    source = Path(manifest["source_root"])

    watched = {str(manifest_path): sha(manifest_path), str(checkpoint): sha(checkpoint),
               str(pair / "eval_repeats.json"): sha(pair / "eval_repeats.json"),
               str(pair / "summary.json"): sha(pair / "summary.json"),
               str(pair_root / "protocol.json"): sha(pair_root / "protocol.json")}
    for path, digest in watched.items():
        if sha(path) != digest:
            raise ValueError("Historical artifact changed: " + path)

    fold_data = manifest["folds"][str(fold)]
    val_ids = list(fold_data["val"])
    if len(val_ids) != fold_data["counts"]["val"]:
        raise ValueError("Validation coverage differs from the prepared manifest")
    args = make_args(manifest["omic_sizes"], encoding_dim=manifest["encoding_dim"])
    with torch.serialization.safe_globals([TorchVersion]):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if (saved["arm"] != "original" or saved["seed"] != seed
            or saved["optimizer_steps"] != summary["optimizer_steps"]
            or saved["checkpoint_selection"] != "fixed_final_epoch"):
        raise ValueError("Trained native checkpoint identity differs")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested device unavailable; no silent device fallback")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    model = build_model(str(source), args, "original").to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    del saved
    model.eval()
    purity = model_purity(model, source)
    before_state = _state_digest(model)

    def cache_of(pid):
        item = torch.load(run_root / manifest["patients"][pid]["cache_path"],
                          map_location="cpu", weights_only=True, mmap=True)
        if item["patient_id"] != pid:
            raise ValueError("Prepared cache patient identity differs")
        return item

    def batch(ids):
        items = [cache_of(pid) for pid in ids]
        value = {"x_wsi": torch.stack([i["wsi"] for i in items]).to(device),
                 "y": torch.tensor([fold_data["labels"][pid] for pid in ids], dtype=torch.long, device=device),
                 "c": torch.tensor([manifest["patients"][pid]["censor"] for pid in ids], dtype=torch.float32, device=device),
                 "omic_missing": False}
        for k in range(len(manifest["omic_sizes"])):
            value[f"x_omic{k + 1}"] = torch.stack([i["omics"][k] for i in items]).to(device)
        return value

    def reset_rng(value):
        random.seed(value)
        np.random.seed(value % (2 ** 32))
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)

    def rng_digest():
        state = np.random.get_state()
        return {"python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                "numpy": hashlib.sha256(state[0].encode() + state[1].tobytes() + repr(state[2:]).encode()).hexdigest(),
                "torch_cpu": hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
                "torch_cuda": [hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest() for s in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else []}

    eval_seed = repeats[0]["eval_seed"]
    reset_rng(eval_seed)
    baseline = []
    with torch.no_grad():
        for start in range(0, len(val_ids), batch_size):
            baseline.append(model(**batch(val_ids[start:start + batch_size]))[0].cpu())
    baseline = torch.cat(baseline)
    native_rng = rng_digest()
    if not bool(torch.isfinite(baseline).all()) or not np.isfinite(np.asarray(repeats[0]["logits"], dtype=float)).all():
        raise RuntimeError("Nonfinite historical or native predictions")
    replay_error = float(np.max(np.abs(baseline.numpy().astype(float) - np.asarray(repeats[0]["logits"], dtype=float))))
    if replay_error > 1e-6:
        raise RuntimeError(f"Recorded same-RNG replay failed: {replay_error}")

    output.mkdir(parents=True)
    (output / "raw").mkdir()
    observer = NativeSlotObserver(model)
    reset_rng(eval_seed)
    observed, rows = [], []
    max_update_error = 0.0
    try:
        with torch.no_grad():
            for start in range(0, len(val_ids), batch_size):
                ids = val_ids[start:start + batch_size]
                logits = model(**batch(ids))[0].cpu()
                observed.append(logits)
                max_update_error = max(max_update_error, observer.maximum_update_error())
                for local, pid in enumerate(ids):
                    arrays = observer.arrays(local)
                    name = "raw/" + hashlib.sha256(pid.encode()).hexdigest() + ".npz"
                    np.savez_compressed(output / name, **arrays)
                    rows.append({"patient_id": pid, "manifest_index": manifest["patients"][pid].get("manifest_index", start + local),
                                 "raw_path": name, "raw_sha256": sha(output / name),
                                 "logits": logits[local].tolist(),
                                 "wsi": array_statistics(arrays, "wsi"), "omic": array_statistics(arrays, "omic")})
    finally:
        observer.close()
    observed = torch.cat(observed)
    observed_rng = rng_digest()
    observer_error = float(np.max(np.abs(observed.numpy().astype(float) - baseline.numpy().astype(float))))
    state_unchanged = _state_digest(model) == before_state
    if observer_error > 1e-6 or max_update_error > 1e-6 or not state_unchanged or native_rng != observed_rng:
        raise RuntimeError("Observer altered predictions, state or RNG consumption")
    if any(sha(path) != digest for path, digest in watched.items()):
        raise RuntimeError("Original artifact changed during the read-only export")

    patients = [{"patient_id": pid, "time": manifest["patients"][pid]["time"],
                 "censor": manifest["patients"][pid]["censor"], "label": fold_data["labels"][pid],
                 "split": "val", "index": index}
                for index, pid in enumerate(val_ids)]
    write(output / "patient_provenance.json", {"patients": patients, "omic_sizes": manifest["omic_sizes"],
                                               "binning": fold_data["binning"], "input_csv_sha256": manifest["input_csv_sha256"]})
    report = {"schema": "slotspe_p1_observations_v2", "status": "measured",
              "input_scope": "local_trisurv_gpu_current_training_run_real_BLCA_cached_validation",
              "server_SPE09_evidence": False, "fold": fold, "seed": seed, "device": device,
              "torch": str(torch.__version__), "epochs": summary["epochs"],
              "prior_optimizer_steps": summary["optimizer_steps"], "new_optimizer_steps": 0,
              "patient_count": len(val_ids), "patches": manifest["patches"], "eval_seed": eval_seed,
              "model_purity": purity, "historical_replay_max_logit_difference": replay_error,
              "observer_max_logit_difference": observer_error,
              "reconstructed_native_update_max_error": max_update_error,
              "replay": {"historical_reference_logits": repeats[0]["logits"], "native_logits": baseline.tolist(),
                         "observed_logits": observed.tolist(),
                         "native_final_rng_sha256": native_rng, "observed_final_rng_sha256": observed_rng},
              "checkpoint_tensors_unchanged": state_unchanged, "original_inputs_unchanged": True,
              "source_inputs_sha256": watched,
              "candidate_counts": {m: sorted({r[m]["candidate_slots"] for r in rows}) for m in ("wsi", "omic")},
              "selected_counts": {m: sorted({r[m]["selected_slots"] for r in rows}) for m in ("wsi", "omic")},
              "interpretation": "Native fixed-count and descriptive coverage observations only. No prognostic event count, semantic label, harmful redundancy or patient benefit is inferred.",
              "P1_fixed_count_predictive_harm": "not_established", "P2_functional_redundancy": "not_established",
              "historical_equivalent": False,
              "historical_equivalence_note": "Re-derived from the current local trained checkpoint; the migrated-away Windows P1 root was not available for byte comparison.",
              "recorded_utc": datetime.now(timezone.utc).isoformat(), "records": rows}
    write(output / "observations.json", report)
    files = [p for p in sorted(output.rglob("*")) if p.is_file()]
    write(output / "receipt.json", {"schema": "slotspe_p1_receipt_v1",
                                    "outputs": {p.relative_to(output).as_posix(): sha(p) for p in files},
                                    "observer_source_sha256": sha(REPO / "survot_rank/research/slotspe_proof/p1_diagnostics.py"),
                                    "exporter_source_sha256": sha(Path(__file__).resolve()),
                                    "original_artifacts_unchanged": True})
    return {"status": "completed", "output": str(output), "patients": len(val_ids),
            "replay_error": replay_error, "observer_error": observer_error,
            "max_update_error": max_update_error, "state_unchanged": state_unchanged,
            "new_training_steps": 0, "new_optimizer_steps": 0, "new_slot_generation": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true")
    group.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if args.run:
        if args.run_root is None or args.output_dir is None:
            parser.error("--run requires --run-root and a fresh --output-dir")
        result = export(args.run_root, args.output_dir, args.fold, args.seed, args.device, args.batch_size)
    else:
        if args.run_root is None:
            parser.error("--run-root is required")
        result = plan(args.run_root, args.fold, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
