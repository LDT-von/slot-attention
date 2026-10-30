"""Four new loss arms, reusing audited S0/R4/R3 best-protocol results.

This entry point never prepares new data or trains a reference arm. Missing or
incompatible reference artifacts stop execution. Planning imports no ML runtime.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/slotspe_diversity_queue.json"
NEW_ARMS = ("SCM", "SRO", "R4_SCM", "R4_SRO")
REUSE_ARMS = ("S0", "R4", "R3")
CHECKPOINT_SELECTION = "per_epoch_validation_best_then_reload_and_reevaluate"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def private_output(value):
    path, private = _path(value), (ROOT / "local_evidence").resolve()
    if path == private or not path.is_relative_to(private):
        raise ValueError("output must be a dedicated ignored local_evidence subdirectory")
    return path


def validate_config(config):
    config = json.loads(json.dumps(config))
    if config.get("schema") != "slotspe_diversity_queue_v1":
        raise ValueError("unsupported diversity queue schema")
    if config.get("new_arms") != list(NEW_ARMS) or config.get("reuse_arms") != list(REUSE_ARMS):
        raise ValueError("exactly four new arms and S0/R4/R3 references are required")
    if config.get("folds") != [0, 1, 2, 3, 4]:
        raise ValueError("scope is five folds, seed3 only")
    expected = {"study": "blca", "feature_encoder": "UNI2-h", "seeds": [3],
                "epochs": 30, "patches": 4096, "batch_size": 8, "eval_repeats": 20,
                "lr": 0.0005, "probe_lr": 0.0005, "projection_dim": 256,
                "slot_iters": 10, "content_target_dim": 32, "content_clusters": 8,
                "content_reference_seed": 731, "checkpoint_selection": CHECKPOINT_SELECTION}
    if config.get("required_protocol") != expected:
        raise ValueError("reference protocol differs from the declared BLCA best comparison")
    if config.get("split_policy") != "inherit_verified_reference_config_and_prepared_split_hashes":
        raise ValueError("split identity must come from the verified completed reference")
    if config.get("content_weight") != 5.0:
        raise ValueError("R4 content weight must remain 5 to compare with existing R4")
    options = config.get("diversity", {})
    if set(options) != {"cosine_margin", "read_overlap_margin", "weight", "warmup_epochs"}:
        raise ValueError("unexpected diversity parameters")
    for name in ("cosine_margin", "read_overlap_margin"):
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be finite in [0,1)")
    weight, warmup = options["weight"], options["warmup_epochs"]
    if isinstance(weight, bool) or not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup <= 30:
        raise ValueError("warmup_epochs must be an integer in [0,30]")
    devices = config.get("devices", [])
    if not devices or len(devices) != len(set(devices)) or any(not re.fullmatch(r"cuda:\d+", d) for d in devices):
        raise ValueError("devices must be distinct explicit cuda indices; no CPU patient fallback")
    return config


def make_plan(config):
    config = validate_config(config)
    reference, output = _path(config["reference_root"]), private_output(config["output_root"])
    if output == reference or output.is_relative_to(reference) or reference.is_relative_to(output):
        raise ValueError("new output and reference must be separate nonnested directories")
    return {"schema": "slotspe_diversity_plan_v1", "default_action": "plan_only",
            "patient_work_executed": False, "reference_root": str(reference),
            "output_root": str(output), "new_training_runs": 20, "reused_arm_fold_results": 15,
            "training_arms": list(NEW_ARMS), "reference_arms": list(REUSE_ARMS),
            "folds": config["folds"], "seeds": [3], "protocol": config["required_protocol"],
            "diversity": config["diversity"], "content_weight": 5.0,
            "parameter_status": config.get("parameters_status"),
            "split_identity": "pending audit; do not infer from a directory name",
            "devices": config["devices"], "max_concurrent_arms": len(config["devices"]),
            "reuse_requires": "completed same-protocol artifacts, matching inputs/source/initialization; failure never retrains references",
            "stages": [{"id": f"fold{fold}_{arm}", "fold": fold, "arm": arm}
                       for fold in config["folds"] for arm in NEW_ARMS]}


def implementation_hashes():
    files = list((ROOT / "slotspe_proof").glob("*.py"))
    files += list((ROOT / "third_party/SlotSPE").rglob("*.py"))
    files += [ROOT / "scripts/run_slotspe_diversity.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def audit(config):
    from .diversity_reuse import audit_reference
    make_plan(config)
    return audit_reference(_path(config["reference_root"]), config["required_protocol"], ROOT)


def _runtime(config, receipt, device):
    runtime = dict(receipt["runtime"])
    # Preserve the exact prepared source/data paths; the reuse audit verifies
    # their bytes. Relocating an old receipt is not a silent path rewrite.
    runtime.update(device=device, variants=list(NEW_ARMS), diversity=dict(config["diversity"]))
    return runtime


def _identity(config, receipt):
    payload = {"config": config, "implementation": implementation_hashes(),
               "reference_file_hashes": receipt["file_hashes"],
               "reference_runtime": receipt["runtime"],
               "reference_root": receipt["reference_root"],
               "reference_manifest_path": receipt["manifest_path"],
               "expected_initial_state_hashes": receipt.get("expected_initial_state_hashes")}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest, payload


def _verify_files(hashes, root):
    for name, expected in hashes.items():
        path = Path(name)
        path = path if path.is_absolute() else Path(root) / path
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"pinned file changed: {name}")


def _arm_hashes(folder):
    required = ("summary.json", "curve.json", "eval_repeats.json", "epoch_validation.json",
                "last.pt", "best.pt", "diagnostics.pt", "validation_predictions.csv")
    if any(not (folder / name).is_file() for name in required):
        raise RuntimeError(f"arm artifacts incomplete: {folder.name}")
    if read(folder / "summary.json").get("status") != "completed":
        raise RuntimeError(f"arm not completed: {folder.name}")
    return {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}


def _open_attempt_log(root, stage_id):
    """Keep failed-launch logs when a stage has not yet created arm output."""
    folder = root / "logs"
    folder.mkdir(exist_ok=True)
    attempt = 1
    while True:
        suffix = "" if attempt == 1 else f".attempt{attempt}"
        path = folder / (stage_id + suffix + ".log")
        try:
            return path.open("x", encoding="utf-8"), path.relative_to(root).as_posix()
        except FileExistsError:
            attempt += 1


def worker(root, stage_id, device):
    root = private_output(root)
    state = read(root / "queue_manifest.json")
    if state.get("status") != "running" or state["records"].get(stage_id, {}).get("status") != "running":
        raise RuntimeError("worker requires an explicitly active queue stage")
    config = validate_config(read(root / "queue_config.json"))
    receipt = read(root / "reference_receipt.json")
    identity, payload = _identity(config, receipt)
    if identity != state["identity"] or payload != state["identity_payload"]:
        raise RuntimeError("worker configuration or source drift")
    _verify_files(receipt["file_hashes"], receipt["reference_root"])
    stage = next((s for s in make_plan(config)["stages"] if s["id"] == stage_id), None)
    if stage is None or device not in config["devices"]:
        raise ValueError("unknown arm stage or device")
    # All patient tensor deserialization is below the --run-only worker gate.
    import torch
    from .diversity_engine import run_diversity_arm
    from .redundancy_content import ContentReference
    manifest = read(receipt["manifest_path"])
    runtime = _runtime(config, receipt, device)
    folder = Path(receipt["reference_root"]) / "folds" / str(stage["fold"]) / "initial"
    reference = ContentReference(manifest["omic_sizes"], manifest["encoding_dim"],
                                 target_dim=runtime["content_target_dim"], clusters=runtime["content_clusters"],
                                 seed=runtime["content_reference_seed"])
    reference.load_state_dict(torch.load(folder / "reference.pt", map_location="cpu", weights_only=True))
    core = torch.load(folder / "core.pt", map_location="cpu", weights_only=True)
    probe = torch.load(folder / "probe.pt", map_location="cpu", weights_only=True)
    from .redundancy_engine import state_hash
    expected = receipt["expected_initial_state_hashes"][str(stage["fold"])]
    if state_hash(core) != expected["core"] or state_hash(probe) != expected["probe"]:
        raise RuntimeError("loaded initial states differ from reused reference arms")
    run_diversity_arm(runtime, receipt["reference_root"], manifest, stage["arm"], core, probe, reference,
                      fold=stage["fold"], output_root=root / "folds" / str(stage["fold"]))


def _stats(values):
    values = list(values)
    if len(values) != 5 or any(v is None or not math.isfinite(v) for v in values):
        raise RuntimeError("five defined fold values are required")
    return {"fold_mean": statistics.mean(values), "fold_std_population": statistics.pstdev(values),
            "fold_std_sample": statistics.stdev(values), "folds_defined": 5}


def summarize(root, config, receipt):
    """Summaries preserve the existing unmasked diagnostics for comparability."""
    from .redundancy_multifold import _arm_view
    views, per_arm = {}, {}
    for fold in config["folds"]:
        views[str(fold)] = {}
        for arm in REUSE_ARMS + NEW_ARMS:
            source = Path(receipt["reference_root"]) if arm in REUSE_ARMS else root
            folder = source / "folds" / str(fold) / "arms" / arm
            summary = read(folder / "summary.json")
            if summary.get("status") != "completed":
                raise RuntimeError("cannot summarize incomplete arm")
            view = _arm_view(summary, read(folder / "curve.json"))
            view.update(result_origin="reused_verified_reference" if arm in REUSE_ARMS else "new_training",
                        best_epoch=summary["best_epoch"])
            views[str(fold)][arm] = view
    for arm in REUSE_ARMS + NEW_ARMS:
        scores = [views[str(f)][arm]["cindex_mean_over_repeats"] for f in config["folds"]]
        entry = {"cindex_per_fold": scores, "cindex_five_fold": _stats(scores),
                 "result_origin": views["0"][arm]["result_origin"], "geometry": {}}
        for modality in ("wsi", "omic"):
            entry["geometry"][modality] = {}
            for metric in ("cosine", "effective_rank", "read_overlap", "slot_count"):
                values = [(views[str(f)][arm].get("slot_diagnostics") or {}).get(modality, {}).get(metric)
                          for f in config["folds"]]
                entry["geometry"][modality][metric] = {"per_fold": values,
                    "mean": statistics.mean(values) if all(v is not None for v in values) else None}
        per_arm[arm] = entry
    contrasts = {}
    for a, b in (("R4", "S0"), ("R3", "R4"), ("SCM", "S0"), ("SRO", "S0"),
                 ("R4_SCM", "R4"), ("R4_SRO", "R4"), ("R4_SCM", "SCM"), ("R4_SRO", "SRO")):
        differences = [x-y for x, y in zip(per_arm[a]["cindex_per_fold"], per_arm[b]["cindex_per_fold"])]
        contrasts[a + "_minus_" + b] = {"per_fold": differences, **_stats(differences)}
    result = {"schema": "slotspe_diversity_summary_v1", "status": "completed", "folds": views,
              "per_arm": per_arm, "paired_contrasts": contrasts,
              "new_training_runs": 20, "reused_arm_fold_results": 15,
              "checkpoint_selection": CHECKPOINT_SELECTION,
              "splits_dir": receipt["runtime"]["splits_dir"],
              "claim_boundary": "single-seed validation-selected comparison; no independent-test or clinical claim",
              "reading_metrics": "historical read_overlap remains unchanged; SRO's masked Bhattacharyya training term is separately named in curves"}
    write(root / "summary.json", result)
    lines = ["arm\torigin\tC-index five-fold mean\tstd_population"]
    for arm, entry in per_arm.items():
        stats = entry["cindex_five_fold"]
        lines.append(f"{arm}\t{entry['result_origin']}\t{stats['fold_mean']:.6f}\t{stats['fold_std_population']:.6f}")
    (root / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def execute(config, resume=False):
    plan = make_plan(config)
    root = Path(plan["output_root"])
    receipt = audit(config)
    identity, payload = _identity(config, receipt)
    if root.exists() and any(root.iterdir()):
        if not resume or not (root / "queue_manifest.json").is_file():
            raise RuntimeError("nonempty output; choose a fresh root or verified --resume")
        state = read(root / "queue_manifest.json")
        if state["identity"] != identity or state["identity_payload"] != payload:
            raise RuntimeError("resume identity drift; preserve previous output")
    else:
        root.mkdir(parents=True, exist_ok=True)
        state = {"schema": "slotspe_diversity_execution_v1", "status": "ready", "identity": identity,
                 "identity_payload": payload, "records": {}, "started_at": datetime.now(timezone.utc).isoformat()}
        write(root / "queue_config.json", config)
        write(root / "reference_receipt.json", receipt)
        write(root / "queue_manifest.json", state)
    lock = root / ".queue.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    running, free_devices = {}, list(config["devices"])
    try:
        # Verify full cached bytes once under explicit --run, never in --plan.
        from . import patient
        patient._manifest(_runtime(config, receipt, config["devices"][0]),
                          Path(receipt["reference_root"]), verify_cache=True)
        pending = []
        for stage in plan["stages"]:
            folder = root / "folds" / str(stage["fold"]) / "arms" / stage["arm"]
            old = state["records"].get(stage["id"])
            if old and old.get("status") == "completed":
                if old["artifact_sha256"] != _arm_hashes(folder):
                    raise RuntimeError("completed arm artifacts changed")
            elif folder.exists() and any(folder.iterdir()):
                raise RuntimeError("partial arm output retained; use a new output root (no silent overwrite)")
            else:
                pending.append(stage)
        needed = (len(pending) * 700 * 1024**2) + 5 * 1024**3
        if shutil.disk_usage(root).free < needed:
            raise RuntimeError("insufficient free disk for new arms and 5 GiB headroom")
        state["status"] = "running"
        write(root / "queue_manifest.json", state)
        while pending or running:
            while pending and free_devices:
                if implementation_hashes() != payload["implementation"]:
                    raise RuntimeError("implementation changed during execution")
                stage, device = pending.pop(0), free_devices.pop(0)
                stream, log_name = _open_attempt_log(root, stage["id"])
                previous = state["records"].get(stage["id"], {})
                attempts = list(previous.get("attempt_logs", [])) + [log_name]
                state["records"][stage["id"]] = {"status": "running", "device": device,
                                                 "attempt_logs": attempts}
                write(root / "queue_manifest.json", state)
                command = [sys.executable, "-B", "-m", "slotspe_proof.diversity_queue", "--run",
                           "--worker-stage", stage["id"], "--output-root", str(root), "--worker-device", device]
                try:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1"),
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                except BaseException:
                    stream.close()
                    raise
                running[stage["id"]] = (process, stream, stage, device)
            for sid, (process, stream, stage, device) in list(running.items()):
                if process.poll() is None:
                    continue
                stream.close()
                del running[sid]
                free_devices.append(device)
                if process.returncode:
                    state["records"][sid]["status"] = "failed"
                    raise RuntimeError(f"{sid} failed; inspect its retained log")
                folder = root / "folds" / str(stage["fold"]) / "arms" / stage["arm"]
                state["records"][sid].update(status="completed", artifact_sha256=_arm_hashes(folder))
                write(root / "queue_manifest.json", state)
            if running:
                time.sleep(0.5)
        _verify_files(receipt["file_hashes"], receipt["reference_root"])
        result = summarize(root, config, receipt)
        state["status"] = "completed"
        write(root / "queue_manifest.json", state)
        return {"status": result["status"], "new_training_runs": 20, "output_root": str(root)}
    except BaseException as exc:
        state.update(status="failed", error=str(exc))
        write(root / "queue_manifest.json", state)
        raise
    finally:
        for process, stream, _, _ in running.values():
            if process.poll() is None:
                process.terminate()
            process.wait()
            stream.close()
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--audit", action="store_true", help="read metadata and hashes only; no model/tensor loading")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--reference-root")
    parser.add_argument("--output-root")
    parser.add_argument("--devices", help="comma-separated explicit devices, e.g. cuda:0,cuda:1; one arm per device")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-stage", help=argparse.SUPPRESS)
    parser.add_argument("--worker-device", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_stage:
        if not args.run or not args.output_root or not args.worker_device:
            parser.error("worker requires --run and an active queue")
        worker(args.output_root, args.worker_stage, args.worker_device)
        return
    config = read(args.config)
    for key in ("reference_root", "output_root"):
        if getattr(args, key):
            config[key] = getattr(args, key)
    if args.devices:
        config["devices"] = args.devices.split(",")
    config = validate_config(config)
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    if args.run:
        result = execute(config, args.resume)
    elif args.audit:
        receipt = audit(config)
        result = {"status": "reuse_verified", "patient_work_executed": False,
                  "reference_arms": list(REUSE_ARMS), "reused_arm_fold_results": 15,
                  "splits_dir": receipt["runtime"]["splits_dir"],
                  "checks": receipt["checks"], "next_action": "review before explicitly authorized --run"}
    else:
        result = make_plan(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
