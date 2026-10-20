"""Plan and explicitly execute one independent SlotSPE proof queue.

Planning uses only the standard library and never reads patient inputs. Every
execution stage runs in a separate process with its own log. Completed stages
are reusable only after configuration, source, input CSV and output hashes match.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/slotspe_proof_server.json"
REQUIRED_SOURCE = ("models/SlotSPE.py", "models/slot_attention.py",
    "models/transformer.py", "models/omics_encoder.py", "utils/loss_func.py",
    "utils/core_utils.py", "utils/process_args.py", "dataset/dataset_survival.py")
DEFAULTS = {"source_root": "third_party/SlotSPE", "study": "blca",
    "label_col": "survival_months_dss", "signature": "combine",
    "folds": [0, 1, 2, 3, 4], "seeds": [3, 17, 29], "epochs": 30,
    "patches": 4096, "batch_size": 8, "eval_repeats": 20,
    "encoding_dim": None, "device": "cuda:0", "lr": 0.0005, "torch_threads": 2,
    "experiment": "reproduction"}
PATH_KEYS = ("source_root", "metadata_root", "features_root")
PACKAGE = "survot_rank.research.slotspe_proof"


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    else:
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(encoded, encoding="utf-8")
        temp.replace(path)


def normalize_config(values, base=REPO_ROOT):
    unknown = set(values) - set(DEFAULTS) - {"metadata_root", "features_root"}
    if unknown:
        raise ValueError(f"Unknown proof settings: {sorted(unknown)}")
    config = {**DEFAULTS, **values}
    if config["experiment"] not in ("reproduction", "attribution_abc", "original_audit"):
        raise ValueError("experiment must be reproduction, attribution_abc or original_audit")
    for key in PATH_KEYS:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} is required; supply the server path")
        value = os.path.expandvars(os.path.expanduser(value))
        config[key] = str((Path(base) / value).resolve()) if not Path(value).is_absolute() else str(Path(value).resolve())
    for key in ("epochs", "patches", "batch_size", "eval_repeats", "torch_threads"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["eval_repeats"] < 2:
        raise ValueError("eval_repeats must be at least 2 to measure variation")
    for key in ("folds", "seeds"):
        value = config[key]
        if not isinstance(value, list) or not value or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in value):
            raise ValueError(f"{key} must contain nonnegative integers")
        if len(value) != len(set(value)):
            raise ValueError(f"{key} contains duplicates")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(config["study"])):
        raise ValueError("study must be a cohort name, not a path")
    if config["label_col"] not in {"survival_months_dss", "survival_months_os", "survival_months_pfi", "survival_months_dfs"}:
        raise ValueError("Unsupported survival endpoint")
    if config["signature"] not in {"all", "six", "hallmarks", "combine", "xena"}:
        raise ValueError("Unsupported signature")
    if config["encoding_dim"] is not None and (isinstance(config["encoding_dim"], bool) or not isinstance(config["encoding_dim"], int) or config["encoding_dim"] < 1):
        raise ValueError("encoding_dim must be positive or null (infer from real features)")
    if isinstance(config["lr"], bool) or not isinstance(config["lr"], (int, float)) or not math.isfinite(float(config["lr"])) or float(config["lr"]) <= 0:
        raise ValueError("lr must be positive and finite")
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", str(config["device"])):
        raise ValueError("device must be cpu or cuda[:index]")
    return config


def make_plan(config, output_root, checks_only=False):
    stages = [{"id": "protocol", "kind": "protocol"},
              {"id": "reconstruction", "kind": "reconstruction"},
              {"id": "mechanisms", "kind": "mechanisms"}]
    if config["experiment"] == "original_audit":
        # This path never constructs a repaired or synthetic substitute model.
        stages = [{"id": "protocol", "kind": "protocol"}]
    if config["experiment"] == "attribution_abc":
        stages += [{"id": "attribution", "kind": "attribution"}]
    if not checks_only:
        stages += [{"id": "prepare", "kind": "prepare"}]
        stages += [{"id": f"fold_{fold}_seed_{seed}", "kind": "pair", "fold": fold, "seed": seed}
                   for fold in config["folds"] for seed in config["seeds"]]
    stages += [{"id": "summarize", "kind": "summarize"}]
    unresolved = sorted({name for key in PATH_KEYS for match in re.findall(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)", config[key])
                         for name in match if name})
    return {"schema": "slotspe_proof_queue_v1", "task_name": "slotspe_problem_proof",
        "artifact_contract": ("slotspe_clean_original_v1" if config["experiment"] == "original_audit" else "slotspe_abc_patient_intervention_v2"),
        "config": config, "output_root": str(Path(output_root).resolve()), "checks_only": checks_only,
        "unresolved_environment_variables": unresolved, "stages": stages,
        "training_arms": training_arms(config),
        "training_runs": 0 if checks_only else len(training_arms(config)) * len(config["folds"]) * len(config["seeds"]),
        "same_weights_eval_controls": 0 if checks_only or config["experiment"] != "reproduction" else len(config["folds"]) * len(config["seeds"]),
        "checkpoint_selection": "fixed_final_epoch_no_validation_selection",
        "patient_work_executed": False, "transport_added": False,
        "scope": "Independent task for this window; proof controls are not a transport-plugin performance experiment"}


def training_arms(config):
    if config.get("experiment") == "original_audit":
        return ["original"]
    return ["A", "B", "C"] if config.get("experiment") == "attribution_abc" else ["original", "readout_fix"]


def metadata_paths(config, training=True):
    root = Path(config["metadata_root"])
    study, signature = config["study"], config["signature"]
    signature = "signatures.csv" if signature == "six" else ("combine_signatures.csv" if signature == "all" else signature + "_signatures.csv")
    files = [root / "clinical/all" / (study + ".csv"),
        *[root / "splits/5fold" / study / f"fold_{fold}.csv" for fold in config["folds"]]]
    if training:
        files += [root / "raw_rna_data_inter" / (study + "_rna_inter.csv"), root / "signatures" / signature]
    return files


def preflight(config, checks_only=False, check_runtime=True):
    for key in (PATH_KEYS if not checks_only else PATH_KEYS[:2]):
        if "$" in config[key]:
            raise ValueError(f"Unresolved environment variable in {key}; provide a real server path")
    files = [Path(config["source_root"]) / rel for rel in REQUIRED_SOURCE]
    files += metadata_paths(config, training=not checks_only)
    missing = [str(path) for path in files if not path.is_file()]
    if not checks_only and not Path(config["features_root"]).is_dir():
        missing.append(config["features_root"])
    if missing:
        raise FileNotFoundError("Missing server inputs:\n" + "\n".join(missing))
    runtime = {}
    if check_runtime:
        import inspect
        import torch
        import numpy
        import pandas
        import einops
        from einops import pack, unpack  # noqa: F401
        from sksurv.metrics import concordance_index_censored  # noqa: F401
        runtime = {name: importlib.metadata.version(name) for name in
                   ("torch", "numpy", "pandas", "einops", "scikit-survival")}
        if not {"weights_only", "mmap"}.issubset(inspect.signature(torch.load).parameters):
            raise RuntimeError("This task requires torch.load with weights_only and mmap support (PyTorch 2.1+)")
        if not checks_only and config["device"].startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA is unavailable; no automatic CPU fallback or smaller budget")
            index = int(config["device"].split(":")[1]) if ":" in config["device"] else 0
            if index >= torch.cuda.device_count():
                raise RuntimeError("Requested CUDA device index does not exist")
            runtime["gpu"] = torch.cuda.get_device_name(index)
            # Detect an unsupported GPU/runtime combination before any patient
            # preparation or training. This is a one-element synthetic probe.
            if torch.ones(1, device=config["device"]).add_(1).item() != 2:
                raise RuntimeError("CUDA runtime probe failed")
    return {"status": "ready", "runtime": runtime,
        "source_and_metadata_sha256": {str(path): _sha(path) for path in files},
        "patient_features_read": False, "patient_training_executed": False}


def run_identity(plan, preflight_result):
    # Runtime/library changes also invalidate a resumed scientific comparison.
    package = Path(__file__).parent
    payload = {"config": plan["config"], "checks_only": plan["checks_only"],
        "stages": plan["stages"], "inputs": preflight_result["source_and_metadata_sha256"],
        "runtime": preflight_result["runtime"],
        "implementation": {p.name: _sha(p) for p in sorted(package.glob("*.py"))}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _verify_resolved_config(root, expected):
    path = Path(root) / "resolved_config.json"
    try:
        saved = _json(path)
    except (OSError, ValueError) as exc:
        raise RuntimeError("Saved resolved configuration is missing or invalid; refusing to run this batch") from exc
    if saved != expected:
        raise RuntimeError("Saved resolved configuration changed; refusing to run this batch")
    return saved


def _stage_files(root, stage, config=None):
    kind = stage["kind"]
    if kind in ("protocol", "reconstruction", "mechanisms", "attribution"):
        return [root / "checks" / (kind + "_checks.json")]
    if kind == "prepare":
        return [root / "prepared/manifest.json"]
    if kind == "pair":
        folder = root / "pairs" / f"fold_{stage['fold']}" / f"seed_{stage['seed']}"
        required = [folder / "protocol.json", folder / "shared_initial_state.pt", folder / "summary.json"]
        arms = training_arms(config or {})
        for arm in [*arms, *(["same_weights_eval_fix"] if (config or {}).get("experiment", "reproduction") == "reproduction" else [])]:
            required += [folder / arm / f for f in ("summary.json", "eval_repeats.json", "validation_predictions.csv")]
            if arm != "same_weights_eval_fix":
                required += [folder / arm / f for f in ("last.pt", "curve.json", "trained_selector_gradients.json")]
                required += [folder / arm / "patient_slot_intervention.json"]
                if (config or {}).get("experiment") == "original_audit":
                    required += [folder / arm / name for name in ("native_model_audit.json", "natural_gate_observations.json")]
        if "B" in arms:
            required += [folder / "matched_state_bc_checks.json"]
        return required
    files = [root / "proof_summary.json", root / "proof_summary.txt"]
    if (root / "patient_summary.json").exists():
        files.append(root / "patient_summary.json")
    return files


def execute(plan, resume=False, executor=None, runtime_check=True):
    config, root = plan["config"], Path(plan["output_root"])
    # Never create outputs inside original source/data directories.
    for key in PATH_KEYS:
        source = Path(config[key])
        if root == source or source in root.parents:
            raise ValueError(f"output_root must be outside {key}")
    check = preflight(config, plan["checks_only"], check_runtime=runtime_check)
    identity = run_identity(plan, check)
    manifest_path = root / "queue_manifest.json"
    if root.exists() and any(root.iterdir()):
        if not resume or not manifest_path.is_file():
            raise FileExistsError("Output batch is not empty; use a new --output-root (or --resume for an intact matching queue)")
        saved = _json(manifest_path)
        if (saved.get("run_identity") != identity or
                any(saved.get(key) != plan[key] for key in ("config", "stages", "checks_only"))):
            raise RuntimeError("Config, source, CSV, runtime, or implementation changed; refusing to resume this batch")
    else:
        if resume:
            raise FileNotFoundError("--resume requires an existing queue_manifest.json")
        root.mkdir(parents=True, exist_ok=True)
        _write(manifest_path, {**plan, "run_identity": identity, "preflight": check,
            "implementation_sha256": {p.name: _sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}, exclusive=True)
        _write(root / "resolved_config.json", config, exclusive=True)
    _verify_resolved_config(root, config)
    lock = root / ".queue.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, stream)
    try:
        for index, stage in enumerate(plan["stages"], 1):
            _verify_resolved_config(root, config)
            receipt = root / "receipts" / (stage["id"] + ".json")
            if receipt.exists():
                saved = _json(receipt)
                if saved.get("run_identity") != identity or not saved.get("outputs"):
                    raise RuntimeError(f"Invalid completion receipt: {receipt}")
                for rel, digest in saved["outputs"].items():
                    if _sha(root / rel) != digest:
                        raise RuntimeError(f"Completed stage output changed: {rel}")
                print(f"[{index}/{len(plan['stages'])}] verified, skip {stage['id']}", flush=True)
                continue
            # An interrupted, unreceipted stage may have partial results. Its
            # worker refuses overwriting them; keep the attempt for diagnosis.
            print(f"[{index}/{len(plan['stages'])}] start {stage['id']}", flush=True)
            command = [sys.executable, "-B", "-u", "-m", PACKAGE + ".worker",
                "--config", str(root / "resolved_config.json"), "--output-root", str(root), "--stage", stage["kind"]]
            if stage["kind"] == "pair":
                command += ["--fold", str(stage["fold"]), "--seed", str(stage["seed"])]
            if plan["checks_only"]:
                command += ["--checks-only"]
            _write(root / "queue_status.json", {"status": "running", "stage": stage, "command": command})
            logs = root / "logs"
            logs.mkdir(exist_ok=True)
            n = 1
            while (logs / f"{stage['id']}_attempt_{n}.log").exists():
                n += 1
            log_path = logs / f"{stage['id']}_attempt_{n}.log"
            with log_path.open("x", encoding="utf-8") as log:
                if executor is None:
                    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
                    process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
                    try:
                        for line in process.stdout:
                            print(line, end="", flush=True)
                            log.write(line)
                            log.flush()
                        code = process.wait()
                    except BaseException:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise
                else:
                    code = executor(command, stage, root, log)
            _verify_resolved_config(root, config)
            if code:
                raise RuntimeError(f"Stage {stage['id']} failed (exit {code}); see {log_path}")
            files = _stage_files(root, stage, config)
            if not files or any(not path.is_file() for path in files):
                raise RuntimeError(f"Stage {stage['id']} did not produce its required outputs")
            if stage["kind"] == "pair" and _json(root / "pairs" / f"fold_{stage['fold']}" / f"seed_{stage['seed']}" / "summary.json").get("status") != "completed":
                raise RuntimeError("Patient pair has no completed summary")
            _write(receipt, {"run_identity": identity, "stage": stage,
                "outputs": {str(path.relative_to(root)): _sha(path) for path in files}}, exclusive=True)
        _write(root / "queue_status.json", {"status": "completed", "stages": len(plan["stages"]),
            "patient_training_executed": not plan["checks_only"], "run_identity": identity})
    except BaseException as exc:
        _write(root / "queue_status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "note": "Partial artifacts retained; do not mix or overwrite them"})
        raise
    finally:
        # Only our own tiny lock file is removed, never experiment artifacts.
        lock.unlink()
    return root


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Default: print tasks; do not read patient data")
    mode.add_argument("--doctor", action="store_true", help="Check server paths/runtime without reading feature tensors or training")
    mode.add_argument("--run", action="store_true", help="Explicitly execute the proof queue on this machine")
    for key in PATH_KEYS:
        p.add_argument("--" + key.replace("_", "-"))
    p.add_argument("--output-root", type=Path)
    p.add_argument("--folds", nargs="+", type=int)
    p.add_argument("--seeds", nargs="+", type=int)
    for key in ("epochs", "patches", "batch_size", "eval_repeats", "encoding_dim"):
        p.add_argument("--" + key.replace("_", "-"), type=int)
    p.add_argument("--study")
    p.add_argument("--device")
    p.add_argument("--experiment", choices=("reproduction", "attribution_abc", "original_audit"))
    p.add_argument("--checks-only", action="store_true", help="Run metadata/synthetic checks without patient training")
    p.add_argument("--resume", action="store_true", help="Verify and skip completed stages in an explicitly named prior output")
    args = p.parse_args(argv)
    overrides = {key: getattr(args, key) for key in (*PATH_KEYS, "folds", "seeds", "epochs", "patches", "batch_size",
        "eval_repeats", "encoding_dim", "study", "device", "experiment") if getattr(args, key) is not None}
    config = normalize_config({**_json(args.config), **overrides})
    if args.resume and (not args.run or args.output_root is None):
        p.error("--resume requires --run and the original --output-root")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    root = args.output_root or REPO_ROOT / "results/slotspe_proof" / (config["study"] + "_" + stamp)
    plan = make_plan(config, root, args.checks_only)
    if args.doctor:
        print(json.dumps(preflight(config, args.checks_only), ensure_ascii=False, indent=2))
    elif args.run:
        print("Results:", execute(plan, resume=args.resume), flush=True)
    else:
        print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
