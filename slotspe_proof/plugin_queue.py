"""Independent plugin experiment plans and explicit execution gates.

Importing this module, normalizing a configuration, and printing a plan use
only the standard library. Only --run reaches patient experiment execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess


REPO = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO / "third_party/SlotSPE"
DEFAULT_CONFIG = REPO / "configs/slotspe_plugins_fold0_seed3.json"
DEFAULT_OUTPUT = REPO / "local_evidence/slot_plugins_blca_uni2h_fold0_seed3_run01"
DEFAULT_GENERATED_OUTPUT = REPO / "local_evidence/slot_plugins_generated_check_run01"
PRESETS = ("B0_native", "B1_gram", "V1_mean", "V1_noise", "V1_vib",
           "V2_content", "V2_separate", "V3_capacity", "V3_relation")
STAGES = {"v1": PRESETS[:5], "v2": PRESETS[5:7], "v3": PRESETS[7:]}
_ENV_TOKEN = re.compile(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*|%[^%]+%")
_LOCKED = {
    "experiment": "original_audit", "study": "blca", "feature_encoder": "UNI2-h",
    "folds": [0], "seeds": [3], "epochs": 30, "patches": 4096,
    "batch_size": 8, "eval_repeats": 20, "splits_dir": "5fold_uni2h",
    "projection_dim": 256, "slot_iters": 10, "lr": 0.0005,
    "native_alpha": 0.5, "native_reconstruction_weight": 0.01,
    "gradient_clip_norm": None,
    "checkpoint_selection": "official_best_validation_latest_tie",
    "validation_patch_policy": "all_real", "validation_batch_size": 1,
    "label_col": "survival_months_dss", "signature": "combine",
    "encoding_dim": None, "torch_threads": 2,
}
_INPUT_DEFAULTS = {"source_root": "third_party/SlotSPE",
                   "metadata_root": "${SLOTSPE_METADATA_ROOT}",
                   "features_root": "${SLOTSPE_FEATURES_ROOT}", "device": "cuda:0"}
_METADATA_FILES = ("clinical/all/blca.csv", "raw_rna_data_inter/blca_rna_inter.csv",
                   "signatures/combine_signatures.csv", "splits/5fold_uni2h/blca/fold_0.csv")


def _resolve_path(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be a nonempty path string")
    expanded = os.path.expandvars(os.path.expanduser(value))
    if _ENV_TOKEN.search(expanded):
        return expanded
    path = Path(expanded)
    return str((path if path.is_absolute() else REPO / path).resolve())


def normalize_config(config, metadata_root=None, features_root=None, device=None,
                     require_inputs=False):
    """Return locked runtime values; unresolved input variables stay visible.

    Input requirements check existence only. They never load an array, model,
    clinical dataframe, or checkpoint, and do not create an output directory.
    """
    if not isinstance(config, dict):
        raise ValueError("configuration must be an object")
    unknown = set(config) - set(_LOCKED) - set(_INPUT_DEFAULTS)
    if unknown:
        raise ValueError("unknown configuration keys: " + ", ".join(sorted(unknown)))
    runtime = {**_LOCKED, **_INPUT_DEFAULTS, **config}
    # Exact JSON types also reject bool==0/1 and floating fold/epoch values.
    for key, expected in _LOCKED.items():
        actual = runtime[key]
        same_type = type(actual) is type(expected)
        if isinstance(expected, float):
            same_type = type(actual) in (int, float)
        if isinstance(expected, list):
            same_type = (same_type and len(actual) == len(expected)
                         and all(type(a) is type(b) for a, b in zip(actual, expected)))
        if not same_type or actual != expected:
            raise ValueError("locked configuration value differs: " + key)
    for key, override in (("metadata_root", metadata_root), ("features_root", features_root)):
        if override is not None:
            runtime[key] = str(override)
    if device is not None:
        runtime["device"] = device
    if not isinstance(runtime["device"], str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", runtime["device"]):
        raise ValueError("device must be cpu or cuda[:index]")
    for key in ("source_root", "metadata_root", "features_root"):
        runtime[key] = _resolve_path(runtime[key], key)
    if _ENV_TOKEN.search(runtime["source_root"]) or Path(runtime["source_root"]) != SOURCE_ROOT.resolve():
        raise ValueError("source_root is fixed to this repository's third_party/SlotSPE")
    # Avoid returning references to the mutable lock constants or input lists.
    runtime["folds"], runtime["seeds"] = list(runtime["folds"]), list(runtime["seeds"])
    if require_inputs:
        for key in ("metadata_root", "features_root"):
            if _ENV_TOKEN.search(runtime[key]):
                raise ValueError("unresolved input environment variable: " + key)
        for key in ("source_root", "metadata_root", "features_root"):
            if not Path(runtime[key]).is_dir():
                raise ValueError("required directory is missing: " + key)
        for relative in _METADATA_FILES:
            if not (Path(runtime["metadata_root"]) / relative).is_file():
                raise ValueError("required metadata file is missing: " + relative)
    return runtime


def output_path(value=DEFAULT_OUTPUT):
    """Resolve an ignored local_evidence child without creating it."""
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else REPO / path).resolve()
    private_root = (REPO / "local_evidence").resolve()
    if path == private_root or not path.is_relative_to(private_root):
        raise ValueError("output_root must be a child of this repository's ignored local_evidence/")
    relative = path.relative_to(REPO).as_posix()
    checked = subprocess.run(["git", "check-ignore", "-q", "--", relative], cwd=REPO,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if checked.returncode != 0:
        raise ValueError("output_root must be ignored by Git")
    return path


def make_plan(config, output_root=DEFAULT_OUTPUT, stage="v1"):
    if stage not in STAGES:
        raise ValueError("unknown stage")
    runtime = normalize_config(config)
    output = output_path(output_root)
    stage_runs = len(STAGES[stage]) * len(runtime["folds"]) * len(runtime["seeds"])
    all_runs = len(PRESETS) * len(runtime["folds"]) * len(runtime["seeds"])
    return {"mode": "plan", "config": runtime, "stage": stage,
            "presets": list(PRESETS), "stage_presets": list(STAGES[stage]),
            "stage_training_runs": stage_runs, "total_training_runs": all_runs,
            "stage_training_epoch_budget": stage_runs * runtime["epochs"],
            "total_training_epoch_budget": all_runs * runtime["epochs"],
            "required_prior_stages": list(STAGES)[:list(STAGES).index(stage)],
            "output_root": str(output), "patient_work_executed": False,
            "later_stages_run_automatically": False,
            "checkpoint_policy": {"initial_best": 0.0, "comparison": ">=", "tie": "latest"},
            "unresolved_environment_variables": sorted({token for key in ("metadata_root", "features_root")
                                                        for token in _ENV_TOKEN.findall(runtime[key])})}


def doctor(config):
    """Hash metadata/source files and inspect runtime; no tensor/model I/O."""
    runtime = normalize_config(config, require_inputs=True)
    records = []
    for relative in _METADATA_FILES:
        path = Path(runtime["metadata_root"]) / relative
        data = path.read_bytes()
        records.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    source_hashes = {}
    for path in sorted(Path(runtime["source_root"]).rglob("*.py")):
        source_hashes[path.relative_to(runtime["source_root"]).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    torch = importlib.import_module("torch")
    cuda = bool(torch.cuda.is_available())
    if runtime["device"].startswith("cuda"):
        index = int(runtime["device"].partition(":")[2] or 0)
        if not cuda or index >= torch.cuda.device_count():
            raise ValueError("requested CUDA device is unavailable")
    return {"mode": "doctor", "ok": True, "config": runtime,
            "metadata_files": records, "source_python_sha256": source_hashes,
            "features_directory_exists": True,
            "runtime": {"torch_version": str(torch.__version__), "cuda_available": cuda,
                        "cuda_version": torch.version.cuda, "device": runtime["device"]},
            "patient_arrays_loaded": False, "models_constructed": False,
            "patient_work_executed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true", help="Print the locked plan (default); standard library only")
    action.add_argument("--doctor", action="store_true", help="Check paths, file hashes and runtime without model/array I/O")
    action.add_argument("--run", action="store_true", help="Execute only the requested patient experiment stage")
    action.add_argument("--check-generated", action="store_true", help="Run independent generated-input checks, without server inputs")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=tuple(STAGES), default="v1")
    parser.add_argument("--output-root", type=Path,
                        help="Ignored local_evidence child; generated checks have an independent default")
    parser.add_argument("--metadata-root")
    parser.add_argument("--features-root")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    try:
        output = output_path(args.output_root if args.output_root is not None else
                             DEFAULT_GENERATED_OUTPUT if args.check_generated else DEFAULT_OUTPUT)
        if args.check_generated:
            result = importlib.import_module("slotspe_proof.plugin_engine").generated_smoke(output)
        else:
            raw = json.loads(args.config.read_text(encoding="utf-8"))
            runtime = normalize_config(raw, metadata_root=args.metadata_root,
                                       features_root=args.features_root, device=args.device,
                                       require_inputs=args.run or args.doctor)
            if args.run:
                result = importlib.import_module("slotspe_proof.plugin_engine").execute(runtime, output, stage=args.stage)
            elif args.doctor:
                result = doctor(runtime)
            else:
                result = make_plan(runtime, output, args.stage)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
