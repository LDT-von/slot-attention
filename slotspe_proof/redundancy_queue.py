"""Independent eleven-arm SlotSPE queue; only --run executes patient work.

Planning is standard-library-only. Separate workers isolate preparation,
reference fitting and each arm. Completed stages resume only with matching
configuration, code, metadata and artifact hashes; partial stages are retained.
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
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/slotspe_redundancy_queue.json"
ARMS = ("S0", "S4", "R1", "R2", "R3", "R4", "R5", "S1", "S2", "S3", "S5")
DEFAULT_OUTPUT = "local_evidence/redundancy_blca_uni2h_fold0_seed3_run01"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            value.update(part)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def load_config(path=None):
    return read(path or DEFAULT_CONFIG)


def _resolve(value):
    value = os.path.expandvars(os.path.expanduser(str(value)))
    return str((ROOT / value).resolve()) if not Path(value).is_absolute() else str(Path(value).resolve())


def normalize_runtime(config, metadata_root=None, features_root=None, device=None):
    if "requested_scope" in config:
        scope = config["requested_scope"]
        stage = next(s for s in config["stages"] if s["id"] == "real_data_diagnostic")
        if scope.get("automatic_cross_fold_expansion") is not False:
            raise ValueError("automatic cross-fold expansion is outside this queue")
        for key in ("study", "feature_encoder", "folds", "seeds"):
            if stage.get(key) != scope.get(key):
                raise ValueError(f"stage and requested scope disagree: {key}")
        if stage.get("variants") != scope.get("selected_variants"):
            raise ValueError("stage and requested variants disagree")
        runtime = {
            "study": scope["study"], "feature_encoder": scope["feature_encoder"],
            "folds": scope["folds"], "seeds": scope["seeds"],
            "variants": scope["selected_variants"], "epochs": stage["epochs"],
            "patches": stage["real_patches"], "batch_size": stage["batch_size"],
            "eval_repeats": stage["eval_repeats"],
            **config.get("execution", {}),
        }
    else:
        runtime = dict(config)
    defaults = {
        "source_root": "third_party/SlotSPE", "metadata_root": "${SLOTSPE_METADATA_ROOT}",
        "features_root": "${SLOTSPE_FEATURES_ROOT}", "label_col": "survival_months_dss",
        "signature": "combine", "device": "cuda:0", "lr": 0.0005,
        "probe_lr": 0.0005, "torch_threads": 2, "encoding_dim": None,
        "projection_dim": 256, "slot_iters": 10, "content_target_dim": 32,
        "content_clusters": 8, "content_reference_seed": 731, "dias_distance": 0.05,
        "slotbert_temperature": 0.5, "experiment": "original_audit",
    }
    runtime = {**defaults, **runtime}
    for key, value in (("metadata_root", metadata_root), ("features_root", features_root), ("device", device)):
        if value is not None:
            runtime[key] = value
    if (runtime.get("study") != "blca" or runtime.get("feature_encoder") != "UNI2-h" or
            runtime.get("folds") != [0] or runtime.get("seeds") != [3] or
            runtime.get("variants") != list(ARMS)):
        raise ValueError("scope is exactly BLCA / UNI2-h / fold0 / seed3 / the declared eleven arms")
    for key in ("epochs", "patches", "batch_size", "eval_repeats", "torch_threads", "projection_dim",
                "slot_iters", "content_target_dim", "content_clusters", "content_reference_seed"):
        value = runtime[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if runtime["projection_dim"] != 256 or runtime["slot_iters"] != 10:
        raise ValueError("patient queue must use native 256 dimensions and 10 slot iterations")
    if runtime["eval_repeats"] < 2 or runtime["content_clusters"] != 8:
        raise ValueError("need repeated evaluation and eight reference clusters")
    if (runtime["content_target_dim"] != 32 or runtime["dias_distance"] != 0.05 or
            runtime["slotbert_temperature"] != 0.5):
        raise ValueError("use the declared 32-dimension content, DIAS .05 and paper temperature .5 presets")
    for key in ("lr", "probe_lr", "slotbert_temperature"):
        if isinstance(runtime[key], bool) or not math.isfinite(runtime[key]) or runtime[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if runtime["encoding_dim"] is not None and (isinstance(runtime["encoding_dim"], bool) or
            not isinstance(runtime["encoding_dim"], int) or runtime["encoding_dim"] < 1):
        raise ValueError("encoding_dim must be positive or null")
    if not isinstance(runtime["device"], str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", runtime["device"]):
        raise ValueError("device must be cpu or cuda[:index]")
    for key in ("source_root", "metadata_root", "features_root"):
        runtime[key] = _resolve(runtime[key])
    if Path(runtime["source_root"]).resolve() != (ROOT / "third_party/SlotSPE").resolve():
        raise ValueError("use the pinned original SlotSPE source in this checkout")
    return runtime


def verify_config_pins(config):
    for group in ("evidence_sha256", "protected_existing_queue_sha256", "candidate_implementation_sha256"):
        for name, expected in config.get(group, {}).items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT.resolve()) or sha(path) != expected:
                raise RuntimeError(f"pinned file differs: {name}")


def implementation_hashes():
    package = Path(__file__).parent
    files = [package / n for n in (
        "redundancy_queue.py", "redundancy_engine.py", "redundancy_content.py", "patient.py",
        "model.py", "assignment_router.py", "native_allocation.py", "native_pruning.py",
        "native_variants.py", "prototype_bank.py", "slot_regularization.py")]
    files += sorted((ROOT / "third_party/SlotSPE").rglob("*.py"))
    files += [ROOT / "scripts/run_slotspe_redundancy.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def private_output(path):
    path = Path(path or ROOT / DEFAULT_OUTPUT)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    base = (ROOT / "local_evidence").resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError("outputs must use a dedicated ignored local_evidence subdirectory")
    return path


def stages():
    return [{"id": "prepare", "kind": "prepare"}, {"id": "initialize", "kind": "initialize"},
            *[{"id": "arm_" + arm, "kind": "arm", "arm": arm} for arm in ARMS],
            {"id": "summarize", "kind": "summarize"}]


def expected_artifacts(stage):
    if stage["kind"] == "prepare":
        return ["prepared/manifest.json"]
    if stage["kind"] == "initialize":
        return ["initial/" + name for name in ("core.pt", "probe.pt", "reference.pt", "receipt.json")]
    if stage["kind"] == "arm":
        return [f"arms/{stage['arm']}/" + name for name in (
            "last.pt", "curve.json", "summary.json", "eval_repeats.json", "diagnostics.pt", "validation_predictions.csv")]
    return ["summary.json", "summary.txt"]


def make_plan(config=None, output_root=None):
    config = load_config() if config is None else config
    verify_config_pins(config)
    runtime = normalize_runtime(config)
    root = private_output(output_root)
    return {"schema": "slotspe_native_redundancy_execution_plan_v1", "config": runtime,
            "output_root": str(root), "stages": stages(), "training_runs": len(ARMS),
            "training_arms": list(ARMS), "feature_encoder": "UNI2-h", "folds": [0], "seeds": [3],
            "default_action": "plan_only", "patient_work_executed": False,
            "automatic_cross_fold_expansion": False, "execution_implemented": True,
            "checkpoint_selection": "fixed_final_epoch_no_validation_selection",
            "budget_scope": "three-epoch default is a short diagnostic, not fully trained efficacy evidence",
            "unresolved_variables": [k for k in ("metadata_root", "features_root") if "$" in runtime[k]],
            "content_contract": "train-only frozen reference projection and cluster targets; rectangular matching with unmatched coverage costs",
            "batching": "same per-example compact native forward and one core update per logical batch for all arms"}


def doctor(config, runtime_check=True):
    runtime = normalize_runtime(config)
    from .queue import preflight
    result = preflight(runtime, checks_only=False, check_runtime=runtime_check)
    if runtime_check:
        import h5py
        import scipy
        result["runtime"].update(h5py=h5py.__version__, scipy=scipy.__version__)
    result.update(feature_encoder="UNI2-h", encoder_provenance="declared existing root; actual feature source must match server records",
                  feature_extraction_started=False, patient_training_executed=False)
    return result


def _stage_folder(stage):
    return {"prepare": "prepared", "initialize": "initial", "arm": "arms/" + stage.get("arm", "")}.get(stage["kind"])


def _artifact_hashes(root, stage):
    required = expected_artifacts(stage)
    for name in required:
        if not (root / name).is_file():
            raise RuntimeError(f"stage {stage['id']} missing artifact: {name}")
    folder = _stage_folder(stage)
    files = sorted((root / folder).rglob("*")) if folder else [root / n for n in required]
    return {p.relative_to(root).as_posix(): sha(p) for p in files if p.is_file()}


def _verify_completed(root, stage, record):
    current = _artifact_hashes(root, stage)
    if current != record.get("artifact_sha256"):
        raise RuntimeError(f"completed stage artifacts changed: {stage['id']}")


def _subprocess_stage(stage, runtime, root):
    log = root / "logs" / (stage["id"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", "-m", "slotspe_proof.redundancy_queue",
               "--run", "--worker-stage", stage["id"], "--output-root", str(root)]
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"stage {stage['id']} failed with exit {result.returncode}; retained log: {log}")


def execute(config, output_root, resume=False, executor=None, runtime_check=True):
    plan = make_plan(config, output_root)
    runtime, root = plan["config"], Path(plan["output_root"])
    checked = doctor(runtime, runtime_check=runtime_check)
    code = implementation_hashes()
    identity_payload = {"runtime": runtime, "implementation": code, "stages": plan["stages"],
                        "inputs": checked.get("source_and_metadata_sha256", {}),
                        "runtime_versions": checked.get("runtime", {})}
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    manifest_path = root / "queue_manifest.json"
    if root.exists() and any(root.iterdir()):
        if not resume or not manifest_path.is_file():
            raise RuntimeError("nonempty output: preserve it; use a fresh directory or verified --resume")
        manifest = read(manifest_path)
        if manifest.get("identity") != identity or read(root / "resolved_config.json") != runtime:
            raise RuntimeError("configuration, input metadata or implementation drift; cannot resume")
    else:
        root.mkdir(parents=True, exist_ok=True)
        write(root / "resolved_config.json", runtime)
        manifest = {"schema": "slotspe_native_redundancy_queue_v1", "identity": identity,
                    "identity_payload": identity_payload, "config": runtime, "stages": plan["stages"],
                    "status": "running", "records": {}, "started_utc": datetime.now(timezone.utc).isoformat()}
        write(manifest_path, manifest)
    executor = executor or _subprocess_stage

    def verify_live_identity():
        saved = read(manifest_path)
        if (read(root / "resolved_config.json") != runtime or saved.get("config") != runtime or
                saved.get("identity") != identity or saved.get("identity_payload") != identity_payload):
            raise RuntimeError("saved runtime or queue identity changed during execution")
        if implementation_hashes() != code:
            raise RuntimeError("implementation changed while queue was running")
        if any(sha(Path(name)) != value for name, value in identity_payload["inputs"].items()):
            raise RuntimeError("source or metadata changed while queue was running")

    for stage in plan["stages"]:
        verify_live_identity()
        for prior in plan["stages"]:
            record = manifest["records"].get(prior["id"], {})
            if record.get("status") == "completed":
                _verify_completed(root, prior, record)
        record = manifest["records"].get(stage["id"], {})
        if record.get("status") == "completed":
            continue
        if record.get("status") in ("running", "failed"):
            raise RuntimeError(f"partial/failed stage retained: {stage['id']}; use a fresh batch, never overwrite it")
        folder = _stage_folder(stage)
        if folder and (root / folder).exists() and any((root / folder).iterdir()):
            raise RuntimeError(f"unrecorded partial stage folder retained: {folder}")
        manifest["status"], manifest["active_stage"] = "running", stage["id"]
        manifest["records"][stage["id"]] = {"status": "running"}
        write(manifest_path, manifest)
        try:
            executor(stage, runtime, root)
            verify_live_identity()
            manifest["records"][stage["id"]] = {"status": "completed", "artifact_sha256": _artifact_hashes(root, stage)}
            manifest.pop("active_stage", None)
            write(manifest_path, manifest)
            print("completed stage " + stage["id"], flush=True)
        except BaseException as error:
            manifest["status"] = "failed"
            manifest["records"][stage["id"]] = {"status": "failed", "error": str(error), "type": type(error).__name__}
            write(manifest_path, manifest)
            raise
    manifest["status"] = "completed"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write(manifest_path, manifest)
    return manifest


def _initialize(runtime, root, manifest):
    import torch
    from .patient import _batch
    from .redundancy_content import ContentReference
    from .redundancy_engine import make_initial_states
    torch.set_num_threads(runtime["torch_threads"])
    out = root / "initial"
    out.mkdir()
    core, probe = make_initial_states(manifest, runtime["source_root"], runtime["seeds"][0],
                                     projection_dim=runtime["projection_dim"], slot_iters=runtime["slot_iters"])
    reference = ContentReference(manifest["omic_sizes"], manifest["encoding_dim"],
                                 target_dim=runtime["content_target_dim"], clusters=runtime["content_clusters"],
                                 seed=runtime["content_reference_seed"])
    fold = manifest["folds"]["0"]
    reference.fit(_batch(root, manifest, fold, [pid], "cpu", torch) for pid in fold["train"])
    torch.save(core, out / "core.pt")
    torch.save(probe, out / "probe.pt")
    torch.save(reference.state_dict(), out / "reference.pt")
    write(out / "receipt.json", {"training_examples": len(fold["train"]), "reference_fit_split": "train_only",
          "frozen_during_evaluation": True, "encoding_dim": manifest["encoding_dim"],
          "omic_sizes": manifest["omic_sizes"], "files_sha256": {n: sha(out / n) for n in ("core.pt", "probe.pt", "reference.pt")},
          "target_scope": "train-standardized fixed random projection, input-only per-example clusters; not biological ground truth"})


def _summarize(root, runtime):
    rows = {arm: read(root / "arms" / arm / "summary.json") for arm in ARMS}
    if any(row.get("status") != "completed" for row in rows.values()):
        raise RuntimeError("incomplete arm; cannot publish complete summary")
    result = {"schema": "slotspe_native_redundancy_summary_v1", "status": "completed", "arms": rows,
              "completed_unique_training_runs": len(rows), "scope": {"study": "blca", "feature_encoder": "UNI2-h", "fold": 0, "seed": 3},
              "epochs": runtime["epochs"], "patches": runtime["patches"],
              "interpretation": "Single-fold exploratory diagnostic; repeated evaluations are not independent patients. Low cosine is not proof of content or predictive improvement."}
    write(root / "summary.json", result)
    (root / "summary.txt").write_text("SlotSPE BLCA / UNI2-h / fold0 / seed3: all eleven arms completed.\n" +
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def worker(stage_id, root):
    root = private_output(root)
    manifest = read(root / "queue_manifest.json")
    if manifest.get("status") != "running" or manifest.get("active_stage") != stage_id:
        raise RuntimeError("worker must belong to the active explicitly-run queue stage")
    runtime = read(root / "resolved_config.json")
    expected_identity = hashlib.sha256(json.dumps(manifest["identity_payload"], sort_keys=True).encode()).hexdigest()
    if (runtime != manifest["config"] or runtime != manifest["identity_payload"]["runtime"] or
            expected_identity != manifest["identity"] or normalize_runtime(runtime) != runtime or
            implementation_hashes() != manifest["identity_payload"]["implementation"]):
        raise RuntimeError("worker runtime or implementation differs from its queued identity")
    stage = next((s for s in manifest["stages"] if s["id"] == stage_id), None)
    if stage is None:
        raise ValueError("unknown stage")
    from .patient import prepare, _manifest
    if stage["kind"] == "prepare":
        prepare(runtime, root)
    elif stage["kind"] in ("initialize", "arm"):
        prepared = _manifest(runtime, root, verify_cache=True)
        if stage["kind"] == "initialize":
            _initialize(runtime, root, prepared)
        else:
            import torch
            from .redundancy_content import ContentReference
            from .redundancy_engine import run_arm
            reference = ContentReference(prepared["omic_sizes"], prepared["encoding_dim"],
                target_dim=runtime["content_target_dim"], clusters=runtime["content_clusters"], seed=runtime["content_reference_seed"])
            reference.load_state_dict(torch.load(root / "initial/reference.pt", map_location="cpu", weights_only=True))
            core = torch.load(root / "initial/core.pt", map_location="cpu", weights_only=True)
            probe = torch.load(root / "initial/probe.pt", map_location="cpu", weights_only=True)
            run_arm(runtime, root, prepared, stage["arm"], core, probe, reference)
    else:
        _summarize(root, runtime)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--metadata-root")
    parser.add_argument("--features-root")
    parser.add_argument("--device")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--doctor", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-stage", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_stage:
        if not args.run or not args.output_root:
            parser.error("internal worker requires --run and a queued output directory")
        worker(args.worker_stage, args.output_root)
        return
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    config = load_config(args.config)
    runtime = normalize_runtime(config, args.metadata_root, args.features_root, args.device)
    # Retain all public scope/pin validation while resolving server path overrides.
    resolved = dict(config)
    resolved["execution"] = {**config.get("execution", {}), **runtime}
    if args.run:
        result = execute(resolved, args.output_root or ROOT / DEFAULT_OUTPUT, resume=args.resume)
        print(json.dumps({"status": result["status"], "output_root": str(private_output(args.output_root))}))
    elif args.doctor:
        verify_config_pins(config)
        print(json.dumps(doctor(runtime), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(make_plan(resolved, args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
