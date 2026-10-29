#!/usr/bin/env python3
"""Real-patient slot-deletion and cross-view predictability diagnostics.

The default action prints a plan and touches no patient data.  ``--run``
executes the diagnostics against an *existing* completed cross-view queue
output.  No training happens here: the script re-runs inference with the
saved final checkpoint and then applies the exact finite-table deletion
oracle to the resulting slots.

What is measured
----------------
1.  **Cross-view predictability**: does the patient's own slot content
    predict the held-out view target better than the training-fold constant
    and better than mismatched donors?
2.  **Slot-deletion oracle**: for each patient, map the ``K`` slots to ``K``
    discrete codes with a codebook fitted on the *training fold only*, then
    compare the exact in-sample optimum of the conditional table over the
    full code multiset against the exact optimum over the multiset with one
    slot deleted.
3.  **Slot geometry**: descriptive redundancy metrics of the slot set.

Honest boundaries, repeated because they decide how the numbers may be used
----------------------------------------------------------------------------
*   The deletion gap is an **empirical sample oracle**: the target labels
    fit and score the same finite tables.  It is not a deployable predictor,
    not population mutual information, and not evidence of patient
    prediction benefit.
*   Few patients make the multiset buckets sparse.  A permutation null is
    therefore computed with the *same* bucket structure; a real gap is only
    meaningful relative to that null.
*   A gap can also arise because the multiset is nearly unique per patient,
    i.e. the slots may have collapsed into a patient fingerprint rather than
    into content.  The multiset multiplicity is reported for that reason.
*   Nothing here establishes patient harm or correction efficacy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_RUN = REPO / "local_evidence/crossview_quick_blca_uni2h_fold0_seed3_run01"
DEFAULT_OUTPUT = REPO / "local_evidence/crossview_deletion_diagnostics_20261029"
# Slot spaces compared.  "native" is the untouched encoder slot, "private"
# is the V2 private slot and "content" is the V2 content head.
SLOT_SPACES = ("native", "private", "content")
CODEBOOK_SIZES = (2, 4, 8)
PERMUTATION_REPEATS = 8
ALL_ARMS = ("B0_native", "V2_separate", "V2_crossview_compute", "V2_crossview_train")
CROSSVIEW_ARMS = ("V2_crossview_compute", "V2_crossview_train")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2),
                    encoding="utf-8")


def _resolve_output(value):
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else REPO / path).resolve()
    private = (REPO / "local_evidence").resolve()
    if path == private or not path.is_relative_to(private):
        raise ValueError("output_root must be a child of the ignored local_evidence/")
    return path


def _check_output(value):
    path = _resolve_output(value)
    relative = path.relative_to(REPO).as_posix()
    checked = subprocess.run(["git", "check-ignore", "-q", "--", relative], cwd=REPO,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if checked.returncode != 0:
        raise ValueError("output_root must be ignored by Git")
    return path


def make_plan(run_root, output_root, arms):
    run_root = Path(run_root)
    output = _check_output(output_root)
    available = [arm for arm in arms
                 if (run_root / "arms" / arm / "last.pt").is_file()]
    return {
        "mode": "plan",
        "run_root": str(run_root),
        "output_root": str(output),
        "arms_requested": list(arms),
        "arms_with_checkpoint": available,
        "slot_spaces": list(SLOT_SPACES),
        "codebook_sizes": list(CODEBOOK_SIZES),
        "codebook_fit_population": "training_fold_only",
        "codebook_fit_vectors": "all_slots_of_all_training_patients_full_bag",
        "permutation_null_repeats": PERMUTATION_REPEATS,
        "targets": ["survival_bin_category", "fixed_reference_content_mass_mse"],
        "diagnostics": [
            "slot_deletion_empirical_oracle_category",
            "slot_deletion_empirical_oracle_mse",
            "slot_multiset_structure",
            "permutation_null_control",
            "slot_geometry_summary",
        ],
        "training_runs": 0,
        "optimizer_steps": 0,
        "patient_arrays_loaded": False,
        "checkpoints_loaded": False,
        "limitations": [
            "empirical_sample_oracle_not_population_information",
            "sparse_multiset_buckets_require_the_permutation_null",
            "single_fold_single_seed",
            "no_claim_of_patient_harm_or_correction_efficacy",
        ],
    }


def _rebuild_host(run_root, manifest, config, arm, state, reference_state):
    """Rebuild the exact host that produced the saved checkpoint.

    The rebuild mirrors ``crossview_engine._build_arm`` and then loads the
    saved state, so the diagnostic uses the trained weights and not a fresh
    initialization.
    """
    import torch

    from slotspe_proof import plugin_engine as existing
    from slotspe_proof.crossview_queue import (VIEW_DECODER_SEED, VIEW_TEMPERATURE)

    preset = "B0_native" if arm == "B0_native" else "V2_separate"
    base = existing._build(config, manifest, preset)
    if arm in CROSSVIEW_ARMS:
        if reference_state is None:
            raise RuntimeError("cross-view arm needs the saved content reference")
        from slotspe_proof.crossview_reconstruction import CrossViewV2
        host = CrossViewV2(base, reference_state, temperature=VIEW_TEMPERATURE,
                           decoder_seed=VIEW_DECODER_SEED)
        host.load_state_dict(state, strict=True)
    else:
        host = base
        host.load_state_dict(state, strict=True)
    return host.to(torch.device(config["device"]))


def _collect_slots(host, arm, run_root, manifest, config, ids, *, training=False):
    """Run inference once and collect the slot spaces for the given patients.

    ``training=True`` samples a bag with the original loader instead of using
    the full bag.  It is used only for the *training* patients that supply the
    codebook vectors, where the full bag is unnecessary and expensive; the
    held-out patients always use the full real bag so that the evaluated
    slots are the same ones the training queue stored.
    """
    import numpy as np
    import torch

    from slotspe_proof import patient
    from slotspe_proof import plugin_engine as existing

    host.eval()
    rows = []
    with torch.no_grad():
        for position, pid in enumerate(ids):
            seed = existing.schedule_seed("repeat", 0, position)
            patient._rng(seed, torch, np)
            real_count = None
            if training:
                item = torch.load(Path(run_root) / manifest["patients"][pid]["cache_path"],
                                  map_location="cpu", weights_only=True)
                bag, selected = patient._sample_bag(item["patches"], manifest["patches"],
                                                    np, torch,
                                                    generator_seed=seed + 600_000_000_000)
                real_count = len(selected)
                fold = manifest["folds"]["0"]
                case = {"x_wsi": bag.unsqueeze(0).to(config["device"]),
                        "omic_missing": False,
                        "y": torch.tensor([fold["labels"][pid]], dtype=torch.long,
                                          device=config["device"]),
                        "c": torch.tensor([manifest["patients"][pid]["censor"]],
                                          dtype=torch.float32, device=config["device"])}
                for index, genes in enumerate(item["omics"], 1):
                    case[f"x_omic{index}"] = genes.unsqueeze(0).to(config["device"])
            else:
                case, _ = existing._case(run_root, manifest, pid, config["device"],
                                         training=False)
                real_count = case["x_wsi"].shape[1]
            if arm in CROSSVIEW_ARMS:
                host(**case, plugin_seed=seed + 400_000_000_000,
                     real_count=real_count,
                     aux_seed=seed + 800_000_000_000,
                     mode="train", run_aux=True)
                records = host.host.last_records
            else:
                host(**case, plugin_seed=seed + 400_000_000_000)
                records = host.last_records
            if records is None:
                raise RuntimeError("no slot records for patient " + pid)
            available = sorted(set(records["wsi"]) & set(SLOT_SPACES))
            if not available:
                raise RuntimeError("no comparable slot space in the records")
            rows.append({
                "patient_id": pid,
                "label": int(manifest["folds"]["0"]["labels"][pid]),
                "censor": float(manifest["patients"][pid]["censor"]),
                "time": float(manifest["patients"][pid]["time"]),
                "spaces": available,
                "slots": {modality: {space: np.asarray(
                    records[modality][space].detach().cpu().numpy(),
                    dtype=np.float64).reshape(-1, records[modality][space].shape[-1])
                    for space in available}
                    for modality in ("wsi", "omic")},
            })
    return rows


def _content_mass_targets(rows, run_root, manifest, reference):
    """Fixed training-fold reference mass statistic for each patient.

    This is the same feature-proxy statistic the cross-view objective uses,
    computed on the patient's *full* real bag.  It is a feature proxy, not a
    biological label, and it never enters the encoder or the decoder.
    """
    import numpy as np
    import torch

    from slotspe_proof.crossview_reconstruction import CrossViewV2  # noqa: F401

    mean = reference.means[0]
    std = reference.stds[0]
    projection = reference.projections[0]
    anchors = reference.static_targets["wsi"]
    targets = {}
    for row in rows:
        pid = row["patient_id"]
        item = torch.load(Path(run_root) / manifest["patients"][pid]["cache_path"],
                          map_location="cpu", weights_only=True)
        raw = item["patches"].detach().cpu().double()
        projected = ((raw - mean) / std) @ projection
        distance = ((projected.unsqueeze(1) - anchors.unsqueeze(0)) ** 2).sum(-1)
        assignments = torch.softmax(-distance / 1.0, dim=-1)
        mass = assignments.mean(0)
        targets[pid] = mass.detach().numpy().astype(np.float64).reshape(-1)
    return targets


def _reference_from_run(run_root):
    import torch

    from slotspe_proof.redundancy_content import ContentReference

    manifest = _read(Path(run_root) / "prepared/manifest.json")
    blob = torch.load(Path(run_root) / "initial.pt", map_location="cpu", weights_only=True)
    reference = ContentReference(manifest["omic_sizes"], manifest["encoding_dim"])
    reference.load_state_dict(blob["reference"])
    return manifest, reference, blob.get("reference")


def execute(run_root, output_root, arms):
    import numpy as np
    import torch

    from slotspe_proof.crossview_deletion import (fit_codebook, geometry_summary,
                                                  multiset_structure, patient_codes,
                                                  permuted_codes_control,
                                                  run_deletion_oracle)

    run_root = Path(run_root)
    output = _resolve_output(output_root)
    if output.exists():
        raise FileExistsError("output_root already exists; refusing to overwrite: "
                              + str(output))
    output.mkdir(parents=True)
    started = time.perf_counter()

    manifest, reference, reference_state = _reference_from_run(run_root)
    config = _read(run_root / "resolved_config.json")
    arms = [arm for arm in arms if (run_root / "arms" / arm / "last.pt").is_file()]
    if not arms:
        raise RuntimeError("no arm has a saved checkpoint in " + str(run_root))

    summary = {
        "schema": "slotspe_crossview_deletion_diagnostics_v1",
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_root": str(run_root),
        "output_root": str(output),
        "arms": arms,
        "patient_training_runs": 0,
        "optimizer_steps": 0,
        "checkpoints_loaded": list(arms),
        "patient_inference_passes": 0,
    }
    _dump(output / "summary.json", summary)

    results = {}
    for arm in arms:
        checkpoint = torch.load(Path(run_root) / "arms" / arm / "last.pt",
                                map_location="cpu", weights_only=True)
        state = checkpoint["state_dict"]
        host = _rebuild_host(run_root, manifest, config, arm, state, reference_state)
        train_ids = list(manifest["folds"]["0"]["train"])
        val_ids = list(manifest["folds"]["0"]["val"])
        # The codebook is fitted on training-fold slots only; the held-out
        # fold supplies the patients the oracle is evaluated on.  The
        # training pass samples a bag with the original loader, because the
        # full bag of a training patient can hold tens of thousands of
        # patches and is not needed to place the codebook.
        train_rows = _collect_slots(host, arm, run_root, manifest, config, train_ids,
                                    training=True)
        val_rows = _collect_slots(host, arm, run_root, manifest, config, val_ids)
        summary["patient_inference_passes"] += 2
        order = [row["patient_id"] for row in val_rows]
        labels = {row["patient_id"]: row["label"] for row in val_rows}
        content_targets = _content_mass_targets(val_rows, run_root, manifest, reference)

        arm_result = {
            "arm": arm,
            "checkpoint_sha256": _sha(Path(run_root) / "arms" / arm / "last.pt"),
            "train_patients": len(train_rows),
            "validation_patients": len(val_rows),
            "oracle_population": "validation_fold",
            "codebook_population": "training_fold",
            "diagnostics": {},
        }
        for space in SLOT_SPACES:
            if space not in train_rows[0]["spaces"]:
                # The native arm has no shared/private decomposition, so the
                # V2-only spaces are simply absent for this arm.
                continue
            train_vectors = np.concatenate(
                [np.asarray(row["slots"]["wsi"][space], dtype=np.float64)
                 for row in train_rows], axis=0)
            val_matrices = [np.asarray(row["slots"]["wsi"][space], dtype=np.float64)
                            for row in val_rows]
            arm_result["diagnostics"][f"{space}_geometry"] = geometry_summary(val_matrices)
            for count in CODEBOOK_SIZES:
                codebook = fit_codebook(train_vectors, count)
                codes = {row["patient_id"]: patient_codes(matrix, codebook)
                         for row, matrix in zip(val_rows, val_matrices)}
                entry = {"codebook_size": count,
                         "train_codebook_vectors": int(train_vectors.shape[0])}
                for kind, targets in (("category", labels), ("mse", content_targets)):
                    oracle = run_deletion_oracle(codes, targets, order, target_kind=kind)
                    null = permuted_codes_control(codes, targets, order,
                                                  target_kind=kind,
                                                  repeats=PERMUTATION_REPEATS)
                    gap = float(oracle["empirical_oracle_gap"])
                    entry[kind] = {
                        "empirical_oracle_gap": gap,
                        "full_oracle_loss": float(oracle["full_oracle_loss"]),
                        "drop_oracle_loss": float(oracle["drop_oracle_loss"]),
                        "coverage": oracle["coverage"],
                        "null_gap_mean": null["null_gap_mean"],
                        "null_gap_min": null["null_gap_min"],
                        "null_gap_max": null["null_gap_max"],
                        "gap_minus_null_mean": gap - null["null_gap_mean"],
                        "gap_exceeds_null_max": gap > null["null_gap_max"],
                        "interpretation": oracle["interpretation"],
                    }
                entry["multiset_structure"] = multiset_structure(codes, order)
                arm_result["diagnostics"][f"{space}_codebook{count}"] = entry
        results[arm] = arm_result
        _dump(output / f"arm_{arm}.json", arm_result)
        del host

    summary.update({
        "status": "completed",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
        "claim_boundary": (
            "Descriptive single-fold single-seed diagnostics on an already "
            "trained checkpoint. The deletion gap is an empirical sample "
            "oracle, not population information, not a deployable predictor "
            "and not evidence of patient harm or correction efficacy."),
    })
    _dump(output / "summary.json", summary)
    return summary


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--plan", action="store_true",
                       help="Print the plan without loading any patient data (default).")
    group.add_argument("--run", action="store_true",
                       help="Execute the diagnostics on real patients.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN,
                        help="Completed cross-view queue output to read.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arm", action="append", dest="arms",
                        help="Limit to specific arms; repeatable.")
    args = parser.parse_args(argv)
    arms = args.arms or list(ALL_ARMS)
    try:
        if args.run:
            result = execute(args.run_root, args.output_root, arms)
        else:
            result = make_plan(args.run_root, args.output_root, arms)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc),
                          "type": type(exc).__name__}, ensure_ascii=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
