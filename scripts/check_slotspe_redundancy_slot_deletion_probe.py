#!/usr/bin/env python3
"""Read-only probe: do the 30/4096 checkpoints carry patient-specific slot
information that the quick-pilot checkpoint did not?

This probe reads *already stored* artifacts of the completed five-fold
best-checkpoint redundancy queue.  It executes no model forward, no
training and no patient intervention: it loads the per-patient slot tensors
that ``redundancy_best_engine`` already wrote to disk and applies the
existing finite-table deletion oracle to them.

Why this probe exists
---------------------
The quick four-arm pilot (3 epochs, 256 patches) measured a near-total slot
collapse: mean pairwise cosine 0.977-0.997 and effective rank 1.005-1.041
out of eight slots.  The 30/4096/20 queue measured a very different geometry
(fold 0 / S0 / repeat 0, recomputed here from the stored slots: wsi cosine
0.8499, effective rank 1.393).  The quick-pilot deletion result therefore
cannot be used to reject the reconstruction premise, because it was measured
on a degenerate checkpoint.

What it measures
----------------
For each arm and each of a small set of slot spaces, the patient's ``K``
slots are mapped to ``K`` discrete codes by a codebook fitted on the
evaluated population, and the exact in-sample optimum of the conditional
table over the full code multiset is compared against the exact optimum over
the multiset with one slot deleted.  The same comparison is made against a
permutation null that keeps the bucket structure but destroys the
code/target association.

Honest boundaries
-----------------
*   The codebook is fitted on the *evaluated* population here, because the
    redundancy queue stored slots only for the held-out fold.  That weakens
    the held-out guarantee of the codebook relative to the cross-view
    diagnostic, which fitted it on the training fold.  The permutation null
    is computed with the same bucket structure, so the *comparison* against
    the null stays meaningful, but the absolute gap is not comparable with
    the cross-view numbers.
*   The deletion gap is an empirical same-sample oracle.  It is not
    population mutual information, not a deployable predictor and not
    evidence of patient harm or correction efficacy.
*   The content target is a feature proxy, not a biological label.
*   One fold and one seed make every number descriptive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from slotspe_proof.crossview_deletion import (fit_codebook, multiset_structure,
                                              patient_codes, permuted_codes_control,
                                              run_deletion_oracle)

DEFAULT_QUEUE = REPO / ("local_evidence/redundancy_blca_uni2h_best_folds0to4"
                        "_seed3_run01")
DEFAULT_OUTPUT = REPO / "local_evidence/redundancy_slot_deletion_probe_20261029"
CODEBOOK_SIZES = (2, 4, 8)
PERMUTATION_REPEATS = 8
SPACES = ("wsi", "omic")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2),
                    encoding="utf-8")


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_output(value):
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else REPO / path).resolve()
    private = (REPO / "local_evidence").resolve()
    if path == private or not path.is_relative_to(private):
        raise ValueError("output_root must be a child of the ignored local_evidence/")
    return path


def _fold_dir(queue, fold):
    return Path(queue) / "folds" / str(fold)


def _arm_dirs(queue, fold):
    base = _fold_dir(queue, fold) / "arms"
    if not base.is_dir():
        return {}
    return {path.name: path for path in sorted(base.iterdir()) if path.is_dir()}


def _load_raw(arm_dir, repeat):
    path = Path(arm_dir) / f"raw_validation_repeat_{repeat:02d}.pt"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path, torch.load(path, map_location="cpu", weights_only=False)


def _collect_rows(arm_dir, repeat, space):
    """Return per-patient slot matrices and the stored content target."""
    path, raw = _load_raw(arm_dir, repeat)
    rows = []
    for item in raw:
        slots = item["slots"][space]
        matrix = slots.detach().cpu().numpy()
        if matrix.ndim == 3:
            matrix = matrix[0]
        target = None
        if repeat == 0 and "targets" in item:
            value = item["targets"][space]
            value = value.detach().cpu().numpy()
            target = value.reshape(-1).astype(np.float64)
        rows.append({"patient_id": item["patient_id"],
                     "slots": matrix.astype(np.float64),
                     "content_target": target})
    return path, rows


def _labels(queue, fold):
    manifest = _read_json(Path(queue) / "prepared" / "manifest.json")
    return {pid: int(label)
            for pid, label in manifest["folds"][str(fold)]["labels"].items()}


def _geometry(matrices):
    cosines, ranks = [], []
    for matrix in matrices:
        norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
        unit = matrix / np.maximum(norms, 1e-12)
        gram = unit @ unit.T
        count = len(matrix)
        off = gram[~np.eye(count, dtype=bool)]
        cosines.append(float(off.mean()))
        ranks.append(float(count * count / (gram ** 2).sum()))
    return {"patients": len(matrices),
            "mean_cosine": float(np.mean(cosines)),
            "mean_effective_rank": float(np.mean(ranks))}


def _probe_space(rows, labels, space, count, targets_by_patient):
    order = [row["patient_id"] for row in rows]
    matrices = [row["slots"] for row in rows]
    train_vectors = np.concatenate(matrices, axis=0)
    codebook = fit_codebook(train_vectors, count)
    codes = {row["patient_id"]: patient_codes(matrix, codebook)
             for row, matrix in zip(rows, matrices)}
    entry = {"codebook_size": count,
             "codebook_population": "evaluated_held_out_population",
             "vector_count": int(train_vectors.shape[0])}
    for kind, mapping in (("category", labels), ("mse", targets_by_patient)):
        if mapping is None:
            continue
        oracle = run_deletion_oracle(codes, mapping, order, target_kind=kind)
        null = permuted_codes_control(codes, mapping, order, target_kind=kind,
                                      repeats=PERMUTATION_REPEATS)
        gap = float(oracle["empirical_oracle_gap"])
        entry[kind] = {
            "empirical_oracle_gap": gap,
            "full_oracle_loss": float(oracle["full_oracle_loss"]),
            "drop_oracle_loss": float(oracle["drop_oracle_loss"]),
            "null_gap_mean": null["null_gap_mean"],
            "null_gap_min": null["null_gap_min"],
            "null_gap_max": null["null_gap_max"],
            "gap_minus_null_mean": gap - null["null_gap_mean"],
            "gap_exceeds_null_max": gap > null["null_gap_max"],
            "coverage": oracle["coverage"],
        }
    entry["multiset_structure"] = multiset_structure(codes, order)
    return entry


def execute(queue, output, folds, arms, repeats):
    queue = Path(queue)
    output = _resolve_output(output)
    if output.exists():
        raise FileExistsError("output_root already exists; refusing to overwrite: "
                              + str(output))
    output.mkdir(parents=True)

    summary = {"schema": "slotspe_redundancy_slot_deletion_probe_v1",
               "status": "running",
               "queue_root": str(queue),
               "output_root": str(output),
               "folds": list(folds),
               "arms_requested": list(arms),
               "repeats": list(repeats),
               "model_forwards": 0,
               "training_runs": 0,
               "optimizer_steps": 0,
               "patient_arrays_loaded": True,
               "note": "Reads slot tensors already stored by the completed queue."}
    _dump(output / "summary.json", summary)

    results = {}
    for fold in folds:
        fold_labels = _labels(queue, fold)
        available = _arm_dirs(queue, fold)
        for arm in arms:
            if arm not in available:
                continue
            arm_dir = available[arm]
            if not (arm_dir / "best.pt").is_file():
                continue
            arm_result = {"arm": arm, "fold": fold,
                          "best_checkpoint_sha256": _sha(arm_dir / "best.pt"),
                          "repeats": {}}
            for repeat in repeats:
                per_repeat = {"raw_files": []}
                for space in SPACES:
                    try:
                        raw_path, rows = _collect_rows(arm_dir, repeat, space)
                    except FileNotFoundError:
                        continue
                    per_repeat["raw_files"].append({
                        "space": space, "path": str(raw_path.relative_to(queue)),
                        "sha256": _sha(raw_path)})
                    targets = None
                    if repeat == 0:
                        targets = {row["patient_id"]: row["content_target"]
                                   for row in rows
                                   if row["content_target"] is not None}
                    if not targets:
                        continue
                    per_repeat.setdefault("spaces", {})[space] = {
                        "geometry": _geometry([row["slots"] for row in rows]),
                    }
                    for count in CODEBOOK_SIZES:
                        per_repeat["spaces"][space].setdefault("codebooks", {})[
                            str(count)] = _probe_space(rows, fold_labels, space,
                                                      count, targets)
                arm_result["repeats"][str(repeat)] = per_repeat
            results[f"{fold}/{arm}"] = arm_result
            _dump(output / f"result_fold{fold}_{arm}.json", arm_result)

    summary["status"] = "completed"
    summary["results"] = results
    summary["claim_boundary"] = (
        "Descriptive single-seed read-only probe over already stored slots. "
        "The codebook is fitted on the evaluated population because the "
        "redundancy queue stored slots only for the held-out fold, so the "
        "absolute gaps are not comparable with the training-fold-codebook "
        "cross-view numbers. The deletion gap is an empirical same-sample "
        "oracle and does not establish patient harm or correction efficacy.")
    _dump(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--plan", action="store_true",
                       help="Print the plan without reading any slot tensor (default).")
    group.add_argument("--run", action="store_true",
                       help="Read the stored slots and apply the deletion oracle.")
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", type=int, action="append", dest="folds")
    parser.add_argument("--arm", action="append", dest="arms")
    parser.add_argument("--repeat", type=int, action="append", dest="repeats")
    args = parser.parse_args(argv)
    folds = args.folds or [0]
    arms = args.arms or ["S0", "R4", "R5", "R1", "R3"]
    repeats = args.repeats or [0]
    try:
        if args.run:
            result = execute(args.queue_root, args.output_root, folds, arms, repeats)
        else:
            result = {"mode": "plan", "queue_root": str(args.queue_root),
                      "folds": folds, "arms": arms, "repeats": repeats,
                      "codebook_sizes": list(CODEBOOK_SIZES),
                      "spaces": list(SPACES),
                      "model_forwards": 0, "training_runs": 0,
                      "patient_arrays_loaded": False}
        print(json.dumps(result, ensure_ascii=True, allow_nan=False)[:4000])
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc),
                          "type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
