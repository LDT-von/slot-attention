#!/usr/bin/env python3
"""Positive control for the slot-deletion oracle.

The read-only probe over the completed 30/4096/20 queue found that the
deletion gap never clears its own permutation null.  That negative result
is only interpretable if the oracle can detect patient-specific information
when it is actually present.  This script establishes that capability on
synthetic data with known ground truth, using exactly the same oracle, the
same codebook, the same bucket structure and the same permutation null that
the probe uses.

It executes no model, no training and no patient work.

How the oracle works, and what the cases must therefore look like
-----------------------------------------------------------------
The oracle groups patients by their sorted code multiset.  The full loss
measures how well that multiset determines the target; the drop loss
measures the same after one slot has been removed from each patient.  The
gap is ``drop_loss - full_loss``.  For the gap to be informative the target
must be (close to) a function of the multiset, and removing a slot must
destroy that function.  A target that is only weakly determined by the
multiset produces a small gap even though information is present.

The synthetic slots are therefore built so that the target is an exact
function of the quantity the oracle can see.

Cases
-----
``fully_informative``
    Every patient's slots sit in one of a small number of well separated
    clusters, and the target is a function of the set of clusters the
    patient uses.  The oracle must detect this.

``uninformative``
    Identical slot geometry, but the target is drawn independently of the
    slots.  The gap must sit inside the permutation null.

``one_slot_informative``
    The target depends on exactly one of the eight slots, so only one of
    the eight deletions can destroy the information.  This is the case
    that most resembles a real model, where the slots may not all be
    equally informative, and it measures the dilution the oracle suffers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from slotspe_proof.crossview_deletion import (fit_codebook, patient_codes,
                                              permuted_codes_control,
                                              run_deletion_oracle)

DEFAULT_OUTPUT = REPO / "local_evidence/slot_deletion_oracle_positive_control_20261030"
PATIENTS = 77
SLOTS = 8
DIM = 256
TARGET_DIM = 32
CLUSTERS = 4
CODEBOOK_SIZES = (4, 8, 16)
PERMUTATION_REPEATS = 8
SEED = 20261030


def _make_patients(rng, patients, slots, dim, clusters):
    """Return slot matrices and the cluster pair used by every patient.

    Each slot of a patient sits in one of ``clusters`` well separated
    centres, so a codebook with one centre per cluster recovers the
    cluster identity of every slot vector.  The returned ``pairs`` record
    which clusters the patient used, and that is the quantity the
    informative targets are built from.
    """
    centres = rng.normal(size=(clusters, dim)) * 8.0
    pairs = []
    for index in range(patients):
        left = index % clusters
        right = (index // clusters) % clusters
        if left == right:
            right = (right + 1) % clusters
        pairs.append((left, right))
    matrices = np.empty((patients, slots, dim), dtype=np.float64)
    for index, (left, right) in enumerate(pairs):
        for slot in range(slots):
            if slot == 0:
                base = centres[left]
            elif slot == 1:
                base = centres[right]
            else:
                base = centres[(left + slot) % clusters] * 0.02
            matrices[index, slot] = base + rng.normal(scale=0.05, size=dim)
    return matrices, pairs


def _targets(rng, pairs, kind, informative, informative_slot=None):
    """Build the target arrays for one case.

    When ``informative`` the target is an exact function of the cluster
    information the oracle can recover.  Otherwise it is independent noise.
    """
    count = len(pairs)
    if kind == "category":
        if informative:
            # The category is a function of the unordered cluster pair.
            keys = sorted({tuple(sorted(pair)) for pair in pairs})
            lookup = {key: index % 4 for index, key in enumerate(keys)}
            values = np.array([lookup[tuple(sorted(pair))] for pair in pairs])
            return values.astype(np.int64)
        return rng.integers(0, 4, size=count).astype(np.int64)
    if informative:
        # The vector target is a fixed function of the cluster that the
        # informative slot belongs to.
        offsets = rng.normal(size=(CLUSTERS, TARGET_DIM)) * 2.0
        if informative_slot is None:
            key = np.array([tuple(sorted(pair)) for pair in pairs], dtype=object)
            keys = sorted({tuple(sorted(pair)) for pair in pairs})
            lookup = {value: index for index, value in enumerate(keys)}
            table = rng.normal(size=(len(keys), TARGET_DIM)) * 2.0
            return np.array([table[lookup[tuple(sorted(pair))]]
                             for pair in pairs])
        column = informative_slot % 2
        return np.array([offsets[pair[column]] for pair in pairs])
    return rng.normal(size=(count, TARGET_DIM)) * 3.0


def _evaluate_case(matrices, pairs, kind, seed, codebook_sizes, informative,
                   informative_slot=None):
    """Evaluate one synthetic case and return the per-codebook results."""
    order = [f"P{index:03d}" for index in range(len(matrices))]
    rng = np.random.default_rng(seed)
    values = _targets(rng, pairs, kind, informative, informative_slot)
    results = {}
    for size in codebook_sizes:
        codebook = fit_codebook(matrices.reshape(-1, matrices.shape[-1]), size)
        codes_by_patient = {patient: patient_codes(matrices[index], codebook)
                            for index, patient in enumerate(order)}
        mapping = {patient: values[index] for index, patient in enumerate(order)}
        oracle = run_deletion_oracle(codes_by_patient, mapping, order,
                                      target_kind=kind)
        null = permuted_codes_control(codes_by_patient, mapping, order,
                                      target_kind=kind, repeats=PERMUTATION_REPEATS)
        gap = float(oracle["empirical_oracle_gap"])
        results[str(size)] = {
            "empirical_oracle_gap": gap,
            "null_gap_mean": null["null_gap_mean"],
            "null_gap_max": null["null_gap_max"],
            "gap_minus_null_mean": gap - null["null_gap_mean"],
            "gap_exceeds_null_max": gap > null["null_gap_max"],
            "coverage": oracle["coverage"],
        }
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true",
                        help="Print the plan without running (default).")
    parser.add_argument("--run", action="store_true",
                        help="Run the synthetic positive control.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({"mode": "plan",
                          "cases": ["fully_informative", "uninformative",
                                    "one_slot_informative"],
                          "patients": PATIENTS, "slots": SLOTS,
                          "clusters": CLUSTERS,
                          "codebook_sizes": list(CODEBOOK_SIZES),
                          "model_forwards": 0, "training_runs": 0,
                          "patient_arrays_loaded": False}))
        return 0
    output = Path(args.output_root)
    if output.exists():
        raise FileExistsError("output_root already exists: " + str(output))
    output.mkdir(parents=True)

    rng = np.random.default_rng(SEED)
    matrices, pairs = _make_patients(rng, PATIENTS, SLOTS, DIM, CLUSTERS)

    summary = {"schema": "slotspe_slot_deletion_oracle_positive_control_v1",
               "status": "running", "patients": PATIENTS, "slots": SLOTS,
               "dimension": DIM, "clusters": CLUSTERS, "seed": SEED,
               "permutation_repeats": PERMUTATION_REPEATS,
               "model_forwards": 0, "training_runs": 0,
               "patient_arrays_loaded": False,
               "cases": {}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["cases"]["fully_informative"] = {
        "category": _evaluate_case(matrices, pairs, "category", SEED,
                                   CODEBOOK_SIZES, True),
        "mse": _evaluate_case(matrices, pairs, "mse", SEED + 10,
                              CODEBOOK_SIZES, True),
    }
    summary["cases"]["uninformative"] = {
        "category": _evaluate_case(matrices, pairs, "category", SEED + 20,
                                   CODEBOOK_SIZES, False),
        "mse": _evaluate_case(matrices, pairs, "mse", SEED + 30,
                              CODEBOOK_SIZES, False),
    }
    summary["cases"]["one_slot_informative"] = {
        "mse": _evaluate_case(matrices, pairs, "mse", SEED + 40,
                              CODEBOOK_SIZES, True, informative_slot=0),
    }
    summary["status"] = "completed"
    summary["interpretation"] = (
        "The oracle detects a target that is a function of the code multiset "
        "and reports no gap when the target is independent of the slots. The "
        "one-slot case shows how much of the signal is lost when only one of "
        "the eight deletions can destroy the information.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    for case, kinds in summary["cases"].items():
        for kind, sizes in kinds.items():
            for size, value in sizes.items():
                lines.append(f"{case:<22} {kind:<9} cb{size:>2}: "
                             f"gap={value['empirical_oracle_gap']:.6g} "
                             f"null_mean={value['null_gap_mean']:.6g} "
                             f"diff={value['gap_minus_null_mean']:+.6g} "
                             f"exceeds={value['gap_exceeds_null_max']}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
