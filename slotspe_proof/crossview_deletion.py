"""Slot-deletion oracle diagnostics for already trained cross-view checkpoints.

This module is the *connector* between trained slot tensors and the existing
finite-table oracle in :mod:`slotspe_proof.crossview_diagnostic`.  It contains
no patient access, no model execution and no checkpoint loading; the caller
owns data, weights and execution authorization.

What the oracle computes
-------------------------
Each patient's ``K`` slots are mapped to ``K`` discrete codes by a *fixed*
codebook.  The unordered multiset of codes is ``S``; deleting slot ``j``
leaves ``S^-``.  For a patient-level target ``T`` (a category) or ``Y`` (a
continuous vector) the exact in-sample optimum of a conditional table is

    categorical:  full = H_emp(T|S),   drop = H_emp(T|S^-)
    continuous:   full = E||Y - E[Y|S]||^2,  drop = E||Y - E[Y|S^-]||^2

and the deletion gap is ``drop - full >= 0``.  Because both tables are exact
sample optima over the *same* records, the gap cannot be inflated by a weak
neural decoder; this is the reason the oracle replaces the
delete-a-slot-and-retrain comparison.

Boundary that the caller must not forget
----------------------------------------
The gap is a *sample* quantity.  Target labels are used to fit and to score
the same finite tables, so this is an empirical oracle, not a deployable
predictor, not a population mutual information, and not evidence of patient
prediction benefit.  With few patients the multiset buckets are sparse, many
buckets hold a single record, and the oracle can report a nonzero gap for
purely structural reasons.  :func:`permuted_codes_control` provides the
corresponding null so that a real gap can be compared against the gap that
the *same* bucket sparsity produces when the code/target association is
destroyed.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from .crossview_diagnostic import empirical_categorical_oracle, empirical_mse_oracle

# Codebook sizes evaluated by default.  Each is reported separately because
# the multiset bucket count grows combinatorially with the code count and the
# resulting sparsity changes the interpretation of the gap.
DEFAULT_CODEBOOK_SIZES = (2, 4, 8)


def _as_matrix(vectors, name):
    """Accept a [N,D] array-like of finite floats and return float64 [N,D]."""
    if isinstance(vectors, np.ndarray):
        array = vectors
    else:
        try:
            array = np.asarray(list(vectors), dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a numeric [N,D] array") from error
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must have shape [N,D] with N,D >= 1")
    if array.dtype != np.float64:
        array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def fit_codebook(vectors, count, *, steps=10):
    """Fit a deterministic Lloyd codebook on training-fold slot vectors.

    Initialization is farthest-point (no RNG), so the codebook is a pure
    function of the supplied vectors and the count.  The caller must supply
    *training-fold* vectors only; a codebook fitted on held-out patients
    would make the bucket structure depend on the evaluated population.

    The returned centres are plain float64 and carry no gradient or model
    state; they are a measurement instrument, not a trained module.
    """
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError("count must be a positive integer")
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    data = _as_matrix(vectors, "vectors")
    if data.shape[0] < count:
        raise ValueError(f"fit requires N >= count, got {data.shape[0]} < {count}")
    squared = (data * data).sum(1)
    selected = [int(np.argmax(squared))]
    minimum = ((data - data[selected[0]]) ** 2).sum(1)
    while len(selected) < count:
        candidate = minimum.copy()
        candidate[selected] = -np.inf
        index = int(np.argmax(candidate))
        selected.append(index)
        minimum = np.minimum(minimum, ((data - data[index]) ** 2).sum(1))
    centres = data[selected].copy()
    for _ in range(steps):
        assignment = assign_codes(data, centres)
        revised = centres.copy()
        for index in range(count):
            members = data[assignment == index]
            if members.shape[0]:
                revised[index] = members.mean(0)
        if np.allclose(revised, centres, atol=0.0, rtol=0.0):
            centres = revised
            break
        centres = revised
    return centres


def assign_codes(vectors, codebook):
    """Nearest-code assignment under squared Euclidean distance."""
    data = _as_matrix(vectors, "vectors")
    centres = _as_matrix(codebook, "codebook")
    if centres.shape[0] < 1:
        raise ValueError("codebook must contain at least one code")
    if data.shape[1] != centres.shape[1]:
        raise ValueError("vector and codebook feature dimensions differ")
    distances = (
        (data * data).sum(1)[:, None]
        - 2.0 * data @ centres.T
        + (centres * centres).sum(1)[None, :]
    )
    return np.argmin(distances, axis=1)


def patient_codes(slot_matrix, codebook):
    """Map one patient's [K,D] slot matrix to a tuple of K python ints.

    The order of the returned codes follows the slot order; the oracle
    itself is permutation invariant because it sorts the multiset.
    """
    matrix = _as_matrix(slot_matrix, "slot_matrix")
    if matrix.shape[0] < 1:
        raise ValueError("slot_matrix must contain at least one slot")
    codes = assign_codes(matrix, codebook)
    return tuple(int(value) for value in codes)


def build_rows(codes_by_patient, targets_by_patient, order):
    """Build oracle rows in a fixed patient order.

    ``targets_by_patient`` values are categories (hashable) or float
    sequences.  Every patient must appear in both mappings and must have the
    same number of slots, which the oracle validates itself.
    """
    rows = []
    for patient in order:
        if patient not in codes_by_patient or patient not in targets_by_patient:
            raise ValueError("patient missing from codes or targets: " + str(patient))
        codes = codes_by_patient[patient]
        target = targets_by_patient[patient]
        if isinstance(target, np.ndarray):
            target = target.tolist()
        if isinstance(target, (list, tuple)):
            values = []
            for value in target:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError("continuous targets must contain real numbers")
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("continuous targets must be finite")
                values.append(value)
            target = tuple(values)
        else:
            if isinstance(target, (bool, np.bool_)):
                raise ValueError("categorical targets must not be booleans")
            if isinstance(target, (Integral, np.integer)):
                target = int(target)
        rows.append((target, tuple(int(code) for code in codes)))
    if not rows:
        raise ValueError("rows must be nonempty")
    return rows


def run_deletion_oracle(codes_by_patient, targets_by_patient, order, *, target_kind):
    """Run the categorical or MSE empirical oracle on patient-level codes.

    ``target_kind`` must be ``"category"`` or ``"mse"``.  The returned
    dictionary carries the oracle result verbatim plus the codebook and
    bucket structure that make the number interpretable.
    """
    if target_kind not in {"category", "mse"}:
        raise ValueError("target_kind must be category or mse")
    rows = build_rows(codes_by_patient, targets_by_patient, order)
    if target_kind == "category":
        result = empirical_categorical_oracle(rows)
    else:
        result = empirical_mse_oracle(rows)
    return result


def permuted_codes_control(codes_by_patient, targets_by_patient, order, *,
                           target_kind="mse", seed=20261029, repeats=8):
    """Null control: destroy the code/target association, keep the buckets.

    Each repeat permutes the *code vectors* across patients with a fixed
    seed and keeps the patient order, the target mapping and the number of
    slots untouched.  The resulting bucket structure is statistically the
    same as the real one, so the gap it produces is the gap attributable to
    bucket sparsity and memorization alone.

    The reported value is the mean and the spread of the repeat gaps, not a
    p-value: the repeats are a small deterministic sample of permutations.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 1:
        raise ValueError("seed must be a positive integer")
    if isinstance(repeats, bool) or not isinstance(repeats, Integral) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    codes = [tuple(codes_by_patient[patient]) for patient in order]
    if len(codes) < 2:
        raise ValueError("permutation control needs at least two patients")
    gaps = []
    for repeat in range(repeats):
        generator = np.random.default_rng(seed + repeat)
        index = generator.permutation(len(codes))
        permuted = {patient: codes[int(index[position])]
                    for position, patient in enumerate(order)}
        result = run_deletion_oracle(permuted, targets_by_patient, order,
                                     target_kind=target_kind)
        gaps.append(float(result["empirical_oracle_gap"]))
    array = np.asarray(gaps, dtype=np.float64)
    return {
        "repeats": repeats,
        "seed": seed,
        "target_kind": target_kind,
        "null_gap_mean": float(array.mean()),
        "null_gap_min": float(array.min()),
        "null_gap_max": float(array.max()),
        "null_gap_std_population": float(array.std()),
        "null_gaps": gaps,
        "interpretation": (
            "Codes were permuted across patients, so any gap here is caused by "
            "bucket sparsity and in-sample memorization, not by information."),
    }


def multiset_structure(codes_by_patient, order):
    """Report how identifiable the code multiset is for a single patient.

    A multiset that is nearly unique per patient can produce a large deletion
    gap for structural reasons alone: the deleted slot is then the only thing
    that distinguishes two otherwise identical buckets.  This is exactly the
    "slots collapsed into a patient fingerprint" failure mode, so the
    multiplicity is reported next to every gap.
    """
    counts = {}
    for patient in order:
        key = tuple(sorted(int(code) for code in codes_by_patient[patient]))
        counts[key] = counts.get(key, 0) + 1
    multiplicities = sorted(counts.values(), reverse=True)
    patients = len(order)
    return {
        "patients": patients,
        "distinct_multisets": len(counts),
        "distinct_multiset_fraction": len(counts) / patients if patients else None,
        "largest_multiset_multiplicity": multiplicities[0] if multiplicities else 0,
        "patients_in_unique_multiset": sum(1 for value in multiplicities if value == 1),
        "patients_in_unique_multiset_fraction":
            sum(1 for value in multiplicities if value == 1) / patients if patients else None,
        "mean_multiset_multiplicity": float(np.mean(multiplicities)) if multiplicities else None,
    }


def geometry_summary(slot_matrices):
    """Slot-geometry summary reused from the redundancy engine.

    The import is local so that this module stays importable in the
    stdlib-only plan phase.
    """
    import torch

    from .redundancy_engine import angular_metrics

    rows = []
    for matrix in slot_matrices:
        value = torch.as_tensor(np.asarray(matrix, dtype=np.float64),
                                dtype=torch.float64)
        rows.append(angular_metrics(value))
    if not rows:
        return None

    def mean(name):
        values = [row[name] for row in rows if row.get(name) is not None]
        return float(np.mean(values)) if values else None

    return {
        "patients": len(rows),
        "mean_cosine": mean("cosine"),
        "mean_absolute_cosine": mean("absolute_cosine"),
        "mean_max_cosine": mean("max_cosine"),
        "mean_fraction_cosine_above_09": mean("fraction_cosine_above_09"),
        "mean_effective_rank": mean("effective_rank"),
        "pair_metrics_defined_patients":
            sum(1 for row in rows if row.get("pair_metrics_defined")),
        "note": (
            "Descriptive geometry of the slot set. These are not clinical "
            "endpoints and cannot by themselves show patient harm or benefit."),
    }
