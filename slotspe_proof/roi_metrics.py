"""Pure numerical SlotSPE reading analysis for pre-aligned token masks.

The caller must independently establish token-row provenance, exact slide and
coordinate/annotation alignment. This module never opens patient files or runs
a model. A tumor mask means *known annotated positive*, not a negative label for
its complement. Returned numbers are reading measurements, not causal effects.

``attention`` is the native, final-iteration post-token-L1-normalization
attention, shape ``[heads, slots, tokens]``. Token rows must include real and
padding positions in precisely the order captured at the native module input.
``real_mask`` and ``tumor_mask`` are 1-D boolean arrays of length ``tokens``;
any true tumor position must also be real. Indices in outputs always refer to
the original, unfiltered input token positions. The head average is calculated
*before* real-token conditioning, separately from every individual head.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _boolean_mask(value, length, name):
    mask = np.asarray(value)
    if mask.dtype.kind != "b" or mask.ndim != 1 or mask.size != length:
        raise ValueError(f"{name} must be a 1-D boolean mask of length {length}")
    return mask


def _overlap(left, right):
    return float(np.minimum(left, right).sum())


def _matrix(values):
    return [[float(value) for value in row] for row in values]


def _top_sets(weights, original_indices, k, atol, rtol):
    """Stable ties use original input row order; do not interpret tie ranks."""
    sets = []
    slot_rows = []
    for row in weights:
        order = np.lexsort((original_indices, -row))
        chosen = order[:k]
        chosen_indices = original_indices[chosen].tolist()
        sorted_weights = row[order]
        differences = sorted_weights[:-1] - sorted_weights[1:]
        if k < len(row):
            upper = float(sorted_weights[k - 1])
            lower = float(sorted_weights[k])
            gap = max(upper - lower, 0.0)
            tolerance = float(atol + rtol * max(abs(upper), abs(lower)))
            cutoff = {
                "gap": gap,
                "tolerance": tolerance,
                "exact_tie": bool(upper == lower),
                "near_tie": bool(gap <= tolerance),
            }
        else:
            cutoff = {"gap": None, "tolerance": None, "exact_tie": False, "near_tie": False}
        sets.append(set(chosen_indices))
        slot_rows.append({
            "input_row_indices": chosen_indices,
            "weights": [float(value) for value in row[chosen]],
            "mass": float(row[chosen].sum()),
            "uniform_mass": float(k / len(row)),
            "has_exact_tie_anywhere": bool(np.any(differences == 0)),
            "cutoff": cutoff,
        })
    count = len(sets)
    intersections = [[len(sets[s] & sets[t]) for t in range(count)] for s in range(count)]
    jaccard = [[float(len(sets[s] & sets[t]) / len(sets[s] | sets[t]))
                for t in range(count)] for s in range(count)]
    return {"k": k, "slots": slot_rows, "intersection": intersections, "jaccard": jaccard}


def _one_attention(attention, real_mask, tumor_mask, top_k, min_tumor_patches,
                   tie_atol, tie_rtol):
    slots, _ = attention.shape
    real_indices = np.flatnonzero(real_mask)
    padding_mass = attention[:, ~real_mask].sum(axis=1)
    real_mass = attention[:, real_mask].sum(axis=1)
    report = {
        "status": "ok",
        "total_row_mass": attention.sum(axis=1).tolist(),
        "padding_mass": padding_mass.tolist(),
        "real_mass": real_mass.tolist(),
        "real_input_row_indices": real_indices.tolist(),
        "real_attention": None,
        "overlap": None,
        "pair_overlaps": None,
        "uniform_tv": None,
        "tumor": None,
        "top": None,
    }
    if len(real_indices) == 0:
        report["status"] = "not_evaluable_no_real_patches"
        return report
    if np.any(real_mass <= 0):
        report["status"] = "not_evaluable_zero_real_attention_mass"
        return report

    # Re-normalization is only a derived real-token summary, not model masking.
    weights = attention[:, real_mask] / real_mass[:, None]
    overlap = np.minimum(weights[:, None, :], weights[None, :, :]).sum(axis=-1)
    pairs = [{"slot_a": s, "slot_b": t, "overlap": float(overlap[s, t])}
             for s, t in combinations(range(slots), 2)]
    n_real = len(real_indices)
    tv = 0.5 * np.abs(weights - 1.0 / n_real).sum(axis=1)
    report["real_attention"] = _matrix(weights)
    report["overlap"] = _matrix(overlap)
    report["pair_overlaps"] = pairs
    report["uniform_tv"] = tv.tolist()
    report["top"] = _top_sets(weights, real_indices, min(top_k, n_real), tie_atol, tie_rtol)

    tumor_in_real = tumor_mask[real_mask]
    n_tumor = int(tumor_in_real.sum())
    q = n_tumor / n_real
    tumor = {
        "status": ("no_annotated_tumor" if not n_tumor else
                   "not_evaluable_insufficient_annotated_tumor" if n_tumor < min_tumor_patches else "eligible"),
        "annotated_count": n_tumor,
        "minimum_annotated_count": min_tumor_patches,
        "q": float(q),
        "mass": None,
        "enrichment": None,
        "mass_minus_q": None,
        "unconditioned_overlap": None,
        "conditioned_overlap": None,
        "zero_tumor_mass_slots": [],
    }
    if n_tumor:
        in_tumor = weights[:, tumor_in_real]
        tumor_mass = in_tumor.sum(axis=1)
        raw_overlap = np.minimum(in_tumor[:, None, :], in_tumor[None, :, :]).sum(axis=-1)
        conditional = [[None] * slots for _ in range(slots)]
        valid = tumor_mass > 0
        for s in range(slots):
            for t in range(slots):
                if valid[s] and valid[t]:
                    conditional[s][t] = _overlap(in_tumor[s] / tumor_mass[s],
                                                  in_tumor[t] / tumor_mass[t])
        tumor.update({
            "mass": tumor_mass.tolist(),
            "enrichment": (tumor_mass / q).tolist(),
            "mass_minus_q": (tumor_mass - q).tolist(),
            "unconditioned_overlap": _matrix(raw_overlap),
            "conditioned_overlap": conditional,
            "zero_tumor_mass_slots": np.flatnonzero(~valid).tolist(),
        })
    report["tumor"] = tumor
    return report


def analyze_attention(attention, real_mask, tumor_mask, *, top_k=20,
                      min_tumor_patches=20, tie_atol=1e-12, tie_rtol=1e-6):
    """Summarize one observation's native WSI attention without model changes.

    Returns JSON-compatible dictionaries for ``head_mean`` and ``by_head``.
    Values that cannot be computed use ``None`` and an explicit status, never
    NaN. If annotated positives are fewer than ``min_tumor_patches``, their
    statistics remain descriptive but their status disallows main analysis.
    ``top_k`` never fabricates rows when fewer than K real inputs exist.
    """
    raw = np.asarray(attention)
    if raw.ndim != 3 or min(raw.shape) < 1 or raw.shape[1] < 2:
        raise ValueError("attention must have shape [heads, slots>=2, tokens], all nonempty")
    if raw.dtype.kind not in "fiu":
        raise ValueError("attention must contain real numeric values")
    raw = raw.astype(np.float64, copy=False)
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("attention must be finite and nonnegative")
    if not np.allclose(raw.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("native attention must sum to 1 across tokens for each head and slot")
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if (isinstance(min_tumor_patches, bool) or
            not isinstance(min_tumor_patches, (int, np.integer)) or min_tumor_patches < 1):
        raise ValueError("min_tumor_patches must be a positive integer")
    for name, value in (("tie_atol", tie_atol), ("tie_rtol", tie_rtol)):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise ValueError(f"{name} must be a finite nonnegative number")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite nonnegative number") from exc
        if not np.isfinite(number) or number < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    tie_atol, tie_rtol = float(tie_atol), float(tie_rtol)
    real = _boolean_mask(real_mask, raw.shape[-1], "real_mask")
    tumor = _boolean_mask(tumor_mask, raw.shape[-1], "tumor_mask")
    if np.any(tumor & ~real):
        raise ValueError("annotated tumor rows must be real; padding cannot be tumor")
    args = (real, tumor, int(top_k), int(min_tumor_patches), tie_atol, tie_rtol)
    return {
        "schema": "slotspe_roi_attention_metrics_v1",
        "input_shape": list(raw.shape),
        "n_real": int(real.sum()),
        "n_padding": int((~real).sum()),
        "n_annotated_tumor": int(tumor.sum()),
        "n_unknown_real": int((real & ~tumor).sum()),
        "mask_semantics": "unannotated real patches are unknown, not normal",
        "tie_atol": tie_atol,
        "tie_rtol": tie_rtol,
        "head_mean": _one_attention(raw.mean(axis=0), *args),
        "by_head": [_one_attention(row, *args) for row in raw],
    }
