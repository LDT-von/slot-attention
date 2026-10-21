"""Descriptive SlotSPE mechanism measurements, without model or patient I/O.

These functions do not declare a research problem present or absent. In particular,
feature similarity, attention overlap, and feature-cluster purity are different
measurements; none establishes tissue semantics or predictive harm on its own.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


def summary(values: Iterable[float]) -> dict[str, Any]:
    """Summarize finite observations; preserve an empty sample as undefined."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("summary expects a one-dimensional collection of finite numbers")
    if not len(array):
        return {"n": 0, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    result = {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }
    if not all(value is None or np.isfinite(value) for value in result.values()):
        raise ValueError("summary computation exceeds the finite float64 range")
    return result


def _matrix(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _attention(values: Any, name: str = "attention") -> np.ndarray:
    array = _matrix(values, name)
    if np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    # Scaling before summing prevents overflow for large but finite weights.
    maxima = array.max(axis=1, keepdims=True)
    if np.any(maxima <= 0):
        raise ValueError(f"{name} must have positive mass in every row")
    scaled = array / maxima
    return scaled / scaled.sum(axis=1, keepdims=True)


def _entropy(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    return -terms.sum(axis=axis)


def _pair_records(matrix: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"slot_i": int(i), "slot_j": int(j), "value": float(matrix[i, j])}
        for i in range(len(matrix)) for j in range(i + 1, len(matrix))
    ]


def _effective_rank(slots: np.ndarray) -> dict[str, Any]:
    # Translate before centering so identical rows are exactly zero even when
    # their mean would acquire floating-point summation error.
    centered = slots - slots[0]
    centered -= centered.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    if not len(singular) or singular[0] == 0:
        return {
            "value": None, "reason": "zero_centered_variation", "algebraic_rank": 0,
            "variance_fractions": [],
            "definition": "exp(entropy of normalized squared singular values after centering slots)",
        }
    relative = singular / singular[0]
    fractions = relative ** 2
    fractions /= fractions.sum()
    tolerance = max(centered.shape) * np.finfo(np.float64).eps
    return {
        "value": float(np.exp(_entropy(fractions))),
        "reason": None,
        "algebraic_rank": int(np.sum(relative > tolerance)),
        "variance_fractions": fractions.tolist(),
        "definition": "exp(entropy of normalized squared singular values after centering slots)",
    }


def _group_measurements(attention: np.ndarray, token_groups: Any) -> dict[str, Any]:
    raw_groups = list(token_groups)
    if len(raw_groups) != attention.shape[1]:
        raise ValueError("token_groups must contain one external category per token")
    labels: list[Any] = []
    keys: dict[str, int] = {}
    group_indices: list[int] = []
    for value in raw_groups:
        if isinstance(value, np.generic):
            value = value.item()
        if not (value is None or isinstance(value, (str, int, float, bool))):
            raise ValueError("token group labels must be JSON scalar categories")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("token group labels must be finite")
        key = type(value).__name__ + ":" + json.dumps(value, ensure_ascii=False, allow_nan=False)
        if key not in keys:
            keys[key] = len(labels)
            labels.append(value)
        group_indices.append(keys[key])
    group_indices_array = np.asarray(group_indices)
    masses = np.stack([
        attention[:, group_indices_array == index].sum(axis=1)
        for index in range(len(labels))
    ], axis=1)
    fragmentation = []
    n_slots = len(attention)
    for index, label in enumerate(labels):
        total_mass = float(masses[:, index].sum())
        if total_mass == 0:
            fragmentation.append({
                "group": label, "total_attention_mass": 0.0, "slot_fractions": None,
                "entropy_nats": None, "normalized_entropy": None,
                "effective_slots": None, "largest_slot_fraction": None,
                "reason": "group_has_zero_attention_mass",
            })
            continue
        fractions = masses[:, index] / total_mass
        entropy = float(_entropy(fractions))
        fragmentation.append({
            "group": label, "total_attention_mass": total_mass,
            "slot_fractions": fractions.tolist(), "entropy_nats": entropy,
            "normalized_entropy": float(entropy / np.log(n_slots)) if n_slots > 1 else None,
            "effective_slots": float(np.exp(entropy)),
            "largest_slot_fraction": float(fractions.max()),
            "reason": "normalized_entropy_undefined_for_one_slot" if n_slots == 1 else None,
        })
    slot_entropy = _entropy(masses, axis=1)
    return {
        "labels": labels, "slot_group_mass": masses.tolist(),
        "slot_purity": masses.max(axis=1).tolist(),
        "slot_group_entropy_nats": slot_entropy.tolist(),
        "slot_effective_groups": np.exp(slot_entropy).tolist(),
        "group_token_counts": [int(np.sum(group_indices_array == i)) for i in range(len(labels))],
        "group_fragmentation": fragmentation,
        "interpretation": (
            "Categories were supplied externally. Feature clusters are descriptive groups, "
            "not tissue or biological semantics. Purity depends on group prevalence. "
            "Fragmentation normalizes each group's attention mass across slots."
        ),
    }


def slot_metrics(slots: Any, attention: Any, token_groups: Any = None) -> dict[str, Any]:
    """Measure slot features and row-normalized slot-to-token attention.

    ``slots`` is [S,D]; ``attention`` is nonnegative [S,N] with positive row
    mass. Attention rows are normalized independently. Zero feature vectors
    are allowed, but their cosine similarities are explicitly undefined.
    Pair distributions contain unique unordered pairs, never diagonal entries.
    """
    features = _matrix(slots, "slots")
    weights = _attention(attention)
    n_slots, feature_dim = features.shape
    if n_slots != len(weights):
        raise ValueError("slots and attention must have the same number of slots")
    scale = float(np.max(np.abs(features)))
    scaled = features / scale if scale else features.copy()
    row_scales = np.max(np.abs(features), axis=1)
    row_scaled = np.zeros_like(features)
    nonzero_rows = row_scales > 0
    row_scaled[nonzero_rows] = features[nonzero_rows] / row_scales[nonzero_rows, None]
    row_norms = np.linalg.norm(row_scaled, axis=1)
    with np.errstate(over="ignore"):
        norms = row_norms * row_scales
    if not np.all(np.isfinite(norms)):
        raise ValueError("slot norms exceed the finite float64 range")
    valid = nonzero_rows
    units = np.zeros_like(scaled)
    units[valid] = row_scaled[valid] / row_norms[valid, None]
    cosine = np.clip(units @ units.T, -1, 1)
    cosine_pairs: list[dict[str, Any]] = []
    l2_pairs: list[dict[str, Any]] = []
    for i in range(n_slots):
        for j in range(i + 1, n_slots):
            defined = bool(valid[i] and valid[j])
            cosine_pairs.append({
                "slot_i": i, "slot_j": j,
                "value": float(cosine[i, j]) if defined else None,
                "reason": None if defined else "zero_feature_vector",
            })
            pair_scale = max(row_scales[i], row_scales[j])
            pair_i = features[i] / pair_scale if pair_scale else features[i]
            pair_j = features[j] / pair_scale if pair_scale else features[j]
            denominator = float(np.hypot(np.linalg.norm(pair_i), np.linalg.norm(pair_j)))
            l2_pairs.append({
                "slot_i": i, "slot_j": j,
                "value": float(np.linalg.norm(pair_i - pair_j) / denominator) if denominator else None,
                "reason": None if denominator else "both_feature_vectors_zero",
            })
    attention_units = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    attention_cosine = np.clip(attention_units @ attention_units.T, 0, 1)
    tv = 0.5 * np.abs(weights[:, None, :] - weights[None, :, :]).sum(axis=2)
    js = np.zeros((n_slots, n_slots))
    entropy = _entropy(weights, axis=1)
    for i in range(n_slots):
        for j in range(i + 1, n_slots):
            midpoint_entropy = float(_entropy(0.5 * (weights[i] + weights[j])))
            value = (midpoint_entropy - 0.5 * (entropy[i] + entropy[j])) / np.log(2)
            js[i, j] = js[j, i] = float(np.clip(value, 0, 1))
    token_mass = weights.sum(axis=0)
    claimed = token_mass > 0
    assignments = np.zeros_like(weights)
    assignments[:, claimed] = weights[:, claimed] / token_mass[claimed]
    assignment_entropy = _entropy(assignments, axis=0)
    n_tokens = weights.shape[1]
    return {
        "schema_version": 1,
        "n_slots": int(n_slots), "feature_dim": int(feature_dim), "n_tokens": int(n_tokens),
        "feature": {
            "norms": norms.tolist(), "norm_summary": summary(norms),
            "offdiag_cosine": cosine_pairs,
            "offdiag_cosine_summary": summary(pair["value"] for pair in cosine_pairs if pair["value"] is not None),
            "undefined_cosine_pairs": int(sum(pair["value"] is None for pair in cosine_pairs)),
            "normalized_l2": l2_pairs,
            "normalized_l2_definition": "L2(x_i-x_j) / sqrt(L2(x_i)^2 + L2(x_j)^2)",
            "centered_effective_rank": _effective_rank(scaled),
        },
        "attention": {
            "normalization": "each slot row sums to one",
            "offdiag_cosine": _pair_records(attention_cosine),
            "offdiag_total_variation": _pair_records(tv),
            "offdiag_shared_mass": _pair_records(np.clip(1 - tv, 0, 1)),
            "offdiag_jensen_shannon_bits": _pair_records(js),
            "entropy_nats": entropy.tolist(),
            "normalized_entropy": (entropy / np.log(n_tokens)).tolist() if n_tokens > 1 else [None] * n_slots,
            "normalized_entropy_reason": None if n_tokens > 1 else "one_token_has_zero_log_cardinality",
            "effective_token_support": np.exp(entropy).tolist(),
            "participation_token_support": (1 / np.sum(weights ** 2, axis=1)).tolist(),
            "token_assignment_entropy_nats": [float(value) if claimed[i] else None for i, value in enumerate(assignment_entropy)],
            "token_assignment_normalized_entropy": [
                float(value / np.log(n_slots)) if claimed[i] and n_slots > 1 else None
                for i, value in enumerate(assignment_entropy)
            ],
            "token_assignment_entropy_reason": "unclaimed_tokens_are_null; normalized_entropy_is_null_for_one_slot",
            "mass_weighted_assignment_entropy_nats": float(np.sum(assignment_entropy * token_mass) / n_slots),
            "unclaimed_token_count": int(np.sum(~claimed)),
        },
        "groups": _group_measurements(weights, token_groups) if token_groups is not None else None,
        "interpretation": "Descriptive measurements only; no universal similarity threshold or predictive-harm verdict.",
    }


def matched_stability(
    attention_a: Any, attention_b: Any, *, n_permutations: int = 256, seed: int = 0,
) -> dict[str, Any]:
    """Match slot attention on the SAME tokens by minimum total variation.

    Token order/content must already be aligned; this function cannot infer
    correspondence between different token samples. A random slot permutation
    baseline is descriptive and is not a significance test. Its local NumPy RNG
    does not consume any model RNG state.
    """
    a = _attention(attention_a, "attention_a")
    b = _attention(attention_b, "attention_b")
    if a.shape != b.shape:
        raise ValueError("attention_a and attention_b must have the same [slots,tokens] shape")
    if isinstance(n_permutations, bool) or int(n_permutations) != n_permutations or n_permutations < 0:
        raise ValueError("n_permutations must be a nonnegative integer")
    costs = 0.5 * np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)
    rows, columns = linear_sum_assignment(costs)
    n_slots = len(a)
    optimum = float(costs[rows, columns].sum())
    alternatives = []
    if n_slots > 1:
        for row, column in zip(rows, columns):
            forbidden = costs.copy()
            forbidden[row, column] = np.inf
            alternative_rows, alternative_columns = linear_sum_assignment(forbidden)
            alternatives.append(float(costs[alternative_rows, alternative_columns].sum()))
    gap = max(0.0, (min(alternatives) - optimum) / n_slots) if alternatives else None
    sorted_costs = np.sort(costs, axis=1)
    generator = np.random.default_rng(seed)
    permutation_distances = [
        float(np.mean(costs[np.arange(n_slots), generator.permutation(n_slots)]))
        for _ in range(int(n_permutations))
    ]
    return {
        "schema_version": 1, "n_slots": int(n_slots), "n_tokens": int(a.shape[1]),
        "distance": "total_variation_of_row_normalized_attention",
        "matched_mean_distance": optimum / n_slots,
        "unmatched_mean_distance": float(np.trace(costs) / n_slots),
        "unmatched_definition": "same-index pairing before Hungarian matching",
        "matches": [
            {
                "slot_a": int(row), "slot_b": int(column), "distance": float(costs[row, column]),
                "row_best_vs_second_best_margin": float(sorted_costs[row, 1] - sorted_costs[row, 0]) if n_slots > 1 else None,
            }
            for row, column in zip(rows, columns)
        ],
        "alternative_assignment_mean_distance_gap": gap,
        "assignment_ambiguous": bool(np.isclose(gap, 0, rtol=1e-10, atol=1e-12)) if gap is not None else None,
        "ambiguity_reason": "only_one_slot_no_alternative_assignment" if n_slots == 1 else "numerical_tie_if_gap_close_to_zero",
        "random_permutation_baseline": {
            "seed": int(seed), "summary": summary(permutation_distances),
            "interpretation": "random slot pairing reference, not a p-value or independent patient sample",
        },
        "interpretation": (
            "Attention correspondence over identical ordered tokens only. Matching can conceal slot-index swaps; "
            "inspect both matched and same-index distances, and matching ambiguity. No tissue-semantic claim."
        ),
    }
