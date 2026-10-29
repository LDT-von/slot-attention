"""Finite-table slot-deletion diagnostics for already prepared records.

The table conditions on the *unordered multiset* of discrete slot codes.  Every
record contributes all ``K`` deletions with equal weight, including repeated
deletions that produce the same multiset.  All target-conditioned tables are
fitted and scored on these same records: their exact optimum is an empirical
oracle, not a deployable decoder or a held-out estimate of generalization.

The output is aggregate-only.  A row is treated as one independent record for
coverage reporting, but this function cannot verify patient identities or
independence.  No patients, files, checkpoints, or neural models are loaded.
"""
from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence
from decimal import Decimal
import math
from numbers import Complex, Integral, Real


_INTERPRETATION = (
    "Empirical oracle: target labels are used to fit and score the same finite "
    "tables. The gap is not population mutual information, a continuous-slot "
    "contribution, or evidence of patient prediction benefit. Records are "
    "assumed independent for coverage counts; independence is not verified."
)


def _sequence(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _categorical_target(value):
    """Validate a hashable category, including numeric values inside tuples/sets."""
    if not isinstance(value, Hashable):
        raise ValueError("categorical targets must be hashable")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("categorical targets must not contain NaN or infinity")
    elif isinstance(value, Integral):
        pass
    elif isinstance(value, Real):
        try:
            valid = math.isfinite(value)
        except (TypeError, OverflowError, ValueError) as exc:
            raise ValueError("invalid numeric categorical target") from exc
        if not valid:
            raise ValueError("categorical targets must not contain NaN or infinity")
    elif isinstance(value, Complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("categorical targets must not contain NaN or infinity")
    elif isinstance(value, (tuple, frozenset)):
        for item in value:
            _categorical_target(item)
    try:
        hash(value)
        if value != value:
            raise ValueError("categorical targets must be reflexive")
    except (TypeError, OverflowError) as exc:
        raise ValueError("categorical targets must be hashable and reflexive") from exc
    return value


def _vector_target(value, expected_dimension):
    if not _sequence(value) or not value:
        raise ValueError("MSE targets must be nonempty real vectors")
    if expected_dimension is not None and len(value) != expected_dimension:
        raise ValueError("MSE target vectors must have one consistent dimension")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError("MSE target coordinates must be finite real numbers")
        try:
            converted = float(item)
        except (TypeError, OverflowError, ValueError) as exc:
            raise ValueError("MSE target coordinates must be finite real numbers") from exc
        if not math.isfinite(converted):
            raise ValueError("MSE target coordinates must be finite real numbers")
        result.append(converted)
    return tuple(result)


def _prepare(rows, target_mode):
    if not _sequence(rows) or not rows:
        raise ValueError("rows must be a nonempty sequence of (target, codes)")
    prepared = []
    expected_slots = None
    expected_dimension = None
    for row in rows:
        if not _sequence(row) or len(row) != 2:
            raise ValueError("each row must be (target, codes)")
        target, codes = row
        if not _sequence(codes) or not codes:
            raise ValueError("codes must be a nonempty sequence of nonnegative integers")
        if expected_slots is not None and len(codes) != expected_slots:
            raise ValueError("all rows must have the same number of slots")
        cleaned_codes = []
        for code in codes:
            if isinstance(code, bool) or not isinstance(code, Integral) or code < 0:
                raise ValueError("codes must be nonnegative integers, not booleans")
            cleaned_codes.append(int(code))
        if target_mode == "category":
            target = _categorical_target(target)
        else:
            target = _vector_target(target, expected_dimension)
            expected_dimension = len(target)
        expected_slots = len(cleaned_codes)
        prepared.append((target, tuple(sorted(cleaned_codes))))
    return prepared, expected_slots, expected_dimension


def _group_records(prepared, slots):
    """Each deletion is a separate equally weighted occurrence, not a subject."""
    full_groups = defaultdict(list)
    drop_groups = defaultdict(list)
    drop_support = defaultdict(set)
    per_record_drop_keys = []
    for index, (target, codes) in enumerate(prepared):
        full_groups[codes].append(target)
        keys = []
        for j in range(slots):
            key = codes[:j] + codes[j + 1:]
            keys.append(key)
            drop_groups[key].append(target)
            drop_support[key].add(index)
        per_record_drop_keys.append(keys)
    return full_groups, drop_groups, drop_support, per_record_drop_keys


def _coverage(prepared, slots, full_groups, drop_groups, drop_support, row_keys):
    n = len(prepared)
    full_singleton_rows = sum(len(full_groups[codes]) == 1 for _, codes in prepared)
    drop_singleton_buckets = sum(len(support) == 1 for support in drop_support.values())
    any_singleton_rows = sum(any(len(drop_support[key]) == 1 for key in keys) for keys in row_keys)
    singleton_deletions = sum(
        len(drop_support[key]) == 1 for keys in row_keys for key in keys
    )
    return {
        "records": n,
        "slots_per_record": slots,
        "full_buckets": len(full_groups),
        "full_singleton_buckets": sum(len(group) == 1 for group in full_groups.values()),
        "full_singleton_record_fraction": full_singleton_rows / n,
        "drop_buckets": len(drop_groups),
        "drop_singleton_buckets": drop_singleton_buckets,
        "drop_any_singleton_record_fraction": any_singleton_rows / n,
        "drop_mean_singleton_deletion_probability": singleton_deletions / (n * slots),
        "independent_records_assumed": n,
        "deletion_occurrences": n * slots,
    }


def _gap(full_loss, drop_loss):
    value = drop_loss - full_loss
    tolerance = 1e-12 * max(1.0, abs(full_loss), abs(drop_loss))
    if value < -tolerance:
        raise ArithmeticError("empirical oracle gap became negative beyond roundoff")
    return max(0.0, value)


def _categorical_loss(groups, total_occurrences):
    terms = []
    for labels in groups.values():
        counts = Counter(labels)
        size = len(labels)
        terms.extend((count / total_occurrences) * math.log(size / count)
                     for count in counts.values())
    return math.fsum(terms)


def _mse_loss(groups, total_occurrences, dimension):
    terms = []
    try:
        for vectors in groups.values():
            size = len(vectors)
            # Scale before summing to avoid overflowing a large but finite mean.
            means = [math.fsum(vector[j] / size for vector in vectors)
                     for j in range(dimension)]
            for vector in vectors:
                for j in range(dimension):
                    difference = vector[j] - means[j]
                    squared = difference * difference
                    if not math.isfinite(squared):
                        raise ValueError("MSE target magnitude causes numerical overflow")
                    terms.append(squared / (total_occurrences * dimension))
        result = math.fsum(terms)
    except OverflowError as exc:
        raise ValueError("MSE target magnitude causes numerical overflow") from exc
    if not math.isfinite(result):
        raise ValueError("MSE target magnitude causes numerical overflow")
    return result


def empirical_categorical_oracle(rows):
    """Return full/drop in-sample optimal categorical log loss, in nats.

    Targets can be hashable categories; Python category equality applies (for
    example, ``1`` and ``True`` compare equal).  No neural decoder is used.
    """
    prepared, slots, _ = _prepare(rows, "category")
    full, drop, support, row_keys = _group_records(prepared, slots)
    full_loss = _categorical_loss(full, len(prepared))
    drop_loss = _categorical_loss(drop, len(prepared) * slots)
    return {
        "metric": "categorical_log_loss_nats",
        "full_oracle_loss": full_loss,
        "drop_oracle_loss": drop_loss,
        "empirical_oracle_gap": _gap(full_loss, drop_loss),
        "coverage": _coverage(prepared, slots, full, drop, support, row_keys),
        "interpretation": _INTERPRETATION,
    }


def empirical_mse_oracle(rows):
    """Return full/drop in-sample optimal mean squared error per target element."""
    prepared, slots, dimension = _prepare(rows, "vector")
    full, drop, support, row_keys = _group_records(prepared, slots)
    full_loss = _mse_loss(full, len(prepared), dimension)
    drop_loss = _mse_loss(drop, len(prepared) * slots, dimension)
    return {
        "metric": "mse_per_target_element",
        "target_dimension": dimension,
        "full_oracle_loss": full_loss,
        "drop_oracle_loss": drop_loss,
        "empirical_oracle_gap": _gap(full_loss, drop_loss),
        "coverage": _coverage(prepared, slots, full, drop, support, row_keys),
        "interpretation": _INTERPRETATION,
    }
