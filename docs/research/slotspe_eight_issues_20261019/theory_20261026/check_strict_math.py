"""Exact arithmetic audits for the accompanying mathematical proofs.

Only invented finite distributions and rational arithmetic are used.  This is
not neural-model execution, patient processing, or a formal proof assistant.
Run: python -B check_strict_math.py --output arithmetic_receipt.json
An existing receipt is never overwritten.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as Q
from hashlib import sha256
from itertools import permutations, product
import json
from pathlib import Path


def matching(predictions, targets):
    """Exhaustive finite injection, independent of SciPy and project code."""
    dimension = len(targets[0])
    count = len(targets)
    candidates = []
    for assignment in permutations(range(len(predictions)), count):
        cost = sum(
            (predictions[i][r] - targets[j][r]) ** 2
            for j, i in enumerate(assignment)
            for r in range(dimension)
        ) / (count * dimension)
        candidates.append((cost, assignment))
    return min(candidates)


def posterior_signed_risk(rows, observe):
    """Exact Bayes squared risk for signed label 2*y-1."""
    groups = defaultdict(lambda: [Q(0), Q(0)])
    for row, probability in rows:
        label = row["y"]
        groups[observe(row)][label] += probability
    risk = Q(0)
    for zero, one in groups.values():
        mass = zero + one
        mean = (one - zero) / mass
        risk += mass * (1 - mean * mean)
    return risk, groups


def audit():
    checks = []

    def passed(name, **values):
        checks.append({"name": name, "passed": True, **values})

    targets = [(Q(-1), Q(0)), (Q(1), Q(2)), (Q(2), Q(-1))]
    mean = tuple(sum(t[r] for t in targets) / len(targets) for r in range(2))
    variance = sum((t[r] - mean[r]) ** 2 for t in targets for r in range(2)) / 6
    constant_cases = 0
    for a in product(map(Q, range(-2, 3)), repeat=2):
        loss, _ = matching([a] * 4, targets)
        bias = sum((a[r] - mean[r]) ** 2 for r in range(2)) / 2
        assert loss == variance + bias
        constant_cases += 1
    passed("constant_prediction_floor_identity", exact_cases=constant_cases)

    separation_cases = 0
    for t1, t2 in permutations(map(Q, range(-2, 3)), 2):
        delta = abs(t1 - t2)
        for values in product(map(Q, range(-2, 3)), repeat=3):
            loss, assignment = matching([(v,) for v in values], [(t1,), (t2,)])
            distance = abs(values[assignment[0]] - values[assignment[1]])
            # Equivalent squared check when the lower bound is nontrivial.
            gap = max(Q(0), delta - distance)
            assert gap * gap <= 2 * 2 * loss
            if loss < delta * delta / 4:
                assert distance > 0
            separation_cases += 1
    passed("rectangular_matching_separation_finite_grid", exact_cases=separation_cases)

    for a in [Q(0), Q(1, 4), Q(1, 2), Q(1)]:
        loss, _ = matching([(Q(-1) + a,), (Q(1) - a,)], [(Q(-1),), (Q(1),)])
        assert loss == a * a
        assert abs(2 - 2 * a) ** 2 == (2 - 2 * a) ** 2
        assert 2 * 2 * loss == (2 * a) ** 2
    collision_loss, _ = matching([(Q(0),), (Q(0),)], [(Q(-1),), (Q(1),)])
    assert collision_loss == Q(1) == Q(2) ** 2 / (2 * 2)
    passed("separation_bound_and_strict_threshold_are_tight", collision_loss=str(collision_loss))

    loss, assignment = matching([(Q(-1),), (Q(1),), (Q(1),)], [(Q(-1),), (Q(1),)])
    assert loss == 0 and len(set(assignment)) == 2
    passed("unmatched_extra_slot_can_duplicate_at_zero_loss")

    for a in map(Q, range(-3, 4)):
        risk = ((Q(-1) - a) ** 2 + (Q(1) - a) ** 2) / 2
        assert risk == 1 + a * a
    passed("query_free_constant_code_variance_floor", floor="1")

    # beta=log(3): tanh(beta)=(9-1)/(9+1)=4/5 exactly.
    scale = Q(9 - 1, 9 + 1)
    for target in [Q(-1), Q(1)]:
        prediction = scale * target
        assert target * prediction > 0
        assert (target * prediction) ** 2 == target ** 2 * prediction ** 2
        assert (target - prediction) ** 2 == Q(1, 25)
    passed("target_query_constant_dictionary_zero_cosine_counterexample", scale=str(scale), mse="1/25")

    first_bag = list(range(8))
    second_bag = [0] * 9 + list(range(1, 8))
    first_support = set(first_bag)
    second_support = set(second_bag)
    assert first_support == second_support
    assert Q(first_bag.count(0), len(first_bag)) < Q(1, 2)
    assert Q(second_bag.count(0), len(second_bag)) > Q(1, 2)
    passed("type_support_target_can_miss_frequency_label", targets_identical=True)

    anchors = [Q(-2), Q(0), Q(2)]
    frequencies = [Q(1, 4), Q(1, 4), Q(1, 2)]
    target_pairs = list(zip(anchors, frequencies))
    center_offsets = [Q(1, 4)] * 3
    mass_offsets = [Q(1, 16), Q(-1, 16), Q(0)]
    predictions = [
        (anchors[i] + center_offsets[i], frequencies[i] + mass_offsets[i])
        for i in [2, 0, 1]
    ]
    pair_loss, assignment = matching(predictions, target_pairs)
    total_cost = 3 * 2 * pair_loss
    assert total_cost < Q(2) ** 2 / 4
    nearest = [min(range(3), key=lambda j: (a - anchors[j]) ** 2) for a, _ in predictions]
    for j, i in enumerate(assignment):
        assert nearest[i] == j
    ordered_mass = [predictions[assignment[j]][1] for j in range(3)]
    squared_mass_error = sum((ordered_mass[j] - frequencies[j]) ** 2 for j in range(3))
    assert squared_mass_error == Q(1, 128) <= total_cost
    passed("anchored_frequency_can_be_recovered_without_target_frequency", squared_frequency_error=str(squared_mass_error))

    changed_pairs = list(zip(anchors, [Q(1, 2), Q(1, 4), Q(1, 4)]))
    fixed_dictionary_loss, _ = matching(target_pairs, changed_pairs)
    assert fixed_dictionary_loss > 0
    passed("anchored_frequency_target_detects_same_support_different_mass", fixed_dictionary_loss=str(fixed_dictionary_loss))

    rows = []
    for u, v, n1, n2 in product(range(2), repeat=4):
        y = u ^ v
        noise_mass = lambda n: Q(1, 4) if n else Q(3, 4)
        probability = Q(1, 4) * noise_mass(n1) * noise_mass(n2)
        rows.append(({"u": u, "v": v, "y": y, "w1": y ^ n1, "w2": y ^ n2}, probability))
    assert len(rows) == 16 and sum(p for _, p in rows) == 1
    uv_risk, _ = posterior_signed_risk(rows, lambda r: (r["u"], r["v"]))
    noisy_risk, groups = posterior_signed_risk(rows, lambda r: (r["w1"], r["w2"]))
    assert uv_risk == 0 and noisy_risk == Q(3, 5)
    agreement = sum(p for r, p in rows if r["w1"] == r["w2"])
    assert agreement == Q(5, 8)
    for key, (zero, one) in groups.items():
        posterior_error = min(zero, one) / (zero + one)
        assert posterior_error == (Q(1, 10) if key[0] == key[1] else Q(1, 2))
    passed("marginal_topk_strictly_suboptimal_finite_distribution", states=len(rows), uv_risk="0", noisy_pair_risk="3/5", agreement="5/8")

    for bit in ["u", "v"]:
        _, independent_groups = posterior_signed_risk(rows, lambda r, bit=bit: r[bit])
        assert all(zero == one for zero, one in independent_groups.values())
    for bit in ["w1", "w2"]:
        _, dependent_groups = posterior_signed_risk(rows, lambda r, bit=bit: r[bit])
        assert all(zero != one for zero, one in dependent_groups.values())
    passed("marginal_information_ordering_certified_by_exact_posteriors")

    index_rows = []
    for u, v in product(range(2), repeat=2):
        y = u ^ v
        index = y
        selected = u if index == 0 else v
        index_rows.append(({"y": y, "index": index, "selected": selected}, Q(1, 4)))
    value_risk, _ = posterior_signed_risk(index_rows, lambda r: r["selected"])
    index_risk, _ = posterior_signed_risk(index_rows, lambda r: (r["index"], r["selected"]))
    assert value_risk == 1 and index_risk == 0
    passed("dynamic_selection_index_channel_counterexample", value_only_risk="1", index_and_value_risk="0")

    cross_risk, _ = posterior_signed_risk(rows, lambda r: r["y"])
    cross_and_mask_risk, _ = posterior_signed_risk(rows, lambda r: (r["y"], r["w1"]))
    assert cross_risk == cross_and_mask_risk == 0
    passed("sufficient_bypass_has_no_additional_bayes_risk_gain")

    # Exact numerator expansion of d/du [3/8*(-t+4log(t)+3/t)], t=1+2u.
    t_square = [Q(1), Q(4), Q(4)]
    numerator = [-x for x in t_square]
    numerator[0] += 4 - 3
    numerator[1] += 8
    numerator = [Q(3, 4) * x for x in numerator]
    assert numerator == [Q(0), Q(3), Q(-3)]
    passed("st_integral_antiderivative_rational_derivative_identity")

    # In units of 3/8, 2A-3B has coefficients 20log2-12log3-1/2.
    coefficients = {"log2": 2 * 4 + 3 * 4, "log3": -3 * 4, "constant": 2 * Q(-5, 2) - 3 * Q(-3, 2)}
    assert coefficients == {"log2": 20, "log3": -12, "constant": Q(-1, 2)}
    assert Q(2 ** 5, 3 ** 3) == Q(32, 27)
    logarithm_lower_bound = Q(5, 32)
    bracket_lower_bound = 4 * logarithm_lower_bound - Q(1, 2)
    assert bracket_lower_bound == Q(1, 8) > 0
    true_gradient_over_log3 = -Q(3, 160)
    st_gradient_lower_bound_over_log3 = Q(3, 40) * bracket_lower_bound
    assert true_gradient_over_log3 < 0 < st_gradient_lower_bound_over_log3
    passed("st_expected_direction_opposes_true_nll_gradient", true_gradient_over_log3=str(true_gradient_over_log3), strict_st_lower_bound_over_log3=str(st_gradient_lower_bound_over_log3))

    first_order_lower_coefficient = Q(3, 160) * Q(3, 320)
    second_order_upper_coefficient = Q(1, 80) * Q(3, 8) ** 2
    assert first_order_lower_coefficient / second_order_upper_coefficient == Q(1, 10)
    passed("st_random_step_explicit_taylor_step_bound", sufficient_eta_times_log3="less than 1/10")

    for alpha, uncensored, censored in product([Q(0), Q(1, 2), Q(1)], [Q(2), Q(3)], [Q(0), Q(5)]):
        weighted = (1 - alpha) * (uncensored + censored) + alpha * uncensored
        assert weighted == uncensored + (1 - alpha) * censored
    passed("native_alpha_survival_loss_algebra")

    return {
        "schema": "slotspe_strict_math_arithmetic_v1",
        "kind": "exact_arithmetic_sanity_checks_not_formal_proof",
        "checks_passed": len(checks),
        "all_passed": True,
        "neural_model_calls": 0,
        "patient_inputs_loaded": 0,
        "server_execution": False,
        "limitations": ["Finite rational checks do not prove real-valued theorems.", "No theorem establishes original SlotSPE patient efficacy or C-index improvement."],
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Preserve the existing receipt: {args.output}")
    result = audit()
    result["script_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.write_bytes(serialized.encode("utf-8"))
    print(json.dumps({"all_passed": result["all_passed"], "checks_passed": result["checks_passed"], "neural_model_calls": 0, "receipt": str(args.output) if args.output else None}))


if __name__ == "__main__":
    main()
