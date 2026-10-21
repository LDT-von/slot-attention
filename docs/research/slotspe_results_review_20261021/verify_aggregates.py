"""Recompute aggregate evidence only; no model, checkpoint or patient data access."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json

import numpy as np


def main():
    root = Path(__file__).resolve().parents[3]
    relative = "docs/research/slotspe_p6_addendum_20261020/slotspe_evalfix_evidence.json"
    source = root / relative
    bundle = json.loads(source.read_text(encoding="utf-8"))
    pairs = bundle["per_pair"]
    deltas, peaks, finals, epochs = [], [], [], []
    train_first, train_last, val_first, val_last = [], [], [], []
    errors = []
    fixed_distinct = []
    fixed_ci_excludes_constant = 0
    for pair in pairs:
        means = {}
        for arm in ("original", "same_weights_eval_fix"):
            row = pair["arms"][arm]
            values = np.asarray(row["cindex_values"], dtype=float)
            means[arm] = float(values.mean())
            errors.append(abs(means[arm] - row["mean"]))
            if arm == "same_weights_eval_fix":
                fixed_distinct.append(len(set(values.tolist())))
                lo, hi = row["mean_95ci"]
                fixed_ci_excludes_constant += int(
                    len(set(values.tolist())) == 1 and not lo <= values[0] <= hi
                )
        delta = means["same_weights_eval_fix"] - means["original"]
        deltas.append(delta)
        errors.append(abs(delta - pair["delta_cindex"]))
        curve = [r for r in pair["curve"]
                 if r.get("val_cindex_status") == "computed"
                 and r.get("val_cindex_exploratory") is not None]
        best = max(curve, key=lambda r: r["val_cindex_exploratory"])
        peaks.append(best["val_cindex_exploratory"])
        finals.append(curve[-1]["val_cindex_exploratory"])
        epochs.append(best["epoch"])
        train_first.append(curve[0]["train_main_nll"])
        train_last.append(curve[-1]["train_main_nll"])
        val_first.append(curve[0]["val_nll_exploratory"])
        val_last.append(curve[-1]["val_nll_exploratory"])

    arr = np.asarray(deltas)
    indices = np.random.default_rng(0).integers(0, len(arr), size=(10000, len(arr)))
    ci = np.percentile(arr[indices].mean(axis=1), [2.5, 97.5]).tolist()
    drop = np.asarray(peaks) - np.asarray(finals)
    result = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_file": relative,
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "execution_scope": "Aggregate JSON arithmetic only; no model execution or patient processing.",
        "n_pairs": len(pairs),
        "n_unique_fold_seed_keys": len({(p["fold"], p["seed"]) for p in pairs}),
        "n_unique_checkpoint_hashes_recorded": len({p["checkpoint_sha256_original"] for p in pairs}),
        "pairs_per_fold": dict(Counter(str(p["fold"]) for p in pairs)),
        "selector_disconnection_server_status": dict(Counter(p["disconnected_selector_status"] for p in pairs)),
        "all_pairs_report_completed_and_source_unchanged": all(
            p["pair_status"] == "completed" and p["source_unchanged"] is True for p in pairs),
        "largest_mean_or_delta_recalculation_error": max(errors),
        "eval_fix_same_weights": {
            "mean_delta_cindex": float(arr.mean()),
            "bootstrap_ci95": ci,
            "bootstrap_samples": 10000,
            "bootstrap_seed": 0,
            "n_positive": int((arr > 0).sum()),
            "n_negative": int((arr < 0).sum()),
            "range": [float(arr.min()), float(arr.max())],
            "all_fixed_repeat_values_identical_within_pair": all(n == 1 for n in fixed_distinct),
            "original_mean": float(np.mean([p["arms"]["original"]["mean"] for p in pairs])),
            "fixed_mean": float(np.mean([p["arms"]["same_weights_eval_fix"]["mean"] for p in pairs])),
            "interpretation": "No consistent direction detected in these final checkpoints; not an equivalence test. Shared patients/folds mean these are not 18 independent cohorts."
        },
        "training_curves": {
            "mean_observed_peak_cindex": float(np.mean(peaks)),
            "mean_epoch30_cindex": float(np.mean(finals)),
            "mean_peak_to_final_drop": float(drop.mean()),
            "pairs_dropping_over_0_01": int((drop > 0.01).sum()),
            "observed_peak_epoch_range": [min(epochs), max(epochs)],
            "observed_peak_epoch_median": float(np.median(epochs)),
            "mean_first_train_nll": float(np.mean(train_first)),
            "mean_final_train_nll": float(np.mean(train_last)),
            "pairs_with_train_nll_decreasing": sum(b < a for a, b in zip(train_first, train_last)),
            "mean_first_val_nll": float(np.mean(val_first)),
            "mean_final_val_nll": float(np.mean(val_last)),
            "pairs_with_val_nll_increasing": sum(b > a for a, b in zip(val_first, val_last)),
            "interpretation": "Strong evidence of late-stage validation deterioration under the recorded 30-epoch protocol. Observed validation maxima include selection bias and stochastic evaluation; not population-optimal epochs."
        },
        "export_errors": {
            "fixed_arm_mean_95ci_excludes_its_identical_repeat_value_count": fixed_ci_excludes_constant,
            "protocol_sha256_input_contains_arm_names_not_digest": all(
                p["protocol_sha256_input"] == ["original", "readout_fix"] for p in pairs),
            "note": "Faulty per-arm wilson_ci intervals are excluded from acceptance. The headline bootstrap is separately recomputed."
        },
        "limitations": [
            "Per-patient risks/logits, event/time arrays, original run logs and checkpoints not received in this public bundle.",
            "Recorded source/checkpoint hashes are provenance claims, not locally rehashed server checkpoints.",
            "Stored C-indices were aggregated; patient-level C-index calculations were not independently reconstructed.",
            "Same-weight eval_fix comparison used final checkpoints, not independently selected best checkpoints."
        ]
    }
    target = Path(__file__).with_name("aggregate_verification.json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
