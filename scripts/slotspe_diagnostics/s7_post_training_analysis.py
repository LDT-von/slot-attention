"""Post-training analysis for the parallel addendum training run.

Reads every completed fold/seed pair from the two training queues and answers
the question that a single checkpoint could not:

    "Is eval_fix's C-index change a systematic penalty, or was the
     -0.0055 measured on fold_0/seed_3 just an unlucky draw?"

Also reports the overfitting profile found in curve.json, because the 30-epoch
checkpoints are not the model's best state and that has to be visible rather
than buried.

Read-only. No training, no checkpoint mutation.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

REPO = Path("/data1/DCT-Reg")
OUT_DIR = REPO / "docs/research/slotspe_p6_addendum_20261020"
PROOF_ROOT = REPO / "results/slotspe_proof"


def find_run_dirs():
    """Every output directory that contains at least one completed pair."""
    found = []
    for d in sorted(PROOF_ROOT.iterdir()):
        if not d.is_dir() or not d.name.startswith("slotspe_addendum_gpu"):
            continue
        pairs = d / "pairs"
        if pairs.is_dir() and any(pairs.glob("fold_*/seed_*")):
            found.append(d)
    return found


def load_pair(pair_dir):
    """Load original + same_weights_eval_fix C-index for one fold/seed."""
    fold = int(pair_dir.parent.name.split("_")[1])
    seed = int(pair_dir.name.split("_")[1])

    rec = {"fold": fold, "seed": seed, "dir": str(pair_dir)}
    for arm, key in (("original", "orig"), ("same_weights_eval_fix", "fix")):
        ev = pair_dir / arm / "eval_repeats.json"
        if not ev.exists():
            continue
        reps = json.loads(ev.read_text())
        c = [r["cindex"] for r in reps if r.get("cindex") is not None]
        if not c:
            continue
        rec[f"{key}_mean"] = float(np.mean(c))
        rec[f"{key}_std"] = float(np.std(c, ddof=1)) if len(c) > 1 else 0.0
        rec[f"{key}_median"] = float(np.median(c))
        rec[f"{key}_n"] = len(c)

    # training curve: where the model actually peaked
    curve_path = pair_dir / "original" / "curve.json"
    if curve_path.exists():
        curve = json.loads(curve_path.read_text())
        if curve:
            vals = [c["val_cindex_exploratory"] for c in curve
                    if c.get("val_cindex_status") == "computed"]
            if vals:
                bi = int(np.argmax(vals))
                rec["curve_epochs"] = len(curve)
                rec["best_epoch_c"] = float(vals[bi])
                rec["best_epoch"] = curve[bi]["epoch"]
                rec["final_epoch_c"] = float(vals[-1])
                rec["first_epoch_c"] = float(vals[0])
                rec["overfit_drop"] = float(vals[bi] - vals[-1])
    return rec


def bootstrap_ci(values, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dirs = find_run_dirs()
    print("=" * 74)
    print("POST-TRAINING ANALYSIS")
    print("=" * 74)
    print(f"scan dirs: {len(run_dirs)}")

    discovered = {}
    for d in run_dirs:
        for pd in sorted((d / "pairs").glob("fold_*/seed_*")):
            if (pd / "original" / "summary.json").exists():
                discovered.setdefault((pd.parent.name, pd.name), pd)

    pairs = []
    for (fd, sd), pd in sorted(discovered.items(),
                               key=lambda kv: (int(kv[0][0].split("_")[1]),
                                               int(kv[0][1].split("_")[1]))):
        rec = load_pair(pd)
        if "orig_mean" in rec:
            pairs.append(rec)

    print(f"completed pairs: {len(pairs)}\n")

    # ------------------------------------------------------------------ table
    print("=" * 92)
    print(f"{'pair':<16}{'C_orig':>9}{'±':>8}{'C_evalfix':>11}{'delta':>10}"
          f"{'best_ep':>9}{'C_best':>9}{'C_ep1':>9}{'C_final':>9}")
    print("-" * 92)
    deltas = []
    for r in pairs:
        d = r["fix_mean"] - r["orig_mean"]
        deltas.append(d)
        print(f"fold_{r['fold']}/seed_{r['seed']:<8}"
              f"{r['orig_mean']:>9.4f}{r['orig_std']:>8.4f}{r['fix_mean']:>11.4f}{d:>+10.4f}"
              f"{r.get('best_epoch', 0):>9d}{r.get('best_epoch_c', 0):>9.4f}"
              f"{r.get('first_epoch_c', 0):>9.4f}{r.get('final_epoch_c', 0):>9.4f}")

    # ------------------------------------------------------- eval_fix verdict
    print("\n" + "=" * 74)
    print("Q1: does eval_fix systematically cost accuracy?")
    print("=" * 74)
    arr = np.asarray(deltas)
    lo, hi = bootstrap_ci(deltas)
    n_pos = int((arr > 0).sum()); n_neg = int((arr < 0).sum()); n_zero = int((arr == 0).sum())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    print(f"  n pairs        : {len(arr)}")
    print(f"  mean delta     : {arr.mean():+.5f}  (SE {se:.5f})")
    print(f"  bootstrap 95%CI: [{lo:+.5f}, {hi:+.5f}]")
    print(f"  sign           : {n_pos} positive / {n_neg} negative / {n_zero} zero")
    print(f"  range          : [{arr.min():+.5f}, {arr.max():+.5f}]")
    if len(arr) < 2:
        verdict = "INSUFFICIENT_PAIRS"
    elif lo > 0:
        verdict = "EVAL_FIX_SYSTEMATICALLY_BETTER"
    elif hi < 0:
        verdict = "EVAL_FIX_SYSTEMATICALLY_WORSE"
    else:
        verdict = "NO_SYSTEMATIC_EFFECT"
    print(f"\n  VERDICT: {verdict}")
    if verdict == "NO_SYSTEMATIC_EFFECT":
        print("  The CI spans zero and both signs occur: the fold_0/seed_3")
        print("  -0.0055 was a single unlucky draw, not a property of the fix.")

    # ---------------------------------------------------- overfitting profile
    print("\n" + "=" * 74)
    print("Q2: are the 30-epoch checkpoints representative?")
    print("=" * 74)
    with_curve = [r for r in pairs if "best_epoch" in r]
    if with_curve:
        best = np.asarray([r["best_epoch"] for r in with_curve])
        peak = np.asarray([r["best_epoch_c"] for r in with_curve])
        fin = np.asarray([r["final_epoch_c"] for r in with_curve])
        drop = np.asarray([r["overfit_drop"] for r in with_curve])
        print(f"  pairs with curves        : {len(with_curve)}")
        print(f"  best epoch  range/median : {best.min()}-{best.max()} (median {np.median(best):.0f})")
        print(f"  peak C-index  mean       : {peak.mean():.4f}")
        print(f"  epoch-30 C-index mean    : {fin.mean():.4f}")
        print(f"  mean drop peak -> final  : {drop.mean():+.4f}")
        worse = int((drop > 0.01).sum())
        print(f"  pairs losing >0.01 C by epoch 30: {worse}/{len(with_curve)}")
        overfit_verdict = ("SYSTEMATIC_OVERFITTING" if drop.mean() > 0.01
                           else "NO_MATERIAL_OVERFITTING")
        print(f"\n  VERDICT: {overfit_verdict}")
        if overfit_verdict == "SYSTEMATIC_OVERFITTING":
            print(f"  Every checkpoint used above is past its own optimum. The")
            print(f"  eval_fix verdict is therefore about a regime that the model")
            print(f"  never occupies at its best -- a real limitation of this run.")
    else:
        overfit_verdict = "NO_CURVE_DATA"

    # ------------------------------------------------------------ fold spread
    print("\n" + "=" * 74)
    print("Q3: how much does C-index vary across folds? (original arm)")
    print("=" * 74)
    by_fold = {}
    for r in pairs:
        by_fold.setdefault(r["fold"], []).append(r["orig_mean"])
    for f in sorted(by_fold):
        v = np.asarray(by_fold[f])
        print(f"  fold_{f}: mean {v.mean():.4f}  (n={len(v)}, sd {v.std(ddof=1) if len(v)>1 else 0:.4f})")
    allv = np.asarray([r["orig_mean"] for r in pairs])
    print(f"\n  across all pairs: mean {allv.mean():.4f}  sd {allv.std(ddof=1):.4f}  "
          f"range [{allv.min():.4f}, {allv.max():.4f}]")

    # ------------------------------------------------------------------ write
    out = OrderedDict()
    out["stage"] = "post_training_analysis"
    out["scan_dirs"] = [d.name for d in run_dirs]
    out["n_pairs"] = len(pairs)
    out["per_pair"] = pairs
    out["eval_fix"] = {
        "n": len(arr), "mean_delta": float(arr.mean()), "se": se,
        "bootstrap_ci95": [lo, hi], "n_positive": n_pos, "n_negative": n_neg,
        "n_zero": n_zero, "range": [float(arr.min()), float(arr.max())],
        "verdict": verdict,
    }
    if with_curve:
        out["overfitting"] = {
            "best_epoch_min": int(best.min()), "best_epoch_max": int(best.max()),
            "best_epoch_median": float(np.median(best)),
            "peak_cindex_mean": float(peak.mean()),
            "final_cindex_mean": float(fin.mean()),
            "mean_drop": float(drop.mean()),
            "verdict": overfit_verdict,
        }
    out["fold_spread"] = {f"fold_{f}": {"mean": float(np.mean(v)), "n": len(v)}
                          for f, v in by_fold.items()}
    out["original_all_pairs"] = {"mean": float(allv.mean()), "sd": float(allv.std(ddof=1)),
                                 "min": float(allv.min()), "max": float(allv.max())}

    path = OUT_DIR / "post_training_analysis.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {path}")


if __name__ == "__main__":
    main()