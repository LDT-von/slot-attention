"""Stage 6 (task B): does the eval_fix arm actually deliver determinism?

WHAT THIS TESTS
---------------
Every addendum stage used a frozen-RNG harness to obtain bit-exact repeats.
That harness is a measurement instrument, not the product.  The question this
stage answers is whether the *shipped* `eval_fix` arm -- the repair that already
exists in `model.apply_repairs` -- delivers determinism without the harness.

Three configurations, evaluated on the same 77 patients:

  original   -- as shipped. Re-draws all 8 slot initialisations per forward.
  eval_fix   -- uses the frozen `_proof_eval_noise` buffer. Deterministic by
                construction, no harness needed.
  readout_fix -- eval_fix plus a permutation-invariant readout.

Measured WITHOUT any RNG freeze for original/eval_fix, to see the difference
the harness was hiding:

  A. determinism: 5 repeats at one seed, bitwise or not?
  B. C-index: single value, or a distribution?
  C. accuracy cost: is eval_fix's C-index different from original's median?

Decision rule (stated before running):
  If eval_fix is bitwise deterministic AND its C-index is within the original's
  own spread, then the fix is free and should be adopted.
  If eval_fix costs accuracy, report the trade-off rather than recommending it.

No training, no checkpoint mutation.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path("/data1/DCT-Reg")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from _gpu_guard import require_gpu_env

_ENV = require_gpu_env()

import numpy as np
import torch

from survot_rank.research.slotspe_proof.patient import _batch
from survot_rank.research.slotspe_proof.model import build_model, make_args

RUN_DIR = REPO / "results/slotspe_proof/slotspe_clean_original_blca_run_20261020"
PREPARED = RUN_DIR / "prepared"
CKPT = RUN_DIR / "pairs/fold_0/seed_3/original/last.pt"
SOURCE_ROOT = REPO / "third_party/SlotSPE"
OUT_DIR = REPO / "docs/research/slotspe_p6_addendum_20261020"
DEVICE = "cuda:0"


def concordance(events, times, scores):
    valid = np.asarray(times) > 0
    times, events, scores = times[valid], events[valid], scores[valid]
    pairs, conc = 0, 0.0
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if times[i] == times[j]: continue
            ie = i if times[i] < times[j] else j
            if events[ie] == 0: continue
            pairs += 1
            if scores[i] != scores[j] and (scores[i] > scores[j]) == (times[i] < times[j]):
                conc += 1
    return (conc / pairs if pairs else float("nan")), pairs


def _risk(logits):
    return -(1 - logits.sigmoid()).cumprod(-1).sum(-1)


def _load(arm):
    manifest = json.loads((PREPARED / "manifest.json").read_text())
    fold_data = manifest["folds"]["0"]
    args = make_args(manifest["omic_sizes"], encoding_dim=manifest["encoding_dim"])
    args.rna_format = "Pathways"
    model = build_model(str(SOURCE_ROOT), args, arm).to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return manifest, fold_data, model


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    out = OrderedDict()
    out["stage"] = "6_eval_fix_determinism"
    out["env"] = _ENV
    out["note"] = ("evaluated WITHOUT the frozen-RNG harness, so any nondeterminism "
                   "in `original` is visible as the product behaves")

    results = {}
    for arm in ("original", "eval_fix", "readout_fix"):
        manifest, fold_data, model = _load(arm)
        val_ids = fold_data["val"]
        events = np.asarray([manifest["patients"][p]["censor"] < 1 for p in val_ids], dtype=bool)
        times = np.asarray([manifest["patients"][p]["time"] for p in val_ids], dtype=float)
        batches = [_batch(str(RUN_DIR), manifest, fold_data, val_ids[s:s+8], DEVICE, torch)
                   for s in range(0, len(val_ids), 8)]

        def run():
            outs = []
            with torch.no_grad():
                for b in batches:
                    lg, _ = model(**b)
                    outs.append(lg.detach().cpu())
            return torch.cat(outs)

        # --- A. determinism: 5 repeats, no RNG control ---
        reps = [run() for _ in range(5)]
        bitwise = all(torch.equal(r, reps[0]) for r in reps[1:])
        maxdiff = float(np.max([float((r - reps[0]).abs().max()) for r in reps[1:]]))

        # --- B/C. C-index over 40 plain repeats ---
        cidx = [concordance(events, times, _risk(r).numpy())[0] for r in
                ([run() for _ in range(40)] if not bitwise else [reps[0]])]
        arr = np.asarray(cidx)
        results[arm] = {
            "deterministic_without_harness": bitwise,
            "max_logit_diff_across_repeats": maxdiff,
            "cindex_is_single_value": bool(arr.std() == 0.0),
            "cindex_median": float(np.median(arr)),
            "cindex_min": float(arr.min()),
            "cindex_max": float(arr.max()),
            "cindex_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        }
        print(f"{arm:12s} deterministic={bitwise!s:5s} maxdiff={maxdiff:.3e} "
              f"C median={results[arm]['cindex_median']:.4f} "
              f"std={results[arm]['cindex_std']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    out["results"] = results
    orig, fix = results["original"], results["eval_fix"]
    out["comparison"] = {
        "original_cindex_spread": [orig["cindex_min"], orig["cindex_max"]],
        "eval_fix_cindex": fix["cindex_median"],
        "eval_fix_inside_original_spread": bool(
            orig["cindex_min"] <= fix["cindex_median"] <= orig["cindex_max"]),
        "determinism_gain": {
            "before": orig["max_logit_diff_across_repeats"],
            "after": fix["max_logit_diff_across_repeats"],
        },
    }
    deterministic = fix["deterministic_without_harness"] and fix["max_logit_diff_across_repeats"] == 0.0
    free = out["comparison"]["eval_fix_inside_original_spread"]
    out["decision_rule"] = ("adopt <=> deterministic AND C-index inside original's own spread")
    out["verdict"] = "ADOPT_FREE_FIX" if (deterministic and free) else (
        "DETERMINISTIC_BUT_COSTS_ACCURACY" if deterministic else "FIX_DOES_NOT_DELIVER_DETERMINISM")

    print(f"\nverdict: {out['verdict']}")
    print(f"  eval_fix C = {fix['cindex_median']:.4f}, original spread "
          f"[{orig['cindex_min']:.4f}, {orig['cindex_max']:.4f}]")
    p = OUT_DIR / "stage6_eval_fix.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {p}")


if __name__ == "__main__":
    main()