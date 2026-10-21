"""Stage 4A-2: corrected end-to-end test.

Stage 4A reported "MATERIAL_HARM" but every one of the 40 perturbations scored
*d higher* than the baseline (all dC > 0).  That pattern is not a perturbation
effect; it means the seed-0 reference is an unlucky low draw and the perturbation
arms are ordinary re-draws.  Comparing perturbations to one arbitrary draw is
the wrong test.

Corrected design:
  1. Characterise the model's C-index distribution over 60 plain re-draws
     (identical input, varying slot-init seed).  This is the model's own
     irreducible inference variance.
  2. Test whether patch permutation moves C-index *beyond* that distribution,
     by comparing the patch arm against the re-init arm from the same seeds.
  3. Report harm as discordant patient pairs flipped, which is the clinically
     interpretable quantity, rather than rank displacement.

Decision rule, stated before looking:
  A perturbation is material only if the patch arm's C-index distribution is
  shifted *downward* relative to the re-init arm with a bootstrap CI excluding
  zero.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gpu_guard import require_gpu_env  # hard-fail if CUDA cannot run here

REPO = Path("/data1/DCT-Reg")
sys.path.insert(0, str(REPO))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

_ENV = require_gpu_env()

from survot_rank.research.slotspe_proof.patient import _batch  # noqa: E402
from survot_rank.research.slotspe_proof.model import build_model, make_args  # noqa: E402

RUN_DIR = REPO / "results" / "slotspe_proof" / "slotspe_clean_original_blca_run_20261020"
PREPARED = RUN_DIR / "prepared"
CKPT = RUN_DIR / "pairs" / "fold_0" / "seed_3" / "original" / "last.pt"
SOURCE_ROOT = REPO / "third_party" / "SlotSPE"
OUT_DIR = REPO / "docs" / "research" / "slotspe_p6_addendum_20261020"
DEVICE = "cuda:0"


@contextlib.contextmanager
def frozen_rng(seed):
    py_state, np_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    cd, cb = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(py_state); np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cd, cb


def _load():
    manifest = json.loads((PREPARED / "manifest.json").read_text())
    fold_data = manifest["folds"]["0"]
    args = make_args(manifest["omic_sizes"], encoding_dim=manifest["encoding_dim"])
    args.rna_format = "Pathways"
    model = build_model(str(SOURCE_ROOT), args, "original").to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return manifest, fold_data, model


def _risk(logits):
    return -(1 - logits.sigmoid()).cumprod(-1).sum(-1)


def _pairs(events, times):
    """Enumerate comparable (earlier, event-observed) patient pairs once."""
    out = []
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if times[i] == times[j]:
                continue
            ie = i if times[i] < times[j] else j
            if events[ie] == 0:
                continue
            out.append((ie, il := (j if times[i] < times[j] else i)))
    return np.asarray(out, dtype=int)


def concordance(events, times, scores):
    valid = np.asarray(times) > 0
    times, events, scores = times[valid], events[valid], scores[valid]
    pairs, conc, tied = 0, 0.0, 0.0
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if times[i] == times[j]:
                continue
            ie = i if times[i] < times[j] else j
            if events[ie] == 0:
                continue
            pairs += 1
            if scores[i] != scores[j]:
                if (scores[i] > scores[j]) == (times[i] < times[j]):
                    conc += 1
            else:
                tied += 1
    return (conc / pairs if pairs else float("nan")), pairs, tied


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    manifest, fold_data, model = _load()
    val_ids = fold_data["val"]
    events = np.asarray([manifest["patients"][p]["censor"] < 1 for p in val_ids], dtype=bool)
    times = np.asarray([manifest["patients"][p]["time"] for p in val_ids], dtype=float)
    PR = _pairs(events, times)
    print(f"[load] val={len(val_ids)}  comparable_pairs={len(PR)}", flush=True)

    batches = [_batch(str(RUN_DIR), manifest, fold_data, val_ids[s:s + 8], DEVICE, torch)
               for s in range(0, len(val_ids), 8)]

    def sweep(seed, permute):
        with torch.no_grad(), frozen_rng(seed):
            outs = []
            for b in batches:
                bb = dict(b)
                if permute:
                    idx = torch.randperm(b["x_wsi"].size(1), device=DEVICE)
                    bb["x_wsi"] = b["x_wsi"][:, idx]
                lg, _ = model(**bb)
                outs.append(lg.detach().cpu())
        return _risk(torch.cat(outs)).numpy()

    N = 60
    rein = [sweep(700000 + i, False) for i in range(N)]
    c_rein = np.asarray([concordance(events, times, r)[0] for r in rein])
    med_c = float(np.median(c_rein))
    print(f"[reinits] C median={med_c:.4f} min={c_rein.min():.4f} "
          f"max={c_rein.max():.4f} std={c_rein.std(ddof=1):.4f}", flush=True)

    pat = [sweep(700000 + i, True) for i in range(N)]
    c_pat = np.asarray([concordance(events, times, r)[0] for r in pat])
    print(f"[patches] C median={np.median(c_pat):.4f} min={c_pat.min():.4f} "
          f"max={c_pat.max():.4f} std={c_pat.std(ddof=1):.4f}", flush=True)

    d = c_pat - c_rein
    rng = np.random.default_rng(0)
    bs = np.asarray([rng.choice(d, size=len(d), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sd = float(d.std(ddof=1))
    print(f"[paired] dC(patch-reinit) mean={d.mean():+.4f} "
          f"CI95=[{lo:+.4f},{hi:+.4f}]", flush=True)

    med_risk = np.median(np.stack(rein), axis=0)
    ref_conc = (med_risk[PR[:, 0]] > med_risk[PR[:, 1]])
    flips = [int(np.sum(ref_conc != (r[PR[:, 0]] > r[PR[:, 1]])))
             for r in rein + pat]
    print(f"[flips] discordant-pair flips: median={np.median(flips):.0f} "
          f"max={max(flips)} of {len(PR)} ({100*np.mean(flips)/len(PR):.2f}%)", flush=True)

    worst = float(med_c - np.minimum(c_rein.min(), c_pat.min()))

    out = OrderedDict()
    out["stage"] = "4A2_corrected_end_to_end"
    out["n_val"] = len(val_ids)
    out["n_comparable_pairs"] = int(len(PR))
    out["n_reps_per_arm"] = N
    out["reinit_distribution"] = {
        "c_median": med_c, "c_mean": float(c_rein.mean()),
        "c_min": float(c_rein.min()), "c_max": float(c_rein.max()),
        "c_std": float(c_rein.std(ddof=1)),
        "interpretation": "model's irreducible inference variance, identical input",
    }
    out["patch_distribution"] = {
        "c_median": float(np.median(c_pat)), "c_mean": float(c_pat.mean()),
        "c_min": float(c_pat.min()), "c_max": float(c_pat.max()),
        "c_std": float(c_pat.std(ddof=1)),
    }
    out["paired_delta_patch_minus_reinit"] = {
        "mean": float(d.mean()), "std": float(sd),
        "bootstrap_ci95": [float(lo), float(hi)],
        "p_value": float(abs(d.mean() / (sd / np.sqrt(len(d))))) if sd > 0 else 0.0,
        "interpretation": "if CI contains 0, patch order is not distinguishable from re-draw noise",
    }
    out["discordant_pair_flips"] = {
        "median": float(np.median(flips)), "max": int(max(flips)),
        "of_total": int(len(PR)),
        "percent_mean": float(100 * np.mean(flips) / len(PR)),
        "reference": "median-risk draw, compared pair-by-pair",
    }
    out["worst_drop_vs_median"] = worst
    out["decision_rule"] = ("material only if patch arm is shifted DOWN vs re-init "
                            "with CI excluding 0")
    out["verdict"] = "MATERIAL_HARM" if (hi < 0 and d.mean() < 0) else "NO_MEASURABLE_HARM"

    print(f"\n[verdict] {out['verdict']}")
    print(f"          worst drop vs median C = {worst:.4f}")

    p = OUT_DIR / "stage4a2_corrected_cindex.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {p}")


if __name__ == "__main__":
    main()