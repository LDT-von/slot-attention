"""Stage 4A: end-to-end impact. Does the output jitter actually hurt patients?

This is the question that decides the project.  Stages 1-3 established:

  * the `original` arm re-draws all 8 slot initialisations from Gaussian noise
    on every forward, so the model is genuinely stochastic at inference;
  * a pure re-initialisation change moves logits by up to ~7.4e-02;
  * a patch-order change moves logits by ~4.6e-02;
  * slot permutation is equivariant in the attention blocks (~1e-06) but NOT in
    the readout, because the top-k gate tie-breaks among 8 near-identical slots.

None of that is automatically a patient-level harm.  A number moving while the
ranking of patients stays correct is not harm.  So measure the thing that
matters: the concordance index, and the per-patient risk ranking.

Perturbation arms, all on the same 77 val patients:
  base      seed 0, unperturbed (reference)
  reinits   seed varies, input identical  -> pure slot re-init stochasticity
  patches   seed fixed, patch order permuted -> patch-order sensitivity

Reported:
  C-index distribution per arm, and C(arm) - C(base) per repeat
  bootstrap 95% CI on the mean delta
  p-value (two-sided, normal approximation)
  rank flips: per-patient |risk_rank_change| and the fraction of patients whose
  predicted rank moves by more than 10 positions
  discordant-pair delta: how many comparable patient pairs flip concordance

Decision rule stated in advance:
  |dC| < 0.01  => perturbation does not measurably harm ranking -> P6 is a
                  numerical curiosity, not a clinical risk -> report and stop.
  |dC| >= 0.01 => material -> proceed to the remediation arm.

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
N_REPS = 20


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


def concordance(events, times, scores):
    """Producer's Harrell C, verbatim from _p1246_diag.py / _s2026_1020_addendum.py."""
    from scipy.stats import rankdata
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
    print(f"[load] val={len(val_ids)}  comparable_pairs={concordance(events,times,np.zeros(len(val_ids)))[1]}",
          flush=True)

    # pre-load batches once; the cache is on disk and small
    batches = []
    for s in range(0, len(val_ids), 8):
        batches.append(_batch(str(RUN_DIR), manifest, fold_data,
                               val_ids[s:s + 8], DEVICE, torch))

    def sweep(seed, permute_patches):
        with torch.no_grad(), frozen_rng(seed):
            outs = []
            for b in batches:
                bb = dict(b)
                if permute_patches:
                    idx = torch.randperm(b["x_wsi"].size(1), device=DEVICE)
                    bb["x_wsi"] = b["x_wsi"][:, idx]
                lg, _ = model(**bb)
                outs.append(lg.detach().cpu())
        return torch.cat(outs)

    # ---- reference ----
    base_logits = sweep(0, False)
    base_risk = _risk(base_logits).numpy()
    c_base, n_pairs, n_tied = concordance(events, times, base_risk)
    base_rank = np.argsort(np.argsort(base_risk))
    print(f"[base] C={c_base:.4f}  pairs={n_pairs}  tied={n_tied:.1f}", flush=True)

    arms = {"reinits": [], "patches": []}
    for r in range(N_REPS):
        for arm, seed, perm in (("reinits", 900000 + r, False),
                                ("patches", 500000 + 7 * r, True)):
            lg = sweep(seed, perm)
            rk = _risk(lg).numpy()
            c, _, _ = concordance(events, times, rk)
            rank = np.argsort(np.argsort(rk))
            arms[arm].append({
                "rep": r, "seed": seed, "cindex": c,
                "d_cindex": c - c_base,
                "logit_max_abs_diff": float(np.abs(lg.numpy() - base_logits.numpy()).max()),
                "mean_abs_rank_shift": float(np.abs(rank - base_rank).mean()),
                "max_abs_rank_shift": int(np.abs(rank - base_rank).max()),
                "n_patients_shifted_gt10": int((np.abs(rank - base_rank) > 10).sum()),
            })
        print(f"  rep {r+1}/{N_REPS} reinit_dC={arms['reinits'][-1]['d_cindex']:+.4f} "
              f"patch_dC={arms['patches'][-1]['d_cindex']:+.4f}", flush=True)

    rng = np.random.default_rng(0)
    summary = OrderedDict()
    summary["stage"] = "4A_end_to_end_cindex"
    summary["n_val"] = len(val_ids)
    summary["n_reps"] = N_REPS
    summary["baseline_cindex"] = c_base
    summary["comparable_pairs"] = n_pairs
    for arm, rows in arms.items():
        d = np.asarray([x["d_cindex"] for x in rows])
        bs = np.asarray([rng.choice(d, size=len(d), replace=True).mean() for _ in range(10000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sd = float(d.std(ddof=1))
        summary[arm] = {
            "cindex_mean": float(np.mean([x["cindex"] for x in rows])),
            "cindex_min": float(np.min([x["cindex"] for x in rows])),
            "cindex_max": float(np.max([x["cindex"] for x in rows])),
            "d_cindex_mean": float(d.mean()),
            "d_cindex_min_abs": float(np.min(np.abs(d))),
            "d_cindex_max_abs": float(np.max(np.abs(d))),
            "d_cindex_std": sd,
            "bootstrap_ci95_mean_delta": [float(lo), float(hi)],
            "p_value_normal_approx": (float(abs(d.mean() / (sd / np.sqrt(len(d)))))
                                      if sd > 0 else 0.0),
            "logit_max_abs_diff_max": float(np.max([x["logit_max_abs_diff"] for x in rows])),
            "mean_abs_rank_shift_mean": float(np.mean([x["mean_abs_rank_shift"] for x in rows])),
            "max_abs_rank_shift_max": int(np.max([x["max_abs_rank_shift"] for x in rows])),
            "n_patients_shifted_gt10_max": int(np.max([x["n_patients_shifted_gt10"] for x in rows])),
        }

    worst = max(summary[a]["d_cindex_max_abs"] for a in arms)
    summary["decision_rule"] = "material if |dC| >= 0.01"
    summary["worst_abs_d_cindex"] = worst
    summary["verdict"] = "MATERIAL_HARM" if worst >= 0.01 else "NO_MEASURABLE_HARM"

    for a in arms:
        s = summary[a]
        print(f"\n[{a}] C mean={s['cindex_mean']:.4f} "
              f"dC mean={s['d_cindex_mean']:+.4f} max|dC|={s['d_cindex_max_abs']:.4f} "
              f"CI95=[{s['bootstrap_ci95_mean_delta'][0]:+.4f},{s['bootstrap_ci95_mean_delta'][1]:+.4f}]")
        print(f"       rank shift mean={s['mean_abs_rank_shift_mean']:.1f} "
              f"max={s['max_abs_rank_shift_max']} "
              f"patients>10 places={s['n_patients_shifted_gt10_max']}")
    print(f"\n[verdict] {summary['verdict']}  (worst |dC| = {worst:.4f})")

    p = OUT_DIR / "stage4a_cindex_impact.json"
    p.write_text(json.dumps(dict(summary, raw=arms), indent=2))
    print(f"[done] wrote {p}")


if __name__ == "__main__":
    main()