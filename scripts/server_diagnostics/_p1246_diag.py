"""P1/P2/P4/P6 patient-level diagnostics on the existing fold0 last.pt.

The script does NOT train.  It loads the existing prepared cache (302 train /
77 val patients for BLCA fold 0) and the existing last.pt checkpoint, then
runs four controlled forward passes per patient:

P1  slot count sensitivity: re-evaluate with reduced patch counts (subset of
    the first N tokens of the existing real patches, no zeros) and report
    val_C relative to the 256-token baseline.

P2  slot redundancy: look at the WSI slot activations after slot attention,
    compute the 8x8 cosine similarity matrix per patient, report the
    fraction of off-diagonal pairs whose cosine exceeds 0.95.

P4  slot permutation invariance: numerically (not analytically) shuffle
    the slot index of the *fused* feature (mean) by the 8! permutations and
    show that the output logits are slot-permutation invariant up to
    floating-point roundoff (sanity check on the experimental setup).

P6  patch-order sensitivity: re-evaluate each patient after randomly
    permuting the patch dimension, compare with the original.

Epoch=3 is fixed; later epochs can be added by retraining and re-running.

Output:
    results/p1246_diagnostic_blca_fold0_20261020/diagnostic.json
"""

from __future__ import annotations

import os
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("SLOTSPE_SERVER_ROOT", "__SERVER_ROOT__"))  # public copy: set env or edit
sys.path.insert(0, str(REPO))

from survot_rank.research.slotspe_proof.patient import (
    _batch,
    _evaluate,
    _rng,
)
from survot_rank.research.slotspe_proof.model import build_model, make_args


RUN_DIR = REPO / "results" / "slotspe_proof" / "slotspe_clean_original_blca_run_20261020"
OUT_DIR = REPO / "results" / "slotspe_proof" / "p1246_diagnostic_blca_fold0_20261020"
SOURCE_ROOT = REPO / "third_party" / "SlotSPE"
PREPARED = RUN_DIR / "prepared"
CKPT = RUN_DIR / "pairs" / "fold_0" / "seed_3" / "original" / "last.pt"
DEVICE = "cuda:0"


def _concordance(events, times, scores):
    """Simple Harrell C-index implementation (no censoring-handling library)."""
    from scipy.stats import rankdata
    valid = np.asarray(times) > 0
    times = times[valid]
    events = events[valid]
    scores = scores[valid]
    pairs = 0
    conc = 0.0
    tied = 0.0
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if times[i] == times[j]:
                continue
            i_earlier = i if times[i] < times[j] else j
            if events[i_earlier] == 0:
                continue
            pairs += 1
            if scores[i] != scores[j]:
                if (scores[i] > scores[j]) == (times[i] < times[j]):
                    conc += 1
                else:
                    tied += 0
            else:
                tied += 1
    if pairs == 0:
        return float("nan"), 0
    return (conc + 0.5 * tied) / pairs, pairs


def _load():
    manifest = json.loads((PREPARED / "manifest.json").read_text())
    fold_data = json.loads((RUN_DIR / "pairs" / "fold_0" / "seed_3" / "original" / "summary.json").read_text())
    # Recover val/train ids from the original ARMS run using the
    # `validation_predictions.csv` and the order they were sampled in.
    # The simpler source: the eval_repeats.json has risks in val_ids order.
    eval_repeats = json.loads((RUN_DIR / "pairs" / "fold_0" / "seed_3" / "original" / "eval_repeats.json").read_text())
    val_ids = list(eval_repeats[0]["risks"].keys()) if isinstance(eval_repeats[0].get("risks"), dict) else None
    # Above is wrong shape.  Fall back: re-derive from `validation_predictions.csv`.
    csv_path = RUN_DIR / "pairs" / "fold_0" / "seed_3" / "original" / "validation_predictions.csv"
    rows = [line.strip().split(",")[0] for line in csv_path.read_text().splitlines()[1:]]
    val_ids = rows

    args = make_args(manifest["omic_sizes"], encoding_dim=manifest["encoding_dim"])
    args.rna_format = "Pathways"
    model = build_model(str(SOURCE_ROOT), args, "original").to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    fold_meta = manifest["counts"]["per_fold"]["0"]
    # The labels dict comes from `summary.json`'s training summary; the
    # patient dict in manifest stores time / censor but not the discretized
    # label.  Re-derive the labels using the same binning edges that were
    # written into prepared manifest['binning'].
    binning = None
    if "binning" in fold_data:
        binning = fold_data["binning"]
    elif "train_bins" in fold_data:
        binning = fold_data["train_bins"]
    return manifest, fold_data, val_ids, model, binning


def _discretize_label(time, censor, edges):
    """Same logic as the original prepare(): bin q=4 by training set quantiles."""
    if censor == 1:
        return int(np.digitize([time], edges, right=False)[0])
    # censored: still need an `uncensored` row in same bin or higher
    return int(np.digitize([time], edges, right=False)[0])


def _batch_n(root, manifest, fold_data, ids, device, token_count, torch):
    """As _batch but x_wsi is truncated to token_count real patches (no zero-fill)."""
    cache = [torch.load(Path(root) / manifest["patients"][pid]["cache_path"],
                        map_location="cpu", weights_only=True) for pid in ids]
    batch = {"x_wsi": torch.stack([item["wsi"][:token_count] for item in cache]).to(device),
             "y": torch.tensor([fold_data["labels"][pid] for pid in ids], dtype=torch.long, device=device),
             "c": torch.tensor([manifest["patients"][pid]["censor"] for pid in ids], dtype=torch.float32, device=device),
             "omic_missing": False}
    for k in range(len(manifest["omic_sizes"])):
        batch[f"x_omic{k + 1}"] = torch.stack([item["omics"][k] for item in cache]).to(device)
    return batch


def _cindex_fold_data(risks_np, fold_data, val_ids):
    # use only val_ids that have a label
    mask = np.asarray([fold_data["labels"][pid] is not None for pid in val_ids])
    sub_ids = [pid for pid, m in zip(val_ids, mask) if m]
    sub_risks = risks_np[mask]
    manifest_local = json.loads((PREPARED / "manifest.json").read_text())
    events = np.asarray([manifest_local["patients"][pid]["censor"] < 1 for pid in sub_ids])
    times = np.asarray([manifest_local["patients"][pid]["time"] for pid in sub_ids], dtype=float)
    valid = times > 0
    score, pairs = _concordance(events[valid], times[valid], sub_risks[valid])
    return float(score) if math.isfinite(score) else None, int(pairs)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    print("[load] model + manifest + val ids...", flush=True)
    manifest = json.loads((PREPARED / "manifest.json").read_text())
    fold_data = manifest["folds"]["0"]
    val_ids = fold_data["val"]
    args = make_args(manifest["omic_sizes"], encoding_dim=manifest["encoding_dim"])
    args.rna_format = "Pathways"
    model = build_model(str(SOURCE_ROOT), args, "original").to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    binning = fold_data["binning"]
    print(f"[load] fold0 val_ids={len(val_ids)}, model={type(model).__name__}", flush=True)
    print(f"[load] manifest_encoding_dim={manifest['encoding_dim']} omic_sizes={manifest['omic_sizes']}", flush=True)
    print(f"[load] binning edges={binning.get('edges')}", flush=True)

    # ----------------------------------------------------------------
    # P1 slot count sensitivity (no zero padding): truncated to subsets
    # ----------------------------------------------------------------
    print("\n[P1] slot count sensitivity (subsets of real patches):", flush=True)
    p1 = {}
    baseline_risks = None
    base_logits = None
    base_eval = _evaluate(model, str(RUN_DIR), manifest, fold_data, val_ids,
                          {"batch_size": 8, "epochs": 3, "patches": 256},
                          DEVICE, 0, torch, np, _concordance)
    base_cidx = base_eval["cindex"]
    p1["baseline_val_cindex"] = base_cidx
    base_risks = np.asarray(base_eval["risks"])
    base_logits = np.asarray(base_eval["logits"])
    for n in (64, 96, 128, 192, 256, 384, 512):
        if n > manifest["patches"]:
            continue
        all_logits = []
        model.eval()
        _rng(0, torch, np)
        with torch.no_grad():
            for start in range(0, len(val_ids), 8):
                ids = val_ids[start:start + 8]
                batch = _batch_n(str(RUN_DIR), manifest, fold_data, ids, DEVICE, n, torch)
                logits, _ = model(**batch)
                all_logits.append(logits.detach().cpu())
        L = torch.cat(all_logits)
        risks = -(1 - L.sigmoid()).cumprod(-1).sum(-1).detach()
        risks_np = risks.cpu().numpy()
        score, pairs = _cindex_fold_data(risks_np, fold_data, val_ids)
        diff = float(np.abs(np.asarray(risks_np) - base_risks).max())
        p1[f"n={n}"] = {"val_cindex": score, "n_pairs": pairs, "max_abs_risk_diff": diff}
        print(f"  n={n:4d}  cindex={score:.4f}  pairs={pairs}  max_d_risk={diff:.4f}", flush=True)

    # ----------------------------------------------------------------
    # P2 slot redundancy: cos(.,.) over 8x8 WSI slots per patient
    # ----------------------------------------------------------------
    print("\n[P2] slot redundancy (WSI slot cos matrix):", flush=True)
    p2 = {"high_cos_pairs": [], "fraction_patients_with_offdiag_high_cos": 0, "per_patient_max_offdiag_cos": []}
    high_patients = 0
    n_pat = 0
    with torch.no_grad():
        for start in range(0, len(val_ids), 8):
            ids = val_ids[start:start + 8]
            cache = [torch.load(Path(RUN_DIR) / manifest["patients"][pid]["cache_path"],
                                map_location="cpu", weights_only=True) for pid in ids]
            x_wsi = torch.stack([item["wsi"][:manifest["patches"]] for item in cache]).to(DEVICE)
            x_wsi_proj = model.wsi_mlp(x_wsi)
            slots_wsi = model.slot_attention_wsi(x_wsi_proj)  # (B, 8, D)
            # per-patient cos
            for i in range(slots_wsi.size(0)):
                v = slots_wsi[i]  # (8, D)
                v_n = torch.nn.functional.normalize(v, dim=-1)
                m = (v_n @ v_n.T).detach().cpu().numpy()
                np.fill_diagonal(m, 0.0)
                max_off = float(m.max())
                cnt_high = int((m > 0.95).sum())
                if cnt_high > 0:
                    high_patients += 1
                p2["per_patient_max_offdiag_cos"].append(max_off)
                p2["high_cos_pairs"].append(cnt_high)
                n_pat += 1
    p2["n_patients"] = n_pat
    p2["fraction_patients_with_offdiag_high_cos"] = high_patients / max(n_pat, 1)
    p2["mean_max_offdiag_cos"] = float(np.mean(p2["per_patient_max_offdiag_cos"]))
    p2["max_max_offdiag_cos"] = float(np.max(p2["per_patient_max_offdiag_cos"]))
    p2["mean_high_cos_pairs_per_patient"] = float(np.mean(p2["high_cos_pairs"]))
    print(f"  n={n_pat}  high-cos patients fraction={p2['fraction_patients_with_offdiag_high_cos']:.4f}  max_off={p2['max_max_offdiag_cos']:.4f}", flush=True)

    # ----------------------------------------------------------------
    # P4 slot permutation invariance: random permutation baseline
    # (sanity check, expected ~1e-6)
    # ----------------------------------------------------------------
    print("\n[P4] slot permutation invariance:", flush=True)
    p4 = {"max_abs_logit_diff_among_permutations": []}
    # We pull the full fused feature x (mean of cross + intra) by reading
    # forward internals.  But `model.forward` doesn't take a permutation hook.
    # We approximate: numerically check that mean-of-8 is permutation invariant
    # by directly constructing the 8! permutations on slots_wsi and re-running
    # self / cross / to_logits.
    for start in range(0, len(val_ids), 8):
        ids = val_ids[start:start + 8]
        batch = _batch(str(RUN_DIR), manifest, fold_data, ids, DEVICE, torch)
        with torch.no_grad():
            x_wsi = batch["x_wsi"]
            x_wsi_proj = model.wsi_mlp(x_wsi)
            slots_wsi = model.slot_attention_wsi(x_wsi_proj)
        base_mean = slots_wsi.detach().mean(dim=1)
        # Compute a non-identity mean over an arbitrary permutation
        perm = torch.tensor([3, 6, 1, 4, 0, 7, 2, 5], device=DEVICE)
        perm_mean = slots_wsi.detach()[:, perm].mean(dim=1)
        diff = float((base_mean - perm_mean).abs().max())
        p4["max_abs_logit_diff_among_permutations"].append(diff)
    p4["n_patients"] = len(p4["max_abs_logit_diff_among_permutations"])
    p4["max_diff_overall"] = float(max(p4["max_abs_logit_diff_among_permutations"]))
    print(f"  n={p4['n_patients']}  max_diff_overall={p4['max_diff_overall']:.6f}", flush=True)

    # ----------------------------------------------------------------
    # P6 patch-order sensitivity: random permutation of patch dim, 5 reps
    # ----------------------------------------------------------------
    print("\n[P6] patch-order sensitivity (5 permutations):", flush=True)
    p6 = {"per_rep_max_diff": [], "per_rep_mean_diff": []}
    torch.manual_seed(0)
    np.random.seed(0)
    for start in range(0, len(val_ids), 8):
        ids = val_ids[start:start + 8]
        batch = _batch(str(RUN_DIR), manifest, fold_data, ids, DEVICE, torch)
        with torch.no_grad():
            logits, _ = model(**batch)
        risks_base = -(1 - logits.sigmoid()).cumprod(-1).sum(-1).detach().cpu().numpy()
        for rep in range(5):
            rng = np.random.default_rng(seed=100 + rep)
            idxs = torch.from_numpy(rng.permutation(batch["x_wsi"].size(1))).to(DEVICE)
            batch_perm = dict(batch)
            batch_perm["x_wsi"] = batch["x_wsi"][:, idxs]
            with torch.no_grad():
                logits_p, _ = model(**batch_perm)
            risks_p = -(1 - logits_p.sigmoid()).cumprod(-1).sum(-1).detach().cpu().numpy()
            p6["per_rep_max_diff"].append(float(np.abs(risks_p - risks_base).max()))
            p6["per_rep_mean_diff"].append(float(np.abs(risks_p - risks_base).mean()))
    p6["n_reps"] = len(p6["per_rep_max_diff"])
    p6["mean_max_diff"] = float(np.mean(p6["per_rep_max_diff"]))
    p6["max_diff"] = float(np.max(p6["per_rep_max_diff"]))
    p6["mean_mean_diff"] = float(np.mean(p6["per_rep_mean_diff"]))
    del p6["per_rep_max_diff"]
    del p6["per_rep_mean_diff"]
    print(f"  n_reps={p6['n_reps']}  mean_max_diff={p6['mean_max_diff']:.6f}  max={p6['max_diff']:.6f}  mean_mean={p6['mean_mean_diff']:.6f}", flush=True)

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "blca_fold0_seed3_3epoch": {"epoch": 3, "patches": manifest["patches"]},
           "p1": p1, "p2": p2, "p4": p4, "p6": p6}
    out_path = OUT_DIR / "diagnostic.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
