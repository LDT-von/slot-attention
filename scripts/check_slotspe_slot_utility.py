#!/usr/bin/env python3
"""
Per-slot utility of the Top-K selector: what do the DISCARDED slots contribute?

Pure inference on ONE already-trained checkpoint and ONE patient set.  No
optimizer, no parameter update, no training, no source edit.

The question this answers is the user's Q2: after Top-K keeps 2 of 8 slots,
are the remaining 6 useless, and what actually drives the keep score?

Measured per patient and modality (wsi / omic), from one forward pass:

  keep_score_*      the raw score each of the 8 slots receives
                     (decoder.map -> pred_keep_slot -> squeeze).
  slot_logits_*     each slot's own 4-bin logit vector, i.e. what the slot
                     would have predicted alone.
  slot_cindex_*     C-index of that single slot pushed alone through the same
                     risk definition as the recorded harness.
  slot_contribution the slot's actual share of the native masked fusion logit.

Selection randomness is separated from learned ordering, because
``parallel_topk_st`` adds Gumbel noise BEFORE taking the top-k:

  chosen_by_native_rng   which slots the native forward actually kept.
  chosen_by_score_only   top-2 by raw score with no noise (the 'what the
                         scorer wanted' control).
  overlap                how many of the two agree.

Default is PLAN ONLY.  Pass --run to execute.
"""
import argparse, hashlib, json, math, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


DEFAULTS = {
    "checkpoint": "/data1/SlotSPE/results/slotspe_proof/blca_gpu_trisurv_a512169/pairs/fold_0/seed_3/original/last.pt",
    "prepared": "/data1/SlotSPE/results/slotspe_proof/blca_gpu_trisurv_a512169/prepared",
    "out": "/data1/SlotSPE/local_evidence/slot_utility_20261023",
    "fold": 0,
    "seed": 3,
    "slots": 8,
    "patches": 256,
}


def build(args):
    import torch
    from survot_rank.research.slotspe_proof import patient as P

    manifest = json.loads((Path(args.prepared) / "manifest.json").read_text())
    fold_data = manifest["folds"][str(args.fold)]
    val_ids = fold_data["val"]
    print(f"[plan] fold={args.fold} seed={args.seed} val patients={len(val_ids)} "
          f"train={len(fold_data['train'])}")

    source = os.environ.get("SLOTSPE_SOURCE_ROOT", str(REPO / "third_party" / "SlotSPE"))
    if source not in sys.path:
        sys.path.insert(0, source)
    from models.SlotSPE import SlotSPE

    import argparse as _ap
    margs = _ap.Namespace()
    for name in ("omic_sizes", "n_classes", "encoding_dim", "wsi_projection_dim", "rna_format",
                 "slot_num_wsi", "slot_num_omics", "slot_iters", "temperature", "topk_ratio",
                 "top_k_method", "bag_loss", "alpha_surv", "lambda_recon_loss"):
        setattr(margs, name, None)
    margs.omic_sizes = manifest["omic_sizes"]
    margs.n_classes = 4
    margs.encoding_dim = manifest["encoding_dim"]
    margs.wsi_projection_dim = 256
    margs.rna_format = "Pathways"
    margs.slot_num_wsi = args.slots
    margs.slot_num_omics = args.slots
    margs.slot_iters = 10
    margs.temperature = 0.01
    margs.topk_ratio = 0.25
    margs.top_k_method = "parallel_topk_st"
    margs.bag_loss = "nll_surv"
    margs.alpha_surv = 0.0
    margs.lambda_recon_loss = 0.1

    model = SlotSPE(margs, omic_names=[], omic_input_dim=None)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state.get("state_dict", state))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[plan] load_state_dict missing={len(missing)} unexpected={len(unexpected)} params={sum(p.numel() for p in model.parameters())}")
    if missing or unexpected:
        print("  missing:", missing[:6])
        print("  unexpected:", unexpected[:6])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # RTX 5090 is sm_120; only the trisurv build has matching kernels.
    device = "cpu"
    if torch.cuda.is_available():
        try:
            probe = torch.randn(8, 8, device="cuda")
            probe @ probe
            device = "cuda"
        except Exception as exc:
            print(f"[plan] cuda probe failed ({type(exc).__name__}), using cpu")
    model.to(device)
    print(f"[plan] device={device} torch={torch.__version__}")
    return torch, P, model, manifest, fold_data, val_ids, device


def measure(args):
    import numpy as np
    import torch
    from sksurv.metrics import concordance_index_censored

    torch, P, model, manifest, fold_data, val_ids, device = build(args)
    root = args.prepared.rsplit("/prepared", 1)[0]

    from survot_rank.research.slotspe_proof.patient import _rng

    # Same batch order and eval seed as the recorded fold_0/seed_3 run.
    _rng(args.seed, torch, np)

    per_modality = {"wsi": [], "omic": []}
    events = np.asarray([manifest["patients"][pid]["censor"] < 1 for pid in val_ids], dtype=bool)
    times = np.asarray([manifest["patients"][pid]["time"] for pid in val_ids], dtype=float)

    batch_size = 8
    for start in range(0, len(val_ids), batch_size):
        ids = val_ids[start:start + batch_size]
        batch = P._batch(root, manifest, fold_data, ids, device, torch)

        captured = {}
        keeps = {}
        hooks = []
        for mod in ("wsi", "omic"):
            dec = getattr(model, "slot_decoder_" + mod)

            def mk(mod):
                def hook(_m, inp, out):
                    captured[mod] = inp[0].detach()
                    if isinstance(out, tuple) and len(out) == 3:
                        keeps[mod] = out[2].detach()
                return hook
            hooks.append(dec.register_forward_hook(mk(mod)))

        with torch.no_grad():
            model(**batch)
        for h in hooks:
            h.remove()

        for mod in ("wsi", "omic"):
            slots = captured[mod]
            dec = getattr(model, "slot_decoder_" + mod)
            mapped = dec.map(slots)
            slot_logits = dec.decoder(mapped)                      # (B, S, C)
            scores = dec.pred_keep_slot(mapped).squeeze(-1)       # (B, S)
            hard_keep = keeps[mod]                                 # (B, S) native 0/1

            for b in range(slots.shape[0]):
                s_scores = scores[b].double()
                s_logits = slot_logits[b].double()
                keep_b = hard_keep[b].double()

                chosen_native = [int(i) for i in torch.nonzero(keep_b > 0.5).flatten()]
                order = torch.argsort(s_scores, descending=True)
                chosen_score = [int(i) for i in order[:dec.k].tolist()]

                # Actual share of the native masked fusion output.
                gate = torch.softmax(s_scores / dec.temperature, dim=-1) * keep_b
                gate = gate / (gate.sum(-1, keepdim=True) + 1e-8)
                contrib = torch.einsum("s,sc->c", gate, s_logits)

                per_modality[mod].append({
                    "patient_id": ids[b],
                    "keep_score": s_scores.tolist(),
                    "slot_logits": s_logits.tolist(),
                    "chosen_native_rng": chosen_native,
                    "chosen_by_score_only": chosen_score,
                    "native_matches_score_only": sorted(chosen_native) == sorted(chosen_score),
                    "fused_contribution": contrib.tolist(),
                })
        done = min(start + batch_size, len(val_ids))
        print(f"[run] {done}/{len(val_ids)} patients", flush=True)

    # Per-slot stand-alone C-index must span the whole cohort: C-index is a
    # cohort-level ranking statistic, so it cannot be computed one patient at a
    # time (an earlier version did that and silently returned None for every
    # slot because the risk vector did not match the event vector).
    slot_cindex = {}
    for mod in ("wsi", "omic"):
        rows = per_modality[mod]
        slot_cindex[mod] = []
        for i in range(args.slots):
            risks = []
            for r in rows:
                lg = torch.tensor(r["slot_logits"][i], dtype=torch.float64)
                risks.append(float(-(1 - lg.sigmoid()).cumprod(-1).sum(-1)))
            arr = np.asarray(risks, dtype=float)
            try:
                c = float(concordance_index_censored(events, times, arr)[0])
                c = c if math.isfinite(c) else None
            except ValueError as exc:
                c = "undefined: " + str(exc)
            slot_cindex[mod].append({
                "slot": i,
                "cindex": c,
                "risk_min": float(arr.min()),
                "risk_max": float(arr.max()),
                "risk_std": float(arr.std()),
                "unique_risks": int(len(np.unique(np.round(arr, 9)))),
            })
        print(f"[run] {mod} per-slot stand-alone C-index: "
              f"{[None if isinstance(x['cindex'], str) else round(x['cindex'], 4) for x in slot_cindex[mod]]}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps({
        "provenance": {
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256(args.checkpoint),
            "prepared_manifest_sha256": sha256(Path(args.prepared) / "manifest.json"),
            "fold": args.fold, "seed": args.seed, "slots": args.slots,
            "n_patients": len(val_ids), "patches": args.patches,
            "mode": "inference only; parameters frozen; no optimizer; no training; source unchanged",
            "single_slot_cindex_definition":
                "each slot's own decoder output alone over the whole cohort, risk = "
                "-(1-sigmoid).cumprod(-1).sum(-1), same definition as the recorded harness",
        },
        "single_slot_cindex": slot_cindex,
        "wsi": per_modality["wsi"],
        "omic": per_modality["omic"],
    }, indent=1))
    print(f"[done] wrote {out/'rows.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULTS["checkpoint"])
    ap.add_argument("--prepared", default=DEFAULTS["prepared"])
    ap.add_argument("--out", default=DEFAULTS["out"])
    ap.add_argument("--fold", type=int, default=DEFAULTS["fold"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--slots", type=int, default=DEFAULTS["slots"])
    ap.add_argument("--patches", type=int, default=DEFAULTS["patches"])
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    if not args.run:
        print(json.dumps({"mode": "plan", "command": "--run", **DEFAULTS}, indent=1))
        print("\nPlan only. No forward pass, no training, no parameter update.")
        return
    measure(args)


if __name__ == "__main__":
    main()