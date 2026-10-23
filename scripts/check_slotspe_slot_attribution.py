#!/usr/bin/env python3
"""
Exact per-slot attribution of the final fused logits on ONE frozen checkpoint.

Continues Q2: after Top-K keeps 2 of 8 slots, what do the other 6 contribute?

Two structural facts make this exactly computable, with no sampling and no
retraining:

  1. models/SlotSPE.py line 293 calls cross_attention WITHOUT a mask, and
     IterativeCrossAttTransformer.forward defaults mask1=mask2=None.  So the
     cross branch always spans all 2S slots.
  2. The fusion head is affine over three concatenated branch means:

         logits = W_c @ mean_cross + W_w @ mean_wsi + W_o @ mean_omic + bias

     Each branch mean is a plain sum over slots, so every slot owns an exact
     additive piece.  Subtracting slot j's piece is a leave-one-slot-out
     counterfactual with no approximation.

The decomposition is self-verifying: the pieces plus the bias must reproduce
the native logits, and that residual is reported.

Note on the selection mask: Transformer.forward masks only the ATTENTION
weights between non-kept slots (transformer.py line 92), but line 131 keeps the
residual ``x = x + attn`` for every slot, and the branch mean at SlotSPE.py
line 300 averages over all S slots.  A "discarded" slot therefore still
contributes; this script measures by how much.

Pure inference. No optimizer, no parameter update, no training, no source edit.
Default is PLAN ONLY. Pass --run to execute.
"""
import argparse, hashlib, json, os, sys
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
    "out": "/data1/SlotSPE/local_evidence/slot_utility_20261023/cross_branch_attribution.json",
    "fold": 0,
    "seed": 3,
    "slots": 8,
}


def build(args):
    import torch
    from survot_rank.research.slotspe_proof import patient as P

    manifest = json.loads((Path(args.prepared) / "manifest.json").read_text())
    fold_data = manifest["folds"][str(args.fold)]
    val_ids = fold_data["val"]
    print(f"[plan] fold={args.fold} seed={args.seed} val={len(val_ids)}")

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
    print(f"[plan] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

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
    from survot_rank.research.slotspe_proof.patient import _rng

    torch, P, model, manifest, fold_data, val_ids, device = build(args)
    root = args.prepared.rsplit("/prepared", 1)[0]
    _rng(args.seed, torch, np)

    S = args.slots
    D = model.wsi_projection_dim
    W = model.to_logits.weight.double()
    bias = model.to_logits.bias.double()
    Wc, Ww, Wo = W[:, 0:D], W[:, D:2 * D], W[:, 2 * D:3 * D]

    def risk_of(logits):
        return -(1 - logits.sigmoid()).cumprod(-1).sum(-1)

    rows = []
    worst_recon = 0.0
    batch_size = 8
    for start in range(0, len(val_ids), batch_size):
        ids = val_ids[start:start + batch_size]
        batch = P._batch(root, manifest, fold_data, ids, device, torch)

        grab, keeps, hooks = {}, {}, []

        def branch_hook(name):
            def h(_m, _i, out):
                grab[name] = out.detach()
            return h
        for name in ("cross_attention", "self_attention_wsi", "self_attention_omic"):
            hooks.append(getattr(model, name).register_forward_hook(branch_hook(name)))

        for mod in ("wsi", "omic"):
            def mk(mod):
                def h(_m, _i, out):
                    if isinstance(out, tuple) and len(out) == 3:
                        keeps[mod] = out[2].detach()
                return h
            hooks.append(getattr(model, "slot_decoder_" + mod).register_forward_hook(mk(mod)))

        with torch.no_grad():
            native_logits, _ = model(**batch)
        for h in hooks:
            h.remove()

        x_inter = grab["cross_attention"].double()
        wsi_intra = grab["self_attention_wsi"].double()
        omic_intra = grab["self_attention_omic"].double()

        for b in range(len(ids)):
            nat = native_logits[b].double()
            pieces = {}
            for i in range(S):
                pieces[f"wsi_{i}"] = (x_inter[b, i] @ Wc.T) / (2 * S) + (wsi_intra[b, i] @ Ww.T) / S
            for i in range(S):
                pieces[f"omic_{i}"] = (x_inter[b, S + i] @ Wc.T) / (2 * S) + (omic_intra[b, i] @ Wo.T) / S

            total = sum(pieces.values()) + bias
            recon_err = float((total - nat).abs().max())
            worst_recon = max(worst_recon, recon_err)

            base_risk = risk_of(nat)
            rec = {"patient_id": ids[b],
                   "kept_wsi": [int(i) for i in torch.nonzero(keeps["wsi"][b] > 0.5).flatten()],
                   "kept_omic": [int(i) for i in torch.nonzero(keeps["omic"][b] > 0.5).flatten()],
                   "native_risk": float(base_risk),
                   "decomposition_max_abs_error": recon_err,
                   "piece_l1": {k: float(v.abs().sum()) for k, v in pieces.items()},
                   "delta_risk": {}}
            for key, piece in pieces.items():
                loo = nat - piece
                rec["delta_risk"][key] = float(risk_of(loo) - base_risk)
            rows.append(rec)
        print(f"[run] {min(start + batch_size, len(val_ids))}/{len(val_ids)}", flush=True)

    print(f"[run] exact decomposition max abs error: {worst_recon:.3e}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "provenance": {
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256(args.checkpoint),
            "prepared_manifest_sha256": sha256(Path(args.prepared) / "manifest.json"),
            "fold": args.fold, "seed": args.seed, "slots": S, "n_patients": len(val_ids),
            "mode": "inference only; parameters frozen; no training; source unchanged",
            "method": "closed-form affine decomposition of to_logits over three branch "
                      "means; cross branch is unmasked and spans all 2S slots; the selection "
                      "mask only zeroes attention between non-kept slots while the residual "
                      "keeps every slot",
            "decomposition_max_abs_error": worst_recon,
        },
        "rows": rows,
    }, indent=1))
    print(f"[done] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULTS["checkpoint"])
    ap.add_argument("--prepared", default=DEFAULTS["prepared"])
    ap.add_argument("--out", default=DEFAULTS["out"])
    ap.add_argument("--fold", type=int, default=DEFAULTS["fold"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--slots", type=int, default=DEFAULTS["slots"])
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if not args.run:
        print(json.dumps({"mode": "plan", "command": "--run", **DEFAULTS}, indent=1))
        print("\nPlan only. No forward pass, no training.")
        return
    measure(args)


if __name__ == "__main__":
    main()