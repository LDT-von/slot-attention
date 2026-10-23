#!/usr/bin/env python3
"""
Joint slot-redundancy vs gate-collapse diagnostic on ONE checkpoint and ONE patient set.

Reads existing prepared cache + a trained SlotSPE checkpoint. Inference only:
params frozen, no optimizer, no training, no source edit.

For every validation patient we compute, from the same forward pass:
  redundancy : mean pairwise cosine of the final slots (slot-content collapse)
  gate noise : how often the truly highest-scoring slot survives Gumbel-TopK
               (selection randomness), plus the rank1-rank2 keep_score gap
               normalised by the Gumbel noise scale.

Default is PLAN ONLY. Pass --run to execute.
"""
import argparse, hashlib, json, math, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
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
    "out": "/data1/SlotSPE/local_evidence/joint_redundancy_gate_20261022",
    "fold": 0,
    "seed": 3,
    "gate_seeds": 8,
    "patches": 256,
}


def build(args):
    import torch
    from survot_rank.research.slotspe_proof import patient as P

    manifest = json.loads((Path(args.prepared) / "manifest.json").read_text())
    fold_data = manifest["folds"][str(args.fold)]
    val_ids = fold_data["val"]
    print(f"[plan] val patients: {len(val_ids)} (train {len(fold_data['train'])})")

    source = os.environ.get("SLOTSPE_SOURCE_ROOT", "/data1/SlotSPE/third_party/SlotSPE")
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
    margs.slot_num_wsi = 8
    margs.slot_num_omics = 8
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
    if missing or unexpected:
        print("  missing:", missing[:6])
        print("  unexpected:", unexpected[:6])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # RTX 5090 is sm_120; only the trisurv env's torch build has sm_120 kernels.
    # Falls back to CPU elsewhere so results stay reproducible.
    device = "cpu"
    if torch.cuda.is_available():
        try:
            probe = torch.randn(8, 8, device="cuda")
            probe @ probe
            device = "cuda"
        except Exception as exc:  # no kernel image for this arch
            print(f"[plan] cuda probe failed ({type(exc).__name__}), using cpu")
    model.to(device)
    print(f"[plan] device={device} torch={torch.__version__}")
    return torch, P, model, manifest, fold_data, val_ids, device


def measure(args):
    torch, P, model, manifest, fold_data, val_ids, device = build(args)
    # cache_path in the manifest is relative to the run root, not the prepared dir
    root = args.prepared.rsplit("/prepared", 1)[0]

    from models.slot_attention import gumbel_noise, parallel_topk_st

    rows = []
    batch_size = 8
    for start in range(0, len(val_ids), batch_size):
        ids = val_ids[start:start + batch_size]
        batch = P._batch(root, manifest, fold_data, ids, device, torch)

        captured = {}
        hooks = []
        for mod in ("wsi", "omic"):
            dec = getattr(model, "slot_decoder_" + mod)
            sa = getattr(model, "slot_attention_" + mod)

            def mk(mod):
                def hook(_m, inp, out):
                    # MoESlotDecoder.forward returns (logits, slot_gate, hard_keep)
                    captured[mod] = inp[0].detach()
                return hook
            hooks.append(dec.register_forward_hook(mk(mod)))

        with torch.no_grad():
            model(**batch)
        for h in hooks:
            h.remove()

        for mod in ("wsi", "omic"):
            slots = captured[mod]
            # ---- redundancy: mean pairwise cosine of raw final slots
            flat = slots[0].double() if slots.shape[0] == 1 else slots.double()
            per_patient = []
            for b in range(slots.shape[0]):
                s = slots[b].double()
                u = s / s.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                C = (u @ u.T).clamp(-1, 1)
                i, j = torch.triu_indices(C.shape[0], C.shape[0], offset=1)
                per_patient.append(float(C[i, j].mean()))
            dec = getattr(model, "slot_decoder_" + mod)
            mapped = dec.map(slots)
            scores_all = dec.pred_keep_slot(mapped).squeeze(-1).double()

            for b in range(slots.shape[0]):
                scores = scores_all[b]
                order = torch.argsort(scores, descending=True)
                gap = float(scores[order[0]] - scores[order[1]])
                kept_counts = torch.zeros(slots.shape[1], dtype=torch.float64)
                overlap_sum = 0.0
                gate_sums = []
                for gseed in range(args.gate_seeds):
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(1000 + gseed)
                        noised = scores + gumbel_noise(scores)
                        top = noised.topk(k=dec.k, dim=-1).indices
                        kept = set(top.tolist())
                        for t in top.tolist():
                            kept_counts[t] += 1
                        overlap_sum += len(kept & {int(order[0]), int(order[1])})
                        gate_sums.append(float((torch.softmax(noised / dec.temperature, dim=-1)
                                               * torch.isin(torch.arange(slots.shape[1]), top).double()).sum()))
                rows.append({
                    "patient_id": ids[b],
                    "modality": mod,
                    "redundancy_mean_pair_cos": per_patient[b],
                    "top1_keep_rate": float(kept_counts[order[0]] / args.gate_seeds),
                    "overlap_with_true_top2": overlap_sum / args.gate_seeds,
                    "score_gap_rank1_rank2": gap,
                    "gumbel_noise_std": float(gumbel_noise(torch.zeros(1, dtype=torch.float64)).std()),
                    "gap_over_noise": gap / 1.28255,
                    "gate_sum_mean": sum(gate_sums) / len(gate_sums),
                })
        done = min(start + batch_size, len(val_ids))
        print(f"[run] {done}/{len(val_ids)} patients")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": {
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256(args.checkpoint),
            "prepared": args.prepared,
            "manifest_sha256": sha256(Path(args.prepared) / "manifest.json"),
            "fold": args.fold, "seed": args.seed, "gate_seeds": args.gate_seeds,
            "n_patients": len(val_ids), "patches": args.patches,
            "mode": "inference only; parameters frozen; no optimizer; no training; source unchanged",
        },
        "rows": rows,
    }
    (out / "joint_rows.json").write_text(json.dumps(payload, indent=1))
    print(f"[done] wrote {out/'joint_rows.json'} rows={len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULTS["checkpoint"])
    ap.add_argument("--prepared", default=DEFAULTS["prepared"])
    ap.add_argument("--out", default=DEFAULTS["out"])
    ap.add_argument("--fold", type=int, default=DEFAULTS["fold"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--gate-seeds", type=int, default=DEFAULTS["gate_seeds"])
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