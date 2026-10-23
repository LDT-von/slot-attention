#!/usr/bin/env python3
"""
Does slot redundancy cost predictive performance? A CLEAN forward-iters counterfactual.

This continues the user's Q1 ("run fewer iterations so the slots are less
similar, then look at the score"), but removes the confound in the naive form.
Training a model for fewer iterations changes the budget, the convergence
state AND the redundancy at once, so a worse score would be uninterpretable.

Here the trained weights are FROZEN and ONLY the number of forward attention
iterations changes, on ONE checkpoint and ONE patient set.  Everything else is
held fixed, and each arm sees an identical initial-slot draw because
``MultiHeadSlotAttention.forward`` samples ``slots = mu + sigma * randn`` OUTSIDE
the iteration loop (slot_attention.py line 80), so a shorter loop does not shift
the random stream.

Arms: iters = 10 (native), 5, 2, 1.  Measured per patient and modality:

  all_pair_cosine        mean pairwise cosine of the final slots
  centered_effective_rank  centered singular-spectrum effective rank
  centered_energy_fraction
  cindex                 cohort C-index of the final fused risk
  nll                    native discrete NLL

Redundancy rises monotonically with iteration count inside a single forward pass,
so if redundancy were harmful the C-index should fall as iters grows.  This is a
within-checkpoint counterfactual on one split, so it bounds the effect under this
budget; it does not estimate a trained model's performance.

Pure inference. No optimizer, no parameter update, no training, no source edit.
Default is PLAN ONLY. Pass --run to execute.
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
    "out": "/data1/SlotSPE/local_evidence/slot_iters_counterfactual_20261023",
    "fold": 0,
    "seed": 3,
    "arms": [10, 5, 2, 1],
    "batch_size": 8,
    "patches": 256,
}


def build(args):
    import torch
    from survot_rank.research.slotspe_proof import patient as P

    manifest = json.loads((Path(args.prepared) / "manifest.json").read_text())
    fold_data = manifest["folds"][str(args.fold)]
    val_ids = fold_data["val"]
    print(f"[plan] fold={args.fold} seed={args.seed} val={len(val_ids)} arms={args.arms}")

    source = os.environ.get("SLOTSPE_SOURCE_ROOT", str(REPO / "third_party" / "SlotSPE"))
    if source not in sys.path:
        sys.path.insert(0, source)
    from models.SlotSPE import SlotSPE
    from models.slot_attention import MultiHeadSlotAttention
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
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # The only mutation anywhere in this script: the native integer attribute.
    attentions = [m for m in model.modules() if isinstance(m, MultiHeadSlotAttention)]
    print(f"[plan] found {len(attentions)} native MultiHeadSlotAttention modules")
    for m in attentions:
        print(f"[plan]   {m.__class__.__name__} native iters={m.iters}")

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
    return torch, P, model, manifest, fold_data, val_ids, device, attentions


def measure(args):
    import numpy as np
    import torch
    from sksurv.metrics import concordance_index_censored
    from survot_rank.research.slotspe_proof.patient import _rng
    from survot_rank.research.slotspe_proof.gate_pair import pair_statistics
    from survot_rank.research.slotspe_proof.problem_trace import NativeSlotTrace

    torch, P, model, manifest, fold_data, val_ids, device, attentions = build(args)
    root = args.prepared.rsplit("/prepared", 1)[0]
    events = np.asarray([manifest["patients"][pid]["censor"] < 1 for pid in val_ids], dtype=bool)
    times = np.asarray([manifest["patients"][pid]["time"] for pid in val_ids], dtype=float)

    trace = NativeSlotTrace(model)
    # NativeSlotTrace records the attention path; the gate and the hard keep mask
    # are returned by the DECODER, so they need their own hooks.
    captured = {}
    handles = [getattr(model, "slot_decoder_" + m).register_forward_hook(
        lambda module, inputs, output, key=m: captured.__setitem__(key, output))
        for m in ("wsi", "omic")]
    arms = []
    try:
        for iters in args.arms:
            for m in attentions:
                m.iters = iters
            per_modality = {"wsi": [], "omic": []}
            gate_eff = {"wsi": [], "omic": []}
            kept_n = {"wsi": [], "omic": []}
            per_patient_risks, losses = [], 0.0

            for index, start in enumerate(range(0, len(val_ids), args.batch_size)):
                ids = val_ids[start:start + args.batch_size]
                # One entry seed per batch, shared by every arm: the initial-slot
                # draw and the Gumbel noise are identical across arms.
                _rng(args.seed * 1000 + index, torch, np)
                trace.reset()
                captured.clear()
                batch = P._batch(root, manifest, fold_data, ids, device, torch)
                with torch.no_grad():
                    logits, _ = model(**batch)
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError(f"nonfinite logits at iters={iters}")
                losses += float(model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None))
                per_patient_risks.extend((-(1 - logits.sigmoid()).cumprod(-1).sum(-1)).cpu().numpy().tolist())

                for modality in ("wsi", "omic"):
                    record = trace.records[modality]
                    slots = record["final"].double().cpu().numpy()
                    attention = record["attentions"][-1].double().cpu().numpy()
                    per_modality[modality].append(pair_statistics(slots, attention))
                    _, gate, keep = captured[modality]
                    row = gate.sum(dim=-1, keepdim=True)
                    if bool((row > 0).all()):
                        w = (gate / row).double().cpu().numpy()
                        gate_eff[modality].extend(float(v) for v in 1.0 / np.sum(w ** 2, axis=-1))
                    kept_n[modality].append(float((keep > 0.5).sum(-1).double().mean()))
                print(f"[run] iters={iters} {min(start + args.batch_size, len(val_ids))}/{len(val_ids)}", flush=True)

            risk_arr = np.asarray(per_patient_risks, dtype=float)
            try:
                c = float(concordance_index_censored(events, times, risk_arr)[0])
                c = c if math.isfinite(c) else None
                status = "computed"
            except ValueError as exc:
                c, status = None, "undefined: " + str(exc)

            def avg(rows, key):
                vals = [v for r in rows for v in r[key]]
                return float(np.mean(vals)) if vals else None

            summary = {}
            for modality in ("wsi", "omic"):
                rows = per_modality[modality]
                all_cos = [v for r in rows for v in r["all_pair_cosine"]]
                summary[modality] = {
                    "patients": len(rows),
                    "all_pair_cosine_mean": float(np.mean(all_cos)) if all_cos else None,
                    "all_pair_cosine_min": float(np.min(all_cos)) if all_cos else None,
                    "pairs_cosine_above_0_95_fraction": avg(rows, "pairs_above_0_95"),
                    "all_pair_reading_overlap_mean": avg(rows, "all_pair_reading_overlap"),
                    "centered_effective_rank_mean": avg(rows, "centered_effective_rank"),
                    "centered_energy_fraction_mean": avg(rows, "centered_energy_fraction"),
                    "gate_weight_effective_slots_mean": float(np.mean(gate_eff[modality])) if gate_eff[modality] else None,
                    "mean_hard_selected_slots": float(np.mean(kept_n[modality])) if kept_n[modality] else None,
                }
            arms.append({
                "iters": iters, "cindex": c, "cindex_status": status,
                "nll": losses / len(val_ids), "redundancy": summary,
            })
            print(f"[run] iters={iters} C-index={c} WSI cos={summary['wsi']['all_pair_cosine_mean']:.6f} "
                  f"rank={summary['wsi']['centered_effective_rank_mean']:.4f}", flush=True)
    finally:
        for m in attentions:
            m.iters = 10
        for h in handles:
            h.remove()
        trace.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "arms.json").write_text(json.dumps({
        "provenance": {
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256(args.checkpoint),
            "prepared_manifest_sha256": sha256(Path(args.prepared) / "manifest.json"),
            "fold": args.fold, "seed": args.seed, "arms": args.arms,
            "n_patients": len(val_ids), "patches": args.patches,
            "mode": "inference only; parameters frozen; no optimizer; no training; source unchanged",
            "only_mutation": "native integer attribute MultiHeadSlotAttention.iters; restored to 10 after the sweep",
            "initial_slot_draw_note": "slots = mu + sigma*randn is sampled OUTSIDE the iteration loop "
                                      "(slot_attention.py line 80), so all arms share the identical "
                                      "initial-slot draw and random stream",
            "risk_definition": "-(1-sigmoid).cumprod(-1).sum(-1), identical to the recorded harness",
        },
        "arms": arms,
    }, indent=1))
    print(f"[done] wrote {out/'arms.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULTS["checkpoint"])
    ap.add_argument("--prepared", default=DEFAULTS["prepared"])
    ap.add_argument("--out", default=DEFAULTS["out"])
    ap.add_argument("--fold", type=int, default=DEFAULTS["fold"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--arms", type=int, nargs="+", default=DEFAULTS["arms"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--patches", type=int, default=DEFAULTS["patches"])
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if not args.run:
        print(json.dumps({"mode": "plan", "command": "--run", **DEFAULTS}, indent=1))
        print("\nPlan only. No forward pass, no training.")
        return
    measure(args)


if __name__ == "__main__":
    main()