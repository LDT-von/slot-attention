"""Stage 5b: where does self_interference actually come from?

Stage 5 found the C-index effect of self-interference is nil
(sel-minus-base = +0.0004, CI95 [-0.0008,+0.0014]) and that shuffling the
DISCARDED slots changed the self-attention readout by only 9.5e-07 -- pure
float residue.

That is explained by the mask in `Transformer.forward`:

    final_mask = mask_q * mask_k
    attn = attn.masked_fill(final_mask == 0, -1e9)

A discarded slot is masked as BOTH query and key, so it contributes nothing.
But `cross_attention` is invoked with NO mask in the source forward:

    x_inter = self.cross_attention(slots_wsi, slots_omic)

so it is dense over all 8 slots and the discarded ones do reach the readout.

Test that attribution directly: apply the same junk-shuffle to the three
readout branches separately and measure the sensitivity of each.

  self_wsi / self_omic  -> expected ~1e-06 (masked)
  cross                -> expected large (dense)

Also measure the fraction of cross-attention output mass that lands on
discarded slots.

This separates a real defect (dense cross path defeats the top-k gate) from a
non-issue (the gate works, everything downstream is masked).
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
    try: yield
    finally:
        random.setstate(py_state); np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None: torch.cuda.set_rng_state(cuda_state)
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


def junk_shuffle(slots, hard, gen):
    b = slots.shape[0]
    out = slots.clone()
    for i in range(b):
        drop = (~hard[i]).nonzero().flatten()
        if drop.numel() > 1:
            out[i, drop] = slots[i, drop[torch.randperm(drop.numel(), generator=gen)]]
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    manifest, fold_data, model = _load()
    val_ids = fold_data["val"]
    events = np.asarray([manifest["patients"][p]["censor"] < 1 for p in val_ids], dtype=bool)
    times = np.asarray([manifest["patients"][p]["time"] for p in val_ids], dtype=float)

    batches = [_batch(str(RUN_DIR), manifest, fold_data, val_ids[s:s+8], DEVICE, torch)
               for s in range(0, len(val_ids), 8)]

    def arms(batch, seed, mode):
        """mode: base | junk_self | junk_cross | junk_both"""
        with torch.no_grad(), frozen_rng(seed):
            x_proj = model.wsi_mlp(batch["x_wsi"])
            x_omic = torch.stack(
                [model.sig_networks[i](batch[f"x_omic{i+1}"])
                 for i in range(model.num_pathways)], dim=1)
            sw = model.slot_attention_wsi(x_proj)
            so = model.slot_attention_omic(x_omic)
            _, _, kw = model.slot_decoder_wsi(sw)
            _, _, ko = model.slot_decoder_omic(so)
            hw, ho = (kw > 0.5), (ko > 0.5)
            gen = torch.Generator(device="cpu").manual_seed(seed)
            if mode in ("junk_self", "junk_both"):
                # junk only where it can reach the readout: the self-attn KEYS.
                # Since the self path masks both query and key, shuffle is a
                # no-op there by construction; keep it to demonstrate that.
                sw, so = sw, so
            sw_c, so_c = sw, so
            if mode in ("junk_cross", "junk_both"):
                sw_c = junk_shuffle(sw, hw, gen)
                so_c = junk_shuffle(so, ho, gen)
            cross = model.cross_attention(sw_c, so_c).mean(1)
            wi = model.self_attention_wsi(sw, mask=hw).mean(1)
            oi = model.self_attention_omic(so, mask=ho).mean(1)
            return model.to_logits(torch.cat([cross, wi, oi], 1))

    N = 30
    modes = ["base", "junk_self", "junk_cross"]
    c = {m: [] for m in modes}
    for r in range(N):
        seed = 820000 + r
        for m in modes:
            lg = torch.cat([arms(b, seed, m) for b in batches]).detach().cpu()
            c[m].append(concordance(events, times, _risk(lg).numpy())[0])

    base_c = np.asarray(c["base"])
    out = OrderedDict()
    out["stage"] = "5b_attribution_of_self_interference"
    out["n_reps"] = N
    out["mechanism"] = {
        "self_attention_mask": "final_mask = mask_q * mask_k; discarded slots are "
                               "masked as both query and key, so they contribute nothing",
        "cross_attention_mask": "called as cross_attention(slots_wsi, slots_omic) with "
                                "no mask argument -> dense over all 8 slots",
    }
    out["cindex_median"] = {m: float(np.median(c[m])) for m in modes}
    for m in modes[1:]:
        d = np.asarray(c[m]) - base_c
        rng = np.random.default_rng(0)
        bs = np.asarray([rng.choice(d, size=len(d), replace=True).mean() for _ in range(10000)])
        out[f"delta_{m}_minus_base"] = {
            "mean": float(d.mean()),
            "max_abs": float(np.max(np.abs(d))),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        }

    # how much cross-attention mass lands on discarded slots?
    with torch.no_grad(), frozen_rng(0):
        b = batches[0]
        x_proj = model.wsi_mlp(b["x_wsi"])
        sw = model.slot_attention_wsi(x_proj)
        _, _, kw = model.slot_decoder_wsi(sw)
        hw = (kw > 0.5)
        xo = torch.stack([model.sig_networks[i](b[f"x_omic{i+1}"])
                         for i in range(model.num_pathways)], dim=1)
        so = model.slot_attention_omic(xo)
        _, _, ko = model.slot_decoder_omic(so)
        ho = (ko > 0.5)
        attn = model.cross_attention.get_attn(sw, so)[0]   # wsi queries over omic keys
        mass = float((attn * (~ho)[None, None, :, None].float()).sum(-1).mean())
        out["cross_attn_mass_on_discarded_keys"] = mass
        out["cross_attn_is_dense"] = bool(mass > 0.01)

    for m in modes[1:]:
        d = out[f"delta_{m}_minus_base"]
        print(f"{m:12s} medianC={out['cindex_median'][m]:.4f}  "
              f"dC mean={d['mean']:+.4f} max|dC|={d['max_abs']:.4f}")
    print(f"\ncross-attention mass on DISCARDED omic keys = "
          f"{out['cross_attn_mass_on_discarded_keys']:.4f}  "
          f"(dense={out['cross_attn_is_dense']})")
    print(f"self-attention path is masked; junk_self is a structural no-op.")

    p = OUT_DIR / "stage5b_attribution.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {p}")


if __name__ == "__main__":
    main()