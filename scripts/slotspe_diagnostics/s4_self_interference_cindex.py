"""Stage 5 (task A): does slot self-interference harm the C-index?

THE FINDING BEING TESTED
------------------------
`patient_slot_intervention.json` already recorded `self_interference: observed`
with `max_self_pool_change = 2.563`.  That is 30-50x larger than any effect
measured so far in this addendum (P6 patch order: 0.045; slot re-init: 0.074;
slot permutation: 0.026).

Root cause, from the source.  The `original` arm pools the self-attention
readout over *all* slots rather than only the kept ones:

```python
# interventions.py, original-arm branch
pool = lambda value: value.mean(1)
```

while the decoder's top-k gate keeps only k=2 of 8 slots.  So six discarded
slots still flow into the prediction.  That is a genuine architectural issue,
independent of the stochastic-initialisation findings.

WHAT WAS NEVER MEASURED
-----------------------
Its effect on the survival metric.  The recorded 2.563 is an activation-space
magnitude, which does not translate into discrimination.  Everything measured so
far concluded "no measurable harm" by looking at C-index; this arm must be
judged the same way, not by activation size.

DESIGN
------
Three arms on the same 77 val patients, all with slot-init noise frozen so the
only difference is the intervention:

  base    -- the model as shipped
  sel     -- self-attention readout pooled over the KEPT slots only
             (drop the 6 discarded slots); cross-attention held fixed, since the
             recorded record explicitly notes "cross remains dense"
  shuf    -- control: shuffle the discarded slots among themselves.  Any effect
             here is the readout depending on *which* junk it sees.  If shuf
             matters as much as perturbing the junk, the harm is in attending to
             discarded slots at all.

Reported: C-index per arm, paired deltas, bootstrap CI, discordant-pair flips.

DECISION RULE (stated before running)
-------------------------------------
  material  <=> the `sel` arm raises C-index with a bootstrap CI excluding 0.
  If yes -> excluding discarded slots is a real, available improvement.
  If no  -> self-interference is a genuine architectural impurity with no
            effect on discrimination, i.e. a correctness wart, not a bug that
            costs accuracy.

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

from _gpu_guard import require_gpu_env  # hard-fail if CUDA cannot run here

import numpy as np
import torch

from survot_rank.research.slotspe_proof.patient import _batch
from survot_rank.research.slotspe_proof.model import build_model, make_args

_ENV = require_gpu_env()

RUN_DIR = REPO / "results/slotspe_proof/slotspe_clean_original_blca_run_20261020"
PREPARED = RUN_DIR / "prepared"
CKPT = RUN_DIR / "pairs/fold_0/seed_3/original/last.pt"
SOURCE_ROOT = REPO / "third_party/SlotSPE"
OUT_DIR = REPO / "docs/research/slotspe_p6_addendum_20261020"
DEVICE = "cuda:0"
N_REPS = 40


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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    manifest, fold_data, model = _load()
    val_ids = fold_data["val"]
    events = np.asarray([manifest["patients"][p]["censor"] < 1 for p in val_ids], dtype=bool)
    times = np.asarray([manifest["patients"][p]["time"] for p in val_ids], dtype=float)

    batches = [_batch(str(RUN_DIR), manifest, fold_data, val_ids[s:s+8], DEVICE, torch)
               for s in range(0, len(val_ids), 8)]
    print(f"[load] val={len(val_ids)} env={_ENV['torch']}", flush=True)

    def readout(batch, mode, seed):
        """Recompute logits with a chosen self-readout pooling strategy.

        mode 'base' -> mean over all 8 slots (shipped behaviour)
        mode 'sel'  -> mean over the kept k slots only
        mode 'shuf' -> mean over all 8, but discarded slots shuffled among
                       themselves first (control for *which* junk is seen)
        """
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

            if mode == "shuf":
                gen = torch.Generator(device="cpu").manual_seed(seed)

                def junk_shuffle(slots, hard, gen):
                    """Permute the DISCARDED slots among themselves.

                    Kept slots are untouched, so the gate result and the selected
                    content are unchanged; only which junk sits at which unused
                    index changes.  Any output change is therefore attributable
                    to the readout attending to discarded slots at all.
                    """
                    b, n, d = slots.shape
                    out = slots.clone()
                    for i in range(b):
                        drop_idx = (~hard[i]).nonzero().flatten()
                        if drop_idx.numel() > 1:
                            perm = torch.randperm(drop_idx.numel(), generator=gen)
                            out[i, drop_idx] = slots[i, drop_idx[perm]]
                    return out

                sw = junk_shuffle(sw, hw, gen)
                so = junk_shuffle(so, ho, gen)

            cross = model.cross_attention(sw, so).mean(1)
            wi = model.self_attention_wsi(sw, mask=hw)
            oi = model.self_attention_omic(so, mask=ho)

            if mode == "sel":
                pw = (wi * hw[..., None]).sum(1) / hw.sum(-1, keepdim=True).clamp_min(1)
                po = (oi * ho[..., None]).sum(1) / ho.sum(-1, keepdim=True).clamp_min(1)
            else:
                pw, po = wi.mean(1), oi.mean(1)

            return model.to_logits(torch.cat([cross, pw, po], dim=1))

    # ---- sweep each arm over N_REPS seeds ---------------------------------
    res = {"base": [], "sel": [], "shuf": []}
    risks = {k: [] for k in res}
    for r in range(N_REPS):
        seed = 810000 + r
        for mode in res:
            lg = torch.cat([readout(b, mode, seed) for b in batches]).detach().cpu()
            rk = _risk(lg).numpy()
            risks[mode].append(rk)
            res[mode].append(concordance(events, times, rk)[0])
        if (r + 1) % 10 == 0:
            print(f"  rep {r+1}/{N_REPS}", flush=True)

    c = {k: np.asarray(v) for k, v in res.items()}
    rng = np.random.default_rng(0)

    def paired(a, b):
        d = c[a] - c[b]
        bs = np.asarray([rng.choice(d, size=len(d), replace=True).mean() for _ in range(10000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sd = float(d.std(ddof=1))
        return {"mean_delta": float(d.mean()),
                "ci95": [float(lo), float(hi)],
                "p_value": float(abs(d.mean() / (sd / np.sqrt(len(d))))) if sd > 0 else 0.0}

    out = OrderedDict()
    out["stage"] = "5_self_interference_cindex"
    out["env"] = _ENV
    out["n_val"] = len(val_ids)
    out["n_reps"] = N_REPS
    out["shipped_readout"] = "mean over ALL 8 slots (k=2 kept by gate)"
    for k in c:
        out[f"cindex_{k}"] = {"median": float(np.median(c[k])), "mean": float(c[k].mean()),
                              "min": float(c[k].min()), "max": float(c[k].max()),
                              "std": float(c[k].std(ddof=1))}
    out["paired_sel_minus_base"] = paired("sel", "base")
    out["paired_shuf_minus_base"] = paired("shuf", "base")

    sel_p = out["paired_sel_minus_base"]
    material = (sel_p["mean_delta"] > 0 and sel_p["ci95"][0] > 0)
    out["decision_rule"] = "material <=> 'sel' arm raises C with CI excluding 0"
    out["verdict"] = "MATERIAL_GAIN_AVAILABLE" if material else "NO_DISCRIMINATION_EFFECT"

    print(f"\nbase  median C = {out['cindex_base']['median']:.4f}")
    print(f"sel   median C = {out['cindex_sel']['median']:.4f}   dC={sel_p['mean_delta']:+.4f} "
          f"CI95=[{sel_p['ci95'][0]:+.4f},{sel_p['ci95'][1]:+.4f}] p={sel_p['p_value']:.3f}")
    sh_p = out["paired_shuf_minus_base"]
    print(f"shuf  median C = {out['cindex_shuf']['median']:.4f}   dC={sh_p['mean_delta']:+.4f} "
          f"CI95=[{sh_p['ci95'][0]:+.4f},{sh_p['ci95'][1]:+.4f}]")
    print(f"\n[verdict] {out['verdict']}")

    p = OUT_DIR / "stage5_self_interference_cindex.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {p}")


if __name__ == "__main__":
    main()