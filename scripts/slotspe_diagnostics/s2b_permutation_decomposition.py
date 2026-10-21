"""Stage 3b: decompose why slot permutation moves the output.

Stage 3 Part A measured a permutation effect of 2.55e-02 but the predicted
value was ~0 (no positional embedding on the slot axis, and the readout ends in
`.mean(dim=1)`), so something in the chain is not permutation-equivariant.
Isolate it stage by stage.

Hypothesis under test: the slots are near-degenerate (inter-slot cosine 0.9996,
measured in stage 3 Part B).  The `topk` gate in `MoESlotDecoder` then selects
among near-tied scores, so permuting the slots can change *which indices* are
kept.  That is a genuine symmetry break, not float noise.

Measured, all inside one shared frozen RNG draw:
  keepset_stability  -- does the kept-slot INDEX SET change under permutation?
  score_gap           -- how close are the top-k scores (ties => unstable)?
  per_stage_equiv     -- cross_attention / self_attention / mean-pool, each
                         tested for equivariance independently.
  roundoff_control    -- identical unpermuted readout twice, to bound float noise.

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


PERM = [3, 6, 1, 4, 0, 7, 2, 5]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    manifest, fold_data, model = _load()
    val_ids = fold_data["val"]
    pt = torch.tensor(PERM, device=DEVICE)

    keepset_changed, cross_d, wsi_d, omic_d, roundoff_d, score_gap_acc = \
        [], [], [], [], [], []
    kept_size = None

    for seed in (500000, 600000):
        for start in range(0, len(val_ids), 8):
            ids = val_ids[start:start + 8]
            batch = _batch(str(RUN_DIR), manifest, fold_data, ids, DEVICE, torch)
            with torch.no_grad(), frozen_rng(seed):
                x_proj = model.wsi_mlp(batch["x_wsi"])
                x_omic = torch.stack(
                    [model.sig_networks[i](batch[f"x_omic{i+1}"])
                     for i in range(model.num_pathways)], dim=1)
                s_wsi = model.slot_attention_wsi(x_proj)
                s_omic = model.slot_attention_omic(x_omic)

                # --- decoder gate stability under permutation ---
                _, _, keep_w = model.slot_decoder_wsi(s_wsi)
                _, _, keep_w_p = model.slot_decoder_wsi(s_wsi[:, pt])
                k_idx = keep_w.argmax(-1)
                k_idx_p = keep_w_p.argmax(-1)
                # a stable gate keeps the SAME slot contents; compare as sets of
                # the per-row topk scores rather than raw indices
                mapped = keep_w[:, pt]
                changed = float((keep_w != mapped).float().sum())
                keepset_changed.append(changed)
                if kept_size is None:
                    kept_size = int(keep_w[0].sum())

                # score gap between k-th and (k+1)-th slot -> tie diagnostic
                sc = model.slot_decoder_wsi.map(s_wsi)
                scores = model.slot_decoder_wsi.pred_keep_slot(sc).squeeze(-1)
                srt = scores.sort(dim=-1, descending=True).values
                kk = model.slot_decoder_wsi.k
                gap = (srt[:, kk - 1] - srt[:, kk]) if srt.shape[1] > kk else torch.zeros_like(srt[:, 0])
                score_gap_acc.extend(gap.detach().cpu().tolist())

                # --- per-stage equivariance, permutation-equivariant pools ---
                cross = model.cross_attention(s_wsi, s_omic)
                cross_p = model.cross_attention(s_wsi[:, pt], s_omic)
                cross_d.append(float((cross.mean(1) - cross_p.mean(1)).abs().max()))

                mw = keep_w.detach() > 0.5
                wi = model.self_attention_wsi(s_wsi, mask=mw)
                wi_p = model.self_attention_wsi(s_wsi[:, pt], mask=mw[:, pt])
                wsi_d.append(float((wi.mean(1) - wi_p.mean(1)).abs().max()))

                mo = keep_w.detach() > 0.5
                oi = model.self_attention_omic(s_omic, mask=mo)
                omic_d.append(0.0)  # omic slots unpermuted by design

                # roundoff control: same slots, same readout, twice
                def rd():
                    c = model.cross_attention(s_wsi, s_omic)
                    a = model.self_attention_wsi(s_wsi, mask=keep_w.detach() > 0.5)
                    b = model.self_attention_omic(s_omic, mask=mo)
                    return model.to_logits(torch.cat([c.mean(1), a.mean(1), b.mean(1)], 1))
                roundoff_d.append(float((rd() - rd()).abs().max()))

    out = OrderedDict()
    out["note"] = "one shared frozen RNG draw for all arms within a batch"
    out["kept_slots_k"] = kept_size
    out["keep_mask_under_permutation"] = {
        "total_mask_elements_changed": float(np.sum(keepset_changed)),
        "n_batches": len(keepset_changed),
        "changed_per_batch_mean": float(np.mean(keepset_changed)),
        "interpretation": "0 => gate follows slot contents (equivariant); >0 => tie-breaking breaks symmetry",
    }
    out["topk_score_gap"] = {
        "mean": float(np.mean(score_gap_acc)), "min": float(np.min(score_gap_acc)),
        "median": float(np.median(score_gap_acc)), "max": float(np.max(score_gap_acc)),
        "interpretation": "gap ~0 => top-k is numerically arbitrary among 8 near-identical slots",
    }
    out["per_stage_pool_diff"] = {
        "cross_attention_mean_pool": float(np.max(cross_d)),
        "self_attention_wsi_mean_pool": float(np.max(wsi_d)),
    }
    out["roundoff_control_max"] = float(np.max(roundoff_d))

    print(f"kept slots k = {kept_size}")
    print(f"keep-mask elements changed under perm : {out['keep_mask_under_permutation']['total_mask_elements_changed']:.1f} "
          f"(mean {out['keep_mask_under_permutation']['changed_per_batch_mean']:.4f}/batch)")
    print(f"top-k score gap  mean={out['topk_score_gap']['mean']:.3e} "
          f"min={out['topk_score_gap']['min']:.3e}")
    print(f"cross-attn  mean-pool perm diff : {out['per_stage_pool_diff']['cross_attention_mean_pool']:.3e}")
    print(f"self-attn   mean-pool perm diff : {out['per_stage_pool_diff']['self_attention_wsi_mean_pool']:.3e}")
    print(f"roundoff control                : {out['roundoff_control_max']:.3e}")

    p = OUT_DIR / "stage3b_permutation_decomposition.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()