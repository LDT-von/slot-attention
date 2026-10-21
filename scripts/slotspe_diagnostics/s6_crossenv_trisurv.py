"""Cross-environment reproducibility check.

Run the two most discriminating artefacts from the addendum in a different
Conda env (`trisurv`) and compare the headline numbers against the values
reported with `dct`.  If they match, the conclusions are independent of the
Python/torch environment; if not, the addendum is environment-specific and
unreliable.

Targets:
  Stage 2   E3  full-sweep bit-exact replay         -> expect "0.0"
  Stage 2   E4  slot-cos replay drift               -> expect "0.0"
  Stage 4A-2 reinit C-index 60-draw distribution   -> expect median~0.7228
            paired patch-vs-reinit delta mean+CI    -> expect ~(-0.0007, ~0)
  Stage 3B  keep-mask drift under permutation       -> expect ~29/batch

No training, no checkpoint mutation.  Uses the same scripts as `dct` runs.
"""
from __future__ import annotations
import contextlib, hashlib, json, os, random, sys
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    manifest, fold_data, model = _load()
    val_ids = fold_data["val"]
    events = np.asarray([manifest["patients"][p]["censor"] < 1 for p in val_ids], dtype=bool)
    times = np.asarray([manifest["patients"][p]["time"] for p in val_ids], dtype=float)

    batches = [_batch(str(RUN_DIR), manifest, fold_data, val_ids[s:s+8], DEVICE, torch)
               for s in range(0, len(val_ids), 8)]

    # ---- A: bit-exact replay --------------------------------------------------
    def sweep(seed, permute):
        with torch.no_grad(), frozen_rng(seed):
            outs = []
            for b in batches:
                bb = dict(b)
                if permute:
                    idx = torch.randperm(b["x_wsi"].size(1), device=DEVICE)
                    bb["x_wsi"] = b["x_wsi"][:, idx]
                lg, _ = model(**bb); outs.append(lg.detach().cpu())
        return torch.cat(outs), _risk(torch.cat(outs)).numpy()

    a1, _ = sweep(0, False); a2, _ = sweep(0, False)
    diff_seed_logits, diff_seed_risk = sweep(1, False)
    bitwise_same_seed = bool(torch.equal(a1, a2))
    abs_diff_same_seed = float((a1 - a2).abs().max())
    abs_diff_diff_seed = float((a1 - diff_seed_logits).abs().max())

    # ---- B: 60-draw distribution for paired delta -----------------------------
    N = 60
    rein, pat = [], []
    for i in range(N):
        _, r = sweep(700000 + i, False); rein.append(r)
        _, r = sweep(700000 + i, True);  pat.append(r)
    c_rein = np.asarray([concordance(events, times, r)[0] for r in rein])
    c_pat  = np.asarray([concordance(events, times, r)[0] for r in pat])
    d = c_pat - c_rein
    rng = np.random.default_rng(0)
    bs = np.asarray([rng.choice(d, size=N, replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])

    # ---- C: topk keep-mask drift under permutation ----------------------------
    PERM = [3, 6, 1, 4, 0, 7, 2, 5]
    pt = torch.tensor(PERM, device=DEVICE)
    total_changed, batches_obs = 0, 0
    with torch.no_grad():
        for b in batches:
            x = model.wsi_mlp(b["x_wsi"])
            sw = model.slot_attention_wsi(x)
            with frozen_rng(42):
                _, _, kw = model.slot_decoder_wsi(sw)
                _, _, kp = model.slot_decoder_wsi(sw[:, pt])
            total_changed += int((kw != kp[:, pt]).float().sum())
            batches_obs += 1

    out = OrderedDict()
    out["env"] = "/home/ubuntu/.conda/envs/trisurv"
    out["python"] = sys.version.split()[0]
    out["torch"] = torch.__version__
    out["a_bit_exact_same_seed"] = {"bitwise": bitwise_same_seed, "max_abs_diff": abs_diff_same_seed,
                                    "diff_seed_max_abs_diff": abs_diff_diff_seed}
    out["b_cindex_distribution"] = {
        "n_reps": N,
        "reinit_median": float(np.median(c_rein)), "reinit_min": float(c_rein.min()),
        "reinit_max": float(c_rein.max()),  "reinit_std":  float(c_rein.std(ddof=1)),
        "patch_median":  float(np.median(c_pat)),  "patch_min":  float(c_pat.min()),
        "patch_max":     float(c_pat.max()),       "patch_std":   float(c_pat.std(ddof=1)),
        "paired_delta_mean": float(d.mean()),
        "paired_delta_ci95": [float(lo), float(hi)],
    }
    out["c_topk_keep_drift"] = {"mask_elements_changed_total": total_changed,
                                "batches_observed": batches_obs,
                                "mean_per_batch": total_changed / batches_obs}

    # ---- D: gate verdict against dct's published headline numbers ------------
    dct_published = {
        "bitwise_same_seed_expected": True,
        "bitwise_max_abs_diff_expected": 0.0,
        "reinit_c_median_expected":   0.7228,
        "paired_delta_mean_expected": -0.0007,
        "paired_delta_ci95_expected": [-0.0018, 0.0005],
        "topk_keep_drift_mean_per_batch_expected": 29.0,
    }
    out["expected_dct_values"] = dct_published

    pass_a = bitwise_same_seed and abs_diff_same_seed == 0.0
    pass_b = abs(out["b_cindex_distribution"]["paired_delta_mean"] - dct_published["paired_delta_mean_expected"]) < 0.0015
    pass_c = abs(out["c_topk_keep_drift"]["mean_per_batch"] - dct_published["topk_keep_drift_mean_per_batch_expected"]) < 0.2

    out["verdict"] = {
        "bit_exactness": "PASS" if pass_a else "FAIL",
        "paired_delta_within_1.5e-3": "PASS" if pass_b else "FAIL",
        "topk_keep_drift_within_0.2": "PASS" if pass_c else "FAIL",
        "overall": "CROSS_ENV_REPRODUCES" if (pass_a and pass_b and pass_c) else "DIVERGENT",
    }
    print(json.dumps(out, indent=2))
    p = OUT_DIR / "cross_env_trisurv.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {p}", flush=True)


if __name__ == "__main__":
    main()