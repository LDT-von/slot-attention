"""Stage 3: corrected S3 slot-permutation test, plus slot-init collapse probe.

Two parts.

PART A -- corrected P4 test.
The previous S3 compared a hand-reconstructed permuted path against the
model's own full forward, with both arms evaluated under *different* slot
re-initialisation draws (the `original` arm re-draws all 8 slots from Gaussian
noise on every forward, and S3 ran no RNG freeze).  The resulting 0.0774 was
therefore dominated by slot re-init noise, not by permutation.

Here both arms are produced inside ONE `_freeze_all_rng` block, so the slot
noise draw is identical.  Prediction: the readout is permutation invariant by
construction (no positional embedding on the slot axis, and the fuse path ends
in `.mean(dim=1)`), so the corrected difference must be ~0 (float roundoff only).

Two controls are run so a non-zero result cannot be misread:
  * control_reconstruction  -- permuted vs unpermuted *reconstruction*, same RNG.
    Any residual drift is pure float roundoff from a different reduction order.
  * control_rng_noise       -- two unpermuted forwards with different RNG, to
    size the mechanism that previously contaminated the measurement.

PART B -- slot-init collapse probe (stage B1).
Slot init is `mu + sigma * randn(...)` with a single shared `mu` for all 8
slots.  If training drove `sigma` small the slots are genuinely collapsed; if
`sigma` is order-1 the slots start close because the parameterisation forces
them to.  Measure `slots_logsigma` directly and compare the initial cosine
spread against the post-training one.

Eval-only. No training, no checkpoint mutation.
"""

from __future__ import annotations

import contextlib
import hashlib
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
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    cd = torch.backends.cudnn.deterministic
    cb = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
        torch.backends.cudnn.deterministic = cd
        torch.backends.cudnn.benchmark = cb


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


def part_a(model, manifest, fold_data, val_ids):
    print("\n[A] corrected P4 slot-permutation test (single shared RNG draw)", flush=True)
    out = OrderedDict()
    perms = [
        [3, 6, 1, 4, 0, 7, 2, 5],
        [7, 6, 5, 4, 3, 2, 1, 0],
        [1, 0, 3, 2, 5, 4, 7, 6],
    ]
    perm_diffs = {str(p): [] for p in perms}
    recon_diffs, rng_noise_diffs, model_vs_recon = [], [], []

    for seed in (500000, 600000):
        for start in range(0, len(val_ids), 8):
            ids = val_ids[start:start + 8]
            batch = _batch(str(RUN_DIR), manifest, fold_data, ids, DEVICE, torch)
            with torch.no_grad(), frozen_rng(seed):
                # ONE frozen block: every arm below shares one slot-noise draw.
                logits_true, _ = model(**batch)

                x_proj = model.wsi_mlp(batch["x_wsi"])
                x_omic = torch.stack(
                    [model.sig_networks[i](batch[f"x_omic{i+1}"])
                     for i in range(model.num_pathways)], dim=1)
                s_wsi = model.slot_attention_wsi(x_proj)
                s_omic = model.slot_attention_omic(x_omic)

                def readout(sw, so):
                    inter = model.cross_attention(sw, so)
                    _, _, keep_w = model.slot_decoder_wsi(sw)
                    _, _, keep_o = model.slot_decoder_omic(so)
                    wi = model.self_attention_wsi(sw, mask=keep_w.detach() > 0.5)
                    oi = model.self_attention_omic(so, mask=keep_o.detach() > 0.5)
                    return model.to_logits(
                        torch.cat([inter.mean(1), wi.mean(1), oi.mean(1)], dim=1))

                base_recon = readout(s_wsi, s_omic)
                model_vs_recon.append(float((base_recon - logits_true).abs().max()))

                for p in perms:
                    pt = torch.tensor(p, device=DEVICE)
                    lp = readout(s_wsi[:, pt], s_omic)
                    perm_diffs[str(p)].append(float((base_recon - lp).abs().max()))

                # control: two identical unpermuted forwards, different RNG
                pass
            # control outside the block: advances RNG -> different slot noise
            with torch.no_grad(), frozen_rng(seed + 1):
                logits_noise, _ = model(**batch)
            rng_noise_diffs.append(float((logits_true - logits_noise).abs().max()))

    out["n_patients"] = len(val_ids)
    out["n_permutations"] = len(perms)
    out["permutation_logit_diff"] = {
        str(p): {"max": float(np.max(v)), "mean": float(np.mean(v)),
                 "min": float(np.min(v))}
        for p, v in perm_diffs.items()
    }
    out["permutation_overall"] = {
        "max": float(max(np.max(v) for v in perm_diffs.values())),
        "mean": float(np.mean([np.mean(v) for v in perm_diffs.values()])),
    }
    out["control_rng_noise_max"] = float(np.max(rng_noise_diffs))
    out["control_rng_noise_mean"] = float(np.mean(rng_noise_diffs))
    out["reconstruction_matches_model_max"] = float(np.max(model_vs_recon))
    out["verdict"] = (
        "PERMUTATION_INVARIANT" if out["permutation_overall"]["max"] < 1e-5
        else "NOT_INVARIANT")
    out["interpretation"] = (
        "Slot permutation changes the output by %.3e, versus %.3e for a pure "
        "slot re-initialisation change under the same harness."
        % (out["permutation_overall"]["max"], out["control_rng_noise_max"]))

    print(f"  [A] permutation max diff = {out['permutation_overall']['max']:.3e}", flush=True)
    print(f"  [A] rng-noise  max diff = {out['control_rng_noise_max']:.3e}", flush=True)
    print(f"  [A] recon vs model      = {out['reconstruction_matches_model_max']:.3e}", flush=True)
    print(f"  [A] VERDICT             = {out['verdict']}", flush=True)
    return out


def part_b(model, manifest, fold_data, val_ids):
    print("\n[B] slot-init collapse probe (B1)", flush=True)
    out = OrderedDict()
    mods = {}
    for name in ("slot_attention_wsi", "slot_attention_omic"):
        m = getattr(model, name)
        mods[name] = m
        mu = m.slots_mu.detach().float().cpu().numpy()
        sig = m.slots_logsigma.detach().float().exp().cpu().numpy()
        out[name] = {
            "slots_mu_shape": list(mu.shape),
            "slots_mu_std": float(mu.std()),
            "slots_mu_absmean": float(np.abs(mu).mean()),
            "sigma_min": float(sig.min()),
            "sigma_max": float(sig.max()),
            "sigma_mean": float(sig.mean()),
            "sigma_std": float(sig.std()),
        }
        print(f"  [B] {name}: sigma_mean={sig.mean():.4f} "
              f"sigma_std={sig.std():.4f} mu_std={mu.std():.4f}", flush=True)

    # initial-only cosine spread, no attention iteration
    torch.manual_seed(0)
    spread = {}
    for name, m in mods.items():
        g = torch.Generator(device="cpu").manual_seed(1234)
        mu = m.slots_mu.detach()
        sig = m.slots_logsigma.detach().exp()
        z = torch.randn(1, m.num_slots, m.dim, generator=g).to(DEVICE)
        s0 = (mu + sig * z)[0]
        v = torch.nn.functional.normalize(s0, dim=-1)
        c = (v @ v.T).cpu().numpy()
        np.fill_diagonal(c, 0.0)
        spread[name] = {"init_max_offdiag_cos": float(c.max()),
                        "init_min_offdiag_cos": float(c.min()),
                        "init_mean_offdiag_cos": float(c.mean())}
        print(f"  [B] {name}: init cos max={c.max():.6f} mean={c.mean():.6f}", flush=True)
    out["init_only_cosine"] = spread

    # post-training cosine, frozen RNG
    with frozen_rng(42), torch.no_grad():
        post = []
        for s in range(0, len(val_ids), 8):
            cache = [torch.load(Path(RUN_DIR) / manifest["patients"][pid]["cache_path"],
                                map_location="cpu", weights_only=True)
                     for pid in val_ids[s:s + 8]]
            x = torch.stack([it["wsi"] for it in cache]).to(DEVICE)
            sl = model.slot_attention_wsi(model.wsi_mlp(x))
            for i in range(sl.size(0)):
                v = torch.nn.functional.normalize(sl[i], dim=-1)
                c = (v @ v.T).cpu().numpy()
                np.fill_diagonal(c, 0.0)
                post.append(float(c.max()))
    out["post_training_max_offdiag_cos_mean"] = float(np.mean(post))
    out["verdict"] = (
        "COLLAPSE_IS_INIT_ARTEFACT"
        if out["post_training_max_offdiag_cos_mean"] - list(spread.values())[0]["init_mean_offdiag_cos"] < 0.01
        else "TRAINING_COLLAPSED_SLOTS")
    print(f"  [B] post-training cos mean = {out['post_training_max_offdiag_cos_mean']:.6f}", flush=True)
    print(f"  [B] VERDICT                 = {out['verdict']}", flush=True)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    manifest, fold_data, model = _load()
    val_ids = fold_data["val"]
    print(f"[load] val={len(val_ids)}", flush=True)

    res = OrderedDict()
    res["stage"] = "3_corrected_s3_and_b1"
    res["checkpoint_sha256"] = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    res["checkpoint_note"] = "checkpoint unchanged; read-only"
    res["part_a_corrected_permutation"] = part_a(model, manifest, fold_data, val_ids)
    res["part_b_slot_init_collapse"] = part_b(model, manifest, fold_data, val_ids)

    p = OUT_DIR / "stage3_corrected_s3_b1.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\n[done] wrote {p}", flush=True)
    print(f"[done] ckpt sha256 {res['checkpoint_sha256']}", flush=True)


if __name__ == "__main__":
    main()