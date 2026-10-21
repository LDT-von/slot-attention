"""Stage 2: bit-exactness probe for the frozen-RNG eval harness.

Question: is a single forward pass, executed twice under the same seed and the
same batch construction, *bit-identical*?  Earlier measurements reported a
~1e-2 drift between two full runs (residual cos 0.9667 -> 0.9777) that the
frozen-RNG control could not explain.  Stage 1 traced all randomness sources;
this probe isolates the remaining candidate: non-bit-exact float accumulation
under cuDNN autotuning.

Design:
  - 1 patient, 1 batch, 5 repeats of the identical forward.
  - Compare raw logits bitwise (not approximately): every float must match the
    reference exactly.
  - Toggle `torch.use_deterministic_algorithms` to test whether it is the fix.
  - Separately, replay the *whole 77-patient sweep* twice to reproduce or
    refute the original drift at the level it was originally observed.

No training. No checkpoint mutation.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
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
    """Verbatim discipline from _s2026_1020_addendum._freeze_all_rng."""
    import random

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


def _bits(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().numpy().tobytes()


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


def _forward_once(model, manifest, fold_data, ids, seed):
    with frozen_rng(seed):
        with torch.no_grad():
            out = []
            for s in range(0, len(ids), 8):
                batch = _batch(str(RUN_DIR), manifest, fold_data,
                               ids[s:s + 8], DEVICE, torch)
                logits, _ = model(**batch)
                out.append(logits.detach().cpu())
            return torch.cat(out)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    print("[load] ...", flush=True)
    manifest, fold_data, model = _load()
    val_ids = fold_data["val"]
    print(f"[load] val={len(val_ids)}", flush=True)

    results = {"stage": "2_bitexact_probe", "checkpoint_sha256": None, "experiments": {}}

    # ---- E1: single patient, 5 repeats, default determinism off -------------
    probe_ids = val_ids[:8]
    runs = [_forward_once(model, manifest, fold_data, probe_ids, 42) for _ in range(5)]
    ref = _bits(runs[0])
    pairwise = []
    for i in range(1, 5):
        pairwise.append(float(np.abs(runs[i].numpy() - runs[0].numpy()).max()))
    e1 = {
        "n_repeats": 5,
        "ids": len(probe_ids),
        "pairwise_max_abs_diff": pairwise,
        "all_bitwise_identical": all(_bits(r) == ref for r in runs[1:]),
        "logit_sha256": [hashlib.sha256(_bits(r)).hexdigest()[:16] for r in runs],
    }
    results["experiments"]["E1_default_determinism"] = e1
    print(f"[E1] pairwise max diffs = {pairwise}", flush=True)
    print(f"[E1] bitwise identical  = {e1['all_bitwise_identical']}", flush=True)

    # ---- E2: same, with use_deterministic_algorithms ----------------------
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        runs2 = [_forward_once(model, manifest, fold_data, probe_ids, 42) for _ in range(5)]
        ref2 = _bits(runs2[0])
        e2 = {
            "use_deterministic_algorithms": True,
            "pairwise_max_abs_diff": [
                float(np.abs(runs2[i].numpy() - runs2[0].numpy()).max()) for i in range(1, 5)
            ],
            "all_bitwise_identical": all(_bits(r) == ref2 for r in runs2[1:]),
            "logit_sha256": [hashlib.sha256(_bits(r)).hexdigest()[:16] for r in runs2],
        }
        print(f"[E2] bitwise identical  = {e2['all_bitwise_identical']}", flush=True)
    except Exception as exc:  # pragma: no cover
        e2 = {"error": repr(exc)}
        print(f"[E2] raised: {exc}", flush=True)
    results["experiments"]["E2_use_deterministic_algorithms"] = e2
    torch.use_deterministic_algorithms(False)

    # ---- E3: full 77-patient sweep replayed twice -------------------------
    sweep_a = _forward_once(model, manifest, fold_data, val_ids, 42)
    sweep_b = _forward_once(model, manifest, fold_data, val_ids, 42)
    sweep_c = _forward_once(model, manifest, fold_data, val_ids, 43)
    d_ab = float(np.abs(sweep_a.numpy() - sweep_b.numpy()).max())
    d_ac = float(np.abs(sweep_a.numpy() - sweep_c.numpy()).max())
    e3 = {
        "n_patients": len(val_ids),
        "same_seed_max_abs_diff": d_ab,
        "same_seed_bitwise_identical": _bits(sweep_a) == _bits(sweep_b),
        "diff_seed_max_abs_diff": d_ac,
        "interpretation": (
            "same_seed ~0 and diff_seed >0 => harness isolates order effects cleanly; "
            "same_seed >0 => residual non-determinism remains"
        ),
    }
    results["experiments"]["E3_full_sweep_replay"] = e3
    print(f"[E3] same-seed max diff = {d_ab:.3e} (bitwise={e3['same_seed_bitwise_identical']})",
          flush=True)
    print(f"[E3] diff-seed max diff = {d_ac:.3e}", flush=True)

    # ---- E4: reproduce the ORIGINAL reported drift ------------------------
    # Original S2 residual-cos was computed from slot activations, not logits.
    # Recompute it here under one seed to see whether 0.9667 vs 0.9777 recurs.
    def _slot_cos(seed):
        with frozen_rng(seed):
            with torch.no_grad():
                vals = []
                for s in range(0, len(val_ids), 8):
                    cache = [torch.load(Path(RUN_DIR) / manifest["patients"][pid]["cache_path"],
                                        map_location="cpu", weights_only=True)
                             for pid in val_ids[s:s + 8]]
                    x = torch.stack([it["wsi"] for it in cache]).to(DEVICE)
                    sl = model.slot_attention_wsi(model.wsi_mlp(x))
                    for i in range(sl.size(0)):
                        v = torch.nn.functional.normalize(sl[i], dim=-1)
                        m = (v @ v.T).cpu().numpy()
                        np.fill_diagonal(m, 0.0)
                        vals.append(float(m.max()))
        return np.asarray(vals)

    c1, c2 = _slot_cos(42), _slot_cos(42)
    e4 = {
        "residual_max_offdiag_cos_run1": float(c1.mean()),
        "residual_max_offdiag_cos_run2": float(c2.mean()),
        "absolute_drift": float(abs(c1.mean() - c2.mean())),
        "bitwise_slot_identical": bool(np.array_equal(
            c1.tobytes(), c2.tobytes())),
    }
    results["experiments"]["E4_slot_cos_replay"] = e4
    print(f"[E4] cos run1={c1.mean():.6f} run2={c2.mean():.6f} "
          f"drift={e4['absolute_drift']:.3e}", flush=True)

    out = OUT_DIR / "bit_exact_probe.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()