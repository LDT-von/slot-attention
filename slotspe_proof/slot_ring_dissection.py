"""Read-only inference-time dissection of the slot-attention collapse rings.

The historical localization (docs/research/slotspe_gate_pair_20261022) named a
four-ring self-reinforcing chain inside third_party/SlotSPE/models/slot_attention.py:

  R1  attn = dots.softmax(dim=-2)          slots compete on the SLOT axis
  R2  attn = F.normalize(attn + eps, p=1, dim=-1)  each slot's total attention
      mass across TOKENS is pinned to 1
  R3  slots = gru(...) ; slots += mlp(...) slots pulled to a shared mean
  R4  slots_mu / slots_logsigma are (1,1,dim)  every slot starts from the SAME
      distribution, so slot identity differs only by the randn noise

This module breaks each ring independently AT INFERENCE ONLY and records the
mechanism readouts (cross-slot divergence, pairwise cosine, effective rank) per
iteration.

Read-only contract
------------------
No optimizer is constructed, no parameter is updated, and no source file is
edited.  Each arm runs an ablation-parameterized replica of the native forward
that is bound to the real trained module, so all weights are the frozen
checkpoint's.  The runner asserts the trained state digest is unchanged after
every arm.

Fidelity contract
-----------------
The ``none`` arm must reproduce the recorded historical validation logits
EXACTLY at eval_seed.  The runner asserts this numerically on the 77 real
validation patients BEFORE any contrast is read; a failure aborts the run.

Scope
-----
Mechanism readout only.  No C-index is computed, nothing here is a
predictive-harm claim, and nothing here is correction-efficacy evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

REPO = Path(__file__).resolve().parents[1]

# The four rings plus the untouched reference arm.
RINGS = ("none", "r1_slot_axis_softmax", "r2_l1_normalize", "r3_gru_residual", "r4_shared_mu")

RING_DEFINITIONS = {
    "none": "untouched native forward; must reproduce the recorded historical logits",
    "r1_slot_axis_softmax": (
        "softmax over the TOKEN axis (dim=-1) instead of the slot axis (dim=-2): "
        "slots no longer compete for one shared attention mass"
    ),
    "r2_l1_normalize": (
        "drop the F.normalize(attn + eps, p=1, dim=-1) that pins each slot's total "
        "attention mass across tokens to 1; the slot-axis softmax alone is kept"
    ),
    "r3_gru_residual": (
        "keep the attention update but write it straight into the slot, dropping the "
        "GRU and the residual MLP that pull slots toward a shared mean"
    ),
    "r4_shared_mu": (
        "give every slot its own deterministic offset from the shared mu, derived from "
        "the trained slots_logsigma with no new parameter and no RNG draw, so slot "
        "identities are independently parameterized instead of sharing one (1,1,dim)"
    ),
}

REPLAY_ATOL = 1e-6


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )


# ---------------------------------------------------------------------------
# Plan: imports nothing heavy, never loads torch or a model
# ---------------------------------------------------------------------------
def plan(config):
    rings = list(config.get("dissection_rings", list(RINGS)))
    if not rings:
        raise ValueError("dissection_rings must be a nonempty list")
    if "none" not in rings:
        raise ValueError("dissection_rings must include the 'none' reference arm")
    unknown = sorted({r for r in rings if r not in RINGS})
    if unknown:
        raise ValueError("unknown dissection ring(s): " + ", ".join(unknown))
    if len(set(rings)) != len(rings):
        raise ValueError("dissection_rings must be unique")

    epochs = config.get("dissection_epochs", 1)
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("dissection_epochs must be a positive integer")

    batch_size = config.get("dissection_batch_size", 8)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("dissection_batch_size must be a positive integer")

    patients = config.get("dissection_patients", [0, 1])
    if not isinstance(patients, list) or not patients:
        raise ValueError("dissection_patients must be a nonempty list")
    indices = []
    for value in patients:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("dissection_patients entries must be nonnegative ints")
        indices.append(int(value))

    return {
        "schema": "slotspe_slot_ring_dissection_plan_v1",
        "phase": "slot_ring_dissection_inference",
        "rings": rings,
        "ring_definitions": {ring: RING_DEFINITIONS[ring] for ring in rings},
        "patient_indices": indices,
        "epochs": epochs,
        "batch_size": batch_size,
        "device_default": config.get("device", "cuda:0"),
        "primary_readout": "per_iteration_cross_slot_divergence_and_pairwise_cosine",
        "training_runs": 0,
        "optimizer_steps": 0,
        "parameters_updated": 0,
        "replay_atol": REPLAY_ATOL,
        "source_root": str(REPO / "third_party/SlotSPE"),
        "scope": (
            "Read-only inference intervention on one frozen trained checkpoint. "
            "Mechanism readout only: no C-index primary readout, no predictive-harm "
            "claim, no correction-efficacy claim."
        ),
        "default_action": "plan_only",
    }


# ---------------------------------------------------------------------------
# Ablation replica of the native forward
# ---------------------------------------------------------------------------
def _dissect_forward(module, inputs, ring, collector=None):
    """Replica of MultiHeadSlotAttention.forward with exactly one ring ablated.

    For ring == "none" this is line-for-line the native computation, which is what
    makes the fidelity replay meaningful.
    """
    import torch
    from torch import einsum
    from torch.nn import functional as F
    from einops import repeat, pack, unpack

    b = inputs.shape[0]
    device, dtype = inputs.device, inputs.dtype
    n_s = module.num_slots

    # ---- R4: shared vs per-slot slot initialization ---------------------
    if ring == "r4_shared_mu":
        sigma_per_dim = module.slots_logsigma.exp()                 # (1,1,dim)
        code = torch.linspace(-1.0, 1.0, n_s, device=device, dtype=dtype).view(1, n_s, 1)
        mu = module.slots_mu + sigma_per_dim * code                # (1,n_s,dim)
        sigma = sigma_per_dim.expand(1, n_s, module.dim).contiguous()
        mu = mu.expand(b, n_s, module.dim).contiguous()
        sigma = sigma.expand(b, n_s, module.dim).contiguous()
    else:
        mu = repeat(module.slots_mu, "1 1 d -> b s d", b=b, s=n_s)
        sigma = repeat(module.slots_logsigma.exp(), "1 1 d -> b s d", b=b, s=n_s)

    # Identical RNG consumption to the native path in every arm.
    slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)

    if collector is not None:
        collector.record(slots)

    normed_inputs = module.norm_input(inputs)
    k, v = module.to_k(normed_inputs), module.to_v(normed_inputs)
    k, v = map(module.split_heads, (k, v))

    for _ in range(module.iters):
        slots_prev = slots

        slots = module.norm_slots(slots)

        q = module.split_heads(module.to_q(slots))

        dots = einsum("... i d, ... j d -> ... i j", q, k) * module.scale

        # ---- R1: which axis competes ----------------------------------
        if ring == "r1_slot_axis_softmax":
            attn = dots.softmax(dim=-1)
        else:
            attn = dots.softmax(dim=-2)

        # ---- R2: pin each slot's token-axis mass to 1 --------------------
        if ring != "r2_l1_normalize":
            attn = F.normalize(attn + module.eps, p=1, dim=-1)

        if collector is not None:
            collector.record_attention(attn)

        updates = einsum("... j d, ... i j -> ... i d", v, attn)
        updates = module.combine_heads(module.merge_heads(updates))

        # ---- R3: GRU + residual pull toward a shared mean -------------
        if ring == "r3_gru_residual":
            slots = updates
        else:
            updates_packed, packed_shape = pack([updates], "* d")
            prev_packed, _ = pack([slots_prev], "* d")
            slots = module.gru(updates_packed, prev_packed)
            slots, = unpack(slots, packed_shape, "* d")
            slots = slots + module.mlp(module.norm_pre_ff(slots))

        if collector is not None:
            collector.record(slots)

    return slots


class IterationCollector:
    """Capture initial slots, per-iteration slots and per-iteration attention."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.slot_states = []
        self.attentions = []

    def record(self, slots):
        self.slot_states.append(slots.detach().clone())

    def record_attention(self, attention):
        self.attentions.append(attention.detach().clone())


# ---------------------------------------------------------------------------
# Mechanism readouts
# ---------------------------------------------------------------------------
def pairwise_cosine(torch, slots):
    """Mean slot-PAIR cosine per patient. slots: (B, S, D)."""
    x = slots.double()
    norms = x.norm(dim=-1, keepdim=True)
    if not bool((norms > 0).all()):
        return None
    unit = x / norms
    cosine = (unit @ unit.transpose(-2, -1)).clamp(-1, 1)
    s = slots.shape[1]
    i, j = torch.triu_indices(s, s, offset=1, device=slots.device)
    return cosine[:, i, j].mean(-1)


def share_spread(torch, attention):
    """max-min normalized read share of the same token across slots.

    attention: (B, heads, slots, tokens).  Larger = slots read differently.
    """
    rows = attention.double()
    rows = rows / rows.sum(-1, keepdim=True).clamp(min=1e-12)
    return rows.max(dim=-2).values - rows.min(dim=-2).values


def update_divergence(torch, attention):
    """Mean L1 distance between same-head slot attention distributions.

    This upper-bounds how different the per-slot value updates can be, so it is
    the quantity R1 and R2 most directly control.
    """
    x = attention.double()
    s = x.shape[-2]
    if s < 2:
        return None
    rows = x / x.sum(-1, keepdim=True).clamp(min=1e-12)
    i, j = torch.triu_indices(s, s, offset=1, device=x.device)
    return (rows[..., i, :] - rows[..., j, :]).abs().sum(-1).mean(dim=(-1, -2))


def centered_effective_rank(torch, slots):
    """Entropy effective rank of the centered slot spectrum, per patient."""
    x = slots.double()
    centred = x - x.mean(dim=-2, keepdim=True)
    spectrum = torch.linalg.svdvals(centred) ** 2
    total = spectrum.sum(-1, keepdim=True)
    p = torch.where(total > 0, spectrum / torch.where(total > 0, total, torch.ones_like(total)),
                    torch.zeros_like(spectrum))
    logp = torch.where(p > 0, p * torch.log(p), torch.zeros_like(p))
    return torch.exp(-logp.sum(-1))


def _finite_stats(torch, values):
    """Mean per batch element -> a scalar, skipping nonfinite entries."""
    if values is None:
        return None
    cleaned = torch.where(torch.isfinite(values), values, torch.full_like(values, float("nan")))
    mean = cleaned.mean()
    return None if not bool(torch.isfinite(mean)) else float(mean)


def _summarize(values):
    """Mean/median/min/max over the finite entries of one readout list."""
    import numpy as np
    array = np.asarray([v for v in values if v is not None], dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return None
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "count": int(array.size),
    }


def _delta(candidate, reference, key):
    """Mean difference between one arm's readout and the reference arm's."""
    left, right = _summarize(candidate[key]), _summarize(reference[key])
    if left is None or right is None:
        return None
    return left["mean"] - right["mean"]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(config, *, input_root, observations, checkpoint, output_dir, device):
    """Execute the dissection. Only reachable behind an explicit --run."""
    import numpy as np
    import torch
    from torch.torch_version import TorchVersion

    from slotspe_proof.model import build_model, make_args
    from slotspe_proof.native_audit import model_purity, _state_digest

    plan_value = plan(config)
    output = Path(output_dir).resolve()
    if not output.is_relative_to((REPO / "local_evidence").resolve()):
        raise ValueError("output must live inside ignored local_evidence")
    if output.exists():
        raise ValueError("a fresh output directory is required")

    observations_root = Path(observations).resolve()
    observations_json = read(observations_root / "observations.json")
    checkpoint_path = Path(checkpoint).resolve()
    input_root = Path(input_root).resolve()
    manifest_path = input_root / "prepared" / "manifest.json"
    manifest = read(manifest_path)
    source_root = Path(manifest["source_root"])

    fold = manifest["folds"]["0"]
    val_ids = list(fold["val"])

    # ---- provenance gates -------------------------------------------------
    expected_ckpt = [digest for path, digest in observations_json["source_inputs_sha256"].items()
                     if Path(path.replace("\\", "/")).name == "last.pt"]
    if len(expected_ckpt) != 1:
        raise ValueError("expected exactly one recorded checkpoint digest")
    watched = {
        str(observations_root / "observations.json"): sha(observations_root / "observations.json"),
        str(manifest_path): sha(manifest_path),
        str(checkpoint_path): expected_ckpt[0],
    }
    for path, digest in watched.items():
        if sha(path) != digest:
            raise ValueError("recorded artifact changed: " + path)

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device unavailable; no silent fallback")

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    args = make_args(manifest["omic_sizes"], encoding_dim=manifest["encoding_dim"])
    with torch.serialization.safe_globals([TorchVersion]):
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if saved["arm"] != "original" or saved["seed"] != observations_json["seed"]:
        raise ValueError("checkpoint identity differs from the recorded observation")
    if saved["optimizer_steps"] != observations_json["prior_optimizer_steps"]:
        raise ValueError("checkpoint optimizer steps differ from the recorded observation")

    model = build_model(str(source_root), args, "original").to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    del saved
    model.eval()
    purity = model_purity(model, source_root)
    before_state = _state_digest(model)

    def reset_rng(value):
        random.seed(value)
        np.random.seed(value % (2 ** 32))
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)

    def cache_of(pid):
        item = torch.load(input_root / manifest["patients"][pid]["cache_path"],
                          map_location="cpu", weights_only=True, mmap=True)
        if item["patient_id"] != pid:
            raise ValueError("prepared cache identity differs")
        return item

    def batch(ids):
        items = [cache_of(pid) for pid in ids]
        value = {
            "x_wsi": torch.stack([i["wsi"] for i in items]).to(device),
            "y": torch.tensor([fold["labels"][pid] for pid in ids], dtype=torch.long, device=device),
            "c": torch.tensor([manifest["patients"][pid]["censor"] for pid in ids],
                              dtype=torch.float32, device=device),
            "omic_missing": False,
        }
        for k in range(len(manifest["omic_sizes"])):
            value[f"x_omic{k + 1}"] = torch.stack([i["omics"][k] for i in items]).to(device)
        return value

    eval_seed = observations_json["eval_seed"]
    recorded_logits = np.asarray([row["logits"] for row in observations_json["records"]], dtype=float)

    # ---- bind the ablation replica to the real trained modules -----------
    state = {"ring": "none", "active": {}}
    originals = {}

    for modality in ("wsi", "omic"):
        module = getattr(model, "slot_attention_" + modality)
        originals[modality] = module.forward

        def make_forward(modality_name):
            def forward(self, inputs, num_slots=None):
                return _dissect_forward(
                    self, inputs, state["ring"], state["active"].get(modality_name)
                )
            return forward

        module.forward = make_forward(modality).__get__(module, type(module))

    def restore():
        for modality, original in originals.items():
            getattr(model, "slot_attention_" + modality).forward = original

    # ---- fidelity replay: 'none' must match history exactly --------------
    replay = {"tolerance": REPLAY_ATOL}
    for ring in plan_value["rings"]:
        state["ring"] = ring
        state["active"] = {}
        reset_rng(eval_seed)
        predicted = []
        with torch.no_grad():
            for start in range(0, len(val_ids), plan_value["batch_size"]):
                chunk = val_ids[start:start + plan_value["batch_size"]]
                predicted.append(model(**batch(chunk))[0].cpu())
        predicted = torch.cat(predicted).numpy().astype(float)
        finite = bool(np.isfinite(predicted).all())
        replay[ring + "_finite"] = finite
        if ring == "none":
            if not finite:
                raise RuntimeError("nonfinite native reference prediction")
            error = float(np.max(np.abs(predicted - recorded_logits)))
            replay["none_max_abs_logit_difference"] = error
            if error > REPLAY_ATOL:
                raise RuntimeError("native reference arm failed to replay: " + repr(error))
            replay["status"] = "reproduced"
        elif not finite:
            raise RuntimeError(f"ablated ring {ring} produced nonfinite predictions")

    if _state_digest(model) != before_state:
        restore()
        raise RuntimeError("dissection altered the trained weights; not read-only")

    # ---- per-ring mechanism readout --------------------------------------
    selected_ids = [val_ids[i] for i in plan_value["patient_indices"]]
    collectors = {m: IterationCollector() for m in ("wsi", "omic")}
    per_ring = {
        ring: {
            m: {
                "initial_cosine": [],
                "initial_effective_rank": [],
                "final_cosine": [],
                "final_effective_rank": [],
                "share_spread": [],
                "update_divergence": [],
                "cosine_by_iteration": [],
                "share_spread_by_iteration": [],
                "update_divergence_by_iteration": [],
            }
            for m in ("wsi", "omic")
        }
        for ring in plan_value["rings"]
    }

    for ring in plan_value["rings"]:
        state["ring"] = ring
        state["active"] = collectors
        for start in range(0, len(selected_ids), plan_value["batch_size"]):
            chunk = selected_ids[start:start + plan_value["batch_size"]]
            for collector in collectors.values():
                collector.reset()
            reset_rng(eval_seed + start)
            with torch.no_grad():
                model(**batch(chunk))
            for modality in ("wsi", "omic"):
                collector = collectors[modality]
                bucket = per_ring[ring][modality]
                # slot_states[0] is the pre-iteration initialization and is
                # recorded SEPARATELY: averaging it together with the ten
                # post-iteration states would blend a well-separated init into
                # the collapse readout and make the two incomparable.
                states = collector.slot_states
                bucket["initial_cosine"].append(
                    _finite_stats(torch, pairwise_cosine(torch, states[0]))
                )
                bucket["initial_effective_rank"].append(
                    _finite_stats(torch, centered_effective_rank(torch, states[0]))
                )
                for iteration, slots in enumerate(states[1:], start=1):
                    bucket["final_cosine"].append(_finite_stats(torch, pairwise_cosine(torch, slots)))
                    bucket["final_effective_rank"].append(
                        _finite_stats(torch, centered_effective_rank(torch, slots))
                    )
                    # per-iteration curve, indexed by iteration 1..iters
                    bucket["cosine_by_iteration"].append(
                        (iteration, _finite_stats(torch, pairwise_cosine(torch, slots)))
                    )
                for iteration, attention in enumerate(collector.attentions, start=1):
                    spread = _finite_stats(torch, share_spread(torch, attention))
                    divergence = _finite_stats(torch, update_divergence(torch, attention))
                    bucket["share_spread"].append(spread)
                    bucket["update_divergence"].append(divergence)
                    bucket["share_spread_by_iteration"].append((iteration, spread))
                    bucket["update_divergence_by_iteration"].append((iteration, divergence))
    state["active"] = {}
    restore()

    if _state_digest(model) != before_state:
        raise RuntimeError("trained weights changed during the readout pass")

    def summarize_curve(pairs):
        """Per-iteration mean, sorted by iteration index."""
        rows = [(int(i), float(v)) for i, v in pairs if v is not None and np.isfinite(v)]
        if not rows:
            return None
        rows.sort()
        return {str(i): v for i, v in rows}

    curve_keys = ("cosine_by_iteration", "share_spread_by_iteration", "update_divergence_by_iteration")
    arms = []
    for ring in plan_value["rings"]:
        arm = {"ring": ring, "definition": RING_DEFINITIONS[ring]}
        for modality in ("wsi", "omic"):
            bucket = per_ring[ring][modality]
            arm[modality] = {
                key: (summarize_curve(values) if key in curve_keys else _summarize(values))
                for key, values in bucket.items()
            }
        arms.append(arm)

    contrasts = []
    reference = per_ring["none"]
    for ring in plan_value["rings"]:
        if ring == "none":
            continue
        contrasts.append({
            "ring": ring,
            "definition": RING_DEFINITIONS[ring],
            "wsi": {
                key: _delta(per_ring[ring]["wsi"], reference["wsi"], key)
                for key in ("initial_cosine", "final_cosine", "final_effective_rank",
                            "share_spread", "update_divergence")
            },
            "omic": {
                key: _delta(per_ring[ring]["omic"], reference["omic"], key)
                for key in ("initial_cosine", "final_cosine", "final_effective_rank",
                            "share_spread", "update_divergence")
            },
            "reading": (
                "Negative cosine difference and positive share-spread difference mean "
                "this ring was part of what collapsed the slots toward one direction."
            ),
        })

    output.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    finished = datetime.now(timezone.utc).isoformat()

    protocol = {
        "schema": "slotspe_slot_ring_dissection_v1",
        "status": "measured",
        "plan": plan_value,
        "replay": replay,
        "state_unchanged": True,
        "source_hashes": purity["source_sha256"],
        "interpretation": (
            "Each ring ablated independently at inference on one frozen checkpoint. "
            "These are mechanism readouts on real patient slots. They locate where the "
            "collapse is generated; they are not predictive-harm or correction-efficacy "
            "evidence and imply no patient outcome."
        ),
    }
    write(output / "protocol.json", protocol)
    write(output / "arms.json", {
        "schema": "slotspe_slot_ring_dissection_arms_v1",
        "arms": arms,
        "contrasts_vs_none": contrasts,
    })
    write(output / "execution_receipt.json", {
        "status": "completed",
        "phase": "slot_ring_dissection_inference",
        "prepared_input_root": str(input_root),
        "prepared_manifest_sha256": watched[str(manifest_path)],
        "observations_sha256": watched[str(observations_root / "observations.json")],
        "checkpoint_sha256": expected_ckpt[0],
        "runner_sha256": sha(__file__),
        "source_root": str(REPO / "third_party/SlotSPE"),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        "device": device,
        "replay": replay,
        "state_unchanged": True,
        "training_runs": 0,
        "optimizer_steps": 0,
        "parameters_updated": 0,
        "started_utc": started,
        "finished_utc": finished,
        "patient_outputs_private": True,
    })
    return {"protocol": protocol, "arms": arms, "contrasts": contrasts, "replay": replay}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only slot collapse ring dissection.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input-root", type=Path, help="prepared run root containing manifest.json")
    parser.add_argument("--observations", type=Path, help="retained P1 observation root")
    parser.add_argument("--checkpoint", type=Path, help="trained original last.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)

    if args.plan and args.run:
        parser.error("--plan and --run are mutually exclusive")
    if args.plan:
        if args.config is None:
            parser.error("--plan requires --config")
        print(json.dumps(plan(read(args.config)), ensure_ascii=False, indent=2))
        return 0
    if not args.run:
        parser.error("only --plan (no execution) or --run (execution) is accepted")
    for name, value in (("--config", args.config), ("--input-root", args.input_root),
                        ("--observations", args.observations), ("--checkpoint", args.checkpoint),
                        ("--output-dir", args.output_dir)):
        if value is None:
            parser.error("--run requires " + name)

    run(read(args.config), input_root=args.input_root, observations=args.observations,
        checkpoint=args.checkpoint, output_dir=args.output_dir, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())