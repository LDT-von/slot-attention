"""Disable inference-time sparse selection on exact retained native slots.

Only --run executes. No optimizer, feature encoding or new slot generation is
performed. Native classes/forwards and source files remain unchanged. All raw
inputs and patient predictions stay in ignored local_evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
ARMS = ("native_top2", "dense_wsi", "dense_omic", "dense_both", "uniform_both")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_bytes((json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def summary(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    if not a.size or not np.isfinite(a).all():
        raise ValueError("Metric values must be finite and nonempty")
    return {"mean": float(a.mean()), "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def mean_pair_cosine(slots):
    import torch
    norms = slots.double().norm(dim=-1, keepdim=True)
    if not bool((norms > 0).all()):
        raise ValueError("Zero slot cosine undefined")
    u = slots.double() / norms
    cosine = (u @ u.transpose(-2, -1)).clamp(-1, 1)
    i, j = torch.triu_indices(slots.shape[1], slots.shape[1], offset=1, device=slots.device)
    return cosine[:, i, j].mean(-1)


def original_decoder_from_mask(decoder, slots, keep):
    import torch
    mapped = decoder.map(slots)
    logits = decoder.decoder(mapped)
    scores = decoder.pred_keep_slot(mapped).squeeze(-1)
    gate = torch.softmax(scores / decoder.temperature, dim=-1) * keep
    gate = gate / (gate.sum(-1, keepdim=True) + 1e-8)
    return torch.einsum("bs,bsc->bc", gate, logits), gate, scores, logits


def native_dense_decoder(decoder, slots, seed):
    """Run the actual native decoder at K=S; restore k and every RNG afterward."""
    import torch
    previous = decoder.k
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        decoder.k = slots.shape[1]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            logits, gate, keep = decoder(slots)
        if not torch.allclose(keep, torch.ones_like(keep), atol=1e-6, rtol=0):
            raise RuntimeError("Native K=S did not retain every candidate")
        return logits, gate, keep
    finally:
        decoder.k = previous


def readout(model, wsi, omic, keep_w, keep_o, unmasked=False):
    import torch
    cross = model.cross_attention(wsi, omic)
    wi = model.self_attention_wsi(wsi, mask=None if unmasked else keep_w.bool())
    oi = model.self_attention_omic(omic, mask=None if unmasked else keep_o.bool())
    logits = model.to_logits(torch.cat([cross.mean(1), wi.mean(1), oi.mean(1)], dim=-1))
    return logits, cross, wi, oi


def risk(logits):
    import numpy as np
    x = np.asarray(logits, dtype=float)
    return -np.exp(-np.logaddexp(0, x)).cumprod(axis=-1).sum(axis=-1)


def run(observations_root, checkpoint, output, device="cuda:0", batch_size=8):
    import numpy as np
    import torch
    from torch.torch_version import TorchVersion
    from sksurv.metrics import concordance_index_censored
    from .model import build_model
    from .native_audit import model_purity, _state_digest
    from .p3_diagnostics import rng_state, rng_digest

    root, checkpoint, output = Path(observations_root).resolve(), Path(checkpoint).resolve(), Path(output).resolve()
    if not output.is_relative_to((REPO / "local_evidence").resolve()) or output.exists():
        raise ValueError("Fresh private local_evidence output required")
    o = read(root / "observations.json"); receipt = read(root / "receipt.json")
    provenance = read(root / "patient_provenance.json")
    watched = {str(root / name): digest for name, digest in receipt["outputs"].items()}
    watched[str(root / "receipt.json")] = sha(root / "receipt.json")
    expected = [digest for path, digest in o["source_inputs_sha256"].items() if Path(path.replace("\\", "/")).name == "last.pt"]
    if len(expected) != 1 or sha(checkpoint) != expected[0]:
        raise ValueError("Checkpoint does not match retained native slots")
    watched[str(checkpoint)] = expected[0]
    source_root = REPO / "third_party/SlotSPE"
    for relative, digest in o["model_purity"]["source_sha256"].items():
        watched[str(source_root / relative)] = digest
    for path, digest in watched.items():
        if sha(path) != digest:
            raise ValueError("Historical raw/source content changed")
    if [p["patient_id"] for p in provenance["patients"]] != [p["patient_id"] for p in o["records"]]:
        raise ValueError("Outcome and retained-slot patient order disagree")
    if len({r["patient_id"] for r in o["records"]}) != len(o["records"]):
        raise ValueError("Duplicate patient observations")
    with torch.serialization.safe_globals([TorchVersion]):
        saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if saved["arm"] != "original" or saved["seed"] != o["seed"] or saved["optimizer_steps"] != o["prior_optimizer_steps"]:
        raise ValueError("Historical training recipe mismatch")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no automatic fallback")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = build_model(str(source_root), SimpleNamespace(**o["model_purity"]["model_args"]), "original").to(device)
    model.load_state_dict(saved["state_dict"], strict=True); del saved
    model.eval(); purity = model_purity(model, source_root)
    before = _state_digest(model); before_rng = rng_digest(rng_state())
    output.mkdir(parents=True)
    protocol = {"schema": "slotspe_sparsity_off_v1", "arms": list(ARMS),
                "baseline": "Exact retained native final slots, recorded Top2 masks and original weights",
                "dense": "Actual original decoder at K=S=8; learned soft weights retained; all-true self masks",
                "uniform": "K=S plus uniform 1/S single-modal weights; identical fused self masks to dense_both",
                "one_modality_controls": True, "other_branches": "Original dense cross-attention unchanged",
                "training_steps": 0, "new_slot_generation": 0,
                "scope": "Frozen-checkpoint inference intervention, not a no-gating training ablation",
                "replay_atol": 1e-6, "batch_size": batch_size, "device": device,
                "fused_and_auxiliary_predictions_reported_separately": True}
    write(output / "protocol.json", protocol)
    status = {"status": "started", "protocol": protocol, "checkpoint_sha256": expected[0],
              "source_hashes": purity["source_sha256"], "runner_sha256": sha(__file__),
              "observations_sha256": sha(root / "observations.json"), "torch": str(torch.__version__),
              "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None}
    write(output / "execution_receipt.json", status)
    predictions = {arm: {"fused": [], "aux_wsi": [], "aux_omic": []} for arm in ARMS}
    slot_cos = {arm: {"wsi": [], "omic": []} for arm in ARMS}
    gate_effective = {arm: {"wsi": [], "omic": []} for arm in ARMS}
    post_self_cos = {arm: {"wsi": [], "omic": []} for arm in ARMS}
    baseline_errors = []; original_gate_errors = []; original_score_errors = []
    identity_errors = []; unmasked_errors = []; cross_errors = []; dense_repeat_errors = []
    try:
        for start in range(0, len(o["records"]), batch_size):
            rows = o["records"][start:start + batch_size]
            arrays = []
            for row in rows:
                path = root / row["raw_path"]
                if sha(path) != row["raw_sha256"]:
                    raise ValueError("Raw slot hash changed")
                with np.load(path, allow_pickle=False) as z:
                    arrays.append({key: z[key] for key in z.files})
            ts = {m + suffix: torch.tensor(np.stack([a[m + suffix] for a in arrays]), device=device)
                  for m in ("wsi", "omic") for suffix in ("_slots", "_keep", "_scores", "_gate")}
            with torch.no_grad():
                original = {}; dense = {}; uniform = {}
                for modality in ("wsi", "omic"):
                    slots, keep = ts[modality + "_slots"], ts[modality + "_keep"]
                    if slots.shape[1] != 8 or not bool(((keep > .5).sum(1) == 2).all()):
                        raise ValueError("Expected exactly eight candidates and two retained slots")
                    decoder = getattr(model, "slot_decoder_" + modality)
                    logit, gate, scores, slot_logits = original_decoder_from_mask(decoder, slots, keep)
                    original_gate_errors.append(float((gate - ts[modality + "_gate"]).abs().max()))
                    original_score_errors.append(float((scores - ts[modality + "_scores"]).abs().max()))
                    original[modality] = (logit, gate, keep)
                    dense[modality] = native_dense_decoder(decoder, slots, 901000 + start)
                    repeat = native_dense_decoder(decoder, slots, 902000 + start)
                    dense_repeat_errors.append(float((dense[modality][0] - repeat[0]).abs().max()))
                    uniform[modality] = (slot_logits.mean(1), torch.full_like(gate, 1 / slots.shape[1]), torch.ones_like(keep))
                baseline_parts = None
                for arm in ARMS:
                    values = {m: uniform[m] if arm == "uniform_both" else dense[m] if arm in ("dense_" + m, "dense_both") else original[m] for m in ("wsi", "omic")}
                    logits, cross, wi, oi = readout(model, ts["wsi_slots"], ts["omic_slots"], values["wsi"][2], values["omic"][2])
                    if arm == "native_top2":
                        baseline_parts = (logits, cross)
                        expected_logits = torch.tensor([row["logits"] for row in rows], device=device)
                        error = float((logits - expected_logits).abs().max()); baseline_errors.append(error)
                        if error > 1e-6:
                            raise RuntimeError("Native fused readout does not reproduce retained baseline")
                    else:
                        cross_errors.append(float((cross - baseline_parts[1]).abs().max()))
                    predictions[arm]["fused"].append(logits.cpu().numpy())
                    for m, post in (("wsi", wi), ("omic", oi)):
                        predictions[arm]["aux_" + m].append(values[m][0].cpu().numpy())
                        weights = values[m][1].double(); weights = weights / weights.sum(1, keepdim=True)
                        gate_effective[arm][m].extend((1 / weights.square().sum(1)).cpu().tolist())
                        selected_slots = torch.stack([ts[m + "_slots"][i, values[m][2][i] > .5] for i in range(len(rows))])
                        slot_cos[arm][m].extend(mean_pair_cosine(selected_slots).cpu().tolist())
                        post_self_cos[arm][m].extend(mean_pair_cosine(post).cpu().tolist())
                again = readout(model, ts["wsi_slots"], ts["omic_slots"], original["wsi"][2], original["omic"][2])[0]
                identity_errors.append(float((again - baseline_parts[0]).abs().max()))
                full = readout(model, ts["wsi_slots"], ts["omic_slots"], dense["wsi"][2], dense["omic"][2], unmasked=True)[0]
                unmasked_errors.append(float((full.cpu() - torch.from_numpy(predictions["dense_both"]["fused"][-1])).abs().max()))
            print(json.dumps({"completed_patients": min(start + batch_size, len(o["records"])), "total_patients": len(o["records"])}), flush=True)
        if max(original_gate_errors + original_score_errors + identity_errors + unmasked_errors + cross_errors + dense_repeat_errors) > 1e-6:
            raise RuntimeError("Native score/gate/control replay tolerance failed")
        arrays = {arm: {key: np.concatenate(value) for key, value in heads.items()} for arm, heads in predictions.items()}
        if not np.array_equal(arrays["dense_both"]["fused"], arrays["uniform_both"]["fused"]):
            raise RuntimeError("Dense versus uniform changed fused predictions unexpectedly")
        for heads in arrays.values():
            if not all(np.isfinite(v).all() for v in heads.values()):
                raise RuntimeError("Nonfinite prediction")
        y = torch.tensor([p["label"] for p in provenance["patients"]], device=device)
        c = torch.tensor([p["censor"] for p in provenance["patients"]], device=device)
        event = np.asarray([p["censor"] < 1 for p in provenance["patients"]], dtype=bool)
        times = np.asarray([p["time"] for p in provenance["patients"]], dtype=float)
        def metric(logits):
            value = concordance_index_censored(event, times, risk(logits))
            nll = model.loss_fn(torch.tensor(logits, device=device), y=y, c=c, t=None) / len(y)
            return {"cindex": float(value[0]), "concordant": int(value[1]), "discordant": int(value[2]), "tied_risk": int(value[3]), "tied_time": int(value[4]), "native_nll_per_patient": float(nll)}
        result = {"task_id": "SPE-08", "protocol": protocol, "patients": len(y), "fold": o["fold"], "seed": o["seed"],
                  "historical_training_epochs": o["epochs"], "historical_optimizer_steps": o["prior_optimizer_steps"],
                  "new_training_steps": 0, "new_slot_generation": 0, "new_full_model_forward_calls": 0,
                  "new_native_downstream_readout_sweeps": 7, "checkpoint_sha256": expected[0],
                  "controls_max_abs_errors": {"historical_logits": max(baseline_errors), "original_gate": max(original_gate_errors), "original_scores": max(original_score_errors), "identical_input": max(identity_errors), "dense_all_true_vs_none_mask": max(unmasked_errors), "cross_branch": max(cross_errors), "dense_decoder_different_rng": max(dense_repeat_errors)},
                  "slot_vectors_before_gate_are_identical_by_design": True, "arms": {}}
        for arm, heads in arrays.items():
            record = {"heads": {key: metric(logits) for key, logits in heads.items()},
                      "activated_slot_cosine": {m: summary(slot_cos[arm][m]) for m in ("wsi", "omic")},
                      "gate_weight_effective_slots": {m: summary(gate_effective[arm][m]) for m in ("wsi", "omic")},
                      "all_eight_post_self_slot_cosine": {m: summary(post_self_cos[arm][m]) for m in ("wsi", "omic")}, "differences_vs_native": {}}
            for key, logits in heads.items():
                base = arrays["native_top2"][key]
                record["differences_vs_native"][key] = {"per_patient_max_abs_logit": summary(np.abs(logits - base).max(1)), "per_patient_abs_risk": summary(np.abs(risk(logits) - risk(base))), "delta_cindex": record["heads"][key]["cindex"] - metric(base)["cindex"]}
            result["arms"][arm] = record
        if _state_digest(model) != before or rng_digest(rng_state()) != before_rng:
            raise RuntimeError("Weights or external RNG changed")
        model_purity(model, source_root)
        for path, digest in watched.items():
            if sha(path) != digest:
                raise RuntimeError("Historical source or inputs changed during execution")
        result["conclusion_boundary"] = ["Frozen weights and exact same retained slots; effects concern reading/prediction only.", "Does not estimate how training without sparse selection changes learned slots.", "One 3-epoch local checkpoint, 71 BLCA patients, one retained draw; no independent test or efficacy claim.", "Cross branch remains dense as designed; dense learned versus uniform gates only differ in auxiliary modal predictions."]
        write(output / "aggregate_summary.json", result)
        np.savez_compressed(output / "predictions.npz", **{arm + "__" + head: x for arm, heads in arrays.items() for head, x in heads.items()})
        write(output / "patient_order.json", {"patients": provenance["patients"]})
        status.update(status="completed", model_state_unchanged=True, original_inputs_unchanged=True, external_rng_unchanged=True, finished_utc=datetime.now(timezone.utc).isoformat())
        write(output / "execution_receipt.json", status)
        return result
    except BaseException as exc:
        status.update(status="failed_retained", error=type(exc).__name__ + ": " + str(exc))
        write(output / "execution_receipt.json", status)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--run", action="store_true")
    args = p.parse_args(argv)
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    if not args.run:
        print(json.dumps({"action": "plan_only", "arms": ARMS, "training_steps": 0, "new_slot_generation": 0, "counterfactual": "Disable Top-K at inference on retained native final slots"}, indent=2))
        return 0
    if args.output_dir is None:
        p.error("--run requires a fresh --output-dir inside ignored local_evidence")
    result = run(args.observations_root, args.checkpoint, args.output_dir, args.device, args.batch_size)
    print(json.dumps({"patients": result["patients"], "controls": result["controls_max_abs_errors"], "arms": {arm: {"fused": record["heads"]["fused"], "delta_fused_cindex": record["differences_vs_native"]["fused"]["delta_cindex"]} for arm, record in result["arms"].items()}}, indent=2))
    return 0
