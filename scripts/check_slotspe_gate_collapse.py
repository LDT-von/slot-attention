"""Read-only native gate audit on hash-verified retained patient slots.

Default is plan-only. --run authorizes frozen decoder forwards, never training.
Patient rows are written only to a new directory inside ignored local_evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def distribution(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    if not a.size or not np.isfinite(a).all():
        raise ValueError("Expected nonempty finite observations")
    return {"n": int(a.size), "median": float(np.median(a)),
            "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
            "below_0_5_count": int((a < .5).sum()),
            "below_0_5_fraction": float((a < .5).mean()),
            "below_0_01_count": int((a < .01).sum()),
            "below_0_01_fraction": float((a < .01).mean())}


def run(args):
    import numpy as np
    import torch
    from torch.torch_version import TorchVersion
    from survot_rank.research.slotspe_proof.model import build_model, load_source
    from survot_rank.research.slotspe_proof.native_audit import model_purity, _state_digest

    root, checkpoint, out = (Path(p).resolve() for p in
                             (args.input_root, args.checkpoint, args.output_dir))
    if not out.is_relative_to((REPO / "local_evidence").resolve()) or out.exists():
        raise ValueError("A fresh output directory in ignored local_evidence is required")
    if args.repeats < 8:
        raise ValueError("At least eight repeats required")
    source_root = REPO / "third_party/SlotSPE"
    official = read(args.source_verification)
    comparison = official.get("comparison", {})
    if not (comparison.get("runtime_difference_count") == 0 and
            comparison.get("runtime_file_equal_count") == comparison.get("runtime_file_count") == 10 and
            official.get("current_head_matches_pinned_commit")):
        raise ValueError("Current official runtime-source comparison must pass first")
    o, receipt = read(root / "observations.json"), read(root / "receipt.json")
    expected = [v for p, v in o["source_inputs_sha256"].items()
                if Path(p.replace("\\", "/")).name == "last.pt"]
    if len(expected) != 1 or sha(checkpoint) != expected[0]:
        raise ValueError("Checkpoint hash differs from retained slot provenance")
    watched = {checkpoint: expected[0], root / "receipt.json": sha(root / "receipt.json")}
    for rel, digest in receipt["outputs"].items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Receipt path escapes input root")
        watched[path] = digest
    for rel, digest in o["model_purity"]["source_sha256"].items():
        watched[source_root / rel] = digest
    for p, digest in watched.items():
        if sha(p) != digest:
            raise ValueError("Historical input or source hash mismatch: " + str(p))
    if len(o["records"]) != o["patient_count"] or len({r["patient_id"] for r in o["records"]}) != o["patient_count"]:
        raise ValueError("Patient count or uniqueness mismatch")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with torch.serialization.safe_globals([TorchVersion]):
        saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if saved["arm"] != "original" or saved["seed"] != o["seed"] or saved["optimizer_steps"] != o["prior_optimizer_steps"]:
        raise ValueError("Retained checkpoint training recipe mismatch")
    devices = list(range(torch.cuda.device_count())) if args.device.startswith("cuda") else []
    native_calls = 0
    rows = [{"patient_id": r["patient_id"], "input_raw_sha256": r["raw_sha256"],
             "modalities": {m: {"repeats": []} for m in ("wsi", "omic")}} for r in o["records"]]
    errors = Counter()
    source = load_source(source_root)
    sampler_module = source.modules["models.slot_attention"]
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        model = build_model(str(source_root), SimpleNamespace(**o["model_purity"]["model_args"]), "original").to(args.device)
        model.load_state_dict(saved["state_dict"], strict=True)
        del saved
        model.eval()
        purity = model_purity(model, source_root)
        before = _state_digest(model)
        for start in range(0, len(rows), args.batch_size):
            originals = o["records"][start:start + args.batch_size]
            arrays = []
            for record in originals:
                path = (root / record["raw_path"]).resolve()
                if not path.is_relative_to(root) or sha(path) != record["raw_sha256"]:
                    raise ValueError("Retained patient arrays differ")
                watched[path] = record["raw_sha256"]
                with np.load(path, allow_pickle=False) as z:
                    arrays.append({m + suffix: z[m + suffix].copy() for m in ("wsi", "omic")
                                   for suffix in ("_slots", "_scores", "_gate", "_keep")})
            for mi, modality in enumerate(("wsi", "omic")):
                decoder = getattr(model, "slot_decoder_" + modality)
                if decoder.temperature != .01 or decoder.k != 2 or decoder.num_slots != 8 or decoder.top_k_method != "parallel_topk_st":
                    raise ValueError("Expected original eight-slot Top2 temperature=.01 decoder")
                ts = {s: torch.as_tensor(np.stack([a[modality + s] for a in arrays]), device=args.device)
                      for s in ("_slots", "_scores", "_gate", "_keep")}
                if not all(bool(torch.isfinite(v).all()) for v in ts.values()):
                    raise ValueError("Nonfinite retained patient inputs")
                slots = ts["_slots"]
                scores = decoder.pred_keep_slot(decoder.map(slots)).squeeze(-1)
                errors["historical_score_max_abs"] = max(errors["historical_score_max_abs"], float((scores - ts["_scores"]).abs().max()))
                deterministic = scores.topk(decoder.k, dim=-1).indices

                def measure(gate, keep):
                    mask = keep > .5
                    if not bool((mask.sum(-1) == decoder.k).all()) or not bool(torch.isfinite(gate).all()):
                        raise ValueError("Invalid native hard mask or gate")
                    errors["hard_mask_max_deviation"] = max(errors["hard_mask_max_deviation"], float((keep - mask.to(keep.dtype)).abs().max()))
                    p32 = torch.softmax(scores / .01, -1)
                    mass = (p32 * keep).sum(-1)
                    direct = p32 * keep / (mass.unsqueeze(-1) + 1e-8)
                    errors["native_formula_max_abs"] = max(errors["native_formula_max_abs"], float((gate - direct).abs().max()))
                    effective = scores.double() / .01
                    logmass = torch.logsumexp(effective.masked_fill(~mask, -torch.inf), -1) - torch.logsumexp(effective, -1)
                    raw_logmass = torch.logsumexp(scores.double().masked_fill(~mask, -torch.inf), -1) - torch.logsumexp(scores.double(), -1)
                    overlap = mask.gather(1, deterministic).sum(-1)
                    stable_gate = torch.softmax((scores / .01).masked_fill(~mask, -torch.inf), -1)
                    result = []
                    for i in range(len(scores)):
                        result.append({"gate_sum": float(gate[i].sum()), "keep_count": int(mask[i].sum()),
                            "selected_indices": mask[i].nonzero().flatten().tolist(),
                            "selected_softmax_mass_tau_0_01_float32": float(mass[i]),
                            "selected_softmax_mass_tau_0_01_float64": float(logmass[i].exp()),
                            "log_selected_softmax_mass_tau_0_01_float64": float(logmass[i]),
                            "selected_softmax_mass_raw_tau_1_float64": float(raw_logmass[i].exp()),
                            "deterministic_top2_overlap": int(overlap[i]),
                            "top1_retained": bool(mask[i, scores[i].argmax()]),
                            "masked_softmax_same_mask_gate_sum": float(stable_gate[i].sum()),
                            "gate_weights": gate[i].tolist(), "hard_keep": keep[i].tolist()})
                    return result

                historical = measure(ts["_gate"], ts["_keep"])
                # Historical outputs replayed with the exact saved hard/ST masks.
                historical_formula = torch.softmax(scores / .01, -1) * ts["_keep"]
                historical_formula /= historical_formula.sum(-1, keepdim=True) + 1e-8
                errors["historical_gate_max_abs"] = max(errors["historical_gate_max_abs"], float((historical_formula - ts["_gate"]).abs().max()))
                for i, h in enumerate(historical):
                    rows[start + i]["modalities"][modality].update({"scores": scores[i].tolist(),
                        "deterministic_top2_indices": deterministic[i].tolist(), "historical_saved_draw": h})
                for repeat in range(args.repeats):
                    seed = args.base_seed + 10000 * repeat + 1000 * mi + start
                    torch.manual_seed(seed)
                    prediction, gate, keep = decoder(slots)
                    native_calls += 1
                    metrics = measure(gate, keep)
                    # Only temperature changes; the identical RNG regenerates the same noise.
                    try:
                        decoder.temperature = 1.
                        torch.manual_seed(seed)
                        _, warm_gate, warm_keep = decoder(slots)
                        native_calls += 1
                    finally:
                        decoder.temperature = .01
                    if not torch.equal(keep > .5, warm_keep > .5):
                        raise RuntimeError("Temperature control changed the hard selection")
                    # Scoped zero-noise control calls the unchanged original forward.
                    # Python module binding is restored before the next native call.
                    with patch.object(sampler_module, "gumbel_noise", lambda x: torch.zeros_like(x)):
                        _, no_noise_gate, no_noise_keep = decoder(slots)
                        native_calls += 1
                    if not torch.equal(torch.sort((no_noise_keep > .5).nonzero()[:, 1].reshape(len(scores), -1), -1).values,
                                       deterministic.sort(-1).values):
                        raise RuntimeError("No-noise selection differs from deterministic score Top2")
                    for i, item in enumerate(metrics):
                        item.update({"repeat": repeat, "seed": seed,
                            "temperature_1_same_noise_gate_sum": float(warm_gate[i].sum()),
                            "no_noise_temperature_0_01_gate_sum": float(no_noise_gate[i].sum())})
                        rows[start + i]["modalities"][modality]["repeats"].append(item)
                    if repeat == 0:
                        torch.manual_seed(seed)
                        p2, g2, k2 = decoder(slots)
                        native_calls += 1
                        errors["same_seed_replay_max_abs"] = max(errors["same_seed_replay_max_abs"],
                            float((p2 - prediction).abs().max()), float((g2 - gate).abs().max()), float((k2 - keep).abs().max()))
        after = _state_digest(model)
        if before != after or any(getattr(model, "slot_decoder_" + m).temperature != .01 for m in ("wsi", "omic")):
            raise RuntimeError("Model state or original temperatures changed")
    for path, digest in watched.items():
        if sha(path) != digest:
            raise RuntimeError("Read-only input changed: " + str(path))
    if max(errors.values(), default=0) > 1e-6:
        raise RuntimeError("Native replay or numeric checks exceed 1e-6: " + str(errors))
    summaries = {}
    for modality in ("wsi", "omic"):
        patient_rows = [r["modalities"][modality] for r in rows]
        for patient in patient_rows:
            reps = patient["repeats"]
            patient["unique_retained_sets"] = len({tuple(r["selected_indices"]) for r in reps})
            patient["gate_sum_across_rng"] = distribution([r["gate_sum"] for r in reps])
            patient["slot_selection_frequency"] = [sum(i in r["selected_indices"] for r in reps) / len(reps) for i in range(8)]
        first = [p["repeats"][0] for p in patient_rows]
        pooled = [r for p in patient_rows for r in p["repeats"]]
        collapsed = [r for r in pooled if r["gate_sum"] < .01]
        summaries[modality] = {
            "primary_first_prespecified_rng_draw_unique_patients": distribution([r["gate_sum"] for r in first]),
            "historical_saved_draw_unique_patients": distribution([p["historical_saved_draw"]["gate_sum"] for p in patient_rows]),
            "all_rng_patient_draws_not_independent_patients": distribution([r["gate_sum"] for r in pooled]),
            "unique_patients_ever_below_0_5": sum(any(r["gate_sum"] < .5 for r in p["repeats"]) for p in patient_rows),
            "unique_patients_ever_below_0_01": sum(any(r["gate_sum"] < .01 for r in p["repeats"]) for p in patient_rows),
            "zero_gate_sum_draws": sum(r["gate_sum"] == 0 for r in pooled),
            "nonfinite_gate_draws": 0,
            "selected_temperature_mass_all_draws": distribution([r["selected_softmax_mass_tau_0_01_float64"] for r in pooled]),
            "selected_raw_score_mass_all_draws": distribution([r["selected_softmax_mass_raw_tau_1_float64"] for r in pooled]),
            "collapsed_draws_temperature_mass": distribution([r["selected_softmax_mass_tau_0_01_float64"] for r in collapsed]) if collapsed else None,
            "top2_overlap_histogram": dict(sorted(Counter(str(r["deterministic_top2_overlap"]) for r in pooled).items())),
            "top2_overlap_mean": float(np.mean([r["deterministic_top2_overlap"] for r in pooled])),
            "top1_not_retained_fraction": float(np.mean([not r["top1_retained"] for r in pooled])),
            "unique_retained_sets_across_rng": distribution([p["unique_retained_sets"] for p in patient_rows]),
            "unique_retained_sets_histogram": dict(sorted(Counter(str(p["unique_retained_sets"]) for p in patient_rows).items())),
            "controls_all_draws": {key: distribution([r[key] for r in pooled]) for key in
                ("temperature_1_same_noise_gate_sum", "no_noise_temperature_0_01_gate_sum", "masked_softmax_same_mask_gate_sum")}}
    result = {"schema": "slotspe_readonly_gate_collapse_v1", "status": "completed",
        "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "source_verification": official, "source_verification_receipt_sha256": sha(args.source_verification),
        "protocol": {"cohort": "BLCA historical local validation", "patients": len(rows),
            "fold": o["fold"], "training_seed": o["seed"], "training_epochs": o["epochs"],
            "historical_optimizer_steps": o["prior_optimizer_steps"], "real_patches_per_patient": o["patches"],
            "retained_slot_generation_eval_seed": o["eval_seed"], "checkpoint_sha256": expected[0],
            "observations_sha256": sha(root / "observations.json"), "retained_receipt_sha256": sha(root / "receipt.json"),
            "slots_per_modality": 8, "k": 2, "temperature": .01, "epsilon": 1e-8,
            "repeats_per_patient": args.repeats, "base_seed": args.base_seed,
            "seed_rule": "base_seed + 10000*repeat + 1000*modality_index + batch_start_index",
            "batch_size": args.batch_size, "device": args.device, "dtype": "float32 native forward; float64 logsumexp mass audit",
            "torch": str(torch.__version__), "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(args.device) if devices else None,
            "fixed": "checkpoint and exact retained final patient slots; only decoder RNG varies",
            "primary_unit": "one unique patient at repeat=0; all 20 draws reported separately",
            "gate_sum_definition": "sum of returned slot_gate AFTER epsilon normalization, not hard mask cardinality",
            "temperature_mass_definition": "sum(exp(score/.01)[keep])/sum(exp(score/.01)); stable float64 logsumexp also reported",
            "raw_mass_definition": "sum(exp(raw_score)[keep])/sum(exp(raw_score)); distinct from temperature-scaled gate mass",
            "controls": "same RNG temporary temperature=1; scoped zero Gumbel noise with native forward; same-mask masked-softmax arithmetic",
            "source_api": {"decoder": "third_party/SlotSPE/models/SlotSPE.py:11-70 MoESlotDecoder.forward keep_score/slot_gate/hard_keep",
                           "sampler": "third_party/SlotSPE/models/slot_attention.py:164-185 parallel_topk_st",
                           "fusion": "third_party/SlotSPE/models/SlotSPE.py:287-301 discards slot_gate; retains Boolean hard mask"}},
        "summary": summaries, "per_patient": rows,
        "checks": {"error_maxima": dict(errors), "verified_unchanged_input_files": len(watched),
            "native_model_purity": purity["checks"], "native_source_sha256": purity["source_sha256"],
            "model_state_sha256_before": before, "model_state_sha256_after": after,
            "model_sources_weights_inputs_unchanged": True, "control_settings_restored": True,
            "new_training_runs": 0, "optimizer_steps": 0, "new_slot_generation": 0,
            "new_full_model_forward_passes": 0, "native_decoder_batch_forward_passes": native_calls,
            "server_execution": False, "new_server_raw_receipt": False,
            "runner_sha256": sha(__file__)},
        "interpretation_limits": [
            "One historical 3-epoch local checkpoint and 71 validation patients, not the separate 77-case server protocol.",
            "Fixed-slot decoder repetition isolates gate noise; it does not measure full-model slot-initialization variation.",
            "gate_sum collapse attenuates the modality-specific auxiliary decoder output. The final fused prediction does not consume slot_gate.",
            "Low overlap and multiple sets demonstrate stochastic selection, not by themselves complete noise domination.",
            "No survival-performance damage or trained correction efficacy is inferred; no outcome scoring performed."]}
    out.mkdir(parents=True)
    write(out / "gate_diagnostic.json", result)
    print(json.dumps({"output": str(out / "gate_diagnostic.json"), "summary": summaries,
                      "checks": result["checks"]}, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source-verification", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--base-seed", type=int, default=4226000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if not args.run:
        print(json.dumps({"action": "plan_only", "repeats": args.repeats,
                          "new_model_forward_passes": 0, "training": False}))
        return
    if args.output_dir is None:
        p.error("--run requires --output-dir")
    run(args)


if __name__ == "__main__":
    main()
