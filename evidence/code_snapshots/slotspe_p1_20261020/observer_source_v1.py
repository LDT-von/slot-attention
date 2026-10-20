"""Read-only native SlotSPE slot occupancy/coverage observations, not a repair.

No model or tensor is loaded on import or in plan mode. Observers return None,
never replace a forward, and never draw random numbers. Array statistics are
descriptive; none is interpreted as a biological event count or predictive harm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class NativeSlotObserver:
    """Observe actual projected queries/keys and validate native value updates."""
    def __init__(self, model):
        self.handles = []
        self.data = {}
        for name in ("wsi", "omic"):
            slot = getattr(model, "slot_attention_" + name)
            decoder = getattr(model, "slot_decoder_" + name)
            def key_hook(_m, _i, output, key=name, module=slot):
                self.data[key] = {"k": module.split_heads(output.detach()), "iterations": [], "update_errors": []}
            def value_hook(_m, _i, output, key=name, module=slot):
                self.data[key]["v"] = module.split_heads(output.detach())
            def query_hook(_m, _i, output, key=name, module=slot):
                import torch
                from torch.nn import functional as F
                value = self.data[key]
                q = module.split_heads(output.detach())
                competition = (torch.einsum("... i d, ... j d -> ... i j", q, value["k"]) * module.scale).softmax(dim=-2)
                attention = F.normalize(competition + module.eps, p=1, dim=-1)
                expected = module.merge_heads(torch.einsum("... j d, ... i j -> ... i d", value["v"], attention))
                value["expected_update"] = expected
                value["iterations"].append(attention.mean(1).cpu())
                value.setdefault("mass", []).append(competition.mean((1, 3)).cpu())
                value["last_attention"] = attention.cpu()
                value["last_competition"] = competition.cpu()
            def update_hook(_m, inputs, _output, key=name):
                value = self.data[key]
                value["update_errors"].append(float((inputs[0].detach() - value.pop("expected_update")).abs().max()))
            def gru_hook(_m, inputs, key=name):
                value = self.data[key]
                if "initial_slots" not in value:
                    b, s = value["k"].shape[0], value["mass"][0].shape[-1]
                    value["initial_slots"] = inputs[1].detach().reshape(b, s, -1).cpu()
            def slots_hook(_m, _i, output, key=name):
                self.data[key]["slots"] = output.detach().cpu()
            def score_hook(_m, _i, output, key=name):
                self.data[key]["scores"] = output.detach().squeeze(-1).cpu()
            def decoder_hook(_m, _i, output, key=name):
                self.data[key]["gate"] = output[1].detach().cpu()
                self.data[key]["keep"] = output[2].detach().cpu()
            self.handles.extend([
                slot.to_k.register_forward_hook(key_hook), slot.to_v.register_forward_hook(value_hook),
                slot.to_q.register_forward_hook(query_hook), slot.combine_heads.register_forward_hook(update_hook),
                slot.gru.register_forward_pre_hook(gru_hook), slot.register_forward_hook(slots_hook),
                decoder.pred_keep_slot.register_forward_hook(score_hook), decoder.register_forward_hook(decoder_hook)])

    def arrays(self, index):
        import numpy as np
        result = {}
        for name, value in self.data.items():
            result[name + "_attention_iterations"] = np.stack([t[index].numpy() for t in value["iterations"]])
            result[name + "_competition_mass_iterations"] = np.stack([t[index].numpy() for t in value["mass"]])
            for target, source in (("last_attention_by_head", "last_attention"), ("last_competition_by_head", "last_competition"),
                                   ("slots", "slots"), ("initial_slots", "initial_slots"), ("scores", "scores"), ("gate", "gate"), ("keep", "keep")):
                result[name + "_" + target] = value[source][index].numpy()
        return result

    def maximum_update_error(self):
        return max(e for row in self.data.values() for e in row["update_errors"])

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def array_statistics(arrays, name):
    import numpy as np
    slots = np.asarray(arrays[name + "_slots"], dtype=np.float64)
    attention = np.asarray(arrays[name + "_attention_iterations"], dtype=np.float64)
    full = np.asarray(arrays[name + "_last_attention_by_head"], dtype=np.float64)
    competition = np.asarray(arrays[name + "_last_competition_by_head"], dtype=np.float64)
    mass_iterations = np.asarray(arrays[name + "_competition_mass_iterations"], dtype=np.float64)
    keep = np.asarray(arrays[name + "_keep"], dtype=np.float64)
    s, d = slots.shape
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Nonfinite raw observation")
    if attention.ndim != 3 or attention.shape[1] != s or full.shape[1:] != attention.shape[1:] or competition.shape != full.shape:
        raise ValueError("Attention/slot dimensions differ")
    if (attention < 0).any() or not np.allclose(attention.sum(-1), 1, atol=2e-6, rtol=0):
        raise ValueError("Native per-slot attention normalization differs")
    if (competition < 0).any() or not np.allclose(competition.sum(1), 1, atol=2e-6, rtol=0):
        raise ValueError("Native token competition normalization differs")
    mass = competition.mean((0, 2))
    if not np.allclose(mass, mass_iterations[-1], atol=2e-6, rtol=0) or not np.allclose(full.mean(0), attention[-1], atol=2e-6, rtol=0):
        raise ValueError("Recorded head/iteration values disagree")
    if keep.shape != (s,) or not np.allclose(keep, np.round(keep), atol=1e-6, rtol=0):
        raise ValueError("Expected native hard Top-K observations")
    normalized = slots / np.maximum(np.linalg.norm(slots, axis=1, keepdims=True), 1e-30)
    pairs = []
    for a in range(s):
        for b in range(a + 1, s):
            pairs.append({"slots": [a, b], "vector_cosine": float(normalized[a] @ normalized[b]),
                "attention_overlap": float(np.minimum(attention[-1, a], attention[-1, b]).sum()),
                "attention_max_abs_difference": float(np.abs(attention[-1, a] - attention[-1, b]).max()),
                "slot_vector_max_abs_difference": float(np.abs(slots[a] - slots[b]).max())})
    positive = mass[mass > 0]
    return {"candidate_slots": s, "selected_slots": int((keep > .5).sum()), "input_tokens": int(attention.shape[-1]),
        "slot_width": d, "iterations": len(attention), "competition_mass": mass.tolist(),
        "mass_entropy_effective_count": float(np.exp(-(positive * np.log(positive)).sum())),
        "pairs": pairs,
        "interpretation": "Mass effective count is a descriptive occupancy index, not a prognostic event count. Similarity/overlap alone does not establish harmful redundancy."}


def plan(root, seed=3):
    root = Path(root).resolve()
    names = ["manifest.json", "training_protocol.json", "cache/paired_blca_fold0.pt",
             f"seed_{seed}/original/last.pt", f"seed_{seed}/original/summary.json", f"seed_{seed}/original/eval_repeats.json"]
    missing = [name for name in names if not (root / name).is_file()]
    return {"schema": "slotspe_p1_readonly_legacy_v1", "run_root": str(root), "seed": seed,
        "layout": "historical_local_patient_cache", "ready_for_readonly_check": not missing, "missing": missing,
        "training_runs": 0, "optimizer_steps": 0, "new_input_sampling": False,
        "native_validation_passes": 2, "default_action": "plan_only", "server_SPE09_evidence": False}


def run_legacy(root, output, seed=3, device="cuda:0"):
    """Replay one historical original checkpoint and observe all validation cases."""
    from .native_audit import model_purity, _state_digest
    from .model import build_model, make_args
    root, output = Path(root).resolve(), Path(output).resolve()
    description = plan(root, seed)
    if description["missing"]:
        raise FileNotFoundError("Missing historical originals: " + ", ".join(description["missing"]))
    if output == root or root in output.parents or output.exists():
        raise ValueError("Use a new output directory outside the original run")
    manifest = _read(root / "manifest.json")
    protocol = _read(root / "training_protocol.json")
    summary = _read(root / f"seed_{seed}/original/summary.json")
    repeats = _read(root / f"seed_{seed}/original/eval_repeats.json")
    source = Path(manifest["source"])
    watched = {str(root / name): _sha(root / name) for name in ["manifest.json", "training_protocol.json", "cache/paired_blca_fold0.pt",
        f"seed_{seed}/original/last.pt", f"seed_{seed}/original/summary.json", f"seed_{seed}/original/eval_repeats.json", "run_patient_proof.py"]}
    if watched[str(root / "cache/paired_blca_fold0.pt")] != manifest["cache_sha256"] or watched[str(root / "manifest.json")] != protocol["manifest_sha256"]:
        raise ValueError("Historical cache/manifest hash differs")
    if watched[str(root / f"seed_{seed}/original/last.pt")] != summary["checkpoint_sha256"] or watched[str(root / "run_patient_proof.py")] != protocol["runner_sha256"]:
        raise ValueError("Historical checkpoint/runner hash differs")
    if manifest["source_hashes"] != protocol["source_hashes_before"] or manifest["source_hashes"] != protocol["source_hashes_after"]:
        raise ValueError("Historical source identity differs")
    for name, digest in manifest["source_hashes"].items():
        path = source / name
        if _sha(path) != digest:
            raise ValueError("Current native source differs: " + name)
        watched[str(path)] = digest
    import numpy as np
    import torch
    import random
    torch.set_num_threads(2)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested device unavailable; no silent device fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Historical torch.__version__ was serialized as the installed TorchVersion
    # metadata type. Keep weights_only enabled; never load arbitrary classes.
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        data = torch.load(root / "cache/paired_blca_fold0.pt", map_location="cpu", weights_only=True, mmap=True)
        saved = torch.load(root / f"seed_{seed}/original/last.pt", map_location="cpu", weights_only=True, mmap=True)
    records = manifest["patients"]
    if data["patient_id"] != [r["patient_id"] for r in records] or data["split"] != [r["split"] for r in records]:
        raise ValueError("Patient identity/order differs")
    if saved["arm"] != "original" or saved["seed"] != seed or saved["epochs"] != summary["epochs"] or saved["optimizer_steps"] != summary["optimizer_steps"]:
        raise ValueError("Checkpoint training identity differs")
    if saved["checkpoint_selection"] != "fixed_final_epoch" or saved["epochs"] <= 0 or saved["optimizer_steps"] <= 0:
        raise ValueError("Requires a trained, fixed-final native checkpoint")
    selected = [i for i, row in enumerate(records) if row["split"] == "val"]
    if len(selected) != manifest["counts"]["val"] or len(set(data["patient_id"])) != len(records):
        raise ValueError("Validation coverage/unique identities differ")
    model = build_model(source, make_args(data["omic_sizes"], encoding_dim=data["wsi"].shape[-1]), "original").to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    purity = model_purity(model, source)
    before_state = _state_digest(model)
    del saved
    eval_seed = repeats[0]["eval_seed"]
    bs = protocol["batch_size"]
    def reset_rng():
        random.seed(eval_seed)
        np.random.seed(eval_seed % (2 ** 32))
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
    def batch(indices):
        ix = torch.tensor(indices, dtype=torch.long)
        value = {"x_wsi": data["wsi"][ix].to(device), "y": data["label"][ix].to(device), "c": data["censor"][ix].to(device), "omic_missing": False}
        value.update({f"x_omic{i+1}": tensor[ix].to(device) for i, tensor in enumerate(data["omics"])})
        return value
    reset_rng()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(selected), bs):
            predictions.append(model(**batch(selected[start:start + bs]))[0].cpu())
    baseline = torch.cat(predictions)
    old_error = float((baseline - torch.tensor(repeats[0]["logits"])).abs().max())
    if old_error > 1e-6:
        raise RuntimeError(f"Historical same-RNG replay failed: {old_error}; retain prior evidence and inspect environment/device")
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    observer = NativeSlotObserver(model)
    reset_rng()
    observations, observed_predictions, max_update_error = [], [], 0.
    try:
        with torch.no_grad():
            for start in range(0, len(selected), bs):
                indices = selected[start:start + bs]
                logits = model(**batch(indices))[0].cpu()
                observed_predictions.append(logits)
                max_update_error = max(max_update_error, observer.maximum_update_error())
                for local, index in enumerate(indices):
                    arrays = observer.arrays(local)
                    filename = "raw/" + hashlib.sha256(records[index]["patient_id"].encode()).hexdigest() + ".npz"
                    np.savez_compressed(output / filename, **arrays)
                    observations.append({"patient_id": records[index]["patient_id"], "manifest_index": index,
                        "raw_path": filename, "raw_sha256": _sha(output / filename), "logits": logits[local].tolist(),
                        "wsi": array_statistics(arrays, "wsi"), "omic": array_statistics(arrays, "omic")})
    finally:
        observer.close()
    observed = torch.cat(observed_predictions)
    observer_error = float((observed - baseline).abs().max())
    state_unchanged = _state_digest(model) == before_state
    if observer_error > 1e-6 or max_update_error > 1e-6 or not state_unchanged:
        raise RuntimeError("Observer altered predictions/state or reconstructed a different attention update")
    if any(_sha(Path(path)) != digest for path, digest in watched.items()):
        raise RuntimeError("Original artifact/source changed during read-only check")
    _write(output / "patient_provenance.json", {"patients": [records[i] for i in selected], "omic_sizes": data["omic_sizes"],
        "binning": manifest["binning"], "input_csv_sha256": manifest["input_csv_sha256"]})
    report = {"schema": "slotspe_p1_observations_v1", "status": "measured", "input_scope": "historical_local_real_BLCA_cached_validation",
        "server_SPE09_evidence": False, "fold": 0, "seed": seed, "device": device, "torch": str(torch.__version__),
        "epochs": summary["epochs"], "prior_optimizer_steps": summary["optimizer_steps"], "new_optimizer_steps": 0,
        "patient_count": len(selected), "patches": manifest["patches"], "eval_seed": eval_seed,
        "model_purity": purity, "historical_replay_max_logit_difference": old_error,
        "observer_max_logit_difference": observer_error, "reconstructed_native_update_max_error": max_update_error,
        "checkpoint_tensors_unchanged": state_unchanged, "original_inputs_unchanged": True, "source_inputs_sha256": watched,
        "candidate_counts": {m: sorted({r[m]["candidate_slots"] for r in observations}) for m in ("wsi", "omic")},
        "selected_counts": {m: sorted({r[m]["selected_slots"] for r in observations}) for m in ("wsi", "omic")},
        "interpretation": "Native fixed-count and descriptive coverage observations only. No prognostic event count, semantic label, harmful redundancy or patient benefit is inferred.",
        "P1_fixed_count_predictive_harm": "not_established", "P2_functional_redundancy": "not_established",
        "records": observations}
    _write(output / "observations.json", report)
    files = [p for p in sorted(output.rglob("*")) if p.is_file()]
    _write(output / "receipt.json", {"schema": "slotspe_p1_receipt_v1", "outputs": {p.relative_to(output).as_posix(): _sha(p) for p in files},
        "observer_source_sha256": _sha(__file__), "original_artifacts_unchanged": True})
    return {k: v for k, v in report.items() if k not in ("records", "model_purity", "source_inputs_sha256")}


def audit_export(output):
    import numpy as np
    output = Path(output).resolve()
    receipt = _read(output / "receipt.json")
    for name, digest in receipt["outputs"].items():
        path = (output / name).resolve()
        if output not in path.parents or _sha(path) != digest:
            raise ValueError("Output hash/path differs: " + name)
    report = _read(output / "observations.json")
    if not report["checkpoint_tensors_unchanged"] or not report["original_inputs_unchanged"]:
        raise ValueError("Read-only invariants failed")
    ids = [r["patient_id"] for r in report["records"]]
    provenance = _read(output / "patient_provenance.json")
    if ids != [r["patient_id"] for r in provenance["patients"]] or len(ids) != len(set(ids)) or len(ids) != report["patient_count"]:
        raise ValueError("Raw patient coverage differs")
    for row in report["records"]:
        if row["raw_path"] not in receipt["outputs"] or row["raw_sha256"] != receipt["outputs"][row["raw_path"]]:
            raise ValueError("Unreceipted patient arrays")
        with np.load(output / row["raw_path"], allow_pickle=False) as arrays:
            for modality in ("wsi", "omic"):
                calculated = array_statistics(arrays, modality)
                if calculated != row[modality]:
                    raise ValueError("Raw arrays disagree with descriptive statistics")
    return {"status": "accepted_raw_observations", "patient_count": len(ids), "candidate_counts": report["candidate_counts"],
        "selected_counts": report["selected_counts"], "server_SPE09_evidence": report["server_SPE09_evidence"],
        "P1_predictive_harm_proved": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true")
    group.add_argument("--audit", action="store_true")
    args = parser.parse_args(argv)
    if args.audit:
        if args.output is None:
            parser.error("--audit requires --output")
        result = audit_export(args.output)
    else:
        if args.run_root is None:
            parser.error("--run-root is required")
        if args.run:
            if args.output is None:
                parser.error("--run requires a fresh --output")
            result = run_legacy(args.run_root, args.output, args.seed, args.device)
        else:
            result = plan(args.run_root, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
