"""Recheck retained native slot arrays; never import torch or run a model.

Default action is metadata planning. --run only reads already saved observations
and NPZ arrays. Private patient rows remain inside ignored local_evidence.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    import numpy as np
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite aggregate metric")
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())}


def measure(z, modality):
    import numpy as np
    slots = z[modality + "_slots"].astype(float)
    keep = z[modality + "_keep"].astype(float)
    gate = z[modality + "_gate"].astype(float)
    attention = z[modality + "_last_attention_by_head"].astype(float)
    if not all(np.isfinite(a).all() for a in (slots, keep, gate, attention)):
        raise ValueError("Nonfinite retained input")
    count = len(slots)
    selected = np.flatnonzero(keep > 0.5)
    if slots.ndim != 2 or count < 2 or len(selected) != 2 or attention.shape[1] != count or len(gate) != count:
        raise ValueError("This audit requires two selected slots from a valid native bag")
    if not np.allclose(keep, (keep > 0.5).astype(float), atol=1e-6, rtol=0) or (gate < 0).any() or np.any(gate[keep < 0.5] != 0):
        raise ValueError("Saved hard keep or weighted gate violates the declared native contract")
    norms = np.linalg.norm(slots, axis=1, keepdims=True)
    if (norms <= 0).any() or (attention < 0).any() or (attention.sum(-1) <= 0).any():
        raise ValueError("Zero slots or invalid attention rows; cosine/reading metric undefined")
    units = slots / norms
    cosine = np.clip(units @ units.T, -1, 1)
    pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
    pair_values = np.asarray([cosine[i, j] for i, j in pairs])
    attention = attention / attention.sum(-1, keepdims=True)
    overlaps = np.asarray([np.minimum(attention[:, i], attention[:, j]).sum(-1).mean() for i, j in pairs])
    a, b = selected
    selected_overlap = float(np.minimum(attention[:, a], attention[:, b]).sum(-1).mean())
    centered = slots - slots.mean(0)
    spectrum = np.linalg.svd(centered, compute_uv=False) ** 2
    probabilities = spectrum[spectrum > 0] / spectrum.sum() if spectrum.sum() else []
    total_gate = float(gate.sum())
    normalized = gate / total_gate if total_gate else None
    return {"candidate_slots": count, "selected_slots": len(selected), "unique_pairs": len(pairs),
            "all_pair_cosine_mean": float(pair_values.mean()), "all_pair_cosine_min": float(pair_values.min()),
            "selected_pair_cosine": float(cosine[a, b]), "pairs_cosine_above_0_95": int((pair_values > .95).sum()),
            "all_pair_reading_overlap_mean": float(overlaps.mean()), "selected_pair_reading_overlap": selected_overlap,
            "selected_minus_all_pair_cosine": float(cosine[a, b] - pair_values.mean()),
            "selected_minus_all_pair_reading_overlap": float(selected_overlap - overlaps.mean()),
            "centered_energy_fraction": float((centered ** 2).sum() / (slots ** 2).sum()),
            "centered_effective_rank": float(np.exp(-sum(p * np.log(p) for p in probabilities))) if len(probabilities) else None,
            "gate_weight_sum": total_gate, "nonzero_gate_weights": int(np.count_nonzero(gate)),
            "gate_weight_effective_slots": float(1 / np.sum(normalized ** 2)) if normalized is not None else None}


def run(root, output):
    import numpy as np
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to((REPO / "local_evidence").resolve()) or output.exists():
        raise ValueError("A fresh output within ignored local_evidence is required")
    observations = read(root / "observations.json")
    receipt = read(root / "receipt.json")
    verified = 0
    for relative, digest in receipt["outputs"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError("Historical output receipt mismatch")
        verified += 1
    for relative, digest in observations["model_purity"]["source_sha256"].items():
        if sha(REPO / "third_party/SlotSPE" / relative) != digest:
            raise ValueError("Historical native model source differs")
    rows = []
    patients = set()
    for row in observations["records"]:
        if row["patient_id"] in patients:
            raise ValueError("Duplicate patient, not an independent case")
        patients.add(row["patient_id"])
        path = (root / row["raw_path"]).resolve()
        if not path.is_relative_to(root) or sha(path) != row["raw_sha256"]:
            raise ValueError("Raw-array content hash mismatch")
        with np.load(path, allow_pickle=False) as z:
            rows.append({"patient_id": row["patient_id"], "raw_sha256": row["raw_sha256"], **{m: measure(z, m) for m in ("wsi", "omic")}})
    if len(rows) != observations["patient_count"]:
        raise ValueError("Cohort count differs from historical record")
    result = {"task_id": "SPE-08", "recorded_utc": datetime.now(timezone.utc).isoformat(),
              "scope": observations["input_scope"], "server_new_results": False,
              "observations_sha256": sha(root / "observations.json"), "historical_receipt_sha256": sha(root / "receipt.json"),
              "verified_historical_output_files": verified, "historical_native_source_matches_current": True,
              "patient_count": len(rows), "fold": observations["fold"], "seed": observations["seed"],
              "historical_epochs": observations["epochs"], "historical_optimizer_steps": observations["prior_optimizer_steps"],
              "patches": observations["patches"], "eval_seed": observations["eval_seed"],
              "checkpoint_sha256_recorded": [digest for name, digest in observations["source_inputs_sha256"].items() if Path(name.replace("\\", "/")).name == "last.pt"],
              "new_model_forward_passes": 0, "new_training_steps": 0,
              "metric": "Raw final slots before decoder.map; unique unordered pairs only. Reading overlap=sum_token min(p_i,p_j), per-head row-normalized then averaged across heads.",
              "cosine_0_95_is_descriptive_not_clinical_threshold": True, "modalities": {}}
    for modality in ("wsi", "omic"):
        values = [r[modality] for r in rows]
        result["modalities"][modality] = {
            "candidate_counts": dict(Counter(str(v["candidate_slots"]) for v in values)),
            "selected_counts": dict(Counter(str(v["selected_slots"]) for v in values)),
            "nonzero_gate_weight_counts": dict(Counter(str(v["nonzero_gate_weights"]) for v in values)),
            "patients_all_unique_pairs_above_0_95": sum(v["pairs_cosine_above_0_95"] == v["unique_pairs"] for v in values),
            "patients_selected_pair_above_0_95": sum(v["selected_pair_cosine"] > .95 for v in values),
            "patients_zero_gate_weight_sum": sum(v["gate_weight_sum"] == 0 for v in values),
            "distributions": {key: stats(v[key] for v in values) for key in values[0] if key not in ("candidate_slots", "selected_slots", "unique_pairs")}}
    result["conclusion_boundary"] = [
        "Native retained final-slot and reading-distribution redundancy persists after this checkpoint's selection.",
        "These are 71 historical local BLCA cases at one 3-epoch checkpoint and one retained eval draw, not the 77-case server run.",
        "No new inference; decoder.map outputs, biological event labels and per-slot marginal predictive utility were not recomputed.",
        "No without-gating matched training control: this does not identify the gate's causal training effect or predictive harm."]
    output.mkdir(parents=True)
    for name, data in (("patient_metrics.json", rows), ("aggregate_summary.json", result)):
        (output / name).write_bytes((json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if not args.run:
        print(json.dumps({"action": "plan_only", "input_root": str(args.input_root), "new_model_forward_passes": 0, "requires_existing": ["observations.json", "receipt.json", "raw/*.npz"]}, indent=2))
        return
    if args.output_dir is None:
        p.error("--run requires --output-dir inside local_evidence")
    print(json.dumps(run(args.input_root, args.output_dir), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
