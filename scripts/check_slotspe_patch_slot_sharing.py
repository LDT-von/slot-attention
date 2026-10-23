"""Measure how patches are shared across slots in the native SlotSPE attention.

Read-only. It reads already saved native attention arrays (no model import, no
forward pass, no training) and answers one question: does a patch get read by
many slots, and do the slots read different patches or the same ones?

Default action is metadata planning. --run reads existing NPZ arrays only.
Private patient rows stay inside the ignored local_evidence directory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values):
    import numpy as np
    values = np.asarray(list(values), dtype=float)
    return {"n": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max())}


def measure(raw_path, modality):
    """Sharing structure between slots for one patient and one modality."""
    import numpy as np
    with np.load(raw_path, allow_pickle=False) as z:
        attention = z[f"{modality}_last_attention_by_head"].astype(np.float64)
        competition = z[f"{modality}_last_competition_by_head"].astype(np.float64)
        slots = z[f"{modality}_slots"].astype(np.float64)
        iterations = z[f"{modality}_attention_iterations"].astype(np.float64)
    heads, slots_count, tokens = attention.shape
    averaged = attention.mean(axis=0)
    averaged = averaged / averaged.sum(axis=1, keepdims=True)

    # (1) cross-slot share of every patch: is one patch read by many slots?
    share = averaged / np.maximum(averaged.sum(axis=0, keepdims=True), 1e-300)
    entropy = -(share * np.log(np.maximum(share, 1e-300))).sum(axis=0)
    effective_slots = np.exp(entropy)
    uniform = 1.0 / slots_count
    share_deviation = float(np.abs(share - uniform).mean())

    # (2) does one slot dominate a patch?
    top1 = share.max(axis=0)
    near_top = int((share >= top1 * 0.99).sum(axis=0).mean())

    # (3) is each slot's own reading distribution over patches differentiated?
    #     Entropy-based effective count. The native attention is L1-normalized along
    #     the token axis, so exp(entropy) counts effective patches per slot.
    per_slot_entropy = np.array([-(p * np.log(np.maximum(p, 1e-300))).sum() for p in averaged])
    effective_patches = float(np.exp(per_slot_entropy).mean())
    #     Top-decile concentration, compared against the uniform baseline 10% of patches.
    top_decile = float(np.mean([np.sort(p)[::-1][:max(1, tokens // 10)].sum() for p in averaged]))

    # (4) are the 8 slots reading the same distribution or different ones?
    profile = averaged.mean(axis=0)
    gain = (averaged / np.maximum(profile, 1e-300)).mean(axis=1)
    rank_one = np.outer(gain, profile)
    rank_one_fraction = float(1 - ((averaged - rank_one) ** 2).sum() / (averaged ** 2).sum())
    gain_spread = float(gain.std() / gain.mean())
    pair_max_gap = float(np.abs(averaged[:, None, :] - averaged[None, :, :]).max(axis=2)[
        np.triu_indices(slots_count, 1)].max())
    overlap = float(np.mean([np.minimum(averaged[i], averaged[j]).sum()
                              for i in range(slots_count) for j in range(i + 1, slots_count)]))

    # (5) slot-vector similarity (representation side, for contrast)
    units = slots / np.linalg.norm(slots, axis=1, keepdims=True)
    cosine = np.clip(units @ units.T, -1, 1)
    upper = np.triu_indices(slots_count, 1)

    # (6) trajectory: does the reading become more uniform over iterations?
    uniform_matrix = np.full((slots_count, tokens), 1.0 / tokens)
    amplitude = []
    for t in range(iterations.shape[0]):
        current = iterations[t]
        amplitude.append(float(np.linalg.norm(current - uniform_matrix) / np.linalg.norm(uniform_matrix)))

    # (7) raw slot-axis competition: is the softmax itself flat?
    raw = competition / np.maximum(competition.sum(axis=1, keepdims=True), 1e-300)
    raw_deviation = float(np.abs(raw - uniform).mean())
    raw_ratio = float((raw.max(axis=1) / np.maximum(raw.min(axis=1), 1e-300)).mean())

    return {"tokens": int(tokens), "slots": int(slots_count), "heads": int(heads),
            "effective_slots_reading_each_patch_mean": float(effective_slots.mean()),
            "effective_slots_reading_each_patch_min": float(effective_slots.min()),
            "cross_slot_share_deviation_from_uniform": share_deviation,
            "slots_within_99pct_of_top_share_at_a_patch": near_top,
            "effective_patches_read_per_slot": effective_patches,
            "top_decile_share_of_slot_mass": top_decile,
            "uniform_top_decile_mass_share": 0.10,
            "rank_one_separable_fraction": rank_one_fraction,
            "per_slot_gain_spread": gain_spread,
            "max_pairwise_gap_at_a_patch": pair_max_gap,
            "pairwise_reading_overlap": overlap,
            "slot_vector_cosine_mean": float(cosine[upper].mean()),
            "slot_vector_cosine_min": float(cosine[upper].min()),
            "reading_uniform_amplitude_by_iteration": amplitude,
            "raw_slot_softmax_deviation_from_uniform": raw_deviation,
            "raw_slot_softmax_max_min_ratio": raw_ratio}


def run(root, output):
    import numpy as np
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to((REPO / "local_evidence").resolve()) or output.exists():
        raise ValueError("A fresh output directory inside ignored local_evidence is required")
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
    seen = set()
    for row in observations["records"]:
        if row["patient_id"] in seen:
            raise ValueError("Duplicate patient, not an independent case")
        seen.add(row["patient_id"])
        path = (root / row["raw_path"]).resolve()
        if not path.is_relative_to(root) or sha(path) != row["raw_sha256"]:
            raise ValueError("Raw-array content hash mismatch")
        rows.append({"patient_id": row["patient_id"],
                     **{m: measure(path, m) for m in ("wsi", "omic")}})
    if len(rows) != observations["patient_count"]:
        raise ValueError("Cohort count differs from the historical record")
    result = {"task_id": "SPE-08", "recorded_utc": datetime.now(timezone.utc).isoformat(),
              "route": "patch_to_slot_sharing_readonly",
              "scope": observations["input_scope"], "server_new_results": False,
              "observations_sha256": sha(root / "observations.json"),
              "historical_receipt_sha256": sha(root / "receipt.json"),
              "verified_historical_output_files": verified,
              "historical_native_source_matches_current": True,
              "patient_count": len(rows), "fold": observations["fold"], "seed": observations["seed"],
              "historical_epochs": observations["epochs"], "patches": observations["patches"],
              "checkpoint_sha256_recorded": [digest for name, digest in observations["source_inputs_sha256"].items()
                                             if Path(name.replace("\\", "/")).name == "last.pt"],
              "new_model_forward_passes": 0, "new_training_steps": 0, "new_optimizer_steps": 0,
              "question": "Is a patch read by many slots, and do the slots read different patches or the same ones?",
              "metric_note": "Cross-slot share = attention of one slot divided by the total attention on that patch. "
                             "Reading overlap = sum over patches of min(p_i, p_j). Slot vectors are the raw final "
                             "slots before decoder.map.",
              "cosine_0_95_is_descriptive_not_clinical_threshold": True,
              "modalities": {}}
    for modality in ("wsi", "omic"):
        values = [r[modality] for r in rows]
        result["modalities"][modality] = {
            "patients": len(values),
            "tokens": values[0]["tokens"], "slots": values[0]["slots"], "heads": values[0]["heads"],
            "distributions": {key: summarize(v[key] for v in values)
                              for key in values[0] if key not in ("reading_uniform_amplitude_by_iteration",)},
            "reading_uniform_amplitude_by_iteration_mean": [
                float(np.mean([v["reading_uniform_amplitude_by_iteration"][i] for v in values]))
                for i in range(len(values[0]["reading_uniform_amplitude_by_iteration"]))],
        }
    result["conclusion_boundary"] = [
        "Single fold0, single seed3, single frozen 3-epoch checkpoint, 77 validation patients.",
        "Read-only recomputation over already saved native attention arrays. No training, no new forward pass.",
        "Sharing geometry is not predictive harm and not a proof that a slot carries no unique information.",
        "The cross-slot share of a patch is structurally pinned near 1/8 by the native slot-axis softmax, so it "
        "measures the normalization, not a learned competition outcome.",
        "No causal intervention was run, so this does not attribute the reading flatness to any single ring.",
    ]
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
        print(json.dumps({"action": "plan_only", "input_root": str(args.input_root),
                          "new_model_forward_passes": 0, "new_training_steps": 0,
                          "requires_existing": ["observations.json", "receipt.json", "raw/*.npz"]}, indent=2))
        return
    if args.output_dir is None:
        p.error("--run requires --output-dir inside local_evidence")
    print(json.dumps(run(args.input_root, args.output_dir), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
