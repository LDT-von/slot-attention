"""Derive attention overlap descriptions from immutable recorded arrays only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"E:\DCT-Reg\.review\slotspe_p1_20261020\seed3_native_v2")
OUT = Path(r"E:\DCT-Reg\docs\research\slotspe_p1_20261020")
report = json.loads((ROOT / "observations.json").read_text(encoding="utf-8"))
def distribution(values):
    a = np.asarray(values, dtype=float)
    return {"observations": int(a.size), **{name: float(value) for name, value in zip(("min", "p25", "median", "p75", "max"), np.quantile(a, [0, .25, .5, .75, 1]))}}
def pairs(matrix):
    return [float(np.minimum(matrix[a], matrix[b]).sum()) for a in range(len(matrix)) for b in range(a + 1, len(matrix))]

results, per_patient = {}, []
for mode in ("wsi", "omic"):
    all_pairs, heads, first, last, selected, uniform_distances = [], [], [], [], [], []
    evolution = [[] for _ in range(10)]
    for row in report["records"]:
        with np.load(ROOT / row["raw_path"], allow_pickle=False) as z:
            attention = z[mode + "_attention_iterations"].astype(np.float64)
            head_attention = z[mode + "_last_attention_by_head"].astype(np.float64)
            chosen = np.flatnonzero(z[mode + "_keep"] > .5)
            overlaps = pairs(attention[-1])
            head_overlaps = [value for head in head_attention for value in pairs(head)]
            top_overlap = float(np.minimum(attention[-1, chosen[0]], attention[-1, chosen[1]]).sum())
            per_patient.append({"patient_id": row["patient_id"], "modality": mode,
                "selected_indices": chosen.tolist(), "selected_pair_attention_overlap": top_overlap,
                "all_pair_overlap_min": min(overlaps), "all_pair_overlap_median": float(np.median(overlaps)),
                "all_head_all_pair_overlap_min": min(head_overlaps)})
            all_pairs.extend(overlaps)
            heads.extend(head_overlaps)
            first.extend(pairs(attention[0]))
            last.extend(overlaps)
            selected.append(top_overlap)
            uniform_distances.extend((.5 * np.abs(head_attention - 1 / head_attention.shape[-1]).sum(-1)).flatten().tolist())
            for i, value in enumerate(attention):
                evolution[i].extend(pairs(value))
    results[mode] = {"mass_entropy_effective_count": distribution([r[mode]["mass_entropy_effective_count"] for r in report["records"]]),
        "final_head_mean_attention_pair_overlap": distribution(all_pairs), "final_per_head_attention_pair_overlap": distribution(heads),
        "selected_pair_head_mean_overlap": distribution(selected), "first_iteration_head_mean_overlap": distribution(first),
        "final_per_head_total_variation_from_uniform": distribution(uniform_distances),
        "vector_cosine": distribution([p["vector_cosine"] for r in report["records"] for p in r[mode]["pairs"]]),
        "pair_overlap_by_iteration": [distribution(v) for v in evolution],
        "patients_with_all_final_head_mean_pairs_over_0_99": sum(x["all_pair_overlap_min"] > .99 for x in per_patient if x["modality"] == mode),
        "technical_threshold_0_99_not_biological_or_functional_redundancy_cutoff": True}
summary = {"task_id": "SPE-08", "date": "2026-10-20", "stage": "P1 historical trained native checkpoint read-only observation",
    "patient_count": report["patient_count"], "fold": 0, "seed": 3, "epochs": 3, "patches": 256,
    "historical_checkpoint_sha256": report["source_inputs_sha256"][str(Path(r"E:\DCT-Reg\.review\slotspe_proof_20261018\patient\seed_3\original\last.pt"))],
    "raw_export_root": str(ROOT), "raw_observations_sha256": hashlib.sha256((ROOT / "observations.json").read_bytes()).hexdigest(),
    "overlap_definition": "sum_j min(p_j,q_j), on normalized native attention over the exact same sampled input rows; 0=no support overlap, 1=identical distributions",
    "results": results, "per_patient": per_patient,
    "P1_fixed_number_observed": all(report["candidate_counts"][m] == [8] and report["selected_counts"][m] == [2] for m in ("wsi", "omic")), "P1_fixed_capacity_predictive_harm_proved": False,
    "P2_attention_distribution_overlap_observed": any(v["patients_with_all_final_head_mean_pairs_over_0_99"] > 0 for v in results.values()),
    "P2_functional_redundancy_or_information_loss_proved": False,
    "limitations": ["Original local 3-epoch diagnostic checkpoint, not the SPE09 server run or a converged model claim.",
        "Head-specific checks address averaging as a source of apparent overlap; they do not provide tissue labels.",
        "Attention similarity does not establish identical slot vectors, lost independent information, or reduced predictive performance.",
        "Pairs/heads are repeated observations within 71 patients, not independent patient counts.",
        "Mass entropy near eight can coexist with almost identical attention, and is not evidence of eight distinct events."]}
(OUT / "descriptive_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({m: {k: v for k, v in x.items() if k not in ("pair_overlap_by_iteration",)} for m, x in results.items()}, ensure_ascii=False, indent=2))
