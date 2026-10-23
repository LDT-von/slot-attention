"""Independently recompute saved gate JSON; never imports torch or runs a model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dist(values):
    a = np.asarray(values, dtype=float)
    assert len(a) and np.isfinite(a).all()
    return {"n": len(a), "median": float(np.median(a)), "mean": float(a.mean()),
            "min": float(a.min()), "max": float(a.max()),
            "below_0_5_count": int((a < .5).sum()), "below_0_5_fraction": float((a < .5).mean()),
            "below_0_01_count": int((a < .01).sum()), "below_0_01_fraction": float((a < .01).mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    assert out.is_relative_to((REPO / "local_evidence").resolve()) and not out.exists()
    data = json.loads(a.input.read_text(encoding="utf-8"))
    rows = data["per_patient"]
    assert len(rows) == data["protocol"]["patients"] == 71
    assert len({r["patient_id"] for r in rows}) == len(rows)
    for rel, digest in data["checks"]["native_source_sha256"].items():
        assert sha(REPO / "third_party/SlotSPE" / rel) == digest
    assert sha(REPO / "local_evidence/original_tree/.review/slotspe_proof_20261018/patient/seed_3/original/last.pt") == data["protocol"]["checkpoint_sha256"]
    assert sha(REPO / "scripts/check_slotspe_gate_collapse.py") == data["checks"]["runner_sha256"]
    assert data["checks"]["model_state_sha256_before"] == data["checks"]["model_state_sha256_after"]
    result = {"schema": "slotspe_gate_result_recomputation_v1", "status": "passed",
              "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(),
              "input_sha256": sha(a.input), "audit_runner_sha256": sha(__file__),
              "model_forward_passes": 0, "training_steps": 0, "modalities": {}}
    enumeration = []
    for m in ("wsi", "omic"):
        first, pooled, unique, worst_gate, worst_mass = [], [], [], [], []
        formula_error, mass_error, gate_sum_error = 0., 0., 0.
        overlap = Counter()
        for index, patient in enumerate(rows):
            v = patient["modalities"][m]
            s = np.asarray(v["scores"], dtype=float)
            z = s / .01
            q = np.exp(z - z.max()); q /= q.sum()
            qraw = np.exp(s - s.max()); qraw /= qraw.sum()
            pairs = list(combinations(range(8), 2))
            masses = [float(q[list(pair)].sum()) for pair in pairs]
            minimum = min(masses)
            worst_mass.append(minimum)
            worst_gate.append(minimum / (minimum + 1e-8))
            enumeration.append({"patient_index": index, "modality": m,
                "mask_combinations": len(pairs), "minimum_selected_mass_float64": minimum,
                "minimum_gate_sum_float64": worst_gate[-1],
                "label": "exhaustive arithmetic on saved patient scores, not model forwards"})
            reps = v["repeats"]
            assert len(reps) == data["protocol"]["repeats_per_patient"] == 20
            sets = {tuple(r["selected_indices"]) for r in reps}
            assert len(sets) == v["unique_retained_sets"]
            unique.append(len(sets)); first.append(reps[0]["gate_sum"])
            assert dist([r["gate_sum"] for r in reps]) == v["gate_sum_across_rng"]
            assert sorted(v["deterministic_top2_indices"]) == sorted(np.argsort(s)[-2:].tolist())
            for r in reps:
                keep = np.asarray(r["hard_keep"], dtype=float)
                indices = r["selected_indices"]
                assert len(indices) == r["keep_count"] == 2
                assert np.flatnonzero(keep > .5).tolist() == indices
                gates = np.asarray(r["gate_weights"], dtype=float)
                assert np.isfinite(gates).all() and (gates >= 0).all()
                assert np.all(gates[keep <= .5] == 0)
                gate_sum_error = max(gate_sum_error, abs(gates.sum() - r["gate_sum"]))
                mass = float(q[indices].sum())
                mass_error = max(mass_error, abs(mass - r["selected_softmax_mass_tau_0_01_float64"]),
                                 abs(qraw[indices].sum() - r["selected_softmax_mass_raw_tau_1_float64"]))
                # ST masks can differ from exact 0/1 by float32 rounding.
                weighted_mass = float((q * keep).sum())
                formula_error = max(formula_error, abs(weighted_mass / (weighted_mass + 1e-8) - r["gate_sum"]))
                n = len(set(indices) & set(v["deterministic_top2_indices"]))
                assert n == r["deterministic_top2_overlap"]
                overlap[str(n)] += 1
                pooled.append(r)
        summary = data["summary"][m]
        assert dist(first) == summary["primary_first_prespecified_rng_draw_unique_patients"]
        assert dist([r["gate_sum"] for r in pooled]) == summary["all_rng_patient_draws_not_independent_patients"]
        assert dict(overlap) == summary["top2_overlap_histogram"]
        assert dist(unique) == summary["unique_retained_sets_across_rng"]
        for key, expected in summary["controls_all_draws"].items():
            assert dist([r[key] for r in pooled]) == expected
        assert gate_sum_error < 1e-6 and formula_error < 2e-6 and mass_error < 1e-12
        result["modalities"][m] = {"patients": len(rows), "rng_draws": len(pooled),
            "all_reported_distributions_recomputed_match": True,
            "gate_weight_sum_max_abs_error": gate_sum_error,
            "float64_formula_vs_native_max_abs_error": formula_error,
            "float64_mass_max_abs_error": mass_error,
            "all_28_mask_combinations_per_patient": {"patient_count": len(rows),
                "total_combinations": len(rows) * 28, "method": "float64 stable softmax of saved original scores, q/(q+1e-8)",
                "per_patient_worst_gate_sum": dist(worst_gate), "per_patient_minimum_mass": dist(worst_mass)}}
    result["per_patient_mask_enumeration"] = enumeration
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_patient_mask_enumeration"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
