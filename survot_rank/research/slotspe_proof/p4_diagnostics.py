"""P4 anonymous indices versus matched reading consistency; no semantic labels.

Replay retained native modules/states solely to observe last attention weights.
Hungarian assignments are descriptive correspondences, never tissue identities.
Import/plan does not load torch, arrays or checkpoints.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

from .p1_diagnostics import _read, _sha, _write


def protocol():
    return {"schema": "slotspe_p4_protocol_v1", "repeats": 20, "reference_repeat": 0,
        "scope": "same 71 patients and same native checkpoint; within-patient repeat matching only",
        "attention_cost": "mean over heads of token-distribution total variation: sum(abs(p-q))/2",
        "vector_cost": "1-cosine of native final slot vectors; separate matching, not semantic truth",
        "matching": "scipy linear_sum_assignment on all 8 slots; no outcomes or keep used",
        "second_best": "minimum assignment cost when excluding each chosen edge, divided by 8; gap is descriptive",
        "near_tie_mean_cost_gap": 1e-7, "near_tie_meaning": "numerical reporting cutoff, not a biological equivalence threshold",
        "top5": "descending head-mean attention with token-index tie break; overlap fraction and mass above uniform reported together",
        "control": "known reverse permutation of reference slots/attention: matched cost must be zero",
        "native_replay_tolerance": 1e-6, "native_update_tolerance": 1e-6,
        "new_optimizer_steps": 0, "semantic_annotations_available": False,
        "cross_patient_semantic_drift_tested": False, "patient_outcomes_used_for_matching": False,
        "interpretation": "Index rearrangement or a low-cost match does not identify tissue semantics. High reading overlap may make identity assignment ambiguous. No semantic harm or correction efficacy inferred."}


def plan(root):
    root = Path(root).resolve()
    required = ("receipt.json", "observations.json", "native.npz", "encoded_inputs.npz", "native_rng_states.npz")
    missing = [n for n in required if not (root / n).is_file()]
    return {"ready": not missing, "missing": missing, "p3_root": str(root), "default": "plan_only", "protocol": protocol()}


def costs(attention_a, attention_b, vectors_a, vectors_b):
    import numpy as np
    aa, ab = np.asarray(attention_a, dtype=float), np.asarray(attention_b, dtype=float)
    va, vb = np.asarray(vectors_a, dtype=float), np.asarray(vectors_b, dtype=float)
    if aa.shape != ab.shape or aa.ndim != 3 or aa.shape[1] != len(va) or va.shape != vb.shape:
        raise ValueError("Attention/vector shape differs")
    if not all(np.isfinite(v).all() for v in (aa, ab, va, vb)) or (aa < 0).any() or (ab < 0).any():
        raise ValueError("Invalid attention/vector values")
    if not np.allclose(aa.sum(-1), 1, atol=2e-6, rtol=0) or not np.allclose(ab.sum(-1), 1, atol=2e-6, rtol=0):
        raise ValueError("Attention normalization differs")
    na, nb = np.linalg.norm(va, axis=-1), np.linalg.norm(vb, axis=-1)
    if (na <= 0).any() or (nb <= 0).any():
        raise ValueError("Zero vectors cannot be cosine-matched")
    tv = np.abs(aa[:, :, None, :] - ab[:, None, :, :]).sum(-1).mean(0) / 2
    angular = np.maximum(0., 1 - (va/na[:, None]) @ (vb/nb[:, None]).T)
    return tv, angular


def assignment(cost):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1] or cost.shape[0] < 2 or not np.isfinite(cost).all() or (cost < 0).any():
        raise ValueError("Expected a finite nonnegative square cost")
    rows, cols = linear_sum_assignment(cost)
    best = float(cost[rows, cols].sum())
    alternatives = []
    for r, c in zip(rows, cols):
        other = cost.copy()
        other[r, c] = np.inf
        r2, c2 = linear_sum_assignment(other)
        alternatives.append(float(cost[r2, c2].sum()))
    gap = max(0., (min(alternatives)-best)/len(rows))
    return {"columns_for_reference_rows": cols.tolist(), "diagonal_mean_cost": float(np.trace(cost)/len(rows)),
        "matched_mean_cost": best/len(rows), "second_best_mean_cost_gap": gap,
        "near_tie": gap <= protocol()["near_tie_mean_cost_gap"],
        "indices_reassigned": int(np.count_nonzero(cols != rows))}


def top_reading(attention_a, attention_b, columns):
    import numpy as np
    aa, ab = attention_a.astype(float).mean(0), attention_b.astype(float).mean(0)[columns]
    k = min(5, aa.shape[-1])
    def top(x):
        return np.lexsort((np.arange(len(x)), -x))[:k]
    a, b = [top(v) for v in aa], [top(v) for v in ab]
    overlap = [len(set(x) & set(y))/k for x, y in zip(a, b)]
    return {"top5_overlap_fraction_mean": float(np.mean(overlap)),
        "reference_top5_mass_above_uniform_mean": float(np.mean([v[ix].sum()-k/len(v) for v, ix in zip(aa, a)])),
        "current_top5_mass_above_uniform_mean": float(np.mean([v[ix].sum()-k/len(v) for v, ix in zip(ab, b)]))}


def patient_analysis(attention, vectors):
    import numpy as np
    if attention.ndim != 4 or len(attention) != 20 or vectors.shape != (20, 8, 256) or attention.shape[2] != 8 or attention.shape[1] < 1 or attention.shape[3] < 1:
        raise ValueError("Expected native twenty-repeat, eight-slot observations")
    records, raw = [], {"attention_costs": [], "vector_costs": []}
    for r in range(1, 20):
        tv, angular = costs(attention[0], attention[r], vectors[0], vectors[r])
        am, vm = assignment(tv), assignment(angular)
        records.append({"repeat": r, "attention_match": am, "vector_match": vm,
            "attention_vector_assignment_agreement_fraction": float(np.mean(np.array(am["columns_for_reference_rows"]) == np.array(vm["columns_for_reference_rows"]))),
            "reading": top_reading(attention[0], attention[r], am["columns_for_reference_rows"])})
        raw["attention_costs"].append(tv)
        raw["vector_costs"].append(angular)
    rev = np.arange(7, -1, -1)
    tv, angular = costs(attention[0], attention[0][:, rev], vectors[0], vectors[0][rev])
    reverse = {"attention_match": assignment(tv), "vector_match": assignment(angular),
        "known_reverse_attention_cost": float(tv[np.arange(8), rev].mean())}
    if reverse["attention_match"]["matched_mean_cost"] != 0 or reverse["known_reverse_attention_cost"] != 0:
        raise ValueError("Known permutation matching control failed")
    flatness = np.abs(attention.astype(float)-1/attention.shape[-1]).sum(-1)/2
    return {"repeats_against_reference": records, "reverse_permutation_control": reverse,
        "mean_per_head_TV_from_uniform": {"min": float(flatness.min()), "median": float(np.median(flatness)), "max": float(flatness.max())}}, {
            k: np.stack(v) for k, v in raw.items()}


def summarize(records):
    import numpy as np
    result = {}
    def dist(values):
        return {"min": float(np.min(values)), "median": float(np.median(values)), "max": float(np.max(values))}
    for m in ("wsi", "omic"):
        pairs = [v for r in records for v in r[m]["repeats_against_reference"]]
        result[m] = {"patients": len(records), "paired_repeats": len(pairs), "paired_repeats_are_independent_patients": False,
            "attention_diagonal_mean_TV": dist([v["attention_match"]["diagonal_mean_cost"] for v in pairs]),
            "attention_matched_mean_TV": dist([v["attention_match"]["matched_mean_cost"] for v in pairs]),
            "attention_second_best_mean_gap": dist([v["attention_match"]["second_best_mean_cost_gap"] for v in pairs]),
            "attention_near_tie_pairs": sum(v["attention_match"]["near_tie"] for v in pairs),
            "attention_reassigned_slots_fraction": sum(v["attention_match"]["indices_reassigned"] for v in pairs)/(8*len(pairs)),
            "vector_diagonal_mean_angular_cost": dist([v["vector_match"]["diagonal_mean_cost"] for v in pairs]),
            "vector_matched_mean_angular_cost": dist([v["vector_match"]["matched_mean_cost"] for v in pairs]),
            "attention_vector_assignment_agreement": dist([v["attention_vector_assignment_agreement_fraction"] for v in pairs]),
            "matched_top5_overlap_fraction": dist([v["reading"]["top5_overlap_fraction_mean"] for v in pairs]),
            "top5_mass_above_uniform": dist([v["reading"]["current_top5_mass_above_uniform_mean"] for v in pairs]),
            "all_known_reverse_attention_costs_zero": all(r[m]["reverse_permutation_control"]["attention_match"]["matched_mean_cost"] == 0 for r in records)}
    return result


def run(p3_root, output, device="cuda:0"):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from torch.torch_version import TorchVersion
    from .model import build_model
    from .native_audit import model_purity, _state_digest
    from .p1_diagnostics import NativeSlotObserver
    from .p3_diagnostics import audit as audit_p3, replay_modules, rng_state, rng_digest
    p3_root, output = Path(p3_root).resolve(), Path(output).resolve()
    if output.exists() or output == p3_root or p3_root in output.parents:
        raise ValueError("Fresh output outside original evidence required")
    probe_source = p3_root.parent / "probe_source_v1.py"
    audit_p3(p3_root, probe_source)
    p3 = _read(p3_root / "observations.json")
    watched = {}
    for path, digest in p3["source_inputs_sha256"].items():
        if _sha(path) == digest:
            watched[path] = digest
        elif Path(path).name == "p3_diagnostics.py" and _sha(probe_source) == digest:
            watched[str(probe_source)] = digest
            watched[path] = _sha(path)
        else:
            raise ValueError("Original source identity differs")
    watched.update({str(p): _sha(p) for p in p3_root.iterdir() if p.is_file()})
    watched[str(Path(__file__).resolve())] = _sha(__file__)
    p1_root = Path(p3["p1_root"])
    p1 = _read(p1_root / "observations.json")
    provenance = _read(p3_root / "provenance.json")
    if device != p3["device"] or device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Use same available device as original replay")
    checkpoint, = [Path(p) for p in watched if Path(p).name == "last.pt"]
    manifest_path, = [Path(p) for p in watched if Path(p).name == "manifest.json"]
    source = _read(manifest_path)["source"]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with torch.serialization.safe_globals([TorchVersion]):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if saved["arm"] != "original" or saved["seed"] != p3["seed"]:
        raise ValueError("Native checkpoint differs")
    model = build_model(source, SimpleNamespace(**p1["model_purity"]["model_args"]), "original").to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    del saved
    model.eval()
    purity, state_before = model_purity(model, source), _state_digest(model)
    if not all(purity["checks"].values()):
        raise ValueError("Native purity failed")
    output.mkdir(parents=True)
    _write(output / "protocol.json", protocol())
    (output / "raw").mkdir()
    with np.load(p3_root / "native.npz", allow_pickle=False) as f:
        native = {k: f[k] for k in f.files}
    bs, n = p3["batch_size"], p3["patients"]
    batches = (n+bs-1)//bs
    with np.load(p3_root / "encoded_inputs.npz", allow_pickle=False) as e:
        encoded = [{m: e[f"b{b:02d}_{m}"] for m in ("wsi", "omic")} for b in range(batches)]
    with np.load(p3_root / "native_rng_states.npz", allow_pickle=False) as f:
        cuda_n = len(p3["controlled_rng_before"]["cuda"])
        states = [[{m: {"cpu": torch.from_numpy(f[f"r{r:02d}_b{b:02d}_{m}_cpu"]),
            "cuda": [torch.from_numpy(f[f"r{r:02d}_b{b:02d}_{m}_cuda{d}"]) for d in range(cuda_n)]}
            for m in ("slot_wsi", "slot_omic", "decoder_wsi", "decoder_omic")} for b in range(batches)] for r in range(20)]
    observer = NativeSlotObserver(model)
    attention = {m: [] for m in ("wsi", "omic")}
    before_rng = rng_digest(rng_state())
    max_update_error, max_logit_error, max_vector_error, max_p1_attention_error = 0., 0., 0., 0.
    try:
        with torch.no_grad():
            for r in range(20):
                observed = {m: [] for m in ("wsi", "omic")}
                for b, e in enumerate(encoded):
                    start, stop = b*bs, min(n, (b+1)*bs)
                    logits = replay_modules(model, e, states, r, b, "native", device).cpu().numpy()
                    max_logit_error = max(max_logit_error, float(np.abs(logits-native["logits"][r, start:stop]).max()))
                    max_update_error = max(max_update_error, observer.maximum_update_error())
                    for m in ("wsi", "omic"):
                        value = observer.data[m]
                        observed[m].append(value["last_attention"].numpy())
                        max_vector_error = max(max_vector_error, float(np.abs(value["slots"].numpy()-native[m+"_final"][r, start:stop]).max()))
                        if not np.array_equal(value["keep"].numpy(), native[m+"_keep"][r, start:stop]):
                            raise RuntimeError("Native keep replay differs")
                        if r == 0:
                            for local, row in enumerate(p1["records"][start:stop]):
                                with np.load(p1_root / row["raw_path"], allow_pickle=False) as raw:
                                    max_p1_attention_error = max(max_p1_attention_error, float(np.abs(value["last_attention"][local].numpy()-raw[m+"_last_attention_by_head"]).max()))
                for m in attention:
                    attention[m].append(np.concatenate(observed[m]))
                if (r+1) % 5 == 0:
                    print(f"native attention replay {r+1}/20", flush=True)
    finally:
        observer.close()
    after_rng = rng_digest(rng_state())
    if max(max_update_error, max_logit_error, max_vector_error, max_p1_attention_error) > 1e-6 or before_rng != after_rng or state_before != _state_digest(model):
        raise RuntimeError("Native observer/replay/RNG/state control failed")
    attention = {m: np.stack(v) for m, v in attention.items()}
    records = []
    for i, p in enumerate(provenance):
        raw, record = {}, {"patient_id": p["patient_id"]}
        for m in attention:
            a = attention[m][:, i]
            record[m], extra = patient_analysis(a, native[m+"_final"][:, i])
            raw[m+"_attention"] = a
            raw.update({m+"_"+k: v for k, v in extra.items()})
        name = "raw/" + __import__("hashlib").sha256(p["patient_id"].encode()).hexdigest()+".npz"
        np.savez_compressed(output / name, **raw)
        record["raw_path"] = name
        records.append(record)
    if any(_sha(p) != digest for p, digest in watched.items()):
        raise RuntimeError("Original artifacts changed")
    report = {"schema": "slotspe_p4_observations_v1", "p3_root": str(p3_root), "p3_probe_source": str(probe_source),
        "scope": "historical_local_native_within_patient_slot_matching", "patients": n, "seed": p3["seed"], "fold": p3["fold"],
        "device": device, "source_inputs_sha256": watched, "model_purity": purity,
        "replay_max_logit_error": max_logit_error, "replay_max_final_slot_error": max_vector_error,
        "native_attention_update_max_error": max_update_error, "reference_vs_P1_attention_max_error": max_p1_attention_error,
        "external_rng_before": before_rng, "external_rng_after": after_rng, "weights_sources_unchanged": True, "hooks_removed": True,
        "new_optimizer_steps": 0, "new_native_module_replays": 20, "new_full_patient_encoder_passes": 0,
        "server_SPE09_evidence": False, "semantic_annotations_available": False, "semantic_drift_or_harm_proved": False,
        "cross_patient_semantics_tested": False, "summaries": summarize(records), "records": records,
        "interpretation": protocol()["interpretation"]}
    _write(output / "observations.json", report)
    _write(output / "receipt.json", {"schema": "slotspe_p4_receipt_v1", "probe_source_sha256": _sha(__file__),
        "outputs": {p.relative_to(output).as_posix(): _sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    return {"patients": n, "replay_error": max_logit_error, "summaries": report["summaries"]}


def audit(output):
    import numpy as np
    from .p3_diagnostics import audit as audit_p3
    output = Path(output).resolve()
    receipt = _read(output / "receipt.json")
    if not {"protocol.json", "observations.json"}.issubset(receipt["outputs"]):
        raise ValueError("Missing report/protocol")
    for name, digest in receipt["outputs"].items():
        p = (output / name).resolve()
        if output not in p.parents or _sha(p) != digest:
            raise ValueError("Raw hash/path differs")
    report = _read(output / "observations.json")
    if _read(output / "protocol.json") != protocol() or report["schema"] != "slotspe_p4_observations_v1" or report["scope"] != "historical_local_native_within_patient_slot_matching":
        raise ValueError("Scope/protocol differs")
    if report["new_optimizer_steps"] != 0 or any(report[k] is not False for k in ("server_SPE09_evidence", "semantic_annotations_available", "semantic_drift_or_harm_proved", "cross_patient_semantics_tested")):
        raise ValueError("Unsupported execution or semantic conclusion")
    p3_root = Path(report["p3_root"])
    audit_p3(p3_root, Path(report["p3_probe_source"]))
    p3 = _read(p3_root / "observations.json")
    for k in ("patients", "seed", "fold", "device"):
        if report[k] != p3[k]:
            raise ValueError("P3 identity differs")
    if report["new_native_module_replays"] != 20 or report["new_full_patient_encoder_passes"] != 0:
        raise ValueError("Work counts differ")
    for key in ("external_rng_before", "external_rng_after"):
        state = report[key]
        if set(state) != {"cpu", "cuda"} or not isinstance(state["cuda"], list) or report["device"].startswith("cuda") and not state["cuda"]:
            raise ValueError("RNG schema differs")
        if any(not isinstance(v, str) or re.fullmatch("[0-9a-f]{64}", v) is None for v in [state["cpu"], *state["cuda"]]):
            raise ValueError("RNG digest invalid")
    for p, digest in report["source_inputs_sha256"].items():
        if _sha(p) != digest:
            raise ValueError("Original input/source changed")
    required = {"native_model_class", "native_forward", "native_slot_classes", "native_decoder_classes", "no_repaired_eval_buffers"}
    if not required.issubset(report["model_purity"]["checks"]) or any(report["model_purity"]["checks"][k] is not True for k in required):
        raise ValueError("Native purity failed")
    provenance = _read(p3_root / "provenance.json")
    if [r["patient_id"] for r in report["records"]] != [p["patient_id"] for p in provenance] or len(provenance) != report["patients"]:
        raise ValueError("Patient order/count differs")
    with np.load(p3_root / "native.npz", allow_pickle=False) as f:
        vectors = {m: f[m+"_final"] for m in ("wsi", "omic")}
    calculated = []
    for i, record in enumerate(report["records"]):
        if record["raw_path"] not in receipt["outputs"]:
            raise ValueError("Raw record missing from receipt")
        entry = {"patient_id": record["patient_id"], "raw_path": record["raw_path"]}
        with np.load(output / record["raw_path"], allow_pickle=False) as raw:
            expected_keys = {m+s for m in ("wsi", "omic") for s in ("_attention", "_attention_costs", "_vector_costs")}
            if set(raw.files) != expected_keys:
                raise ValueError("Raw keys differ")
            for m in vectors:
                entry[m], extra = patient_analysis(raw[m+"_attention"], vectors[m][:, i])
                for k, value in extra.items():
                    if not np.array_equal(value, raw[m+"_"+k]):
                        raise ValueError("Raw pair cost differs")
        if entry != record:
            raise ValueError("Raw matching statistics differ")
        calculated.append(entry)
    if summarize(calculated) != report["summaries"]:
        raise ValueError("Summary differs")
    for k in ("replay_max_logit_error", "replay_max_final_slot_error", "native_attention_update_max_error", "reference_vs_P1_attention_max_error"):
        if not isinstance(report[k], (int, float)) or not np.isfinite(report[k]) or not 0 <= report[k] <= 1e-6:
            raise ValueError("Native replay control failed")
    if report["external_rng_before"] != report["external_rng_after"] or not report["external_rng_before"].get("cpu") or report["weights_sources_unchanged"] is not True or report["hooks_removed"] is not True:
        raise ValueError("State/RNG preservation failed")
    return {"status": "accepted_within_patient_matching_observations", "patients": report["patients"], "paired_repeats_per_patient": 19,
        "biological_semantic_drift_proved": False, "server_raw_received": False}


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p3-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true")
    group.add_argument("--audit", action="store_true")
    args = parser.parse_args(argv)
    if args.audit:
        if args.output is None:
            parser.error("--audit requires --output")
        result = audit(args.output)
    else:
        if args.p3_root is None or args.run and args.output is None:
            parser.error("requires --p3-root; --run also requires fresh --output")
        result = run(args.p3_root, args.output, args.device) if args.run else plan(args.p3_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
