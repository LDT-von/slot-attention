"""Conditional P2 probes of retained native final slots; no training or repair.

The original selector's recorded hard masks are fixed. These probes test the
downstream use of slot differences, not whether those differences are useful.
Plan/import never loads torch, weights or patient tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .p1_diagnostics import _read, _sha, _write, audit_export as audit_p1

ARMS = ("identity", "permute_wsi", "permute_omic", "permute_both",
        "wsi_closest_pair", "omic_closest_pair", "wsi_selected_pair", "omic_selected_pair",
        "wsi_all_mean", "omic_all_mean", "both_all_mean")
CONTROLS = ARMS[:4]


def protocol():
    return {"schema": "slotspe_p2_protocol_v1", "arms": list(ARMS),
        "input": "P1 audited native final slots and original hard keep masks, identical patient order/batch size",
        "pair_selection": "largest mean of per-head attention overlap sum(min(p,q)); lexicographic tie break; no outcomes used",
        "closest_pair": "replace both members by their arithmetic mean, preserving modal slot count and sum",
        "selected_pair": "replace the two originally selected slots by their arithmetic mean; keep unchanged",
        "all_mean": "replace all 8 slots by their modal arithmetic mean repeated 8 times; keep unchanged",
        "permutation": "reverse slot order and corresponding keep mask together",
        "readout": "original cross_attention, self_attention_wsi/omic and to_logits in eval; no decoder re-selection",
        "baseline_tolerance": 1e-6, "control_tolerance": 1e-6,
        "numerical_logit_threshold": 1e-5, "numerical_risk_threshold": 1e-6,
        "threshold_meaning": "numerical sensitivity only, not a clinical/harm/redundancy threshold",
        "risk": "negative sum of cumprod(1-sigmoid(logits)) over 4 bins, calculated in float64",
        "outcomes_used": False, "new_optimizer_steps": 0,
        "interpretation": "Artificial latent intervention with frozen keep, possibly outside training distribution; no correction efficacy or biological fragmentation conclusion."}


def plan(observations):
    root = Path(observations).resolve()
    missing = [n for n in ("receipt.json", "observations.json", "patient_provenance.json") if not (root / n).is_file()]
    return {"ready": not missing, "missing": missing, "observations": str(root),
        "default_action": "plan_only", "training_runs": 0, "protocol": protocol()}


def pair_choices(arrays, modality):
    import numpy as np
    attention = np.asarray(arrays[modality + "_last_attention_by_head"], dtype=np.float64)
    candidates = [(float(np.minimum(attention[:, a], attention[:, b]).sum(-1).mean()), a, b)
                  for a in range(attention.shape[1]) for b in range(a + 1, attention.shape[1])]
    overlap, a, b = min(candidates, key=lambda x: (-x[0], x[1], x[2]))
    selected = np.flatnonzero(arrays[modality + "_keep"] > .5).tolist()
    if len(selected) != 2:
        raise ValueError("This protocol requires exactly two selected slots")
    return {"closest_pair": [a, b], "closest_pair_mean_head_overlap": overlap, "selected_pair": selected,
            "closest_pair_selected_members": sum(i in selected for i in (a, b))}


def transform(arrays, arm):
    """Deterministic final-slot intervention; never samples or reads outcomes."""
    import numpy as np
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    result = {m + suffix: np.array(arrays[m + suffix], copy=True)
              for m in ("wsi", "omic") for suffix in ("_slots", "_keep")}
    for modality in ("wsi", "omic"):
        slots, keep = result[modality + "_slots"], result[modality + "_keep"]
        if arm in ("permute_" + modality, "permute_both"):
            result[modality + "_slots"] = slots[::-1].copy()
            result[modality + "_keep"] = keep[::-1].copy()
        elif arm in (modality + "_all_mean", "both_all_mean"):
            slots[:] = slots.mean(0, dtype=np.float64).astype(slots.dtype)
        elif arm in (modality + "_closest_pair", modality + "_selected_pair"):
            pair = pair_choices(arrays, modality)[arm[len(modality) + 1:]]
            mean = slots[pair].mean(0, dtype=np.float64).astype(slots.dtype)
            slots[pair] = mean
    return result


def native_readout(model, wsi, omic, wsi_keep, omic_keep):
    """Verbatim native downstream formula; all slots still enter cross attention."""
    import torch
    cross = model.cross_attention(wsi, omic).mean(dim=1)
    wi = model.self_attention_wsi(wsi, mask=wsi_keep.bool()).mean(dim=1)
    oi = model.self_attention_omic(omic, mask=omic_keep.bool()).mean(dim=1)
    return model.to_logits(torch.cat([cross, wi, oi], dim=1))


def perturbation_statistics(original, altered):
    import numpy as np
    result = {}
    for m in ("wsi", "omic"):
        before, after = original[m+"_slots"].astype(float), altered[m+"_slots"].astype(float)
        delta = after - before
        result[m] = {"slot_max_abs_change": float(np.abs(delta).max()),
            "relative_frobenius_change": float(np.linalg.norm(delta) / max(np.linalg.norm(before), 1e-30)),
            "modal_mean_max_abs_change": float(np.abs(delta.mean(0)).max()),
            "original_centered_energy_fraction": float(np.linalg.norm(before-before.mean(0))**2 / max(np.linalg.norm(before)**2, 1e-30))}
    return result


def risks(logits):
    import numpy as np
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all():
        raise ValueError("Expected finite 4-bin logits")
    # Stable 1-sigmoid(x) = sigmoid(-x).
    survival_step = np.exp(-np.logaddexp(0., x))
    return -np.cumprod(survival_step, axis=1).sum(axis=1)


def summarize(baseline, logits, settings):
    import numpy as np
    baseline, logits = np.asarray(baseline, dtype=float), np.asarray(logits, dtype=float)
    if baseline.shape != logits.shape:
        raise ValueError("Mismatched predictions")
    original, changed = risks(baseline), risks(logits)
    ld, rd = np.abs(logits - baseline).max(1), np.abs(changed - original)
    i, j = np.triu_indices(len(original), 1)
    before, after = original[i] - original[j], changed[i] - changed[j]
    margin = settings["numerical_risk_threshold"]
    stable = (np.abs(before) > margin) & (np.abs(after) > margin)
    def stats(x):
        return {"min": float(x.min()), "median": float(np.median(x)), "max": float(x.max())}
    return {"patients": len(original), "max_abs_logit_difference": stats(ld), "absolute_risk_difference": stats(rd),
        "patients_logit_difference_above_numerical_threshold": int((ld > settings["numerical_logit_threshold"]).sum()),
        "patients_risk_difference_above_numerical_threshold": int((rd > margin).sum()),
        "risk_order_reversals": int(((before * after < 0) & stable).sum()),
        "eligible_patient_pairs_above_risk_margin_both_arms": int(stable.sum()),
        "all_patient_pairs": len(i), "original_risk_range": float(np.ptp(original))}


def run(observations, output, device="cuda:0"):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from torch.torch_version import TorchVersion
    from .model import build_model
    from .native_audit import model_purity, _state_digest
    observations, output = Path(observations).resolve(), Path(output).resolve()
    if output.exists() or output == observations or observations in output.parents:
        raise ValueError("Use a fresh output outside original observations")
    audit_p1(observations)
    p1 = _read(observations / "observations.json")
    provenance = _read(observations / "patient_provenance.json")
    watched = dict(p1["source_inputs_sha256"])
    p1_receipt = _read(observations / "receipt.json")
    watched.update({str(observations / n): digest for n, digest in p1_receipt["outputs"].items()})
    watched[str(observations / "receipt.json")] = _sha(observations / "receipt.json")
    watched[str(Path(__file__).resolve())] = _sha(__file__)
    if any(_sha(path) != digest for path, digest in watched.items()):
        raise ValueError("P1 source/input identity changed")
    manifest_paths = [Path(p) for p in watched if Path(p).name == "manifest.json"]
    checkpoint_paths = [Path(p) for p in watched if Path(p).name == "last.pt"]
    if len(manifest_paths) != 1 or len(checkpoint_paths) != 1:
        raise ValueError("Ambiguous original identity")
    manifest = _read(manifest_paths[0])
    with torch.serialization.safe_globals([TorchVersion]):
        saved = torch.load(checkpoint_paths[0], weights_only=True, map_location="cpu", mmap=True)
    if saved["arm"] != "original" or saved["seed"] != p1["seed"] or saved["optimizer_steps"] != p1["prior_optimizer_steps"] or saved["checkpoint_selection"] != "fixed_final_epoch":
        raise ValueError("Trained native checkpoint differs")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU unavailable; no silent fallback")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = build_model(manifest["source"], SimpleNamespace(**p1["model_purity"]["model_args"]), "original").to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    del saved
    model.eval()
    purity = model_purity(model, manifest["source"])
    if not all(purity["checks"].values()):
        raise ValueError("Native purity failed")
    before_state = _state_digest(model)
    def rng():
        return {"cpu": hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
            "cuda": [hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}
    before_rng = rng()
    arrays = []
    for row in p1["records"]:
        with np.load(observations / row["raw_path"], allow_pickle=False) as raw:
            arrays.append({k: raw[k] for k in raw.files})
    ids = [r["patient_id"] for r in p1["records"]]
    if ids != [r["patient_id"] for r in provenance["patients"]] or len(set(ids)) != len(ids):
        raise ValueError("Patient alignment differs")
    bs = _read(manifest_paths[0].parent / "training_protocol.json")["batch_size"]
    output.mkdir(parents=True)
    _write(output / "protocol.json", protocol())  # sealed before the first probe
    (output / "raw").mkdir()
    all_logits = {}
    mean_errors = {}
    with torch.no_grad():
        for arm in ARMS:
            altered = [transform(a, arm) for a in arrays]
            mean_errors[arm] = max(float(np.max(np.abs(a[m+"_slots"].astype(float).mean(0) - v[m+"_slots"].astype(float).mean(0))))
                                   for a, v in zip(arrays, altered) for m in ("wsi", "omic"))
            pred = []
            for start in range(0, len(arrays), bs):
                batch = altered[start:start+bs]
                tensors = [torch.from_numpy(np.stack([a[key] for a in batch])).to(device)
                           for key in ("wsi_slots", "omic_slots", "wsi_keep", "omic_keep")]
                pred.append(native_readout(model, *tensors).cpu().numpy())
            all_logits[arm] = np.concatenate(pred)
            if not np.isfinite(all_logits[arm]).all() or mean_errors[arm] > 1e-6:
                raise RuntimeError("Nonfinite output or modal mean conservation failed")
            if arm == "identity" and float(np.max(np.abs(all_logits[arm].astype(float) - np.asarray([r["logits"] for r in p1["records"]])))) > 1e-6:
                raise RuntimeError("Native downstream replay failed before interventions; retain attempt")
        baseline = all_logits["identity"]
        replay_error = float(np.max(np.abs(baseline.astype(float) - np.asarray([r["logits"] for r in p1["records"]]))))
        summaries = {arm: summarize(baseline, logits, protocol()) for arm, logits in all_logits.items()}
        control_error = max(summaries[a]["max_abs_logit_difference"]["max"] for a in CONTROLS)
        if replay_error > 1e-6 or control_error > 1e-6 or _state_digest(model) != before_state or before_rng != rng():
            raise RuntimeError("Native replay/permutation/state/RNG control failed; retain attempt")
    records = []
    for i, (row, original) in enumerate(zip(p1["records"], arrays)):
        raw = {m+s: original[m+s] for m in ("wsi", "omic") for s in ("_slots", "_keep")}
        for arm in ARMS:
            raw.update({arm + "__" + k: v for k, v in transform(original, arm).items()})
            raw[arm + "__logits"] = all_logits[arm][i]
        name = "raw/" + hashlib.sha256(row["patient_id"].encode()).hexdigest() + ".npz"
        np.savez_compressed(output / name, **raw)
        records.append({"patient_id": row["patient_id"], "p1_raw_path": row["raw_path"], "raw_path": name,
            "pair_choices": {m: pair_choices(original, m) for m in ("wsi", "omic")},
            "perturbations": {a: perturbation_statistics(original, transform(original, a)) for a in ARMS}})
    if any(_sha(p) != digest for p, digest in watched.items()):
        raise RuntimeError("Original files changed during read-only probes")
    report = {"schema": "slotspe_p2_observations_v1", "scope": "historical_local_BLCA_native_final_slot_conditional_interventions",
        "server_SPE09_evidence": False, "new_optimizer_steps": 0, "fold": p1["fold"], "seed": p1["seed"],
        "previous_epochs": p1["epochs"], "patients": len(ids), "batch_size": bs, "device": device, "torch": str(torch.__version__),
        "p1_root": str(observations), "model_purity": purity, "source_inputs_sha256": watched,
        "native_downstream_replay_max_error": replay_error, "permutation_max_logit_error": control_error,
        "model_state_unchanged": True, "rng_before": before_rng, "rng_after": rng(), "original_inputs_unchanged": True,
        "max_modal_mean_error_by_arm": mean_errors, "summaries": summaries, "records": records,
        "conclusion_boundary": protocol()["interpretation"], "harm_or_correction_efficacy_proved": False}
    _write(output / "observations.json", report)
    _write(output / "receipt.json", {"schema": "slotspe_p2_receipt_v1",
        "outputs": {p.relative_to(output).as_posix(): _sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "probe_source_sha256": _sha(__file__)})
    return {k: report[k] for k in ("patients", "native_downstream_replay_max_error", "permutation_max_logit_error", "summaries")}


def audit(output):
    import numpy as np
    output = Path(output).resolve()
    receipt = _read(output / "receipt.json")
    if not {"protocol.json", "observations.json"}.issubset(receipt["outputs"]):
        raise ValueError("Missing report/protocol receipt")
    for name, digest in receipt["outputs"].items():
        path = (output / name).resolve()
        if output not in path.parents or _sha(path) != digest:
            raise ValueError("Output hash/path differs")
    settings, report = _read(output / "protocol.json"), _read(output / "observations.json")
    if settings != protocol() or report["schema"] != "slotspe_p2_observations_v1" or report["scope"] != "historical_local_BLCA_native_final_slot_conditional_interventions" or report["server_SPE09_evidence"] is not False or report["new_optimizer_steps"] != 0 or report["harm_or_correction_efficacy_proved"] is not False:
        raise ValueError("Protocol/scope differs")
    p1_root = Path(report["p1_root"])
    audit_p1(p1_root)
    p1 = _read(p1_root / "observations.json")
    if [r["patient_id"] for r in report["records"]] != [r["patient_id"] for r in p1["records"]] or report["patients"] != len(p1["records"]):
        raise ValueError("Patient identity/count differs")
    all_logits, mean_errors = {a: [] for a in ARMS}, {a: 0. for a in ARMS}
    for record, p1_record in zip(report["records"], p1["records"]):
        if record["raw_path"] not in receipt["outputs"] or record["p1_raw_path"] != p1_record["raw_path"]:
            raise ValueError("Unreceipted raw/patient path")
        with np.load(p1_root / p1_record["raw_path"], allow_pickle=False) as raw:
            original = {k: raw[k] for k in raw.files}
        if record["pair_choices"] != {m: pair_choices(original, m) for m in ("wsi", "omic")}:
            raise ValueError("Pair choice differs")
        if record["perturbations"] != {a: perturbation_statistics(original, transform(original, a)) for a in ARMS}:
            raise ValueError("Perturbation size differs")
        with np.load(output / record["raw_path"], allow_pickle=False) as raw:
            for m in ("wsi", "omic"):
                for suffix in ("_slots", "_keep"):
                    if not np.array_equal(raw[m+suffix], original[m+suffix]):
                        raise ValueError("Original slots/masks differ")
            for arm in ARMS:
                for key, value in transform(original, arm).items():
                    if not np.array_equal(raw[arm+"__"+key], value):
                        raise ValueError("Intervention differs from protocol")
                for m in ("wsi", "omic"):
                    err = float(np.max(np.abs(raw[arm+"__"+m+"_slots"].astype(float).mean(0) - original[m+"_slots"].astype(float).mean(0))))
                    mean_errors[arm] = max(mean_errors[arm], err)
                all_logits[arm].append(raw[arm+"__logits"])
    baseline = np.stack(all_logits["identity"])
    replay_error = float(np.max(np.abs(baseline - np.array([r["logits"] for r in p1["records"]]))))
    summaries = {a: summarize(baseline, np.stack(v), settings) for a, v in all_logits.items()}
    control_error = max(summaries[a]["max_abs_logit_difference"]["max"] for a in CONTROLS)
    if summaries != report["summaries"] or mean_errors != report["max_modal_mean_error_by_arm"] or max(mean_errors.values()) > 1e-6:
        raise ValueError("Raw statistics/mean conservation differs")
    if replay_error != report["native_downstream_replay_max_error"] or control_error != report["permutation_max_logit_error"] or replay_error > 1e-6 or control_error > 1e-6:
        raise ValueError("Native/permutation control differs")
    required = {"native_model_class", "native_forward", "native_slot_classes", "native_decoder_classes", "no_repaired_eval_buffers"}
    checks = report["model_purity"]["checks"]
    if report["model_state_unchanged"] is not True or report["rng_before"] != report["rng_after"] or not required.issubset(checks) or any(checks[k] is not True for k in required):
        raise ValueError("State/RNG/purity failed")
    return {"status": "accepted_conditional_raw_probes", "patients": report["patients"], "arms": len(ARMS),
        "server_SPE09_evidence": False, "harm_or_correction_efficacy_proved": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path)
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
        if args.observations is None:
            parser.error("--observations is required")
        if args.run and args.output is None:
            parser.error("--run requires fresh --output")
        result = run(args.observations, args.output, args.device) if args.run else plan(args.observations)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
