"""P3: native eval replay and separate slot/Gumbel draw controls; no training.

Original classes/parameters/forwards remain untouched. Native eval runs first;
controls replay the same original modules on native encoded inputs, with RNG
states taken at the original module boundaries. No alternative initializer or
selector is installed. Import/plan opens no tensors.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
from pathlib import Path
import random
import re

from .p1_diagnostics import _read, _sha, _write, audit_export as audit_p1

ARMS = ("native", "fix_slot_draws", "fix_gumbel_draws", "fix_both_draws")
MODULES = ("slot_wsi", "slot_omic", "decoder_wsi", "decoder_omic")


def protocol():
    return {"schema": "slotspe_p3_protocol_v1", "repeats": 20, "arms": list(ARMS),
        "reference_repeat": 0, "native": "same original checkpoint, cached patches, batch/order and historical 20 eval seeds",
        "controls": "Replay original slot/decoder/readout modules on identical native encoded inputs; restore natural module-entry RNG states within fork_rng",
        "fix_slot_draws": "Both slot modules use repeat0 module-entry RNG; decoder draws use current native repeat RNG",
        "fix_gumbel_draws": "Both decoder modules use repeat0 module-entry RNG; slots use current native repeat RNG",
        "fix_both_draws": "All four modules use repeat0 module-entry RNG",
        "fixed_gumbel_is_fixed_keep": False, "new_optimizer_steps": 0,
        "native_replay_tolerance": 1e-6, "deterministic_control_tolerance": 1e-6,
        "risk_numerical_threshold": 1e-6, "logit_numerical_threshold": 1e-5,
        "risk": "negative sum cumprod(1-sigmoid(logits)), float32 CPU as historical evaluation",
        "metric": "sksurv.metrics.concordance_index_censored, observed=censor<1; validation descriptive only",
        "interpretation": "Conditional computational attribution, not additive variance decomposition, biological semantic alignment, task-optimal initialization, correction efficacy, or independent test performance."}


def plan(p1_root):
    root = Path(p1_root).resolve()
    missing = [n for n in ("observations.json", "receipt.json", "patient_provenance.json") if not (root / n).is_file()]
    return {"ready": not missing, "missing": missing, "p1_root": str(root), "default": "plan_only", "protocol": protocol()}


def rng_state():
    import torch
    return {"cpu": torch.random.get_rng_state().clone(),
        "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def rng_digest(state):
    return {"cpu": hashlib.sha256(state["cpu"].numpy().tobytes()).hexdigest(),
        "cuda": [hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest() for s in state["cuda"]]}


@contextmanager
def replay_rng(state):
    import torch
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.random.set_rng_state(state["cpu"])
        if devices:
            if len(state["cuda"]) != len(devices):
                raise ValueError("CUDA RNG device count differs")
            torch.cuda.set_rng_state_all(state["cuda"])
        yield


class NativeCapture:
    """Read-only hooks; no new random draws and no output replacement."""
    def __init__(self, model):
        self.handles = []
        self.reset()
        for m in ("wsi", "omic"):
            slot = getattr(model, "slot_attention_" + m)
            decoder = getattr(model, "slot_decoder_" + m)
            def slot_pre(_module, inputs, name=m):
                self.states["slot_"+name] = rng_state()
                self.inputs[name] = inputs[0].detach().cpu().numpy().copy()
            def gru_pre(_module, inputs, name=m):
                key = name+"_initial"
                if key not in self.values:
                    b, _, d = self.inputs[name].shape
                    self.values[key] = inputs[1].detach().cpu().reshape(b, -1, d).numpy().copy()
            def slot_post(_module, _inputs, output, name=m):
                self.values[name+"_final"] = output.detach().cpu().numpy().copy()
            def decoder_pre(_module, _inputs, name=m):
                self.states["decoder_"+name] = rng_state()
            def decoder_post(_module, _inputs, output, name=m):
                self.values[name+"_keep"] = output[2].detach().cpu().numpy().copy()
            self.handles.extend((slot.register_forward_pre_hook(slot_pre), slot.gru.register_forward_pre_hook(gru_pre),
                slot.register_forward_hook(slot_post), decoder.register_forward_pre_hook(decoder_pre), decoder.register_forward_hook(decoder_post)))

    def reset(self):
        self.states, self.inputs, self.values = {}, {}, {}

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def replay_modules(model, encoded, states, repeat, batch_index, arm, device):
    import torch
    from .p2_diagnostics import native_readout
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    values = {}
    for m in ("wsi", "omic"):
        slot_repeat = 0 if arm in ("fix_slot_draws", "fix_both_draws") else repeat
        decoder_repeat = 0 if arm in ("fix_gumbel_draws", "fix_both_draws") else repeat
        with replay_rng(states[slot_repeat][batch_index]["slot_"+m]):
            values[m] = getattr(model, "slot_attention_"+m)(torch.from_numpy(encoded[m]).to(device))
        with replay_rng(states[decoder_repeat][batch_index]["decoder_"+m]):
            values[m+"_keep"] = getattr(model, "slot_decoder_"+m)(values[m])[2]
    return native_readout(model, values["wsi"], values["omic"], values["wsi_keep"], values["omic_keep"])


def array_summary(arrays, provenance):
    import numpy as np
    from sksurv.metrics import concordance_index_censored
    logits, risks = arrays["logits"], arrays["risks"]
    n = len(provenance)
    if logits.shape != (20, n, 4) or risks.shape != (20, n) or not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Nonfinite/mismatched repeat observations")
    expected_keys = {"logits", "risks"} | {m+s for m in ("wsi", "omic") for s in ("_initial", "_final", "_keep")}
    if set(arrays) != expected_keys:
        raise ValueError("Unexpected raw array keys")
    event = np.array([p["censor"] < 1 for p in provenance])
    time = np.array([p["time"] for p in provenance], dtype=float)
    cs = [float(concordance_index_censored(event, time, r)[0]) for r in risks]
    def distribution(x):
        return {"min": float(np.min(x)), "median": float(np.median(x)), "max": float(np.max(x))}
    values = {"patients": n, "eval_repeats": 20, "C_index_validation_exploratory": {
        "values": cs, "min": min(cs), "max": max(cs), "mean": float(np.mean(cs)), "std_population": float(np.std(cs))},
        "patient_risk_range": distribution(np.ptp(risks.astype(float), axis=0)),
        "patient_risk_std_population": distribution(risks.astype(float).std(axis=0)),
        "patients_risk_range_above_numerical_threshold": int((np.ptp(risks.astype(float), axis=0) > 1e-6).sum()),
        "max_logit_range": float(np.ptp(logits.astype(float), axis=0).max()),
        "patients_logit_range_above_numerical_threshold": int((np.ptp(logits.astype(float), axis=0).max(-1) > 1e-5).sum())}
    for m in ("wsi", "omic"):
        if arrays[m+"_initial"].shape != (20, n, 8, 256) or arrays[m+"_final"].shape != (20, n, 8, 256) or arrays[m+"_keep"].shape != (20, n, 8):
            raise ValueError("Native slot shape differs")
        keep = arrays[m+"_keep"]
        if not np.allclose(keep, (keep > .5).astype(float), atol=1e-6, rtol=0) or not np.all((keep > .5).sum(-1) == 2):
            raise ValueError("Native hard keep differs")
        values[m] = {"initial_slot_max_repeat_range": float(np.ptp(arrays[m+"_initial"].astype(float), axis=0).max()),
            "final_slot_max_repeat_range": float(np.ptp(arrays[m+"_final"].astype(float), axis=0).max()),
            "patients_keep_mask_changed": int(np.any(np.any(keep != keep[:1], axis=-1), axis=0).sum())}
    return values


def run(p1_root, output, device="cuda:0"):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from torch.torch_version import TorchVersion
    from .model import build_model
    from .native_audit import model_purity, _state_digest
    p1_root, output = Path(p1_root).resolve(), Path(output).resolve()
    if output.exists() or output == p1_root or p1_root in output.parents:
        raise ValueError("Fresh output outside original evidence required")
    audit_p1(p1_root)
    p1 = _read(p1_root / "observations.json")
    watched = dict(p1["source_inputs_sha256"])
    watched[str(p1_root / "receipt.json")] = _sha(p1_root / "receipt.json")
    watched[str(Path(__file__).resolve())] = _sha(__file__)
    watched[str(Path(__file__).with_name("p2_diagnostics.py"))] = _sha(Path(__file__).with_name("p2_diagnostics.py"))
    for path, digest in watched.items():
        if _sha(path) != digest:
            raise ValueError("Historical identity changed: " + path)
    manifest_path, = [Path(p) for p in watched if Path(p).name == "manifest.json"]
    checkpoint, = [Path(p) for p in watched if Path(p).name == "last.pt"]
    manifest = _read(manifest_path)
    historical = _read(checkpoint.parent / "eval_repeats.json")
    training = _read(manifest_path.parent / "training_protocol.json")
    if len(historical) != 20 or device != p1["device"] or device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Expected historical 20 repeats on same available device")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with torch.serialization.safe_globals([TorchVersion]):
        data = torch.load(manifest_path.parent / "cache/paired_blca_fold0.pt", weights_only=True, map_location="cpu", mmap=True)
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    if saved["arm"] != "original" or saved["seed"] != p1["seed"] or saved["checkpoint_selection"] != "fixed_final_epoch" or saved["optimizer_steps"] != p1["prior_optimizer_steps"]:
        raise ValueError("Trained original checkpoint differs")
    if data["patient_id"] != [p["patient_id"] for p in manifest["patients"]] or data["omic_sizes"] != saved["omic_sizes"]:
        raise ValueError("Cached order/pathway contract differs")
    ix = [i for i, p in enumerate(manifest["patients"]) if p["split"] == "val"]
    if [data["patient_id"][i] for i in ix] != [p["patient_id"] for p in p1["records"]]:
        raise ValueError("Validation coverage differs")
    provenance = [{k: manifest["patients"][i][k] for k in ("patient_id", "time", "censor", "label", "index")} for i in ix]
    for i, p in zip(ix, provenance):
        if float(data["time"][i]) != p["time"] or float(data["censor"][i]) != p["censor"] or int(data["label"][i]) != p["label"]:
            raise ValueError("Clinical identity differs")
    model = build_model(manifest["source"], SimpleNamespace(**p1["model_purity"]["model_args"]), "original").to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    del saved
    model.eval()
    purity, before = model_purity(model, manifest["source"]), _state_digest(model)
    if not all(purity["checks"].values()):
        raise ValueError("Native model purity failed")
    output.mkdir(parents=True)
    _write(output / "protocol.json", protocol())
    _write(output / "provenance.json", provenance)
    bs = training["batch_size"]
    encoded, states, native = [], [], []
    capture = NativeCapture(model)
    def seed(value):
        random.seed(value)
        np.random.seed(value % 2**32)
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
    def stack_batches(batches):
        return {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
    def stack_repeats(repeats):
        return {k: np.stack([r[k] for r in repeats]) for k in repeats[0]}
    def add_risks(arrays):
        logits = torch.from_numpy(arrays["logits"])
        arrays["risks"] = (-(1-logits.sigmoid()).cumprod(-1).sum(-1)).numpy()
        return arrays
    try:
        with torch.no_grad():
            for r, reference in enumerate(historical):
                seed(reference["eval_seed"])
                batches, batch_states = [], []
                for b, start in enumerate(range(0, len(ix), bs)):
                    selected = torch.tensor(ix[start:start+bs])
                    sample = {"x_wsi": data["wsi"][selected].to(device), "omic_missing": False}
                    sample.update({f"x_omic{j+1}": v[selected].to(device) for j, v in enumerate(data["omics"])})
                    capture.reset()
                    logits = model(**sample)[0].cpu().numpy()
                    if r == 0:
                        encoded.append(capture.inputs)
                    elif any(not np.array_equal(encoded[b][m], capture.inputs[m]) for m in ("wsi", "omic")):
                        raise RuntimeError("Eval encoded input changed between repeats")
                    batch_states.append(capture.states)
                    batches.append({"logits": logits, **capture.values})
                arrays = add_risks(stack_batches(batches))
                if not np.allclose(arrays["logits"], reference["logits"], atol=1e-6, rtol=0) or not np.allclose(arrays["risks"], reference["risks"], atol=1e-6, rtol=0):
                    raise RuntimeError("Historical native prediction replay failed")
                states.append(batch_states)
                native.append(arrays)
                if (r+1) % 5 == 0:
                    print(f"native historical replay {r+1}/20", flush=True)
            native = stack_repeats(native)
            np.savez_compressed(output / "native.npz", **native)
            rng_arrays = {}
            for r, batches in enumerate(states):
                for b, modules in enumerate(batches):
                    for name, state in modules.items():
                        prefix = f"r{r:02d}_b{b:02d}_{name}_"
                        rng_arrays[prefix+"cpu"] = state["cpu"].numpy()
                        for d, value in enumerate(state["cuda"]):
                            rng_arrays[prefix+f"cuda{d}"] = value.cpu().numpy()
            np.savez_compressed(output / "native_rng_states.npz", **rng_arrays)
            np.savez_compressed(output / "encoded_inputs.npz", **{f"b{b:02d}_{m}": a for b, e in enumerate(encoded) for m, a in e.items()})
            controlled_rng_before = rng_digest(rng_state())
            summaries = {"native": array_summary(native, provenance)}
            replay_errors = []
            # Replay all 20 native draw states once through the same modular
            # path used by the controls, before attributing any differences.
            for arm in ARMS:
                repeats = []
                for r in range(20):
                    batches = []
                    for b, e in enumerate(encoded):
                        capture.reset()
                        logits = replay_modules(model, e, states, r, b, arm, device).cpu().numpy()
                        batches.append({"logits": logits, **capture.values})
                    a = add_risks(stack_batches(batches))
                    if arm == "native":
                        replay_errors.append(float(np.max(np.abs(a["logits"].astype(float)-native["logits"][r].astype(float)))))
                        if replay_errors[-1] > 1e-6 or any(not np.array_equal(a[k], native[k][r]) for k in a if k not in ("logits", "risks")):
                            raise RuntimeError("Modular/native exact draw replay failed")
                    repeats.append(a)
                if arm != "native":
                    raw = stack_repeats(repeats)
                    np.savez_compressed(output / (arm+".npz"), **raw)
                    summaries[arm] = array_summary(raw, provenance)
                print(f"module-state control complete: {arm}", flush=True)
            controlled_rng_after = rng_digest(rng_state())
    finally:
        capture.close()
    if controlled_rng_before != controlled_rng_after or _state_digest(model) != before or any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
        raise RuntimeError("Controlled replay changed external RNG/weights or left hooks")
    if any(_sha(p) != digest for p, digest in watched.items()):
        raise RuntimeError("Original artifacts changed")
    report = {"schema": "slotspe_p3_observations_v1", "scope": "historical_local_BLCA_trained_native_eval_randomness",
        "p1_root": str(p1_root), "patients": len(ix), "fold": 0, "seed": p1["seed"], "previous_epochs": p1["epochs"],
        "previous_optimizer_steps": p1["prior_optimizer_steps"], "batch_size": bs, "device": device,
        "eval_seeds": [r["eval_seed"] for r in historical], "model_purity": purity, "source_inputs_sha256": watched,
        "historical_reference": {"logits": [r["logits"] for r in historical], "risks": [r["risks"] for r in historical], "cindex": [r["cindex"] for r in historical]},
        "historical_replay_max_logit_error": float(np.abs(native["logits"].astype(float)-np.array([r["logits"] for r in historical])).max()),
        "modular_native_replay_max_logit_error": max(replay_errors), "encoded_inputs_unchanged_across_native_repeats": True,
        "controlled_rng_before": controlled_rng_before, "controlled_rng_after": controlled_rng_after,
        "weights_and_sources_unchanged": True, "hooks_removed": True, "new_optimizer_steps": 0, "server_SPE09_evidence": False,
        "full_native_eval_passes": 20, "additional_module_eval_passes": 80, "independent_patient_count": len(ix),
        "summaries": summaries, "P3_sample_conditioned_initialization_predictive_harm": "not_established",
        "correction_efficacy": "not_tested", "interpretation": protocol()["interpretation"]}
    _write(output / "observations.json", report)
    _write(output / "receipt.json", {"schema": "slotspe_p3_receipt_v1", "source_sha256": _sha(__file__),
        "outputs": {p.name: _sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return {"patients": len(ix), "native_replay_error": report["historical_replay_max_logit_error"], "summaries": summaries}


def recompute_historical_risks(logits):
    """Keep each validation pass's original (patients, bins) CPU kernel shape.

    Flattening repeats into one larger kernel can change float32 sigmoid
    rounding. Exact replay must preserve the historical evaluation shape.
    """
    import numpy as np
    import torch
    torch.set_num_threads(2)
    return np.stack([(-(1-torch.from_numpy(z).sigmoid()).cumprod(-1).sum(-1)).numpy() for z in logits])


def audit(output, probe_source=None):
    import numpy as np
    import torch
    output = Path(output).resolve()
    receipt = _read(output / "receipt.json")
    required = {"protocol.json", "provenance.json", "observations.json", "encoded_inputs.npz", "native_rng_states.npz"} | {a+".npz" for a in ARMS}
    if not required.issubset(receipt["outputs"]):
        raise ValueError("Missing raw evidence")
    for n, digest in receipt["outputs"].items():
        p = (output / n).resolve()
        if output not in p.parents or _sha(p) != digest:
            raise ValueError("Raw hash/path differs")
    report, settings = _read(output / "observations.json"), _read(output / "protocol.json")
    if settings != protocol() or report["schema"] != "slotspe_p3_observations_v1" or report["scope"] != "historical_local_BLCA_trained_native_eval_randomness" or report["new_optimizer_steps"] != 0 or report["server_SPE09_evidence"] is not False:
        raise ValueError("Scope/protocol differs")
    p1_root = Path(report["p1_root"])
    audit_p1(p1_root)
    p1 = _read(p1_root / "observations.json")
    provenance = _read(output / "provenance.json")
    for target, source in (("patients", "patient_count"), ("seed", "seed"), ("fold", "fold"), ("device", "device"), ("previous_epochs", "epochs"), ("previous_optimizer_steps", "prior_optimizer_steps")):
        if report[target] != p1[source]:
            raise ValueError("P1 identity differs")
    if [p["patient_id"] for p in provenance] != [p["patient_id"] for p in p1["records"]]:
        raise ValueError("Patient order differs")
    p1_patients = _read(p1_root / "patient_provenance.json")["patients"]
    if provenance != [{k: p[k] for k in provenance[0]} for p in p1_patients]:
        raise ValueError("Patient clinical metadata differs")
    for path, digest in report["source_inputs_sha256"].items():
        if _sha(path) == digest:
            continue
        if Path(path).resolve() != Path(__file__).resolve() or probe_source is None or _sha(probe_source) != digest or receipt["source_sha256"] != digest:
            raise ValueError("Original input/source hash changed; measured diagnostic requires its frozen --probe-source")
        def bodies(p):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(Path(p).read_text(encoding="utf-8")).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        old, new = bodies(probe_source), bodies(__file__)
        for name in ("protocol", "rng_state", "rng_digest", "replay_rng", "NativeCapture", "replay_modules", "array_summary", "run"):
            if old[name] != new[name]:
                raise ValueError("Measured computation changed after frozen source")
    historical_path, = [Path(p) for p in report["source_inputs_sha256"] if Path(p).name == "eval_repeats.json"]
    historical = _read(historical_path)
    expected_reference = {"logits": [r["logits"] for r in historical], "risks": [r["risks"] for r in historical], "cindex": [r["cindex"] for r in historical]}
    if report["historical_reference"] != expected_reference or report["eval_seeds"] != [r["eval_seed"] for r in historical] or len(historical) != 20:
        raise ValueError("Historical references differ")
    checks = report["model_purity"]["checks"]
    required_checks = {"native_model_class", "native_forward", "native_slot_classes", "native_decoder_classes", "no_repaired_eval_buffers"}
    if not required_checks.issubset(checks) or any(checks[k] is not True for k in required_checks) or report["model_purity"]["model_args"] != p1["model_purity"]["model_args"]:
        raise ValueError("Native purity/architecture differs")
    if report["full_native_eval_passes"] != 20 or report["additional_module_eval_passes"] != 80 or report["independent_patient_count"] != report["patients"]:
        raise ValueError("Work counts differ")
    for key in ("controlled_rng_before", "controlled_rng_after"):
        state = report[key]
        if set(state) != {"cpu", "cuda"} or not isinstance(state["cuda"], list) or report["device"].startswith("cuda") and not state["cuda"]:
            raise ValueError("Invalid external RNG schema")
        if any(not isinstance(s, str) or re.fullmatch("[0-9a-f]{64}", s) is None for s in [state["cpu"], *state["cuda"]]):
            raise ValueError("Invalid external RNG digest")
    bs = report["batch_size"]
    batches = (report["patients"] + bs - 1) // bs
    with np.load(output / "encoded_inputs.npz", allow_pickle=False) as encoded:
        if set(encoded.files) != {f"b{b:02d}_{m}" for b in range(batches) for m in ("wsi", "omic")}:
            raise ValueError("Encoded input coverage differs")
        for b in range(batches):
            batch_n = min(bs, report["patients"] - b*bs)
            for m, tokens in (("wsi", 256), ("omic", len(p1["model_purity"]["model_args"]["omic_sizes"]))):
                value = encoded[f"b{b:02d}_{m}"]
                if value.shape != (batch_n, tokens, 256) or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError("Encoded input shape/dtype differs")
    devices = ["cpu", *[f"cuda{d}" for d in range(len(report["controlled_rng_before"]["cuda"]))]]
    with np.load(output / "native_rng_states.npz", allow_pickle=False) as states:
        keys = {f"r{r:02d}_b{b:02d}_{m}_{d}" for r in range(20) for b in range(batches) for m in MODULES for d in devices}
        if set(states.files) != keys or any(states[k].dtype != np.uint8 or states[k].ndim != 1 or states[k].size < 16 for k in keys):
            raise ValueError("Recorded RNG state coverage/dtype differs")
    raw = {}
    for arm in ARMS:
        with np.load(output / (arm+".npz"), allow_pickle=False) as f:
            raw[arm] = {k: f[k] for k in f.files}
        a = raw[arm]
        calculated_risk = recompute_historical_risks(a["logits"])
        if not np.array_equal(calculated_risk, a["risks"]) or array_summary(a, provenance) != report["summaries"][arm]:
            raise ValueError("Raw metrics differ")
    native = raw["native"]
    if not np.array_equal(native["logits"], np.asarray(report["historical_reference"]["logits"], dtype=np.float32)) or not np.array_equal(native["risks"], np.asarray(report["historical_reference"]["risks"], dtype=np.float32)):
        raise ValueError("Historical exact replay differs")
    if report["summaries"]["native"]["C_index_validation_exploratory"]["values"] != report["historical_reference"]["cindex"]:
        raise ValueError("Historical C-index differs")
    for m in ("wsi", "omic"):
        for suffix in ("_initial", "_final"):
            key = m+suffix
            if not np.array_equal(raw["fix_gumbel_draws"][key], native[key]):
                raise ValueError("Free slot draw control differs")
            for arm in ("fix_slot_draws", "fix_both_draws"):
                if not np.array_equal(raw[arm][key], np.broadcast_to(native[key][:1], native[key].shape)):
                    raise ValueError("Fixed slot draw control differs")
    for k, v in raw["fix_both_draws"].items():
        if not np.array_equal(v, np.broadcast_to(native[k][:1], native[k].shape)):
            raise ValueError("Both-fixed reference repeat not exact")
    if report["controlled_rng_before"] != report["controlled_rng_after"] or not report["controlled_rng_before"].get("cpu"):
        raise ValueError("External RNG control failed")
    for key in ("weights_and_sources_unchanged", "hooks_removed", "encoded_inputs_unchanged_across_native_repeats"):
        if report[key] is not True:
            raise ValueError("Preservation flag failed")
    for key in ("historical_replay_max_logit_error", "modular_native_replay_max_logit_error"):
        if not isinstance(report[key], (int, float)) or not np.isfinite(report[key]) or not 0 <= report[key] <= 1e-6:
            raise ValueError("Replay tolerance failed")
    return {"status": "accepted_native_and_draw_control_observations", "patients": report["patients"], "repeats_per_condition": 20,
        "conditions": 4, "new_training_runs": 0, "server_raw_results_received": False}


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-source", type=Path, help="frozen measurement source when only offline auditor was updated")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true")
    group.add_argument("--audit", action="store_true")
    args = parser.parse_args(argv)
    if args.audit:
        if args.output is None:
            parser.error("--audit requires --output")
        result = audit(args.output, args.probe_source)
    else:
        if args.p1_root is None or args.run and args.output is None:
            parser.error("requires --p1-root; --run additionally requires fresh --output")
        result = run(args.p1_root, args.output, args.device) if args.run else plan(args.p1_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
