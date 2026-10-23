"""Observers and disclosed interventions for an unchanged native SlotSPE."""
import hashlib
import math


def model_purity(model, source_root):
    from .model import load_source
    source = load_source(source_root)
    checks = {"native_model_class": type(model) is source.SlotSPE,
        "native_forward": model.forward.__func__ is source.SlotSPE.forward,
        "native_slot_classes": all(type(getattr(model, "slot_attention_" + key)) is source.MultiHeadSlotAttention for key in ("wsi", "omic")),
        "native_decoder_classes": all(type(getattr(model, "slot_decoder_" + key)) is source.MoESlotDecoder for key in ("wsi", "omic")),
        "no_repaired_eval_buffers": not any("_proof_eval_noise" in name for name, _ in model.named_buffers())}
    if not all(checks.values()):
        raise RuntimeError("Original-only audit received a repaired model")
    return {"checks": checks, "module_paths": source.module_paths, "source_sha256": source.source_hashes,
        "model_class": type(model).__module__ + "." + type(model).__name__, "model_args": vars(model.args)}


def _state_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


class GateObserver:
    """Observe actual final-validation gates; hooks return None, no RNG calls."""
    def __init__(self, model, patient_ids):
        self.ids, self.records, self.hooks = list(patient_ids), [], []
        self.scores, self.cursor = {}, {}
        self.repeat = self.seed = None
        for key in ("wsi", "omic"):
            def scores_hook(_module, _inputs, output, name=key):
                self.scores[name] = output.detach().squeeze(-1).cpu()
            def gate_hook(_module, _inputs, output, name=key):
                gate, keep = output[1].detach().cpu(), output[2].detach().cpu()
                scores = self.scores[name]
                begin = self.cursor[name]
                for i in range(len(gate)):
                    total = float(gate[i].sum())
                    finite = math.isfinite(total) and all(math.isfinite(float(x)) for x in scores[i])
                    number = lambda x: float(x) if math.isfinite(float(x)) else None
                    self.records.append({"patient_id": self.ids[begin + i], "modality": name,
                        "repeat": self.repeat, "eval_seed": self.seed, "gate_sum": total if math.isfinite(total) else None,
                        "finite": finite, "near_zero": finite and total < 1e-6,
                        "scores": [number(x) for x in scores[i]], "gate": [number(x) for x in gate[i]],
                        "selected_indices": (keep[i] > .5).nonzero().flatten().tolist(),
                        "unnoised_top_score_selected": bool(keep[i, scores[i].argmax()] > .5)})
                self.cursor[name] += len(gate)
            decoder = getattr(model, "slot_decoder_" + key)
            self.hooks.append(decoder.pred_keep_slot.register_forward_hook(scores_hook))
            self.hooks.append(decoder.register_forward_hook(gate_hook))

    def begin(self, repeat, seed):
        self.repeat, self.seed = repeat, seed
        self.cursor = {"wsi": 0, "omic": 0}

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def result(self, checkpoint_sha256):
        near_zero = sum(row["near_zero"] for row in self.records)
        nonfinite = sum(not row["finite"] for row in self.records)
        return {"input_scope": "unmodified_prepared_real_validation_inputs_final_checkpoint",
            "checkpoint_sha256": checkpoint_sha256, "patient_ids": self.ids,
            "observations": len(self.records), "near_zero_count": near_zero, "nonfinite_count": nonfinite,
            "near_zero_gate": "observed" if near_zero else "not_observed",
            "rows_are_not_independent_patients": True, "observer_only_no_outputs_replaced": True,
            "records": self.records}


def _controlled_gate_case(source_root, seed):
    """Separate native decoder copy; never inject scores into the patient model."""
    import torch
    from .model import load_source
    torch.manual_seed(seed)
    decoder = load_source(source_root).MoESlotDecoder(32, 8, temperature=.01, topk_ratio=.25).eval()
    slots = torch.randn(1, 8, 32)
    scores = torch.tensor([[[1.]] + [[0.]] * 7])
    hook = decoder.pred_keep_slot.register_forward_hook(lambda _m, _i, _o: scores)
    rows = []
    try:
        with torch.no_grad():
            for index in range(32):
                torch.manual_seed(seed + index)
                _, gate, keep = decoder(slots)
                rows.append({"seed": seed + index, "gate_sum": float(gate.sum()),
                    "top_score_selected": bool(keep[0, 0] > .5)})
    finally:
        hook.remove()
    return {"input_scope": "constructed_scores_separate_native_decoder_not_patient_frequency",
        "scores": scores.flatten().tolist(), "temperature": .01, "k": 2, "records": rows,
        "counterexample": "observed" if any(r["gate_sum"] < 1e-6 for r in rows) else "not_observed"}


def run_final_audit(model, root, manifest, fold_data, config, device, seed, repeats, checkpoint_sha256):
    import numpy as np
    import torch
    from sksurv.metrics import concordance_index_censored
    from .patient import _batch, _rng, _evaluate
    purity = model_purity(model, config["source_root"])
    before_state = _state_digest(model)
    ids = fold_data["val"]
    # Same RNG and immutable patient cache as the first of the 20 native runs.
    replay = _evaluate(model, root, manifest, fold_data, ids, config, device,
        repeats[0]["eval_seed"], torch, np, concordance_index_censored)
    replay_error = float(np.max(np.abs(np.asarray(replay["logits"]) - np.asarray(repeats[0]["logits"]))))
    padding_rows = []
    risk = lambda logits: -(1 - logits.sigmoid()).cumprod(-1).sum(-1)
    for begin in range(0, len(ids), config["batch_size"]):
        batch_ids = ids[begin:begin + config["batch_size"]]
        batch = _batch(root, manifest, fold_data, batch_ids, device, torch)
        probe_seed = seed * 1000000 + 92000 + begin
        _rng(probe_seed, torch, np)
        with torch.no_grad():
            original, _ = model(**batch)
            for count in (16, 64):
                modified = dict(batch)
                modified["x_wsi"] = torch.cat([batch["x_wsi"], batch["x_wsi"].new_zeros(len(batch_ids), count, batch["x_wsi"].shape[-1])], 1)
                _rng(probe_seed, torch, np)
                altered, _ = model(**modified)
                for i, pid in enumerate(batch_ids):
                    padding_rows.append({"patient_id": pid, "added_controlled_zero_rows": count,
                        "probe_seed": probe_seed,
                        "original_zero_feature_rows": int((batch["x_wsi"][i] == 0).all(-1).sum()),
                        "logits_before": original[i].cpu().tolist(), "logits_after": altered[i].cpu().tolist(),
                        "max_logit_difference": float((original[i] - altered[i]).abs().max()),
                        "risk_abs_difference": float((risk(original)[i] - risk(altered)[i]).abs())})
    state_unchanged = _state_digest(model) == before_state
    model_purity(model, config["source_root"])
    if not state_unchanged:
        raise RuntimeError("Diagnostics modified native checkpoint tensors")
    excluded = {}
    for row in fold_data["excluded"]:
        excluded[row["reason"]] = excluded.get(row["reason"], 0) + 1
    controlled = _controlled_gate_case(config["source_root"], seed + 20000)
    return {"schema": "slotspe_native_final_audit_v1", "status": "measured", "checkpoint_sha256": checkpoint_sha256,
        "patient_ids": ids, "purity": purity, "checkpoint_state_unchanged_by_diagnostics": state_unchanged,
        "repaired_models_constructed": False, "transport_added": False,
        "rng_replay": {"eval_seed": repeats[0]["eval_seed"], "max_logit_difference": replay_error,
            "within_1e_minus_6": replay_error <= 1e-6, "logits": replay["logits"], "risks": replay["risks"]},
        "padding": {"input_scope": "real_patient_cache_plus_controlled_zero_rows_not_natural_padding_prevalence",
            "sensitivity": "observed" if any(r["max_logit_difference"] > 1e-7 for r in padding_rows) else "not_observed",
            "threshold": 1e-7, "source_loader_opportunity_counts_from_exclusions": excluded,
            "records": padding_rows, "limitation": "The prepared cohort contains sufficient real patches; do not count added zeros as natural patient padding."},
        "controlled_gate_counterexample": controlled,
        "interpretation": "Actual final-checkpoint observations and explicitly controlled probes; no correction benefit, biological causality or guaranteed failure."}
