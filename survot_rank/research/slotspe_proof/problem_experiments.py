"""Original research-question probes, isolated from historical implementation audits.

Import and --plan use only the standard library. Only --run reads patient tensors.
No optimizer is used by observe; capacity_train is a separate explicit phase.
All patient records, tensors, annotations and fitted dictionaries stay private.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "configs/slotspe_problem_experiments.json"
QUESTIONS = {
    "P1": "Fixed capacity versus independently measured patient complexity",
    "P2": "Redundant input reading and representation; fragmentation needs content labels",
    "P3": "Cold-start aggregation trajectories; own cues versus mismatched cues",
    "P4": "Content-matched stability; independent labels needed for semantic conclusions",
    "P5": "Reviewed candidate distractors competing with a fixed relevant anchor",
    "P6": "Coordinate/FOV/encoder-aligned scale or layer comparisons",
    "P7": "Main/auxiliary/reconstruction gradient alignment on shared slot parameters",
    "P8": "Not applicable: original SlotSPE has no subsequent cross-modal OT"
}
PATHS = ("source_root", "input_root", "checkpoint", "model_args_file", "annotations", "paired_features")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def config_at(path):
    values = read(path)
    expected = set(read(DEFAULT_CONFIG))
    if set(values) != expected or values.get("schema") != "slotspe_original_questions_v1":
        raise ValueError("Use all and only the keys in configs/slotspe_problem_experiments.json")
    for key in PATHS:
        value = values[key]
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Invalid path setting: " + key)
            expanded = os.path.expandvars(os.path.expanduser(value))
            values[key] = str((REPO / expanded).resolve()) if not Path(expanded).is_absolute() else str(Path(expanded).resolve())
    for key in ("repeats", "dictionary_train_patients", "dictionary_groups", "capacity_epochs", "capacity_batch_size", "reference_iterations", "torch_threads"):
        if isinstance(values[key], bool) or not isinstance(values[key], int) or values[key] < 1:
            raise ValueError(key + " must be a positive integer")
    for key in ("fold", "seed"):
        if isinstance(values[key], bool) or not isinstance(values[key], int) or values[key] < 0:
            raise ValueError(key + " must be a nonnegative integer")
    if values["max_patients"] is not None and (isinstance(values["max_patients"], bool) or not isinstance(values["max_patients"], int) or values["max_patients"] < 1):
        raise ValueError("max_patients must be null or a positive integer")
    for key in ("replay_atol", "replay_rtol", "capacity_lr"):
        if isinstance(values[key], bool) or not isinstance(values[key], (int, float)) or not math.isfinite(values[key]) or values[key] <= 0:
            raise ValueError(key + " must be positive and finite")
    counts = values["capacity_counts"]
    if not isinstance(counts, list) or len(counts) < 2 or any(type(n) is not int or n < 2 for n in counts) or len(set(counts)) != len(counts):
        raise ValueError("capacity_counts must be distinct integers >= 2")
    if values["capacity_modality"] not in ("wsi", "omic") or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", values["device"]):
        raise ValueError("Invalid modality or device")
    return values


def plan(config, phase="observe"):
    required = ["source_root", "input_root", "model_args_file"] + (["checkpoint"] if phase == "observe" else [])
    missing = [k for k in required if not config.get(k) or "$" in str(config[k]) or not Path(config[k]).exists()]
    return {"task": "SPE-08-original-question-experiments", "action": "plan_only", "phase": phase,
            "questions": QUESTIONS, "missing_inputs": missing, "ready_for_explicit_execution": not missing,
            "training_runs": len(config["capacity_counts"]) if phase == "capacity_train" else 0,
            "capacity_factor": {"modality": config["capacity_modality"], "counts": config["capacity_counts"], "fixed_keep_k": 2},
            "observe_scope": "Unmodified checkpoint baseline plus disclosed diagnostic latent/input interventions and gradient probes",
            "missing_sidecars": [k for k in ("annotations", "paired_features") if not config.get(k)],
            "automatic_problem_verdict": False, "output_policy": "fresh directory inside this repository's ignored local_evidence only",
            "not_primary_endpoints": ["overfitting", "eval randomness", "C-index"],
            "execution": "Only --run executes; server must retain missing prerequisites as missing, never invent annotations."}


def private_output(path):
    target = Path(path).expanduser().resolve()
    if not target.is_relative_to((REPO / "local_evidence").resolve()):
        raise ValueError("Output must be inside repository local_evidence, never public docs/results")
    if target.exists():
        raise FileExistsError("Refusing to overwrite any existing output directory")
    return target


def load_args(path, manifest):
    obj = read(path)
    if "purity" in obj:
        obj = obj["purity"]["model_args"]
    elif "model_args" in obj:
        obj = obj["model_args"]
    needed = {"omic_sizes", "encoding_dim", "wsi_projection_dim", "slot_num_wsi", "slot_num_omics", "slot_iters", "temperature", "topk_ratio", "top_k_method", "bag_loss", "alpha_surv", "lambda_recon_loss", "rna_format", "n_classes"}
    if not needed <= set(obj) or obj["rna_format"] != "Pathways" or obj["bag_loss"] != "nll_surv":
        raise ValueError("Supply actual native model args; this runner requires Pathways + nll_surv")
    if obj["omic_sizes"] != manifest["omic_sizes"] or obj["encoding_dim"] != manifest["encoding_dim"]:
        raise ValueError("Actual model recipe and prepared inputs disagree")
    result = SimpleNamespace(**obj)
    wrapper = read(path)
    if "decoder_k" in wrapper:
        result._diagnostic_decoder_k = wrapper["decoder_k"]
    return result


def sidecars(config, manifest):
    annotation = read(config["annotations"]) if config["annotations"] else {"patients": {}}
    paired = read(config["paired_features"]) if config["paired_features"] else {"patients": {}}
    digest = sha(Path(config["input_root"]) / "prepared/manifest.json")
    for obj in (annotation, paired):
        if obj.get("patients") and obj.get("prepared_manifest_sha256") != digest:
            raise ValueError("Sidecar not bound to exact prepared manifest")
    return annotation, paired


def check_annotation(item, n):
    """Independent labels are scored separately from the feature dictionary proxy."""
    if not item:
        return None
    if not item.get("review_provenance") or not item.get("label_definition"):
        raise ValueError("External labels need reviewer/provenance and label definition")
    labels = item.get("token_groups")
    if labels is not None and len(labels) != n:
        raise ValueError("External token labels do not match cached row order")
    if item.get("roles") is not None and (len(item["roles"]) != n or set(item["roles"]) - {"signal", "candidate_nuisance", "unknown"}):
        raise ValueError("Invalid annotation roles")
    return labels


@contextmanager
def seeded(seed):
    import torch
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield


def encode(model, batch):
    import torch
    return {"wsi": model.wsi_mlp(batch["x_wsi"]), "omic": torch.stack([
        model.sig_networks[i](batch["x_omic" + str(i + 1)]) for i in range(model.num_pathways)], dim=1)}


def fit_dictionary(features, groups):
    """Training-only deterministic spherical k-means; clusters are NOT tissue labels."""
    import torch
    from torch.nn import functional as F
    x = F.normalize(features, dim=-1)
    k = min(groups, len(x))
    centers = [x[0]]
    for _ in range(1, k):
        centers.append(x[(x @ torch.stack(centers).T).max(1).values.argmin()])
    centers = torch.stack(centers)
    for _ in range(20):
        labels = (x @ centers.T).argmax(1)
        centers = torch.stack([F.normalize(x[labels == j].mean(0), dim=0) if (labels == j).any() else centers[j] for j in range(k)])
    return centers


def feature_groups(x, centers):
    import torch.nn.functional as F
    return (F.normalize(x, dim=-1) @ centers.T).argmax(-1).detach().cpu().numpy()


def warm_initial(module, x, count):
    """Disclosed inference perturbation, not a claimed trained initializer repair."""
    import torch
    from torch.nn import functional as F
    p = F.normalize(x[0].detach(), dim=-1)
    chosen = [0]
    for _ in range(1, count):
        similarity = p @ p[chosen].T
        score = similarity.max(1).values
        score[chosen] = float("inf")
        chosen.append(int(score.argmin()) if len(chosen) < len(p) else chosen[0])
    cues = x[:, chosen].detach()
    # Per-vector moment matching controls gross scale; own and donor use same rule.
    cues = (cues - cues.mean(-1, keepdim=True)) / cues.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return module.slots_mu + module.slots_logsigma.exp() * cues


def distribution(trace):
    return trace["attentions"][-1][0].mean(0).detach().cpu().numpy()


def semantic_stability(a, b, labels):
    """Match using one token partition, score external content on the other."""
    import numpy as np
    from .problem_metrics import matched_stability
    if labels is None:
        return {"status": "not_testable_missing_independent_content_labels"}
    match_columns = np.arange(a.shape[1])[::2]
    score_columns = np.arange(a.shape[1])[1::2]
    if len(match_columns) < 2 or len(score_columns) < 2:
        return {"status": "not_testable_insufficient_disjoint_tokens"}
    matching = matched_stability(a[:, match_columns], b[:, match_columns])
    categories = sorted(set(labels), key=lambda value: (type(value).__name__, str(value)))
    def mass(x):
        x = x[:, score_columns]
        if (x.sum(1) <= 0).any():
            raise ValueError("Semantic scoring partition contains an empty attention row")
        x = x / x.sum(1, keepdims=True)
        return np.stack([x[:, np.asarray(labels, dtype=object)[score_columns] == category].sum(1) for category in categories], axis=1)
    ma, mb = mass(a), mass(b)
    mapping = [r["slot_b"] for r in matching["matches"]]
    return {"status": "measured_external_content_stability", "matching_tokens": len(match_columns), "scoring_tokens": len(score_columns), "class_support": [{"category": category, "matching": sum(labels[i] == category for i in match_columns), "scoring": sum(labels[i] == category for i in score_columns)} for category in categories], "matching_ambiguous": matching["assignment_ambiguous"], "matched_content_total_variation": float(0.5 * np.abs(ma - mb[mapping]).sum(1).mean()), "same_index_content_total_variation": float(0.5 * np.abs(ma - mb).sum(1).mean()), "interpretation": "External content-class stability on disjoint scoring tokens; token columns may be spatially correlated; not independent biological validation or fixed slot identity across patients."}


def gradient_probe(model, batch, seed):
    """Recover the exact native auxiliary terms and measure shared-block gradients."""
    import torch
    from torch.nn import functional as F
    captured, handles = {}, []
    try:
        for name in ("slot_decoder_wsi", "slot_decoder_omic", "reconstruction_head_omic", "reconstruction_head_wsi", "reconstruction_head_joint"):
            def hook(_m, inputs, out, key=name):
                if key in captured:
                    raise RuntimeError("Unexpected repeated native objective head")
                captured[key] = (inputs, out)
            handles.append(getattr(model, name).register_forward_hook(hook))
        model.train()
        with seeded(seed):
            logits, auxiliary = model(**batch)
        n = len(batch["y"])
        main = model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None) / n
        decoder = sum(model.loss_fn(captured["slot_decoder_" + m][1][0], y=batch["y"], c=batch["c"], t=None) for m in ("wsi", "omic")) / n
        inp, out = captured["reconstruction_head_omic"]
        reconstruction = F.mse_loss(out, inp[0].detach())
        _, out_joint = captured["reconstruction_head_joint"]
        reconstruction = reconstruction + F.mse_loss(out_joint, inp[0].detach())
        inp_w, out_w = captured["reconstruction_head_wsi"]
        reconstruction = reconstruction + 1 - F.cosine_similarity(F.normalize(out_w, dim=-1), F.normalize(inp_w[0], dim=-1), dim=-1).mean()
        reconstruction = model.args.lambda_recon_loss * reconstruction
        rebuilt = decoder + reconstruction
        if not torch.allclose(rebuilt, auxiliary, atol=1e-6, rtol=1e-5):
            raise RuntimeError("Auxiliary decomposition does not reproduce native objective")
        groups = {"slots_wsi": "slot_attention_wsi.", "slots_omic": "slot_attention_omic.", "selector_wsi": "slot_decoder_wsi.pred_keep_slot.", "selector_omic": "slot_decoder_omic.pred_keep_slot."}
        names = [(name, p) for name, p in model.named_parameters() if p.requires_grad and any(name.startswith(v) for v in groups.values())]
        terms = {"main_survival": main, "aux_survival": decoder, "weighted_reconstruction": reconstruction}
        gradients = {key: torch.autograd.grad(loss, [p for _, p in names], retain_graph=True, allow_unused=True) for key, loss in terms.items()}
        result = {"losses": {k: float(v.detach()) for k, v in terms.items()}, "aux_rebuild_abs_error": float((rebuilt - auxiliary).abs().detach()), "blocks": {}}
        for block, prefix in groups.items():
            index = [i for i, (name, _) in enumerate(names) if name.startswith(prefix)]
            record = {"n_parameters": len(index), "objectives": {}, "pairwise": {}}
            for key, gs in gradients.items():
                sq = sum(float(gs[i].detach().double().square().sum()) for i in index if gs[i] is not None)
                record["objectives"][key] = {"connected_parameters": sum(gs[i] is not None for i in index), "norm": math.sqrt(sq)}
            for a, b in (("main_survival", "aux_survival"), ("main_survival", "weighted_reconstruction"), ("aux_survival", "weighted_reconstruction")):
                shared = [i for i in index if gradients[a][i] is not None and gradients[b][i] is not None]
                dot = sum(float((gradients[a][i].detach().double() * gradients[b][i].detach().double()).sum()) for i in shared)
                na = math.sqrt(sum(float(gradients[a][i].detach().double().square().sum()) for i in shared))
                nb = math.sqrt(sum(float(gradients[b][i].detach().double().square().sum()) for i in shared))
                record["pairwise"][a + "__" + b] = {"shared_parameters": len(shared), "dot": dot, "cosine": dot / (na * nb) if na * nb else None, "local_first_order_opposition": bool(shared and dot < 0), "undefined_reason": None if na * nb else "no shared nonzero gradients"}
            result["blocks"][block] = record
        return result
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        model.eval()


def nuisance_probe(module, x, initial, annotation, seed=0):
    import numpy as np
    import torch
    from .problem_metrics import matched_stability
    from .problem_trace import replay_slots
    roles = annotation.get("roles")
    scope = annotation.get("role_scope")
    if not roles:
        return {"status": "not_testable_missing_reviewed_roles"}
    if scope not in ("acquisition_artifact", "non_target_content", "prognostic_irrelevance_claim"):
        raise ValueError("P5 roles need an explicit role_scope")
    if scope == "prognostic_irrelevance_claim" and not annotation.get("independent_prognostic_evidence"):
        return {"status": "not_testable_unsubstantiated_prognostic_irrelevance"}
    relevant = [i for i, r in enumerate(roles) if r == "signal"]
    nuisance = [i for i, r in enumerate(roles) if r == "candidate_nuisance"]
    maximum_dose = min(len(relevant) // 3, len(nuisance))
    if maximum_dose < 1:
        return {"status": "not_testable_insufficient_matched_tokens"}
    anchor = relevant[:-maximum_dose]
    baseline = replay_slots(module, x[:, anchor], initial)
    output = {"status": "measured_conditional_input_intervention", "role_scope": scope, "anchor_tokens": len(anchor), "dose_results": [], "scope": "Candidate-input competition, not proven loss of prognostic information"}
    rng = np.random.default_rng(seed)
    for dose in sorted({max(1, maximum_dose // 4), max(1, maximum_dose // 2), maximum_dose}):
        for subset_repeat in range(3):
            record = {"dose_each": dose, "subset_repeat": subset_repeat, "arms": {}}
            samples = (("candidate_nuisance", rng.choice(nuisance, dose, replace=False).tolist()), ("heldout_relevant", rng.choice(relevant[-maximum_dose:], dose, replace=False).tolist()))
            for arm, extra in samples:
                trial = replay_slots(module, x[:, anchor + extra], initial)
                attention = distribution(trial)
                anchor_mass = attention[:, :len(anchor)].sum(-1)
                matched = matched_stability(distribution(baseline), attention[:, :len(anchor)])
                record["arms"][arm] = {"added_indices": extra, "mean_mass_to_added_tokens": float(1 - anchor_mass.mean()), "uniform_added_mass_reference": dose / (len(anchor) + dose), "anchor_reading_change": matched}
            output["dose_results"].append(record)
    return output


def paired_probe(model, x, initial, item, patient_meta, input_root, labels=None):
    """Reject dimensional equality as a substitute for paired feature provenance."""
    import torch
    from .problem_metrics import matched_stability, slot_metrics
    from .problem_trace import replay_slots
    if not item:
        return {"status": "not_testable_missing_aligned_multiscale_or_multilayer_features"}
    contract = item.get("contract", {})
    needed = ("same_patient", "same_coordinates_and_fov_contract", "same_row_order", "common_feature_space", "independently_reviewed")
    if not all(contract.get(k) is True for k in needed) or not item.get("review_provenance"):
        return {"status": "not_testable_unverified_feature_contract"}
    if item.get("base_patch_indices") != patient_meta["patch_indices_concatenated_slide_order"]:
        raise ValueError("P6 base row mapping differs from actual cached patches")
    base = replay_slots(model.slot_attention_wsi, x, initial)
    results = []
    for variant in item.get("variants", []):
        if variant.get("kind") not in ("scale", "encoder_layer") or not variant.get("encoder_provenance") or not variant.get("coordinate_manifest_sha256"):
            raise ValueError("Variant lacks true scale/layer/coordinate provenance")
        path = Path(variant["features_path"])
        if not path.is_absolute():
            path = Path(input_root) / path
        if sha(path) != variant["sha256"]:
            raise ValueError("P6 feature hash mismatch")
        value = torch.load(path, map_location=x.device, weights_only=True)
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape != (x.shape[1], model.args.encoding_dim) or not torch.isfinite(value).all():
            raise ValueError("P6 requires aligned finite original-embedding-space tensors")
        z = model.wsi_mlp(value[None])
        altered = replay_slots(model.slot_attention_wsi, z, initial)
        measurement = {"name": variant["name"], "kind": variant["kind"], "reading_change": matched_stability(distribution(base), distribution(altered)), "interpretation": "Paired frozen-feature control; changes alone do not establish missing useful information."}
        if labels is not None:
            measurement["external_content_readout"] = slot_metrics(altered["final"][0].cpu().numpy(), distribution(altered), labels)["groups"]
        results.append(measurement)
    return {"status": "measured_paired_feature_and_content_control" if results and labels is not None else "sensitivity_only_missing_independent_content_targets" if results else "not_testable_no_variants", "base_content_readout": slot_metrics(base["final"][0].cpu().numpy(), distribution(base), labels)["groups"] if labels is not None else None, "variants": results, "scale_insufficiency_established": False}


def observe(config, model, manifest, args, out):
    import numpy as np
    import torch
    from .patient import _batch
    from .problem_metrics import matched_stability, slot_metrics, summary
    from .problem_trace import NativeSlotTrace, replay_slots
    fold = manifest["folds"][str(config["fold"])]; root = Path(config["input_root"])
    train_ids = fold["train"][:config["dictionary_train_patients"]]
    val_ids = fold["val"][:config["max_patients"]]
    if not train_ids or not val_ids or set(fold["train"]) & set(fold["val"]):
        raise ValueError("Require disjoint nonempty prepared training and validation patients")
    annotation, paired = sidecars(config, manifest)
    dictionary_inputs = []
    donors = []
    with torch.no_grad():
        for pid in train_ids:
            z = encode(model, _batch(root, manifest, fold, [pid], config["device"], torch))
            dictionary_inputs.append(z["wsi"][0])
            if len(donors) < 3:
                donors.append({k: v.detach().clone() for k, v in z.items()})
        centers = fit_dictionary(torch.cat(dictionary_inputs), config["dictionary_groups"])
    torch.save({"centers": centers.cpu(), "training_patient_ids": train_ids, "donor_patient_ids": train_ids[:len(donors)]}, out / "training_dictionary.pt")
    records, replay_errors, gradient_records = [], [], []
    with NativeSlotTrace(model) as observer:
        for p_index, pid in enumerate(val_ids):
            batch = _batch(root, manifest, fold, [pid], config["device"], torch)
            item = annotation["patients"].get(pid, {})
            labels = check_annotation(item, batch["x_wsi"].shape[1])
            if item and item.get("cache_sha256") != manifest["patients"][pid]["cache_sha256"]:
                raise ValueError("Annotation not bound to exact cached tensor")
            baseline_attention = {}
            patient_rows = []
            for repeat in range(config["repeats"]):
                observer.reset()
                trial_seed = config["seed"] * 1000000 + p_index * 100 + repeat
                with torch.no_grad(), seeded(trial_seed):
                    logits, _ = model(**batch)
                row = {"repeat": repeat, "seed": trial_seed, "modalities": {}}
                for modality in ("wsi", "omic"):
                    native = observer.records[modality]
                    module = getattr(model, "slot_attention_" + modality)
                    x, initial = native["inputs"], native["initial"]
                    with torch.no_grad():
                        replay = replay_slots(module, x, initial)
                        error = float((replay["final"] - native["final"]).abs().max())
                        replay_errors.append(error)
                        if not torch.allclose(replay["final"], native["final"], atol=config["replay_atol"], rtol=config["replay_rtol"]):
                            raise RuntimeError("Exact native slot replay failed; no mechanism result accepted")
                        for key in ("states", "attentions"):
                            if len(replay[key]) != len(native[key]) or not all(torch.allclose(a, b, atol=config["replay_atol"], rtol=config["replay_rtol"]) for a, b in zip(replay[key], native[key])):
                                raise RuntimeError("Native iteration-by-iteration replay failed: " + key)
                        attn = distribution(replay)
                        groups = labels if modality == "wsi" and labels is not None else (feature_groups(x[0], centers) if modality == "wsi" else np.arange(x.shape[1]))
                        metric = slot_metrics(native["final"][0].cpu().numpy(), attn, groups)
                        current = {"P2": metric, "group_scope": "independent_content_labels" if modality == "wsi" and labels is not None else "train_only_feature_prototypes" if modality == "wsi" else "pathway_token_identity_not_prognostic_events", "P1_frozen_capacity": []}
                        for count in config["capacity_counts"]:
                            with seeded(trial_seed + 50000):
                                full_noise = torch.randn(1, max(config["capacity_counts"]), initial.shape[-1], device=x.device, dtype=x.dtype)
                            init = module.slots_mu + module.slots_logsigma.exp() * full_noise[:, :count]
                            expanded = replay_slots(module, x, init)
                            current["P1_frozen_capacity"].append({"slots": count, "metrics": slot_metrics(expanded["final"][0].cpu().numpy(), distribution(expanded), groups), "scope": "conditional_frozen_checkpoint_probe_not_capacity_causality"})
                        trajectories = {}
                        common = replay_slots(module, x, initial, iterations=config["reference_iterations"])
                        common_attention = distribution(common)
                        arms = [("native_random", initial), ("own_input_cues", warm_initial(module, x, initial.shape[1]))]
                        for donor_index, donor in enumerate(donors):
                            if donor[modality].shape[1:] != x.shape[1:]:
                                raise ValueError("P3 cue donor token/dimension budget differs")
                            arms.append(("mismatched_train_cues_" + str(donor_index), warm_initial(module, donor[modality], initial.shape[1])))
                        for arm, init in arms:
                            long = replay_slots(module, x, init, iterations=config["reference_iterations"])
                            terminal = distribution(long)
                            distances = [matched_stability(a[0].mean(0).cpu().numpy(), terminal, n_permutations=0)["matched_mean_distance"] for a in long["attentions"]]
                            budget = module.iters
                            trajectories[arm] = {"iteration_to_own_terminal_distance": distances, "native_budget_common_reference_distance": matched_stability(long["attentions"][budget - 1][0].mean(0).cpu().numpy(), common_attention, n_permutations=0)["matched_mean_distance"], "native_budget_metrics": slot_metrics(long["states"][budget][0].cpu().numpy(), long["attentions"][budget - 1][0].mean(0).cpu().numpy(), groups), "terminal_metrics": slot_metrics(long["final"][0].cpu().numpy(), terminal, groups), "reference_is_ground_truth": False}
                        current["P3"] = {"native_budget": module.iters, "long_reference_budget": config["reference_iterations"], "arms": trajectories, "scope": "inference_initialization_counterfactual_not_trained_initializer_efficacy"}
                        if modality in baseline_attention:
                            current["P4"] = matched_stability(baseline_attention[modality], attn)
                            current["P4"]["claim_scope"] = "same_input_computational_reading_stability_only"
                            if modality == "wsi":
                                current["P4"]["external_content_test"] = semantic_stability(baseline_attention[modality], attn, labels)
                        else:
                            baseline_attention[modality] = attn
                        if modality == "wsi" and repeat == 0:
                            current["P5"] = nuisance_probe(module, x, initial, item, trial_seed)
                            current["P6"] = paired_probe(model, x, initial, paired["patients"].get(pid), manifest["patients"][pid], root, labels)
                        if repeat == 0:
                            np.savez_compressed(out / (f"patient_{p_index:04d}_{modality}_native.npz"), encoded=x.cpu().numpy(), initial=initial.cpu().numpy(), final=native["final"].cpu().numpy(), attention=np.stack([a[0].cpu().numpy() for a in replay["attentions"]]), logits=logits.cpu().numpy())
                        row["modalities"][modality] = current
                patient_rows.append(row)
            records.append({"patient_id": pid, "cache_sha256": manifest["patients"][pid]["cache_sha256"], "repeats": patient_rows})
            # Save progress privately before the more memory-intensive gradient probe.
            write(out / "patient_records.json", records)
            gradients = []
            for r in range(config["repeats"]):
                observer.reset()
                gradients.append(gradient_probe(model, batch, config["seed"] * 1000000 + p_index * 100 + 90 + r))
            gradient_records.append({"patient_id": pid, "dropout_repeats_not_independent_patients": gradients})
            write(out / "objective_gradients.json", gradient_records)
            print(json.dumps({"completed_patients": p_index + 1, "total_patients": len(val_ids)}), flush=True)
    return {"patients": len(records), "repeat_rows_are_not_independent_patients": True, "native_replay_error": summary(replay_errors), "question_status": {"P1": "frozen_sensitivity_only_use_separate_matched_training_for_capacity", "P2": "representation_and_input_reading_measured", "P3": "aggregation_trajectories_measured_causal_scope_limited", "P4": "external_content_scoring_when_labels_present_else_computational_only", "P5": "see_patient_specific_role_prerequisites", "P6": "see_patient_specific_aligned_features_and_content_targets", "P7": "exact_native_objective_gradient_geometry_measured", "P8": "not_applicable"}, "main_endpoint_is_not_cindex": True}


def execute(config, phase, output):
    from .model import build_model, MODEL_SOURCE_FILES
    from .native_audit import _state_digest, model_purity
    import torch
    if plan(config, phase)["missing_inputs"]:
        raise ValueError("Required server paths not resolved; run --plan")
    out = private_output(output)
    root = Path(config["input_root"])
    manifest_path = root / "prepared/manifest.json"
    manifest = read(manifest_path)
    fold = manifest["folds"][str(config["fold"])]; args = load_args(config["model_args_file"], manifest)
    if config["reference_iterations"] <= args.slot_iters:
        raise ValueError("reference_iterations must exceed the actual native iteration budget")
    if fold["binning"].get("validation_outcomes_used_for_training_bins") is not False:
        raise ValueError("Training-fold-only bins required")
    ids = set(fold["train"] + fold["val"])
    for pid in ids:
        meta = manifest["patients"][pid]
        if sha(root / meta["cache_path"]) != meta["cache_sha256"]:
            raise ValueError("Prepared cache changed")
    source = Path(config["source_root"])
    hashes = {p: sha(source / p) for p in MODEL_SOURCE_FILES}
    for p, digest in hashes.items():
        if manifest["source_hashes"].get(p) != digest:
            raise ValueError("Prepared source provenance differs: " + p)
    torch.set_num_threads(config["torch_threads"])
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no automatic CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out.mkdir(parents=True)
    receipt = {"status": "started", "phase": phase, "config": config, "args": vars(args), "source_hashes": hashes, "prepared_manifest_sha256": sha(manifest_path), "model_args_sha256": sha(config["model_args_file"]), "runner_sha256": sha(__file__), "helper_sha256": {name: sha(Path(__file__).with_name(name)) for name in ("problem_trace.py", "problem_metrics.py", "problem_capacity.py", "patient.py", "model.py")}, "started_utc": datetime.now(timezone.utc).isoformat(), "patient_outputs_private": True}
    from importlib.metadata import version
    receipt["environment"] = {"torch": str(torch.__version__), "cuda_runtime": torch.version.cuda, "device": config["device"], "gpu": torch.cuda.get_device_name(config["device"]) if config["device"].startswith("cuda") else None, "packages": {name: version(name) for name in ("numpy", "scipy", "einops")}}
    write(out / "execution_receipt.json", receipt)
    try:
        if phase == "capacity_train":
            from .problem_capacity import run_capacity
            result = run_capacity(config, source_root=source, input_root=root, manifest=manifest, model_args=args, output_root=out / "capacity_training")
        else:
            model = build_model(str(source), args, "original").to(config["device"])
            state = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
            model.load_state_dict(state.get("state_dict", state), strict=True)
            if hasattr(args, "_diagnostic_decoder_k"):
                for modality, setting in args._diagnostic_decoder_k.items():
                    if modality not in ("wsi", "omic") or setting.get("actual_value") != 2:
                        raise ValueError("Unexpected controlled-capacity decoder recipe")
                    getattr(model, "slot_decoder_" + modality).k = 2
            model.eval(); model_purity(model, source)
            before = _state_digest(model)
            receipt["checkpoint_sha256"] = sha(config["checkpoint"])
            write(out / "execution_receipt.json", receipt)
            result = observe(config, model, manifest, args, out)
            if before != _state_digest(model):
                raise RuntimeError("Diagnostic execution changed checkpoint state")
            model_purity(model, source)
        if hashes != {p: sha(source / p) for p in MODEL_SOURCE_FILES}:
            raise RuntimeError("Original executable source changed")
        receipt.update(status="completed", result=result, source_unchanged=True, finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        receipt.update(status="failed_retained", error=type(exc).__name__ + ": " + str(exc))
        write(out / "execution_receipt.json", receipt)
        raise
    write(out / "execution_receipt.json", receipt)
    return {"status": receipt["status"], "phase": phase, "output": str(out)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("observe", "capacity_train"), default="observe")
    parser.add_argument("--output-dir", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--run", action="store_true")
    ns = parser.parse_args(argv)
    config = config_at(ns.config)
    if ns.run and ns.output_dir is None:
        parser.error("--run requires a fresh --output-dir inside ignored local_evidence")
    result = execute(config, ns.phase, ns.output_dir) if ns.run else plan(config, ns.phase)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
