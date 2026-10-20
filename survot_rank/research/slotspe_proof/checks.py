"""Portable, read-only SlotSPE protocol and controlled reconstruction probes.

Metadata probes execute unchanged AST bodies from the supplied checkout, without
importing its training stack or torch. Reconstruction imports torch lazily and
uses synthetic inputs only. Neither entry point trains a patient model.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _report_path(output_dir, name):
    path = Path(output_dir).resolve() / name
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing proof report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save(path, report):
    # Exclusive creation also protects against two queue workers racing.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def _tree(path):
    return ast.parse(Path(path).read_text(encoding="utf-8-sig"), filename=str(path))


def _literal(node, default=None):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return default


def _source_defaults(source_root, dataset_tree):
    """Resolve the endpoint/bin defaults from source, never from a cohort name."""
    defaults = {}
    provenance = {}
    candidates = [source_root / "utils" / "process_args.py"]
    candidates += sorted(source_root.glob("*.py"))
    for path in candidates:
        if not path.is_file():
            continue
        for node in ast.walk(_tree(path)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                continue
            names = [_literal(arg) for arg in node.args]
            for name in ("label_col", "n_classes", "n_bins"):
                if f"--{name}" not in names:
                    continue
                keyword = next((kw for kw in node.keywords if kw.arg == "default"), None)
                value = _literal(keyword.value) if keyword is not None else None
                if value is not None and name not in defaults:
                    defaults[name], provenance[name] = value, str(path)
    # Some source distributions have only a runnable dataset example.
    if "label_col" not in defaults:
        assigned = {_literal(node.value) for node in ast.walk(dataset_tree)
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "label_col"
                        for target in node.targets)}
        assigned.discard(None)
        if len(assigned) == 1:
            defaults["label_col"] = assigned.pop()
            provenance["label_col"] = "dataset/dataset_survival.py literal assignment"
    factory_class = next(node for node in dataset_tree.body
                         if isinstance(node, ast.ClassDef)
                         and node.name == "SurvivalDatasetFactory")
    init = next(node for node in factory_class.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    default_args = dict(zip([arg.arg for arg in init.args.args][-len(init.args.defaults):],
                            [_literal(value) for value in init.args.defaults]))
    bins = defaults.get("n_classes", defaults.get("n_bins", default_args.get("n_bins")))
    endpoint = defaults.get("label_col")
    if not isinstance(endpoint, str) or not endpoint.startswith("survival_months_"):
        raise ValueError("Cannot infer an unambiguous survival endpoint from supplied source defaults")
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("Cannot infer a valid number of survival bins from supplied source")
    return endpoint, "censorship_" + endpoint.removeprefix("survival_months_"), bins, {
        "source_default_arguments": defaults,
        "default_provenance": provenance,
        "eps": float(default_args.get("eps", 1e-6)),
        "scope": "Source-default endpoint and bin count; not an inference of prior experiment settings.",
    }


def _metadata_api(source_root, np, pd):
    """Compile original metadata-only methods; avoid the module's torch import."""
    dataset_path = source_root / "dataset" / "dataset_survival.py"
    tree = _tree(dataset_path)
    names = {
        "SurvivalDatasetFactory": ("_setup_clinical_data", "_get_uncensored_data", "_disc_label"),
        "SurvivalDataset": ("__init__", "_load_split", "_get_split_from_df"),
    }
    bodies = []
    lines = {}
    for class_name, wanted in names.items():
        original = next((node for node in tree.body
                         if isinstance(node, ast.ClassDef) and node.name == class_name), None)
        if original is None:
            raise ValueError(f"Unsupported source layout: {class_name} missing in {dataset_path}")
        methods = [copy.deepcopy(node) for node in original.body
                   if isinstance(node, ast.FunctionDef) and node.name in wanted]
        missing = set(wanted) - {node.name for node in methods}
        if missing:
            raise ValueError(f"Unsupported metadata API in supplied source: {sorted(missing)}")
        for node in methods:
            lines[f"{class_name}.{node.name}"] = [node.lineno, node.end_lineno]
        bodies.append(ast.ClassDef(name=class_name, bases=[], keywords=[], body=methods,
                                   decorator_list=[]))
    namespace = {"np": np, "pd": pd, "os": os}
    module = ast.fix_missing_locations(ast.Module(body=bodies, type_ignores=[]))
    exec(compile(module, str(dataset_path), "exec"), namespace)
    return namespace, tree, dataset_path, lines


def _reference_api(source_root, np):
    path = source_root / "utils" / "core_utils.py"
    if not path.is_file():
        return None, {"status": "not_evaluable", "reason": "utils/core_utils.py absent"}
    node = next((node for node in _tree(path).body if isinstance(node, ast.FunctionDef)
                 and node.name == "_extract_survival_metadata"), None)
    if node is None:
        return None, {"status": "not_evaluable", "reason": "_extract_survival_metadata absent"}

    class NumpySurv:
        """Surv.from_arrays equivalent for the two-field metadata-only contract."""
        @staticmethod
        def from_arrays(event, time, name_event=None, name_time=None):
            result = np.empty(len(time), dtype=[(name_event or "event", bool),
                                                (name_time or "time", float)])
            result[result.dtype.names[0]] = event
            result[result.dtype.names[1]] = time
            return result

    try:
        from sksurv.util import Surv
        backend = "sksurv.util.Surv.from_arrays"
    except ImportError:
        Surv = NumpySurv
        backend = "numpy event/time structured array compatibility helper; no metric computation"
    namespace = {"Surv": Surv, "np": np}
    exec(compile(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name], {"path": str(path), "source_lines": [node.lineno, node.end_lineno],
                                  "helper_backend": backend, "function_body": "unchanged AST"}


def run_protocol_checks(source_root, metadata_root, study, folds, output_dir, *, label_col=None, n_bins=None):
    """Probe real server CSVs without loading features, torch, or training code.

    Source defaults determine the endpoint and bin count. Held-out extremes are
    deliberate causal interventions, not estimates of naturally occurring bias.
    Defects are observations, never prerequisites for a successful check run.
    """
    import numpy as np
    import pandas as pd

    output = _report_path(output_dir, "protocol_checks.json")
    source_root, metadata_root = Path(source_root).resolve(), Path(metadata_root).resolve()
    if not study or Path(study).name != study or any(char in study for char in "/\\"):
        raise ValueError("study must be a cohort name, not a path")
    folds = list(dict.fromkeys(int(fold) for fold in folds))
    if not folds or min(folds) < 0:
        raise ValueError("At least one nonnegative fold is required")
    api, tree, dataset_path, source_lines = _metadata_api(source_root, np, pd)
    endpoint, censor, source_bins, settings = _source_defaults(source_root, tree)
    if n_bins is None:
        n_bins = source_bins
    elif isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("Configured n_bins must be an integer >= 2")
    settings["source_default_n_bins"] = source_bins
    settings["configured_n_bins"] = n_bins
    if label_col is not None:
        if label_col not in {"survival_months_dss", "survival_months_os", "survival_months_pfi", "survival_months_dfs"}:
            raise ValueError("Unsupported explicitly configured survival endpoint")
        settings["source_default_endpoint"] = endpoint
        settings["configured_endpoint"] = label_col
        endpoint, censor = label_col, "censorship_" + label_col.removeprefix("survival_months_")
    extract, reference_description = _reference_api(source_root, np)
    paths = [dataset_path, metadata_root / "clinical" / "all" / f"{study}.csv"]
    paths += [metadata_root / "splits" / "5fold" / study / f"fold_{fold}.csv" for fold in folds]
    paths += [path for path in [source_root / "utils" / "core_utils.py",
                               source_root / "utils" / "process_args.py"] if path.is_file()]
    before = {str(path): _digest(path) for path in paths}
    factory = api["SurvivalDatasetFactory"].__new__(api["SurvivalDatasetFactory"])
    for key, value in {"study": study, "data_path": str(metadata_root), "label_col": endpoint,
                       "censorship_var": censor, "n_bins": n_bins, "eps": settings["eps"]}.items():
        setattr(factory, key, value)
    factory._setup_clinical_data()
    full = factory.clinical_df.copy(deep=True)
    case = "case id"
    if full.empty or not full[case].is_unique:
        raise ValueError("Clinical metadata must contain nonempty, unique patient IDs")
    if not np.isfinite(full[[endpoint, censor]].to_numpy(dtype=float)).all():
        raise ValueError("Clinical outcomes must be finite after the source's own filtering")
    if not full[censor].isin([0, 1]).all():
        raise ValueError("Expected the supplied source's binary censorship convention")
    ledger = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        ledger.append({"assertion": name, "passed": True})

    def relabel(frame):
        cloned = copy.copy(factory)
        cloned.clinical_df = frame.drop(columns="label", errors="ignore").copy(deep=True)
        cloned._disc_label(cloned._get_uncensored_data())
        return cloned

    def labels(frame):
        return frame.set_index(case)["label"]

    def edges_json(edges):
        return ["-inf" if value == -np.inf else "+inf" if value == np.inf else float(value)
                for value in edges]

    def fit_edges(train, mode):
        if mode == "source_estimator_train_only":
            edges = np.asarray(relabel(train).bins, dtype=float).copy()
        else:
            _, edges = pd.qcut(train.loc[train[censor] < 1, endpoint], q=n_bins,
                               labels=False, retbins=True)
            edges = np.asarray(edges, dtype=float).copy()
        edges[0], edges[-1] = -np.inf, np.inf
        if len(edges) != n_bins + 1 or not np.all(np.diff(edges) > 0):
            raise ValueError("Train-only bins must be strictly increasing")
        return edges

    def fixed_labels(frame, edges):
        result = pd.cut(frame[endpoint], bins=edges, labels=False, right=False, include_lowest=True)
        check("train-fitted open intervals cover every present patient", not result.isna().any())
        return pd.Series(result.to_numpy(dtype=int), index=frame[case].to_numpy())

    def reference(frame):
        result = np.empty(len(frame), dtype=[("event", bool), ("time", float)])
        result["event"] = (1 - frame[censor].to_numpy()).astype(bool)
        result["time"] = frame[endpoint].to_numpy()
        return result

    def source_reference(frame):
        cloned = copy.copy(factory)
        cloned.clinical_df = frame.copy(deep=True)
        return extract(cloned)

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "completed",
              "scope": "clinical/split metadata only; no features, checkpoints, or model training",
              "source_root": str(source_root), "metadata_root": str(metadata_root), "study": study,
              "endpoint": endpoint, "censorship_column": censor, "n_bins": n_bins,
              "settings": settings, "cohort_count": len(full), "source_methods": source_lines,
              "source_method_execution": "unchanged AST method bodies; imports and feature methods excluded",
              "reference_function": reference_description, "source_hashes_before": before,
              "original_edges": edges_json(factory.bins), "folds": []}
    original_reference = None
    if extract is not None:
        try:
            original_reference = source_reference(full)
        except (AttributeError, NameError, TypeError) as exc:
            reference_description.update(status="not_evaluable", reason=f"{type(exc).__name__}: {exc}")
    for fold in folds:
        train = api["SurvivalDataset"](factory, "not_loaded", "train", fold, 1).label_df.copy(deep=True)
        val = api["SurvivalDataset"](factory, "not_loaded", "val", fold, 1).label_df.copy(deep=True)
        train_ids, val_ids = set(train[case]), set(val[case])
        check(f"fold{fold}: train and validation are nonempty", len(train_ids) and len(val_ids))
        check(f"fold{fold}: train and validation disjoint", not train_ids & val_ids)
        order = train[case].tolist()
        raw_before = full.set_index(case).loc[order, [endpoint, censor]]
        selected = val.loc[val[endpoint].idxmax(), case]
        extreme = full.copy(deep=True)
        extreme_value = float(np.max(np.abs(full[endpoint].to_numpy())) * 1000 + full[endpoint].max() + 1)
        extreme.loc[extreme[case] == selected, endpoint] = extreme_value
        removed = full.loc[~full[case].isin(val_ids)].copy(deep=True)
        permuted = full.copy(deep=True)
        positions = permuted.index[permuted[case].isin(val_ids)]
        permuted.loc[positions, [endpoint, censor]] = (
            permuted.loc[positions, [endpoint, censor]].iloc[::-1].to_numpy())
        reverse = pd.concat([full.loc[~full[case].isin(val_ids)],
                             full.loc[full[case].isin(val_ids)].iloc[::-1]], ignore_index=True)
        interventions = {"one_validation_time_extreme": extreme, "remove_validation_rows": removed,
                         "permute_validation_outcome_pairs": permuted, "reverse_validation_row_order": reverse}
        baseline_fixed = {}
        entry = {"fold": fold, "train_count": len(train), "validation_count": len(val),
                 "split_covers_clinical_cohort": train_ids | val_ids == set(full[case]),
                 "extreme_intervention": {"patient_id": str(selected), "time": extreme_value},
                 "train_only_baselines": {}, "interventions": []}
        for mode in ("source_estimator_train_only", "train_uncensored_quantiles"):
            try:
                edges = fit_edges(train, mode)
            except ValueError as exc:
                # Quantile ties/too few events are eligibility limits, not successes.
                entry["train_only_baselines"][mode] = {"status": "not_evaluable", "reason": str(exc)}
                if mode == "source_estimator_train_only":
                    raise
                continue
            base = fixed_labels(full, edges).loc[order]
            baseline_fixed[mode] = edges, base
            entry["train_only_baselines"][mode] = {"status": "completed", "edges": edges_json(edges),
                                                   "train_label_counts": [int((base == i).sum()) for i in range(n_bins)]}
        fixed_reference = reference(train)
        check(f"fold{fold}: corrected reference excludes validation IDs", not set(train[case]) & val_ids)
        ref_info = {"status": "not_evaluable", "corrected_count": len(fixed_reference),
                    "corrected_validation_overlap": 0}
        if original_reference is not None:
            if np.array_equal(original_reference, reference(full)):
                ref_info.update(status="observed", original_count=len(original_reference),
                                original_validation_overlap=len(val_ids), original_scope="full clinical cohort")
            elif np.array_equal(original_reference, fixed_reference):
                ref_info.update(status="not_observed", original_count=len(original_reference),
                                original_validation_overlap=0, original_scope="training rows")
            else:
                ref_info.update(original_count=len(original_reference),
                                reason="Returned metadata cannot be mapped unambiguously to source patient IDs")
        entry["survival_reference"] = ref_info
        for name, altered in interventions.items():
            check(f"fold{fold}/{name}: training raw outcomes unchanged",
                  raw_before.equals(altered.set_index(case).loc[order, [endpoint, censor]]))
            original_after = relabel(altered)
            changed = int((labels(full).loc[order] != labels(original_after.clinical_df).loc[order]).sum())
            intervention = {"name": name, "training_raw_outcomes_changed": 0,
                            "original_training_labels_changed": changed,
                            "original_edges": edges_json(original_after.bins), "corrections": {}}
            if name in ("permute_validation_outcome_pairs", "reverse_validation_row_order"):
                intervention["negative_control_invariant"] = changed == 0
            for mode, (base_edges, base_labels) in baseline_fixed.items():
                altered_train = altered.loc[altered[case].isin(train_ids)]
                after_edges = fit_edges(altered_train, mode)
                after_labels = fixed_labels(altered, after_edges).loc[order]
                fixed_changed = int((base_labels != after_labels).sum())
                check(f"fold{fold}/{name}/{mode}: train-only boundaries invariant",
                      np.array_equal(base_edges, after_edges))
                check(f"fold{fold}/{name}/{mode}: train-only labels invariant", fixed_changed == 0)
                intervention["corrections"][mode] = {"boundaries_unchanged": True,
                                                       "training_labels_changed": fixed_changed}
            # Preserve reference ordering even when interventions reorder other rows.
            after_train = altered.set_index(case).loc[order].reset_index()
            check(f"fold{fold}/{name}: train-only survival reference invariant",
                  np.array_equal(fixed_reference, reference(after_train)))
            intervention["corrected_reference_unchanged"] = True
            entry["interventions"].append(intervention)
        entry["held_out_label_dependence"] = (
            "observed" if any(item["original_training_labels_changed"] > 0
                              for item in entry["interventions"][:2]) else "not_observed")
        report["folds"].append(entry)
    report["held_out_label_dependence"] = (
        "observed" if any(item["held_out_label_dependence"] == "observed" for item in report["folds"])
        else "not_observed")
    report["source_hashes_after"] = {str(path): _digest(path) for path in paths}
    check("original source and metadata files unchanged", before == report["source_hashes_after"])
    report.update(original_files_unchanged=True, assertions=ledger, assertion_count=len(ledger))
    report["interpretation_limits"] = [
        "An extreme validation time is a causal counterexample, not an estimate of normal-data effect size.",
        "A not_observed result means these interventions did not reveal the issue in this source/cohort/fold.",
        "Reference membership does not measure the resulting metric bias.",
        "This checks isolated target/reference corrections; patient training and predictive improvement are not measured.",
        "These protocol defects and fixes are not evidence that OT is necessary or beneficial.",
    ]
    return _save(output, report)


def run_reconstruction_checks(source_root, output_dir):
    """Two-seed, small, zero-update reconstruction counterexample and controls.

    The random frozen target is a measurement control, not a proposed clinical
    model. Success of the construction never asserts actual training collapse.
    """
    import torch
    from torch import nn
    import torch.nn.functional as F
    from .model import build_model, make_args

    output = _report_path(output_dir, "reconstruction_checks.json")
    started = time.monotonic()
    source_root = Path(source_root).resolve()
    paths = [source_root / name for name in (
        "models/SlotSPE.py", "models/slot_attention.py", "models/transformer.py",
        "models/omics_encoder.py", "utils/loss_func.py")]
    before = {str(path): _digest(path) for path in paths}
    seeds = (0, 1)
    args = make_args([3, 4, 5], 16)
    args.wsi_projection_dim, args.slot_num_wsi, args.slot_num_omics = 32, 8, 8
    args.slot_iters, args.n_classes = 3, 4
    args.rna_format = "Pathways"
    batch = args.n_classes * 2
    atol = 1e-7
    ledger = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        ledger.append({"assertion": name, "passed": True})

    def data_for(seed):
        generator = torch.Generator().manual_seed(seed)
        data = {"x_wsi": torch.randn(batch, 16, args.encoding_dim, generator=generator),
                "omic_missing": False, "y": torch.arange(batch) % args.n_classes,
                "c": (torch.arange(batch) >= args.n_classes).float()}
        data.update({f"x_omic{i + 1}": torch.randn(batch, size, generator=generator)
                     for i, size in enumerate(args.omic_sizes)})
        return data

    def encode(model, data):
        wsi = model.wsi_mlp(data["x_wsi"])
        omic = torch.stack([encoder(data[f"x_omic{i + 1}"])
                            for i, encoder in enumerate(model.sig_networks)], dim=1)
        return wsi, omic

    def measures(wsi, omic, rw, ro, rj, target_wsi=None, target_omic=None, detach_wsi=False):
        tw = wsi if target_wsi is None else target_wsi
        to = omic.detach() if target_omic is None else target_omic
        if detach_wsi:
            tw = tw.detach()
        lw = 1 - F.cosine_similarity(F.normalize(rw, dim=-1), F.normalize(tw, dim=-1), dim=-1).mean()
        lo, lj = F.mse_loss(ro, to), F.mse_loss(rj, to)
        return {"omics_self_mse": float(lo.detach()), "cross_mse": float(lj.detach()),
                "wsi_cosine_loss": float(lw.detach()), "sum": float((lo + lj + lw).detach())}

    def deterministic(model, data):
        wsi, omic = encode(model, data)
        ws, oms = model.slot_attention_wsi(wsi), model.slot_attention_omic(omic)
        _, _, wk = model.slot_decoder_wsi(ws)
        _, _, ok = model.slot_decoder_omic(oms)
        ro = model.reconstruction_head_omic(omic, oms, mask=ok.bool())
        rj = model.reconstruction_head_joint(omic, model.slot_attention_omic(wsi))
        rw = model.reconstruction_head_wsi(wsi, ws, mask=wk.bool())
        return wsi, omic, rw, ro, rj

    def capture_training(model, data, seed):
        captured, handles = {}, []
        def hook(name):
            def capture(module, inputs, result):
                captured[name] = result
            return capture
        for name in ("wsi_mlp", "reconstruction_head_omic", "reconstruction_head_joint",
                     "reconstruction_head_wsi", "slot_decoder_omic", "slot_decoder_wsi"):
            handles.append(getattr(model, name).register_forward_hook(hook(name)))
        # The source calls outer encoder.forward directly; hook its last child.
        for i, encoder in enumerate(model.sig_networks):
            handles.append(encoder[-1].register_forward_hook(hook(f"omic_{i}")))
        try:
            torch.manual_seed(seed)
            logits, auxiliary = model(**data)
        finally:
            for handle in handles:
                handle.remove()
        wsi = captured["wsi_mlp"]
        omic = torch.stack([captured[f"omic_{i}"] for i in range(len(args.omic_sizes))], dim=1)
        result = measures(wsi, omic, captured["reconstruction_head_wsi"],
                          captured["reconstruction_head_omic"], captured["reconstruction_head_joint"])
        decoder_loss = sum(model.loss_fn(captured[name][0], y=data["y"], c=data["c"], t=None)
                           for name in ("slot_decoder_wsi", "slot_decoder_omic")) / batch
        expected_aux = float(decoder_loss.detach()) + args.lambda_recon_loss * result["sum"]
        error = abs(float(auxiliary.detach()) - expected_aux)
        check(f"seed{seed}: original auxiliary formula reproduced", error < 2e-6)
        return dict(result, encoded_wsi_patient_variance=float(wsi.detach().var(0).mean()),
                    encoded_omics_patient_variance=float(omic.detach().var(0).mean()),
                    decoder_survival_loss=float(decoder_loss.detach()),
                    final_survival_loss=float((model.loss_fn(logits, y=data["y"], c=data["c"], t=None) / batch).detach()),
                    auxiliary_formula_absolute_error=error)

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "completed",
              "scope": "controlled synthetic construction, not patient training", "source_root": str(source_root),
              "torch": torch.__version__, "device": "cpu", "seeds": list(seeds), "optimizer_steps": 0,
              "trained_checkpoint_loaded": False, "source_hashes_before": before,
              "probe_configuration": vars(args).copy(), "deterministic_rows": [],
              "training_mode_rows": [], "fixed_reference_rows": []}
    # Keep the caller's random state intact; no process-global thread/mode changes.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(54321)
        reference = build_model(source_root, args, "original").cpu().eval()
        reference.requires_grad_(False)
        reference_before = {name: value.detach().clone() for name, value in reference.state_dict().items()}
        model = build_model(source_root, args, "original").cpu()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            # A nonzero e_1 avoids zero-vector ambiguity in cosine similarity.
            for encoder in [model.wsi_mlp, *model.sig_networks]:
                linears = [module for module in encoder.modules() if isinstance(module, nn.Linear)]
                if not linears or linears[-1].bias is None:
                    raise ValueError("Source encoder does not support this controlled constant construction")
                linears[-1].bias[0] = 1
            for name in ("reconstruction_head_omic", "reconstruction_head_joint", "reconstruction_head_wsi"):
                getattr(model, name).cross_attn.out_proj.bias[0] = 1
        for seed in seeds:
            data = data_for(seed)
            raw_variances = [float(data["x_wsi"].var(0).mean())] + [
                float(data[f"x_omic{i + 1}"].var(0).mean()) for i in range(len(args.omic_sizes))]
            check(f"seed{seed}: distinct synthetic raw inputs", min(raw_variances) > 0)
            with torch.no_grad():
                target_wsi, target_omic = encode(reference, data)
                model.eval()
                torch.manual_seed(seed + 10000)
                tensors = deterministic(model, data)
                wsi, omic = tensors[:2]
                result = measures(*tensors)
                detached = measures(*tensors, detach_wsi=True)
                constant = float(wsi.var(0).mean()) == 0 and float(omic.var(0).mean()) == 0
                observed = constant and abs(result["sum"]) < atol
                report["deterministic_rows"].append({"seed": seed,
                    "raw_patient_variances": raw_variances, "encoded_wsi_patient_variance": float(wsi.var(0).mean()),
                    "encoded_omics_patient_variance": float(omic.var(0).mean()),
                    "dropout_off_original_formula": result, "detach_wsi_only_control": detached,
                    "constant_zero_reconstruction": "observed" if observed else "not_observed"})
                check(f"seed{seed}: detach preserves forward loss", abs(result["sum"] - detached["sum"]) < atol)
                fixed = measures(*tensors, target_wsi=target_wsi, target_omic=target_omic)
                mse_bound = float((target_omic - target_omic.mean(0, keepdim=True)).square().mean())
                cosine_bound = float(1 - F.normalize(target_wsi, dim=-1).mean(0).norm(dim=-1).mean())
                check(f"seed{seed}: random frozen reference is nondegenerate", mse_bound > 1e-8 and cosine_bound > 1e-8)
                check(f"seed{seed}: frozen reference rejects constant predictions",
                      fixed["omics_self_mse"] >= mse_bound - 1e-6 and fixed["cross_mse"] >= mse_bound - 1e-6
                      and fixed["wsi_cosine_loss"] >= cosine_bound - 1e-6)
                report["fixed_reference_rows"].append({"seed": seed, "frozen_before_student": True,
                    "random_reference_not_semantically_validated": True, "constructed_student_loss": fixed,
                    "patient_independent_per_pathway_mse_lower_bound": mse_bound,
                    "patient_independent_per_patch_cosine_loss_lower_bound": cosine_bound})
            for mode in ("train_dropout_disabled_control", "train_original_dropout"):
                model.train()
                if mode == "train_dropout_disabled_control":
                    for module in model.modules():
                        if isinstance(module, (nn.Dropout, nn.AlphaDropout, nn.FeatureAlphaDropout)):
                            module.eval()
                with torch.no_grad():
                    row = capture_training(model, data, seed + 20000)
                report["training_mode_rows"].append(dict(row, seed=seed, mode=mode))
        data = data_for(seeds[-1])
        changed = {name: value + 10 if name.startswith("x_") else value for name, value in data.items()}
        model.train()
        with torch.no_grad():
            torch.manual_seed(9988)
            w1, o1 = encode(model, data)
            torch.manual_seed(9988)
            w2, o2 = encode(model, changed)
        invariant = torch.equal(w1, w2) and torch.equal(o1, o2)
        report["train_dropout_raw_input_intervention"] = {
            "rng_fixed": True, "raw_input_additive_change": 10,
            "encoded_wsi_max_abs_change": float((w1 - w2).abs().max()),
            "encoded_omics_max_abs_change": float((o1 - o2).abs().max()),
            "input_information_absent_at_constructed_weights": bool(invariant)}
        check("reference parameters and buffers unchanged", all(
            torch.equal(reference_before[name], value) for name, value in reference.state_dict().items()))
    report["constant_zero_reconstruction_dropout_off"] = (
        "observed" if all(row["constant_zero_reconstruction"] == "observed" for row in report["deterministic_rows"])
        else "not_observed")
    original_training = [row for row in report["training_mode_rows"] if row["mode"] == "train_original_dropout"]
    report["same_construction_zero_with_original_training_dropout"] = (
        "observed" if all(abs(row["sum"]) < atol for row in original_training) else "not_observed")
    report["source_hashes_after"] = {str(path): _digest(path) for path in paths}
    check("original source files unchanged", before == report["source_hashes_after"])
    report.update(original_files_unchanged=True, assertions=ledger, assertion_count=len(ledger),
                  elapsed_seconds=time.monotonic() - started)
    report["claims_not_supported"] = [
        "Actual collapse during patient training, or a zero optimum of the full stochastic training objective.",
        "Low reconstruction error establishes feature disentanglement or retention of raw-input information.",
        "Detaching a target alone excludes constant representations.",
        "A frozen random reference is a clinically useful innovation.",
        "These probes establish predictive benefit, or necessity of OT.",
    ]
    return _save(output, report)
