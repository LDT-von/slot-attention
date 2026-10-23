"""Opt-in matched direct-diversity experiments on newly generated tensors.

This script has no patient loader, external checkpoint, configurable path, or
server entry point. Without --synthetic it prints the frozen protocol only.
Original SlotSPE source and its native slot tensors are never rewritten.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "local_evidence/direct_diversity_synthetic_20261022_run01"
ARMS = {
    "R0": {"regularizer": "none", "regularizer_weight": 0.0, "set_weight": 0.0},
    "R1": {"regularizer": "off_diagonal_cosine_squared", "regularizer_weight": 10.0, "set_weight": 0.0},
    "R2": {"regularizer": "SlotBERT_Eq11_transplant", "regularizer_weight": 1.0, "set_weight": 0.0},
    "R3": {"regularizer": "SlotBERT_Eq11_transplant", "regularizer_weight": 1.0, "set_weight": 5.0},
    "R4": {"regularizer": "none", "regularizer_weight": 0.0, "set_weight": 5.0},
    "R5": {"regularizer": "off_diagonal_cosine_squared", "regularizer_weight": 10.0, "set_weight": 5.0},
}
PLAN = {
    "data": "new invented 8-component tensors only; no patient input",
    "variants": ARMS, "seeds": [3, 17, 29],
    "train_size": 64, "holdout_size": 32, "steps": 128, "batch_size": 8,
    "eval_repeats": 3, "projection_dim": 32, "slots": 8, "iterations": 3,
    "encoding_dim": 16, "pathways": 8, "pathway_dim": 16, "patches": 32,
    "optimizer": "Adam", "learning_rate": 0.001, "device": "cpu", "threads": 1,
    "native_loss": "unchanged native main NLL / B + native auxiliary loss",
    "regularizer_reduction": "mean over the two prediction-path modality outputs",
    "slotbert_temperature": 0.5,
    "slotbert_scope": "Eq11 loss transplant only; batch examples replace frames; diagonal retained",
    "slotbert_weight_boundary": "weight 1 is this experiment's preset, not the paper default 0.01",
    "probe": "same shared per-slot MLPs in all arms; slots are the only decoder input",
    "probe_updates": "detached-slot probe step, then frozen-probe core step",
    "set_targets": "WSI per-case raw-token farthest/Lloyd clusters; OMIC raw pathway set",
    "target_component_labels": "not used by training; train-mean oracle dictionary is evaluation only",
    "pairing": "same seed-specific initial core/probe states, generated inputs, batches and step RNG",
    "selection": "fixed final step; no heldout tuning or best-checkpoint selection",
    "effective_rank_definition": "angular participation ratio K^2 / sum(G^2), not algebraic rank",
    "content_boundary": "probe and donor checks measure generated content, not survival relevance",
    "patient_execution": False, "server_execution": False,
    "expected_runs": 18, "expected_core_optimizer_steps": 2304,
    "expected_probe_optimizer_steps": 2304, "expected_root_train_forwards": 2304,
    "expected_root_eval_forwards": 216, "expected_train_SA_calls": 6912,
    "expected_eval_SA_calls": 432,
    "comparison_boundary": "new richer benchmark; do not merge historical S0-S6 measurements",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="execute the fixed generated-only protocol")
    options = parser.parse_args()
    if not options.synthetic:
        print(json.dumps(PLAN, ensure_ascii=False, indent=2))
        return
    if OUTPUT.resolve() != (ROOT / "local_evidence/direct_diversity_synthetic_20261022_run01").resolve():
        raise ValueError("only the fixed ignored output directory is permitted")
    if OUTPUT.exists():
        raise FileExistsError("preserve prior evidence: fixed output directory already exists")
    if "/local_evidence/" not in (ROOT / ".gitignore").read_text(encoding="utf-8"):
        raise ValueError("local evidence directory must be ignored")

    import torch
    from scipy.optimize import linear_sum_assignment
    from torch.nn import functional as F
    from survot_rank.research.slotspe_proof.model import build_model, make_args
    from survot_rank.research.slotspe_proof.slot_regularization import (
        QueryFreeSlotDecoder, SlotCapture, matched_set_mse,
        slot_gram_loss, slot_metrics, slotbert_loss,
    )

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protected = list((ROOT / "third_party/SlotSPE").rglob("*.py"))
    for name in ("slotspe_clean_original_server.json", "slotspe_proof_server.json",
                 "slotspe_attribution_server.json", "slotspe_problem_experiments.json",
                 "slotspe_native_variants.json"):
        path = ROOT / "configs" / name
        if path.exists():
            protected.append(path)
    protected_before = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(protected)}
    implementation = [Path(__file__), ROOT / "survot_rank/research/slotspe_proof/slot_regularization.py",
                      ROOT / "survot_rank/research/slotspe_proof/model.py"]
    implementation_before = {p.relative_to(ROOT).as_posix(): digest(p) for p in implementation}
    OUTPUT.mkdir(parents=True)
    snapshot = OUTPUT / "executed_implementation"
    snapshot.mkdir()
    for path in implementation:
        destination = snapshot / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    summary = {
        "schema": "native_slotspe_generated_direct_diversity_v1",
        "status": "running_generated_only", "recorded_at": datetime.now().astimezone().isoformat(),
        "plan": PLAN, "protected_sha256_before": protected_before,
        "implementation_sha256": implementation_before,
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "device": "cpu", "threads": 1},
        "counters": {"root_train_forwards": 0, "root_eval_forwards": 0,
                     "train_SA_calls": 0, "eval_SA_calls": 0,
                     "core_optimizer_steps": 0, "probe_optimizer_steps": 0},
        "runs": [], "patient_runs": 0, "server_runs": 0,
    }

    def flush():
        write_json(OUTPUT / "summary.json", summary)

    def finite_tree(value):
        if torch.is_tensor(value):
            return not value.is_floating_point() or bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite_tree(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_tree(item) for item in value)
        return True

    def state_digest(state):
        result = hashlib.sha256()
        for name in sorted(state):
            tensor = state[name].detach().cpu().contiguous()
            result.update(name.encode("utf-8"))
            result.update(str((tensor.dtype, tuple(tensor.shape))).encode("ascii"))
            result.update(tensor.numpy().tobytes())
        return result.hexdigest()

    def build():
        args = make_args([16] * 8, encoding_dim=16)
        args.wsi_projection_dim = 32
        args.slot_iters = 3
        return build_model(ROOT / "third_party/SlotSPE", args, "original")

    def build_heads():
        return torch.nn.ModuleDict({modality: QueryFreeSlotDecoder(32, 16) for modality in ("wsi", "omic")})

    def sliced(data, indices):
        return {key: value[indices] if torch.is_tensor(value) else value for key, value in data.items()}

    def primary(model, logits, batch):
        return model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None) / len(batch["y"])

    def clusters(tokens):
        # All choices depend solely on this example's unordered raw tokens.
        result = []
        for points in tokens:
            chosen = [int(points.square().sum(-1).argmax())]
            while len(chosen) < 8:
                distances = (points[:, None] - points[chosen][None]).square().sum(-1).min(-1).values
                distances[chosen] = -1
                chosen.append(int(distances.argmax()))
            centers = points[chosen].clone()
            for _ in range(10):
                labels = (points[:, None] - centers[None]).square().sum(-1).argmin(-1)
                updated = torch.stack([points[labels == index].mean(0) if bool((labels == index).any()) else centers[index]
                                       for index in range(8)])
                if torch.equal(updated, centers):
                    break
                centers = updated
            result.append(centers)
        return torch.stack(result)

    def generate(seed):
        generator = torch.Generator(device="cpu").manual_seed(910000 + seed)
        wsi_state = torch.randn(96, 8, generator=generator)
        omic_state = torch.randn(96, 8, generator=generator)
        identity = torch.eye(8)
        wsi_centers = torch.cat([2.0 * identity.expand(96, -1, -1),
                                 0.8 * identity[None] * wsi_state[:, :, None]], dim=-1)
        omic_centers = torch.cat([2.0 * identity.expand(96, -1, -1),
                                  0.8 * identity[None] * omic_state[:, :, None]], dim=-1)
        patches = wsi_centers.repeat_interleave(4, dim=1) + 0.05 * torch.randn(96, 32, 16, generator=generator)
        for index in range(len(patches)):
            patches[index] = patches[index, torch.randperm(32, generator=generator)]
        pathways = omic_centers + 0.05 * torch.randn(96, 8, 16, generator=generator)
        score = wsi_state[:, 0] - 0.7 * wsi_state[:, 3] + 0.8 * omic_state[:, 2] + 0.5 * omic_state[:, 6]
        times = torch.exp(0.15 * (score + 0.05 * torch.randn(96, generator=generator)))
        bins = torch.quantile(times[:64], torch.tensor([0.25, 0.5, 0.75]))
        labels = torch.bucketize(times, bins).long()
        data = {"x_wsi": patches, "omic_missing": False, "y": labels, "c": torch.zeros(96)}
        data.update({f"x_omic{index + 1}": pathways[:, index] for index in range(8)})
        targets = {"wsi": clusters(patches), "omic": pathways.clone()}
        static_dictionary = {"wsi": wsi_centers[:64].mean(0), "omic": omic_centers[:64].mean(0)}
        batches = [torch.randperm(64, generator=generator)[:8] for _ in range(PLAN["steps"])]
        torch.manual_seed(seed * 100000 + 100)
        core_state = copy.deepcopy(build().state_dict())
        torch.manual_seed(seed * 100000 + 101)
        probe_state = copy.deepcopy(build_heads().state_dict())
        payload = {"data": data, "targets": targets, "static_train_dictionary": static_dictionary,
                   "generated_states_evaluation_only": {"wsi": wsi_state, "omic": omic_state},
                   "true_centroids_evaluation_only": {"wsi": wsi_centers, "omic": omic_centers},
                   "times_evaluation_only": times, "bins_train_only": bins, "batches": batches,
                   "initial_core_state": core_state, "initial_probe_state": probe_state}
        if not finite_tree(payload):
            raise RuntimeError("generated-only payload is nonfinite")
        torch.save(payload, OUTPUT / f"generated_seed_{seed}.pt")
        return payload

    def prediction_slots(capture, training):
        expected = {"wsi": 1, "omic": 2 if training else 1}
        if {name: len(values) for name, values in capture.slots.items()} != expected:
            raise RuntimeError("native prediction/reconstruction call pattern differs")
        result = {name: values[0] for name, values in capture.slots.items()}
        if any(value.shape[1:] != (8, 32) for value in result.values()):
            raise RuntimeError("all eight unmodified native slots must be retained")
        return result

    def mean_rows(rows):
        result = {}
        for key in rows[0]:
            values = [row[key] for row in rows]
            if key == "min_norm":
                result[key] = min(values)
            elif key == "max_cosine":
                result[key] = max(values)
            elif key == "slot_count":
                if set(values) != {8}:
                    raise RuntimeError("a reported arm changed slot count")
                result[key] = 8
            elif key == "case_count":
                result[key] = sum(values)
            else:
                result[key] = sum(values) / len(values)
        return result

    def matched_angular_distance(first, second):
        first, second = F.normalize(first.double(), dim=-1), F.normalize(second.double(), dim=-1)
        costs = 1.0 - first @ second.transpose(-1, -2)
        values = []
        for cost in costs:
            rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
            values.append(cost[rows, columns].mean())
        return float(torch.stack(values).mean())

    def evaluate(model, heads, capture, data, targets, static_dictionary, seed):
        model.eval()
        heads.eval()
        measurements = {"wsi": [], "omic": []}
        raw, main_values, auxiliary_values = [], [], []
        donor_index = torch.arange(len(data["y"])).roll(1)
        donor_data = sliced(data, donor_index)
        for repeat in range(PLAN["eval_repeats"]):
            rng_seed = seed * 100000 + 50000 + repeat
            with torch.no_grad():
                torch.manual_seed(rng_seed)
                logits, auxiliary = model(**data)
                slots = {name: value.detach().clone() for name, value in prediction_slots(capture, False).items()}
                main_loss = primary(model, logits, data)
                if not finite_tree((logits, auxiliary, main_loss, slots)):
                    raise RuntimeError("nonfinite generated holdout output")
                main_values.append(float(main_loss))
                auxiliary_values.append(float(auxiliary))
                predictions = {name: heads[name](value) for name, value in slots.items()}
                torch.manual_seed(rng_seed)
                donor_logits, donor_auxiliary = model(**donor_data)
                donor_slots = {name: value.detach().clone() for name, value in prediction_slots(capture, False).items()}
                if not finite_tree((donor_logits, donor_auxiliary, donor_slots)):
                    raise RuntimeError("nonfinite generated donor-control output")
                for name in slots:
                    own = matched_set_mse(predictions[name], targets[name], reduction="none")
                    donor = matched_set_mse(predictions[name][donor_index], targets[name], reduction="none")
                    static = matched_set_mse(static_dictionary[name][None].expand(len(own), -1, -1),
                                             targets[name], reduction="none")
                    row = slot_metrics(slots[name])
                    row.update({"own_set_mse": float(own.mean()), "donor_set_mse": float(donor.mean()),
                                "donor_to_own_ratio": float(donor.mean() / own.mean().clamp_min(1e-12)),
                                "static_dictionary_set_mse": float(static.mean()),
                                "own_to_static_ratio": float(own.mean() / static.mean().clamp_min(1e-12)),
                                "same_noise_input_change_angular_distance": matched_angular_distance(slots[name], donor_slots[name])})
                    measurements[name].append(row)
                raw.append({"repeat": repeat, "logits": logits.detach().clone(), "slots": slots,
                            "probe_predictions": {name: value.detach().clone() for name, value in predictions.items()},
                            "same_noise_donor_input_slots": donor_slots,
                            "per_case_set_mse": {name: {
                                "own": matched_set_mse(predictions[name], targets[name], reduction="none"),
                                "donor": matched_set_mse(predictions[name][donor_index], targets[name], reduction="none"),
                                "static": matched_set_mse(static_dictionary[name][None].expand(len(data["y"]), -1, -1),
                                                          targets[name], reduction="none")}
                                for name in slots}})
        return {"native_main_nll": sum(main_values) / len(main_values),
                "native_auxiliary": sum(auxiliary_values) / len(auxiliary_values),
                "modalities": {name: mean_rows(rows) for name, rows in measurements.items()},
                "per_draw_metrics": measurements}, raw

    def check_gradients(module):
        return all(bool(torch.isfinite(parameter.grad).all()) for parameter in module.parameters()
                   if parameter.grad is not None)

    flush()
    active_model = active_heads = None
    active_losses = []
    try:
        for seed in PLAN["seeds"]:
            payload = generate(seed)
            data, targets = payload["data"], payload["targets"]
            train, holdout = sliced(data, slice(0, 64)), sliced(data, slice(64, 96))
            train_targets = {name: values[:64] for name, values in targets.items()}
            holdout_targets = {name: values[64:] for name, values in targets.items()}
            expected_core_hash = state_digest(payload["initial_core_state"])
            expected_probe_hash = state_digest(payload["initial_probe_state"])
            for arm, config in ARMS.items():
                started = time.perf_counter()
                model, heads = build(), build_heads()
                active_model, active_heads = model, heads
                model.load_state_dict(payload["initial_core_state"], strict=True)
                heads.load_state_dict(payload["initial_probe_state"], strict=True)
                initial_hashes = {"core": state_digest(model.state_dict()), "probe": state_digest(heads.state_dict())}
                if initial_hashes != {"core": expected_core_hash, "probe": expected_probe_hash}:
                    raise RuntimeError("paired seed-specific initial states differ")
                core_optimizer = torch.optim.Adam(model.parameters(), lr=PLAN["learning_rate"])
                probe_optimizer = torch.optim.Adam(heads.parameters(), lr=PLAN["learning_rate"])
                counts_before = copy.deepcopy(summary["counters"])

                def root_counter(module, _args):
                    key = "root_train_forwards" if module.training else "root_eval_forwards"
                    summary["counters"][key] += 1

                def sa_counter(_module, _args, _output):
                    key = "train_SA_calls" if model.training else "eval_SA_calls"
                    summary["counters"][key] += 1

                handles = [model.register_forward_pre_hook(root_counter)]
                handles.extend(getattr(model, "slot_attention_" + name).register_forward_hook(sa_counter)
                               for name in ("wsi", "omic"))
                losses = []
                active_losses = losses
                summary["active_run"] = {"variant": arm, "seed": seed, "completed_core_steps": 0}
                flush()
                try:
                    with SlotCapture(model) as capture:
                        initial, initial_raw = evaluate(model, heads, capture, holdout, holdout_targets,
                                                        payload["static_train_dictionary"], seed)
                        for step, indices in enumerate(payload["batches"]):
                            model.train()
                            heads.train()
                            core_optimizer.zero_grad(set_to_none=True)
                            probe_optimizer.zero_grad(set_to_none=True)
                            batch = sliced(train, indices)
                            batch_targets = {name: value[indices] for name, value in train_targets.items()}
                            # Reset after construction and before every native
                            # forward. The probe MLP has no stochastic layers.
                            torch.manual_seed(seed * 100000 + step)
                            logits, auxiliary = model(**batch)
                            slots = prediction_slots(capture, True)
                            main_loss = primary(model, logits, batch)
                            probe_parts = {name: matched_set_mse(heads[name](value.detach()), batch_targets[name])
                                           for name, value in slots.items()}
                            probe_loss = torch.stack(list(probe_parts.values())).mean()
                            if not finite_tree((logits, auxiliary, main_loss, slots, probe_loss)):
                                raise RuntimeError(f"nonfinite forward {arm}/{seed}/{step}")
                            probe_loss.backward()
                            if not check_gradients(heads):
                                raise RuntimeError("nonfinite detached-probe gradient")
                            if any(parameter.grad is not None for parameter in model.parameters()):
                                raise RuntimeError("detached probe update leaked into core gradients")
                            probe_optimizer.step()
                            summary["counters"]["probe_optimizer_steps"] += 1
                            for parameter in heads.parameters():
                                parameter.requires_grad_(False)
                            try:
                                gram = torch.stack([slot_gram_loss(value) for value in slots.values()]).mean()
                                slotbert = torch.stack([slotbert_loss(value, PLAN["slotbert_temperature"])
                                                        for value in slots.values()]).mean()
                                if config["set_weight"]:
                                    content = torch.stack([matched_set_mse(heads[name](value), batch_targets[name])
                                                           for name, value in slots.items()]).mean()
                                else:
                                    with torch.no_grad():
                                        content = torch.stack([matched_set_mse(heads[name](value), batch_targets[name])
                                                               for name, value in slots.items()]).mean()
                                loss = main_loss + auxiliary
                                if config["regularizer"] == "off_diagonal_cosine_squared":
                                    loss = loss + config["regularizer_weight"] * gram
                                elif config["regularizer"] == "SlotBERT_Eq11_transplant":
                                    loss = loss + config["regularizer_weight"] * slotbert
                                if config["set_weight"]:
                                    loss = loss + config["set_weight"] * content
                                if not finite_tree((loss, gram, slotbert, content)):
                                    raise RuntimeError("nonfinite direct-diversity loss")
                                loss.backward()
                                if not check_gradients(model):
                                    raise RuntimeError("nonfinite core gradient")
                                core_optimizer.step()
                                summary["counters"]["core_optimizer_steps"] += 1
                            finally:
                                for parameter in heads.parameters():
                                    parameter.requires_grad_(True)
                            if not finite_tree((model.state_dict(), heads.state_dict())):
                                raise RuntimeError("nonfinite post-update core/probe state")
                            losses.append({"step": step, "native_main": float(main_loss.detach()),
                                           "native_auxiliary": float(auxiliary.detach()),
                                           "gram_raw": float(gram.detach()), "slotbert_raw": float(slotbert.detach()),
                                           "content_raw": float(content.detach()),
                                           "detached_probe_loss": float(probe_loss.detach()),
                                           "total_core_loss": float(loss.detach())})
                            summary["active_run"]["completed_core_steps"] = len(losses)
                            if (step + 1) % 32 == 0:
                                flush()
                        final, final_raw = evaluate(model, heads, capture, holdout, holdout_targets,
                                                    payload["static_train_dictionary"], seed)
                finally:
                    for handle in handles:
                        handle.remove()
                run = {"seed": seed, "variant": arm, "config": config,
                       "initial_state_sha256": initial_hashes, "core_optimizer_steps": len(losses),
                       "initial_holdout": initial, "final_holdout": final,
                       "final_train": losses[-1], "all_finite": True,
                       "execution_counts": {name: value - counts_before[name]
                                            for name, value in summary["counters"].items()},
                       "seconds": time.perf_counter() - started}
                torch.save({"core_state": model.state_dict(), "probe_state": heads.state_dict(),
                            "core_optimizer_state": core_optimizer.state_dict(),
                            "probe_optimizer_state": probe_optimizer.state_dict(), "losses": losses,
                            "initial_holdout": initial_raw, "final_holdout": final_raw,
                            "initial_state_sha256": initial_hashes,
                            "generated_file_sha256": digest(OUTPUT / f"generated_seed_{seed}.pt")},
                           OUTPUT / f"matched_{arm}_seed_{seed}.pt")
                write_json(OUTPUT / f"numeric_{arm}_seed_{seed}.json", {"run": run, "losses": losses})
                summary["runs"].append(run)
                summary.pop("active_run", None)
                flush()
                print(f"{arm}/seed{seed}: completed 128 core + 128 probe updates; "
                      f"WSI cos={final['modalities']['wsi']['cosine']:.6f}, "
                      f"OMIC cos={final['modalities']['omic']['cosine']:.6f}, "
                      f"native NLL={final['native_main_nll']:.6f}", flush=True)
        expected_counts = {name: PLAN["expected_" + name] for name in summary["counters"]}
        if summary["counters"] != expected_counts or len(summary["runs"]) != PLAN["expected_runs"]:
            raise RuntimeError(f"execution counts differ from fixed protocol: {summary['counters']}")
        summary["aggregate_final"] = {}
        summary["paired_final_vs_R0"] = {}
        for arm in ARMS:
            selected = [run for run in summary["runs"] if run["variant"] == arm]
            summary["aggregate_final"][arm] = {
                "native_main_nll": sum(run["final_holdout"]["native_main_nll"] for run in selected) / len(selected),
                "native_auxiliary": sum(run["final_holdout"]["native_auxiliary"] for run in selected) / len(selected),
                "modalities": {name: mean_rows([run["final_holdout"]["modalities"][name] for run in selected])
                               for name in ("wsi", "omic")},
            }
            if arm != "R0":
                baseline = {run["seed"]: run for run in summary["runs"] if run["variant"] == "R0"}
                summary["paired_final_vs_R0"][arm] = {
                    name: [{"seed": run["seed"],
                            "cosine_delta": run["final_holdout"]["modalities"][name]["cosine"] - baseline[run["seed"]]["final_holdout"]["modalities"][name]["cosine"],
                            "absolute_cosine_delta": run["final_holdout"]["modalities"][name]["absolute_cosine"] - baseline[run["seed"]]["final_holdout"]["modalities"][name]["absolute_cosine"],
                            "native_nll_delta": run["final_holdout"]["native_main_nll"] - baseline[run["seed"]]["final_holdout"]["native_main_nll"]}
                           for run in selected] for name in ("wsi", "omic")}
        summary["protected_sha256_after"] = {name: digest(ROOT / name) for name in protected_before}
        summary["protected_files_unchanged"] = summary["protected_sha256_after"] == protected_before
        summary["implementation_unchanged_during_run"] = implementation_before == {name: digest(ROOT / name) for name in implementation_before}
        if not summary["protected_files_unchanged"] or not summary["implementation_unchanged_during_run"]:
            raise RuntimeError("protected or executed implementation bytes changed during the run")
        summary["raw_file_sha256"] = {path.relative_to(OUTPUT).as_posix(): digest(path)
                                      for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path.name != "summary.json"}
        summary["status"] = "completed_generated_only_matched_six_arm"
        summary["completed_at"] = datetime.now().astimezone().isoformat()
        flush()
        print("completed 18 generated-only runs, 2304 core + 2304 probe updates; patient/server=0", flush=True)
    except BaseException as error:
        summary["status"] = "failed_generated_only_partial_evidence_preserved"
        summary["failure"] = {"type": type(error).__name__, "message": str(error)}
        summary["protected_files_unchanged"] = protected_before == {name: digest(ROOT / name) for name in protected_before}
        if active_model is not None and active_heads is not None:
            torch.save({"core_state": active_model.state_dict(), "probe_state": active_heads.state_dict(),
                        "losses": active_losses, "active_run": summary.get("active_run")},
                       OUTPUT / "failure_partial_state.pt")
        flush()
        raise


if __name__ == "__main__":
    main()
