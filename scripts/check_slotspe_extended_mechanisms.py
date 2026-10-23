"""Run CPU-only generated-tensor checks against unchanged original SlotSPE.

No work occurs at import. This script uses newly initialized weights, generated
tensors, and forward passes only. It does not load checkpoints or patient data.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/research/slotspe_extended_audit_20261022/synthetic_checks.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def fixed_rng(torch, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def equal_initial_counterexamples(torch, source):
    from survot_rank.research.slotspe_proof.problem_trace import replay_slots

    checks = []
    for seed in (3, 17, 29):
        with fixed_rng(torch, seed):
            module = source.MultiHeadSlotAttention(
                num_slots=8, dim=32, heads=8, dim_head=64, iters=10
            ).cpu().eval()
            inputs = torch.randn(2, 16, 32)
            inputs[:, :8, :16] += 3.0
            inputs[:, 8:, 16:] -= 3.0
        states = []
        calls = [0]
        native_randn = torch.randn

        def same_noise(*shape, **kwargs):
            dimensions = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else tuple(shape)
            if dimensions == (2, 8, 32):
                calls[0] += 1
                return torch.zeros(*shape, **kwargs)
            return native_randn(*shape, **kwargs)

        def capture_slots(_module, args):
            states.append(args[0].detach().clone())
            return None

        handle = module.norm_slots.register_forward_pre_hook(capture_slots)
        try:
            with torch.no_grad(), fixed_rng(torch, seed + 1000), patch.object(torch, "randn", same_noise):
                final = module(inputs)
        finally:
            handle.remove()
        if calls[0] != 1 or len(states) != 10:
            raise RuntimeError("Initial-noise interception or native iteration capture did not match source")
        states.append(final.detach().clone())
        with torch.no_grad():
            replay = replay_slots(module, inputs, states[0])
        replay_errors = [float((a - b).abs().max()) for a, b in zip(states, replay["states"])]
        replay_accepted = len(replay["states"]) == len(states) and all(
            torch.allclose(a, b, atol=1e-6, rtol=1e-5)
            for a, b in zip(states, replay["states"])
        )
        if not replay_accepted:
            raise RuntimeError("Native fixed-noise run and deterministic replay disagree")
        spreads = [float((value - value[:, :1]).abs().max()) for value in states]
        checks.append({
            "seed": seed,
            "new_random_weights": True,
            "configuration": {"num_slots": 8, "dim": 32, "heads": 8, "dim_head": 64, "iters": 10},
            "input_shape": list(inputs.shape),
            "input_token_max_difference_from_first": float((inputs - inputs[:, :1]).abs().max()),
            "initial_noise": "exact zeros, intercepted only at native initialization shape [2,8,32]",
            "initialization_calls_intercepted": calls[0],
            "native_iterations": 10,
            "native_replay_all_states_accepted": replay_accepted,
            "native_replay_max_abs_error_by_state": replay_errors,
            "max_slot_difference_from_first_by_state": spreads,
            "exactly_identical_slots_at_all_captured_states": all(value == 0.0 for value in spreads),
        })
    return {
        "status": "generated_counterexample_observed" if all(row["exactly_identical_slots_at_all_captured_states"] for row in checks) else "counterexample_not_observed",
        "standalone_native_sa_forward_calls": len(checks),
        "deterministic_replay_calls": len(checks),
        "runs": checks,
        "claim": "With exactly equal initial slots, this original update cannot split slot identity in these diverse generated inputs, even across ten iterations.",
        "limits": "Conditional symmetry/capability counterexample; it does not measure natural collision frequency, training-induced collapse, or patient harm. Native random initialization is usually unequal.",
    }


def generated_batch(torch):
    return {
        "x_wsi": torch.randn(2, 16, 32),
        "x_omic1": torch.randn(2, 5),
        "x_omic2": torch.randn(2, 7),
        "x_omic3": torch.randn(2, 3),
        "omic_missing": False,
        "y": torch.tensor([1, 2], dtype=torch.long),
        "c": torch.tensor([0.0, 1.0]),
    }


def full_model_trace(torch, model, batch):
    traces = {}
    names = (
        "slot_attention_wsi", "slot_attention_omic", "slot_decoder_wsi", "slot_decoder_omic",
        "cross_attention", "self_attention_wsi", "self_attention_omic",
        "reconstruction_head_omic", "reconstruction_head_joint", "reconstruction_head_wsi",
    )
    for training in (False, True):
        events = []
        handles = []
        projection = {}

        def capture_projection(_module, _args, output):
            projection["wsi"] = output.detach()
            return None

        def make_hook(name):
            def capture(_module, args):
                item = {"order": len(events) + 1, "module": name,
                        "tensor_input_shapes": [list(value.shape) for value in args if isinstance(value, torch.Tensor)]}
                if name.startswith("slot_attention_") and args:
                    projected = projection.get("wsi")
                    matches = projected is not None and args[0].shape == projected.shape and torch.equal(args[0], projected)
                    item["input_equals_current_wsi_projection"] = bool(matches)
                    if name == "slot_attention_omic":
                        item["input_role"] = "wsi_projection_for_joint_reconstruction" if training and matches else "ordinary_omic_tokens"
                events.append(item)
                return None
            return capture

        try:
            handles.append(model.wsi_mlp.register_forward_hook(capture_projection))
            for name in names:
                handles.append(getattr(model, name).register_forward_pre_hook(make_hook(name)))
            model.train(training)
            with torch.no_grad(), fixed_rng(torch, 101):
                logits, auxiliary = model(**batch)
        finally:
            for handle in handles:
                handle.remove()
            model.eval()
        counts = {name: sum(event["module"] == name for event in events) for name in names}
        gate_start = min(event["order"] for event in events if event["module"].startswith("slot_decoder_"))
        later_sa = [event for event in events if event["module"].startswith("slot_attention_") and event["order"] > gate_start]
        expected = counts["slot_attention_wsi"] == 1 and counts["slot_attention_omic"] == (2 if training else 1)
        if not training:
            expected = expected and not later_sa
        else:
            expected = expected and len(later_sa) == 1 and later_sa[0].get("input_role") == "wsi_projection_for_joint_reconstruction"
            joint_order = next(event["order"] for event in events if event["module"] == "reconstruction_head_joint")
            expected = expected and joint_order == later_sa[0]["order"] + 1
        if not expected:
            raise RuntimeError("Observed full-model call path differed from the stated original source contract")
        traces["train" if training else "eval"] = {
            "events": events,
            "counts": counts,
            "logit_shape": list(logits.shape),
            "finite_logits": bool(torch.isfinite(logits).all()),
            "matches_expected_source_path": expected,
            "slot_attention_calls_after_first_decoder": later_sa,
            "observed_gate_selected_input_reaggregation": False,
        }
    return {
        "status": "source_path_observed",
        "modes": traces,
        "claim": "The generated eval path aggregates each modality once before selection. The extra train OMIC-module call reads the WSI projection for joint reconstruction; it is not selected-slot feedback to same-modality inputs.",
        "limits": "Observed execution paths and tensor identity establish this mechanism path only; call counts alone establish neither a representation defect nor patient harm.",
    }


def coordinate_check(torch, model, batch):
    model.eval()
    coords_a = torch.stack((torch.arange(16) % 4, torch.arange(16) // 4), dim=-1)[None].repeat(2, 1, 1)
    coords_b = torch.stack((torch.arange(16) * 13, (torch.arange(16) * 7) % 19), dim=-1)[None].repeat(2, 1, 1)

    def run(extra):
        with torch.no_grad(), fixed_rng(torch, 303):
            return model(**dict(batch, **extra))[0].detach()

    baseline = run({})
    layout_a = run({"coords": coords_a})
    layout_b = run({"coords": coords_b})
    altered = batch["x_wsi"].clone()
    altered[:, :8] += torch.linspace(-4, 4, 32)
    changed = run({"x_wsi": altered})
    a_error = float((baseline - layout_a).abs().max())
    b_error = float((baseline - layout_b).abs().max())
    control_error = float((baseline - changed).abs().max())
    accepted = a_error == 0.0 and b_error == 0.0 and control_error > 0.0
    return {
        "status": "coordinate_nonconsumption_observed" if accepted else "check_not_established",
        "same_feature_bag_same_model_same_rng": True,
        "two_coordinate_layouts_are_distinct": not torch.equal(coords_a, coords_b),
        "without_coords_vs_layout_a_max_logit_abs_difference": a_error,
        "without_coords_vs_layout_b_max_logit_abs_difference": b_error,
        "changed_wsi_feature_positive_control_max_logit_abs_difference": control_error,
        "positive_control_is_nonconstant": control_error > 0.0,
        "claim": "The original forward does not consume this extra coordinate field in these generated calls.",
        "limits": "An ignored extra keyword is not a supported coordinate interface. This does not establish that all spatial information is absent: patch embeddings can retain local morphology, and existing contextual encoders may embed spatial information.",
    }


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch
    from survot_rank.research.slotspe_proof.model import MODEL_SOURCE_FILES, load_source, make_args

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source_root = ROOT / "third_party/SlotSPE"
    before = {str(path): digest(source_root / path) for path in MODEL_SOURCE_FILES}
    receipt = {
        "schema": "slotspe_extended_generated_checks_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": digest(__file__),
        "source_root": "third_party/SlotSPE",
        "source_hashes_before": before,
        "environment": {"python": sys.version.split()[0], "platform": platform.system(), "torch": torch.__version__, "device": "cpu", "torch_threads": torch.get_num_threads(), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]},
        "execution_scope": {"generated_tensors_only": True, "random_new_weights_only": True, "checkpoint_files_loaded": 0, "patient_records_loaded": 0, "optimizer_steps": 0, "backward_calls": 0, "arrays_or_models_saved": False},
        "execution_counts": {"standalone_native_sa_forward_calls": 0, "deterministic_replay_calls": 0, "full_model_forward_calls": 0, "full_model_eval_forward_calls": 0, "full_model_train_mode_forward_calls": 0, "native_sa_calls_inside_full_model": {"wsi": 0, "omic": 0}},
        "checks": {},
    }
    try:
        source = load_source(str(source_root))
        receipt["checks"]["identical_initial_slot_symmetry"] = equal_initial_counterexamples(torch, source)
        counts = receipt["execution_counts"]
        counts["standalone_native_sa_forward_calls"] = receipt["checks"]["identical_initial_slot_symmetry"]["standalone_native_sa_forward_calls"]
        counts["deterministic_replay_calls"] = receipt["checks"]["identical_initial_slot_symmetry"]["deterministic_replay_calls"]
        with fixed_rng(torch, 20261022):
            args = make_args([5, 7, 3], encoding_dim=32)
            args.wsi_projection_dim = 32
            model = source.SlotSPE(args).cpu().eval()
            batch = generated_batch(torch)
        receipt["generated_full_model"] = {"batch": 2, "wsi_tokens": 16, "wsi_feature_dim": 32, "omic_feature_dims": [5, 7, 3], "n_classes": args.n_classes, "args": vars(args)}
        meter_handles = []

        def count_model(module, _args):
            counts["full_model_forward_calls"] += 1
            counts["full_model_train_mode_forward_calls" if module.training else "full_model_eval_forward_calls"] += 1
            return None

        def count_sa(modality):
            def capture(_module, _args):
                counts["native_sa_calls_inside_full_model"][modality] += 1
                return None
            return capture

        try:
            meter_handles.append(model.register_forward_pre_hook(count_model))
            for modality in ("wsi", "omic"):
                meter_handles.append(getattr(model, "slot_attention_" + modality).register_forward_pre_hook(count_sa(modality)))
            receipt["checks"]["native_call_path"] = full_model_trace(torch, model, batch)
            receipt["checks"]["coordinate_field_nonconsumption"] = coordinate_check(torch, model, batch)
        finally:
            for handle in meter_handles:
                handle.remove()
        counts["all_native_sa_forward_calls"] = counts["standalone_native_sa_forward_calls"] + sum(counts["native_sa_calls_inside_full_model"].values())
        counts["count_basis"] = "Forward entry hooks; train-mode is one generated forward, not optimization or training. Replay calls directly apply original submodules and are counted separately."
        receipt["status"] = "completed"
    except Exception as error:
        receipt["status"] = "failed_retained"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        after = {str(path): digest(source_root / path) for path in MODEL_SOURCE_FILES}
        receipt["source_hashes_after"] = after
        receipt["original_source_bytes_unchanged"] = before == after
        if before != after:
            receipt["status"] = "failed_source_changed"
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_bytes((json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps({"status": receipt["status"], "output": str(OUTPUT), "original_source_bytes_unchanged": receipt["original_source_bytes_unchanged"]}, ensure_ascii=False))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
