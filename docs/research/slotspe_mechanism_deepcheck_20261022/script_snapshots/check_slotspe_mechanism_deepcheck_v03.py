"""Bounded CPU checks on generated inputs and new, disclosed native weights.

No patient inputs, checkpoints, optimizer, training loop or downloaded code.
The constructive head check changes newly instantiated in-memory parameters;
it never changes the vendored original source. Existing output is not replaced.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def rng(torch, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def maxdiff(a, b):
    return float((a - b).abs().max())


def spread(x):
    return float((x - x.mean(dim=1, keepdim=True)).square().mean().sqrt())


def native_full_model_checks(torch, source, make_args):
    results = []
    for seed in (3, 17, 29):
        with rng(torch, seed):
            args = make_args([5, 7, 3], encoding_dim=32)
            args.wsi_projection_dim = 256
            model = source.SlotSPE(args).cpu().double().eval()
            batch = {"x_wsi": torch.randn(2, 32, 32, dtype=torch.float64), "omic_missing": False}
            for i, width in enumerate((5, 7, 3), start=1):
                batch[f"x_omic{i}"] = torch.randn(2, width, dtype=torch.float64)
            permutation = torch.randperm(32)
            proto_a = torch.randn(1, 1, 32, dtype=torch.float64)
            proto_b = torch.randn(1, 1, 32, dtype=torch.float64)
        calls = [0]
        sa_calls = [0]

        def meter(_module, _args):
            calls[0] += 1

        def sa_meter(_module, _args):
            sa_calls[0] += 1

        handles = [model.register_forward_pre_hook(meter)]
        handles.extend(getattr(model, "slot_attention_" + mod).register_forward_pre_hook(sa_meter) for mod in ("wsi", "omic"))

        def run(wsi):
            with torch.no_grad(), rng(torch, 1000 + seed):
                return model(**dict(batch, x_wsi=wsi))[0].detach()

        try:
            x = batch["x_wsi"]
            base = run(x)
            repeated = run(x.repeat_interleave(3, dim=1))
            permuted = run(x[:, permutation])
            changed = x.clone()
            changed[:, :8] += torch.linspace(-3, 3, 32, dtype=x.dtype)
            positive = run(changed)
            mixture = []
            for count_a in (8, 16, 24):
                mix = torch.cat((proto_a.expand(2, count_a, -1), proto_b.expand(2, 32 - count_a, -1)), dim=1)
                mixture.append(run(mix))
            # Same 16 A / 16 B patch multiset at a 4x8 grid: a two-region
            # split versus checkerboard. The model receives patch features;
            # the grid labels below define an external geometry diagnostic.
            split_grid = torch.cat((torch.zeros(2, 8), torch.ones(2, 8)), dim=0).long()
            checker_grid = (torch.arange(4)[:, None] + torch.arange(8)[None, :]) % 2
            checker_features = torch.where(checker_grid.reshape(1, 32, 1).bool(), proto_b, proto_a).expand(2, -1, -1)
            checker_output = run(checker_features)

            def unlike_neighbor_fraction(grid):
                unlike = (grid[:, 1:] != grid[:, :-1]).sum() + (grid[1:] != grid[:-1]).sum()
                edges = grid.shape[0] * (grid.shape[1] - 1) + (grid.shape[0] - 1) * grid.shape[1]
                return float(unlike / edges)

            split_relation = unlike_neighbor_fraction(split_grid)
            checker_relation = unlike_neighbor_fraction(checker_grid)
            results.append({
                "seed": seed, "new_random_weights_only": True,
                "mode": "eval", "paired_rng_seed": 1000 + seed,
                "wsi_projection_dim": 256, "native_slot_count": 8, "native_iterations": 10,
                "native_parameter_count": sum(p.numel() for p in model.parameters()),
                "full_model_forward_calls": calls[0], "native_sa_forward_calls": sa_calls[0],
                "repeat_all_tokens_3x_logit_max_difference": maxdiff(base, repeated),
                "permute_same_token_bag_logit_max_difference": maxdiff(base, permuted),
                "change_feature_positive_control_logit_max_difference": maxdiff(base, positive),
                "fixed_32_tokens_A25_vs_A75_logit_max_difference": maxdiff(mixture[0], mixture[2]),
                "same_16A_16B_different_grid_layout_logit_max_difference": maxdiff(mixture[1], checker_output),
                "external_unlike_neighbor_fraction_split": split_relation,
                "external_unlike_neighbor_fraction_checkerboard": checker_relation,
                "external_relation_statistic_rotation_invariant": all(unlike_neighbor_fraction(torch.rot90(grid, 1)) == unlike_neighbor_fraction(grid) for grid in [split_grid, checker_grid]),
                "all_outputs_finite": all(bool(t.isfinite().all()) for t in [base, repeated, permuted, positive, checker_output, *mixture]),
            })
        finally:
            for h in handles:
                h.remove()
    return {
        "runs": results,
        "interpretation": {
            "replication": "All-token replication changes absolute sample cardinality, not composition; fixed-patch protocols do not expose this variable directly.",
            "permutation": "Paired-RNG bag-permutation invariance is tested; this is not a test on real tissue geometry or on a spatially contextual feature encoder.",
            "geometry": "Two generated layouts have the same independent patch-feature multiset but different grid adjacency. Equal outputs show the current feature-bag interface cannot recover that missing binding; this does not test contextual upstream features or patient predictive harm.",
            "composition": "A nonzero difference refutes a universal claim that the whole native model cannot respond to content proportions; random-weight sensitivity is not learned biological accuracy.",
        },
    }


def reconstruction_dictionary_check(torch, source):
    """An admissible native-head parameter construction, not a trained result."""
    import torch.nn.functional as F

    with rng(torch, 4103):
        dim = 256
        heads = 8
        head = source.modules["models.SlotSPE"].ReconstructionHead(dim, num_heads=heads, mode="wsi_patches").cpu().double().eval()
        head_dim = dim // heads
        dictionary = torch.zeros(8, dim, dtype=torch.float64)
        for code in range(8):
            for h in range(heads):
                for bit in range(3):
                    dictionary[code, h * head_dim + bit] = 1.0 if ((code >> bit) & 1) else -1.0
        slots = dictionary[None].repeat(2, 1, 1)
        ids_a = torch.tensor([[0, 7] * 8, [7, 7, 7, 0] * 4])
        ids_b = 7 - ids_a
        target_a, target_b = dictionary[ids_a], dictionary[ids_b]
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[:, [0, 7]] = True

    frozen = head.query_gen.frozen_mlp
    matrix = frozen.weight.detach().clone()
    bias = frozen.bias.detach().clone()
    inverse = torch.linalg.inv(matrix)
    eye = torch.eye(dim, dtype=torch.float64)
    with torch.no_grad():
        head.cross_attn.in_proj_weight.zero_()
        head.cross_attn.in_proj_bias.zero_()
        head.cross_attn.in_proj_weight[:dim].copy_(inverse)
        head.cross_attn.in_proj_bias[:dim].copy_(-inverse @ bias)
        head.cross_attn.in_proj_weight[dim:2 * dim].copy_(128 * eye)
        head.cross_attn.in_proj_weight[2 * dim:].copy_(eye)
        head.cross_attn.out_proj.weight.copy_(eye)
        head.cross_attn.out_proj.bias.zero_()
        for parameter in head.mlp.parameters():
            parameter.zero_()

    calls = [0]

    def count(_module, _args):
        calls[0] += 1

    handle = head.register_forward_pre_hook(count)
    try:
        with torch.no_grad():
            output_a = head(target_a, slots, mask=mask)
            output_b = head(target_b, slots, mask=mask)
            with patch.object(head.query_gen, "forward", lambda x: torch.zeros_like(x)):
                no_content_query = head(target_a, slots, mask=mask)
            identical_slots = head(target_a, torch.zeros_like(slots), mask=mask)
    finally:
        handle.remove()

    def loss(pred, target):
        # Same normalized cosine expression as the native WSI reconstruction objective.
        return float(1 - F.cosine_similarity(F.normalize(pred, dim=-1), F.normalize(target, dim=-1), dim=-1).mean())

    return {
        "scope": "constructive_native_wsi_reconstruction_head_only_not_complete_training_objective",
        "dim": dim, "heads": heads, "batch": 2, "tokens": 16, "candidate_slots": 8, "selected_slots": 2,
        "slot_dictionary_identical_between_pseudo_samples_and_target_cases": bool(torch.equal(slots[0], slots[1])),
        "within_dictionary_unique_vectors": int(torch.unique(dictionary, dim=0).shape[0]),
        "target_varies_between_tokens": maxdiff(target_a, target_a[:, :1]),
        "target_changed_between_cases": maxdiff(target_a, target_b),
        "frozen_query_matrix_condition_number": float(torch.linalg.cond(matrix)),
        "frozen_query_weights_were_not_changed": bool(torch.equal(matrix, frozen.weight) and torch.equal(bias, frozen.bias)),
        "parameter_construction": "Only new head MHA query projection inverts the native frozen query map; key gain128; value/out identity; residual MLP zero. No gradient or optimizer.",
        "head_forward_calls": calls[0],
        "case_a_max_reconstruction_error": maxdiff(output_a, target_a),
        "case_b_max_reconstruction_error": maxdiff(output_b, target_b),
        "case_a_native_wsi_cosine_loss": loss(output_a, target_a),
        "case_b_native_wsi_cosine_loss": loss(output_b, target_b),
        "constant_query_negative_control_cosine_loss": loss(no_content_query, target_a),
        "identical_zero_slots_negative_control_cosine_loss": loss(identical_slots, target_a),
        "constant_query_token_output_spread": spread(no_content_query),
        "identical_slot_token_output_spread": spread(identical_slots),
        "all_outputs_finite": all(bool(t.isfinite().all()) for t in [output_a, output_b, no_content_query, identical_slots]),
        "claim": "Low WSI reconstruction loss alone does not require these slots to change with the current sample: target-derived queries can address a shared constant dictionary.",
        "limits": "Generated two-archetype targets and manually constructed admissible parameters. No assertion that a trained checkpoint uses this solution; no assertion about OMIC/joint or total survival loss; slots still supply dictionary values, so this is not a copy path ignoring slots entirely.",
    }


def near_equal_initialization_check(torch, source):
    results = []
    for seed in (3, 17, 29):
        with rng(torch, seed + 2000):
            module = source.MultiHeadSlotAttention(num_slots=8, dim=256, heads=8, dim_head=64, iters=10).cpu().double().eval()
            x = torch.randn(2, 32, 256, dtype=torch.float64)
            shared_noise = torch.randn(2, 1, 256, dtype=torch.float64).expand(2, 8, 256).clone()
            directions = torch.randn(2, 8, 256, dtype=torch.float64)
        native_randn = torch.randn
        for delta in (None, 0.0, 1e-4, 1e-2, 1.0):
            states = []
            intercepts = [0]

            def noise(*shape, **kwargs):
                dims = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else tuple(shape)
                if dims == (2, 8, 256):
                    intercepts[0] += 1
                    return (shared_noise + delta * directions).to(device=kwargs.get("device", "cpu"), dtype=kwargs.get("dtype", torch.float64))
                return native_randn(*shape, **kwargs)

            def capture(_module, args):
                states.append(args[0].detach().clone())

            h = module.norm_slots.register_forward_pre_hook(capture)
            try:
                with torch.no_grad(), rng(torch, seed + 9000):
                    if delta is None:
                        final = module(x)
                    else:
                        with patch.object(torch, "randn", noise):
                            final = module(x)
                states.append(final.detach().clone())
            finally:
                h.remove()
            if len(states) != 11 or (delta is not None and intercepts[0] != 1):
                raise RuntimeError("Native initialization/iteration capture failed")
            widths = [spread(value) for value in states]
            results.append({
                "seed": seed, "initialization": "native_random" if delta is None else "shared_noise_plus_disclosed_perturbation",
                "delta": delta, "native_forward_calls": 1, "centered_slot_rms_by_state": widths,
                "final_to_initial_spread_ratio": None if maxdiff(states[0], states[0][:, :1]) == 0 else widths[-1] / widths[0],
                "equal_initial_state_ratio_excluded_due_to_centering_roundoff": maxdiff(states[0], states[0][:, :1]) == 0,
                "exactly_equal_at_all_states": all(maxdiff(value, value[:, :1]) == 0 for value in states),
                "all_states_finite": all(bool(value.isfinite().all()) for value in states),
            })
    return {
        "runs": results,
        "source_attention_scale_dim256": 256 ** -0.5,
        "per_head_dot_dimension": 64,
        "inverse_sqrt_head_dimension_reference": 64 ** -0.5,
        "scale_is_design_factor_not_proven_bug": True,
        "limits": "New random weights, generated inputs, no training. Near-state damping/expansion does not measure learned patient collapse or meaningful content recovery. Native random reference distinguishes the injected equal-state boundary from ordinary starts.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New receipt JSON path under the workspace; existing files refused")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.relative_to(ROOT)
    if output.exists():
        parser.error("Output exists; preserve it and select a fresh receipt path")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(ROOT))
    import torch
    from survot_rank.research.slotspe_proof.model import MODEL_SOURCE_FILES, load_source, make_args

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source_root = ROOT / "third_party/SlotSPE"
    before = {str(p): sha(source_root / p) for p in MODEL_SOURCE_FILES}
    receipt = {
        "schema": "slotspe_mechanism_deepcheck_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "SPE-08", "script_sha256": sha(__file__), "source_hashes_before": before,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu", "dtype": "float64", "threads": torch.get_num_threads()},
        "execution_scope": {"generated_inputs_only": True, "new_random_or_disclosed_constructed_weights_only": True,
                            "patient_records_loaded": 0, "checkpoints_loaded": 0, "optimizer_steps": 0, "backward_calls": 0,
                            "server_actions": 0, "arrays_or_weights_saved": False},
        "checks": {},
    }
    try:
        source = load_source(str(source_root))
        receipt["checks"]["native_full_model_information_controls"] = native_full_model_checks(torch, source, make_args)
        receipt["checks"]["target_query_shared_dictionary"] = reconstruction_dictionary_check(torch, source)
        receipt["checks"]["near_equal_initialization"] = near_equal_initialization_check(torch, source)
        receipt["execution_counts"] = {
            "full_model_forward_calls": sum(r["full_model_forward_calls"] for r in receipt["checks"]["native_full_model_information_controls"]["runs"]),
            "native_sa_inside_full_model": sum(r["native_sa_forward_calls"] for r in receipt["checks"]["native_full_model_information_controls"]["runs"]),
            "standalone_native_reconstruction_head_forward_calls": receipt["checks"]["target_query_shared_dictionary"]["head_forward_calls"],
            "standalone_native_sa_forward_calls": len(receipt["checks"]["near_equal_initialization"]["runs"]),
        }
        receipt["status"] = "completed_checks_observations_not_patient_claims"
    except Exception as error:
        receipt["status"] = "failed_retained"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        after = {str(p): sha(source_root / p) for p in MODEL_SOURCE_FILES}
        receipt["source_hashes_after"] = after
        receipt["original_source_bytes_unchanged"] = before == after
        if before != after:
            receipt["status"] = "failed_original_source_changed"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "receipt": str(output), "original_source_unchanged": receipt["original_source_bytes_unchanged"]}, ensure_ascii=False))
    return 0 if receipt["status"] == "completed_checks_observations_not_patient_claims" else 1


if __name__ == "__main__":
    raise SystemExit(main())
