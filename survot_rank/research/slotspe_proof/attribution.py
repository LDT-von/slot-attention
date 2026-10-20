"""Matched-state A/B/C contracts; these checks do not measure patient benefit."""
from pathlib import Path


def describe_gradient(gradient):
    import torch
    if gradient is None:
        return {"kind": "None", "connected": False, "finite": None, "norm": None,
                "nonzero_elements": 0}
    nonzero = int(torch.count_nonzero(gradient))
    return {"kind": "nonzero" if nonzero else "zero", "connected": True,
            "finite": bool(torch.isfinite(gradient).all()),
            "norm": float(gradient.norm()), "nonzero_elements": nonzero}


def matched_bc_check(source_root, args, state, batch, seed, device="cpu"):
    """Compare the same weights/input/RNG, never independently trained B/C.

    Compare auxiliary gradients for EVERY parameter (including None), rather
    than just selector norms. Only keep on the fused self-pool path is detached.
    """
    import torch
    from .model import build_model
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    results = {}
    with torch.random.fork_rng(devices=devices):
        for arm in ("B", "C"):
            model = build_model(source_root, args, arm).to(device)
            model.load_state_dict(state, strict=True)
            model.train()
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            selected = {}
            hooks = []
            for modality in ("wsi", "omic"):
                def capture(module, inputs, output, key=modality):
                    selected[key] = output[2].detach().cpu().clone()
                hooks.append(getattr(model, "slot_decoder_" + modality).register_forward_hook(capture))
            try:
                logits, auxiliary = model(**batch)
            finally:
                for hook in hooks:
                    hook.remove()
            main = model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None) / len(batch["y"])
            named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            parameters = [p for _, p in named]
            main_grads = torch.autograd.grad(main, parameters, retain_graph=True, allow_unused=True)
            aux_grads = torch.autograd.grad(auxiliary, parameters, allow_unused=True)
            results[arm] = {
                "logits": logits.detach().cpu(), "auxiliary": auxiliary.detach().cpu(),
                "selected": selected,
                "main": {name: g.detach().cpu() if g is not None else None for (name, _), g in zip(named, main_grads)},
                "aux": {name: g.detach().cpu() if g is not None else None for (name, _), g in zip(named, aux_grads)},
            }
            del model
    b, c = results["B"], results["C"]
    aux_comparison = {}
    for name, bg in b["aux"].items():
        cg = c["aux"][name]
        same = (bg is None and cg is None) or (bg is not None and cg is not None and torch.equal(bg, cg))
        aux_comparison[name] = {"B": describe_gradient(bg), "C": describe_gradient(cg),
            "exact_equal": same, "max_abs_difference": float((bg - cg).abs().max()) if bg is not None and cg is not None else None}
    selector = {name: {"B_main": describe_gradient(b["main"][name]),
                       "C_main": describe_gradient(c["main"][name])}
                for name in b["main"] if "pred_keep_slot" in name}
    checks = {
        "forward_exact_equal": torch.equal(b["logits"], c["logits"]),
        "auxiliary_loss_exact_equal": torch.equal(b["auxiliary"], c["auxiliary"]),
        "selection_exact_equal": all(torch.equal(b["selected"][key], c["selected"][key]) for key in b["selected"]),
        "B_main_all_selector_parameters_disconnected": bool(selector) and all(not row["B_main"]["connected"] for row in selector.values()),
        "C_main_all_selector_parameters_connected": bool(selector) and all(row["C_main"]["connected"] for row in selector.values()),
        "C_main_nonzero_each_modality": all(any(row["C_main"]["nonzero_elements"] > 0 for name, row in selector.items() if "slot_decoder_" + key in name) for key in ("wsi", "omic")),
        "auxiliary_all_parameter_gradients_exact_equal": all(row["exact_equal"] for row in aux_comparison.values()),
        "all_computed_gradients_finite": all(describe_gradient(g)["finite"] is not False for value in results.values() for kind in ("main", "aux") for g in value[kind].values()),
    }
    return {"status": "passed" if all(checks.values()) else "failed", "seed": seed,
        "comparison": "same_state_same_input_same_rng_not_separately_trained_weights",
        "checks": checks, "logits_max_abs_difference": float((b["logits"] - c["logits"]).abs().max()),
        "selector_parameters": selector, "auxiliary_all_parameters": aux_comparison,
        "interpretation": "Connectivity and identical auxiliary gradients only; no patient efficacy or attribution conclusion"}


def run_attribution_checks(source_root, output_dir, seeds):
    import torch
    from .model import build_model, make_args
    from .queue import _write
    path = Path(output_dir) / "attribution_checks.json"
    if path.exists():
        raise FileExistsError(path)
    records = []
    with torch.random.fork_rng(devices=[]):
        for method in ("parallel_topk_st", "gumbel_topk_st"):
            for seed in seeds:
                torch.manual_seed(seed)
                args = make_args([3, 4, 5], encoding_dim=16)
                args.wsi_projection_dim, args.slot_iters = 32, 3
                args.top_k_method = method
                base = build_model(source_root, args, "A")
                batch = {"x_wsi": torch.randn(4, 24, 16), "omic_missing": False,
                    "y": torch.tensor([0, 1, 2, 3]), "c": torch.tensor([0., 1., 0., 1.])}
                batch.update({f"x_omic{i+1}": torch.randn(4, width) for i, width in enumerate(args.omic_sizes)})
                report = matched_bc_check(source_root, args, base.state_dict(), batch, seed + 1000)
                report["selection_method"] = method
                records.append(report)
    result = {"status": "passed" if all(row["status"] == "passed" for row in records) else "failed",
        "input_scope": "synthetic_tensors_random_weights_zero_optimizer_steps", "records": records,
        "patient_effect": "not_evaluated", "transport_added": False}
    _write(path, result, exclusive=True)
    if result["status"] != "passed":
        raise AssertionError(f"B/C contract failed; retained report: {path}")
    return result


if __name__ == "__main__":
    import argparse
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 17, 29])
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = run_attribution_checks(args.source_root, args.output_dir, args.seeds)
    print(result["status"], "synthetic checks; no patient training")
