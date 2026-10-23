"""Small synthetic model interventions, separately labelled from patient results."""
from pathlib import Path
import hashlib
import json


def run_mechanism_checks(source_root, output_dir, seeds):
    import torch
    from .model import build_model, make_args, selected_pool

    output = Path(output_dir) / "mechanisms_checks.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    args = make_args([3, 4, 5], encoding_dim=16)
    args.wsi_projection_dim = 32
    args.slot_iters = 3
    records = []
    with torch.random.fork_rng(devices=[]):
        for seed in seeds:
            torch.manual_seed(seed)
            a = build_model(source_root, args, "original").cpu()
            b = build_model(source_root, args, "readout_fix").cpu()
            b.load_state_dict(a.state_dict(), strict=True)
            if sum(p.numel() for p in a.parameters()) != sum(p.numel() for p in b.parameters()):
                raise AssertionError("Repair added model parameters")
            data = {"x_wsi": torch.randn(4, 24, 16), "omic_missing": False,
                "y": torch.tensor([0, 1, 2, 3]), "c": torch.tensor([0., 1., 0., 1.])}
            data.update({f"x_omic{i + 1}": torch.randn(4, n) for i, n in enumerate(args.omic_sizes)})
            row = {"seed": seed, "parameter_count": sum(p.numel() for p in a.parameters())}
            for name, net in (("original", a), ("readout_fix", b)):
                net.train()
                torch.manual_seed(seed + 1000)
                logits, _ = net(**data)
                loss = net.loss_fn(logits, y=data["y"], c=data["c"], t=None) / 4
                parameters = [(n, p) for n, p in net.named_parameters() if "pred_keep_slot" in n]
                gradients = torch.autograd.grad(loss, [p for _, p in parameters], allow_unused=True)
                row[name + "_main_selector_gradients"] = {n: None if g is None else float(g.norm())
                    for (n, _), g in zip(parameters, gradients)}
                net.eval()
                with torch.no_grad():
                    predictions = torch.stack([net(**data)[0] for _ in range(8)])
                row[name + "_eval_max_range"] = float((predictions.max(0).values - predictions.min(0).values).max())
            row["original_gradient_disconnect"] = "observed" if all(
                g is None for g in row["original_main_selector_gradients"].values()) else "not_observed"
            row["repair_selector_connected"] = all(g is not None for g in row["readout_fix_main_selector_gradients"].values())
            row["repair_has_nonzero_selector_gradient"] = any(g is not None and g > 1e-12 for g in row["readout_fix_main_selector_gradients"].values())
            slots = torch.randn(4, 8, 32, requires_grad=True)
            keep = torch.zeros(4, 8)
            keep[:, :2] = 1
            mask = keep.bool()
            changed = torch.where(mask[..., None], slots, slots + 2 * torch.randn_like(slots))
            before = a.self_attention_wsi(slots, mask=mask)
            after = a.self_attention_wsi(changed, mask=mask)
            probe = torch.randn(4, 32)
            old_gradient, = torch.autograd.grad((before.mean(1) * probe).sum(), slots, retain_graph=True)
            new_gradient, = torch.autograd.grad((selected_pool(before, keep) * probe).sum(), slots)
            row["self_branch_intervention"] = {
                "original_pool_max_change": float((before.mean(1) - after.mean(1)).abs().max()),
                "selected_pool_max_change": float((selected_pool(before, keep) - selected_pool(after, keep)).abs().max()),
                "original_dropped_gradient_norm": float(old_gradient[~mask].norm()),
                "selected_dropped_gradient_norm": float(new_gradient[~mask].norm())}
            row["original_self_exclusion_failure"] = "observed" if row["self_branch_intervention"]["original_pool_max_change"] > 1e-7 else "not_observed"
            if row["self_branch_intervention"]["selected_pool_max_change"] > 1e-7 or row["self_branch_intervention"]["selected_dropped_gradient_norm"] > 1e-7:
                raise AssertionError("Selected self readout failed exclusion control")
            if row["readout_fix_eval_max_range"] > 1e-7:
                raise AssertionError("Repair evaluation is not repeatable")
            if not row["repair_selector_connected"] or not row["repair_has_nonzero_selector_gradient"]:
                raise AssertionError("Repair selector gradient did not pass connectivity/numerical control")
            records.append(row)
    report = {"status": "completed", "scope": "Synthetic tensors and random weights; zero optimizer steps; no patient performance",
        "source_root": str(Path(source_root).resolve()), "seeds": list(seeds), "records": records,
        "cross_branch": "Original dense branch retained; self exclusion is not a global sparse-model claim",
        "not_observed_rule": "Baseline defects are observations, not required successful outcomes",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report
