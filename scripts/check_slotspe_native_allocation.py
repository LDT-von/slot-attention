"""Opt-in, CPU-only S0--S4 experiment using invented data and new weights.

No patient loader, checkpoint input or server operation exists in this script.
Without --synthetic it prints the fixed plan and performs no model execution.
"""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", default="local_evidence/native_allocation_20261022")
    options = parser.parse_args()
    plan = {
        "data": "invented only; new random weights; CPU float32",
        "seeds": [3, 17, 29], "steps": 24, "batch_size": 8,
        "train_size": 64, "holdout_size": 32, "eval_repeats": 3,
        "toy_projection_dim": 32, "slots_per_modality": 8,
        "toy_slot_iters": 3, "patches": 24, "pathways": 4,
        "optimizer": "Adam", "learning_rate": 0.001,
        "native_default_shape_gate": "5 arms, dim256/8slots/10iters, full loss backward, no optimizer",
        "patient_execution": False, "server_execution": False,
        "selection_rule": "fixed last step; no outcome tuning or best-checkpoint selection",
    }
    if not options.synthetic:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    output = (ROOT / options.output).resolve()
    private_root = (ROOT / "local_evidence").resolve()
    if not output.is_relative_to(private_root) or output == private_root:
        raise ValueError("output must be a new child of ignored local_evidence")
    if output.exists():
        raise FileExistsError("preserve prior runs; choose a new output directory")
    output.mkdir(parents=True)

    import torch
    from torch.nn import functional as F
    from survot_rank.research.slotspe_proof.assignment_router import AssignmentConfig
    from survot_rank.research.slotspe_proof.model import build_model, make_args
    from survot_rank.research.slotspe_proof.native_allocation import NativeAllocation, VARIANTS

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protected = list((ROOT / "third_party/SlotSPE").rglob("*.py"))
    protected += [ROOT / "configs/slotspe_proof_queue.json"] if (ROOT / "configs/slotspe_proof_queue.json").exists() else []
    before_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in protected}
    summary = {
        "schema": "native_slotspe_generated_allocation_smoke_v1",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "plan": plan, "variants": VARIANTS,
        "assignment_config": asdict(AssignmentConfig()),
        "scope": "module-wide native SlotSPE SA replacement; omic module also reused for WSI reconstruction",
        "runtime": {"torch": torch.__version__, "device": "cpu", "dtype": "float32", "threads": 1},
        "source_sha256": before_hashes,
        "code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in [
            Path(__file__), ROOT / "survot_rank/research/slotspe_proof/native_allocation.py",
            ROOT / "survot_rank/research/slotspe_proof/assignment_router.py",
        ]},
        "native_shape_gates": [], "runs": [],
        "patient_runs": 0, "server_runs": 0,
    }

    def flush():
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def build(dim=32, iters=3):
        args = make_args([4] * 4, encoding_dim=8)
        args.wsi_projection_dim = dim
        args.slot_iters = iters
        return build_model(ROOT / "third_party/SlotSPE", args, "original")

    def primary(model, logits, batch):
        return model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None) / len(batch["y"])

    def slice_batch(data, index):
        return {key: value[index] if torch.is_tensor(value) else value for key, value in data.items()}

    def trace_metrics(records):
        result = {}
        for record in records:
            key = f'{record["modality"]}_{record["call_index"]}'
            weights = record["attentions"][-1]
            slots = F.normalize(record["final"], dim=-1)
            pairs = torch.triu_indices(slots.shape[1], slots.shape[1], offset=1)
            cosine = (slots[:, pairs[0]] * slots[:, pairs[1]]).sum(-1).mean()
            overlap = torch.minimum(weights[:, :, pairs[0]], weights[:, :, pairs[1]]).sum(-1).mean()
            result[key + "_slot_cosine"] = float(cosine)
            result[key + "_read_overlap"] = float(overlap)
        return result

    def evaluate(model, adapter, data, seed):
        model.eval()
        metrics, predictions = [], []
        for repeat in range(plan["eval_repeats"]):
            torch.manual_seed(seed * 100000 + 50000 + repeat)
            adapter.set_noise_seed(seed * 100000 + 70000 + repeat)
            with torch.no_grad():
                logits, auxiliary = model(**data)
                row = {"nll": float(primary(model, logits, data))}
                row.update(trace_metrics(adapter.records))
            metrics.append(row)
            predictions.append(logits.detach().clone())
        mean = {key: sum(row[key] for row in metrics) / len(metrics) for key in metrics[0]}
        return mean, torch.stack(predictions)

    # This gate uses native dim/slots/iterations. Only input encoding width and
    # invented token/pathway counts are reduced; all native model branches run.
    torch.manual_seed(7701)
    gate_model = build(256, 10)
    gate_state = copy.deepcopy(gate_model.state_dict())
    gate_batch = {"x_wsi": torch.randn(2, 12, 8), "omic_missing": False,
                  "y": torch.tensor([0, 2]), "c": torch.tensor([0, 1])}
    gate_batch.update({f"x_omic{i + 1}": torch.randn(2, 4) for i in range(4)})
    for arm in VARIANTS:
        start = time.perf_counter()
        gate_model.load_state_dict(gate_state)
        gate_model.zero_grad(set_to_none=True)
        gate_model.train()
        torch.manual_seed(7702)
        with NativeAllocation(gate_model, arm, noise_seed=7703) as adapter:
            logits, auxiliary = gate_model(**gate_batch)
            loss = primary(gate_model, logits, gate_batch) + auxiliary
            loss.backward()
            finite = bool(torch.isfinite(loss)) and all(
                bool(torch.isfinite(p.grad).all()) for p in gate_model.parameters() if p.grad is not None)
            if not finite:
                raise RuntimeError(f"nonfinite native shape gate: {arm}")
            row = {"variant": arm, "loss": float(loss.detach()), "finite": finite,
                   "calls": [{k: r[k] for k in ("modality", "call_index", "source_input_count")} for r in adapter.records],
                   "seconds": time.perf_counter() - start}
            summary["native_shape_gates"].append(row)
        print(f"native shape {arm}: finite full forward/backward", flush=True)
        flush()
    del gate_model, gate_state

    for seed in plan["seeds"]:
        generator = torch.Generator().manual_seed(40000 + seed)
        size = plan["train_size"] + plan["holdout_size"]
        latent = torch.randn(size, 2, generator=generator)
        # A proportion-dependent WSI signal plus a separate omic signal. Bin
        # thresholds are defined from generated TRAIN times only.
        positive = torch.rand(size, plan["patches"], generator=generator) < latent[:, :1].sigmoid()
        x_wsi = 0.12 * torch.randn(size, plan["patches"], 8, generator=generator)
        x_wsi[:, :, 0] += positive.float() * 2 - 1
        x_wsi[:, :, 1] += (positive.float() * 2 - 1) * 0.7
        invented_time = -latent.sum(-1) + 0.1 * torch.randn(size, generator=generator)
        bins = torch.quantile(invented_time[:plan["train_size"]], torch.tensor([0.25, 0.5, 0.75]))
        data = {"x_wsi": x_wsi, "omic_missing": False,
                "y": torch.bucketize(invented_time, bins), "c": torch.zeros(size, dtype=torch.long)}
        for index in range(plan["pathways"]):
            pathway = 0.12 * torch.randn(size, 4, generator=generator)
            pathway[:, 0] += latent[:, 1]
            pathway[:, 1] += latent[:, 1] * 0.5 + (index - 1.5) * 0.3
            data[f"x_omic{index + 1}"] = pathway
        train = slice_batch(data, slice(0, plan["train_size"]))
        holdout = slice_batch(data, slice(plan["train_size"], None))
        batches = [torch.randperm(plan["train_size"], generator=generator)[:plan["batch_size"]] for _ in range(plan["steps"])]
        torch.manual_seed(seed)
        initial_model = build()
        state = copy.deepcopy(initial_model.state_dict())
        torch.save({"invented_data": data, "bins_train_only": bins, "batches": batches,
                    "initial_state": state}, output / f"generated_seed_{seed}.pt")
        for arm in VARIANTS:
            start = time.perf_counter()
            model = build()
            model.load_state_dict(state)
            optimizer = torch.optim.Adam(model.parameters(), lr=plan["learning_rate"])
            with NativeAllocation(model, arm, noise_seed=seed) as adapter:
                initial_metrics, initial_predictions = evaluate(model, adapter, holdout, seed)
                losses = []
                for step, indices in enumerate(batches):
                    model.train()
                    batch = slice_batch(train, indices)
                    torch.manual_seed(seed * 100000 + step)
                    adapter.set_noise_seed(seed * 100000 + 30000 + step)
                    optimizer.zero_grad(set_to_none=True)
                    logits, auxiliary = model(**batch)
                    loss_main = primary(model, logits, batch)
                    loss = loss_main + auxiliary
                    loss.backward()
                    if not torch.isfinite(loss) or not all(
                            torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                        raise RuntimeError(f"nonfinite toy training: {seed}/{arm}/{step}")
                    optimizer.step()
                    losses.append({"step": step, "main": float(loss_main.detach()),
                                   "auxiliary": float(auxiliary.detach()), "total": float(loss.detach())})
                final_metrics, final_predictions = evaluate(model, adapter, holdout, seed)
            row = {"seed": seed, "variant": arm, "optimizer_steps": len(losses),
                   "initial_holdout": initial_metrics, "final_holdout": final_metrics,
                   "final_train": losses[-1], "all_finite": True,
                   "seconds": time.perf_counter() - start}
            summary["runs"].append(row)
            torch.save({"state": model.state_dict(), "losses": losses,
                        "initial_logits": initial_predictions, "final_logits": final_predictions},
                       output / f"toy_{arm}_seed_{seed}.pt")
            print(f"toy {arm}/seed{seed}: NLL {initial_metrics['nll']:.6f} -> {final_metrics['nll']:.6f}", flush=True)
            flush()
    summary["protected_files_unchanged"] = before_hashes == {k: digest(ROOT / k) for k in before_hashes}
    if not summary["protected_files_unchanged"]:
        raise RuntimeError("protected source changed")
    summary["completed_optimizer_steps"] = sum(row["optimizer_steps"] for row in summary["runs"])
    summary["status"] = "completed_generated_data_only"
    summary["raw_file_sha256"] = {p.name: digest(p) for p in sorted(output.glob("*.pt"))}
    flush()
    print(f"completed {len(summary['runs'])} invented-data runs, {summary['completed_optimizer_steps']} optimizer steps; patient/server runs=0", flush=True)


if __name__ == "__main__":
    main()
