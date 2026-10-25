"""Native-size VIB training smoke on invented tensors; never patient work.

No inference about C-index or mechanism efficacy follows from this check.
Use --run-generated to explicitly execute the four declared conditions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_generated():
    import torch
    from slotspe_proof.model import build_model, make_args
    from slotspe_proof.slot_bottleneck import SlotVIBHost

    torch.set_num_threads(2)
    source = ROOT / "third_party/SlotSPE"
    paths = sorted(source.rglob("*.py"))
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    generator = torch.Generator().manual_seed(773)
    # Native K/D/iters/patch budget, invented low-dimensional feature inputs.
    batch = {"x_wsi": torch.randn(1, 4096, 8, generator=generator),
             "x_omic1": torch.randn(1, 3, generator=generator),
             "x_omic2": torch.randn(1, 5, generator=generator),
             "y": torch.tensor([1]), "c": torch.tensor([0.]), "omic_missing": False}
    args = make_args([3, 5], encoding_dim=8)
    torch.manual_seed(3)
    original = build_model(source, args, "original")
    initial = {name: value.detach().clone() for name, value in original.state_dict().items()}
    del original
    results = []
    for name, mode, beta in (("native", "off", 0.), ("mean_control", "mean", 0.),
                             ("stochastic_no_kl", "stochastic", 0.), ("vib", "stochastic", .001)):
        started = time.perf_counter()
        native = build_model(source, args, "original")
        native.load_state_dict(initial, strict=True)
        host = SlotVIBHost(native, 256, mode=mode)
        optimizer = torch.optim.Adam(host.parameters(), lr=.0005)
        torch.manual_seed(4201)  # same dropout/initial-slot/Gumbel draws
        before_rng = torch.get_rng_state().clone()
        host.train()
        logits, auxiliary = host(**batch, vib_seed=801)
        after_rng = torch.get_rng_state().clone()
        main = native.loss_fn(logits, y=batch["y"], c=batch["c"], t=None)
        kl = host.kl_loss()
        loss = main + auxiliary + beta * kl
        loss.backward()
        gradients = [p.grad for p in host.parameters() if p.grad is not None]
        if not bool(torch.isfinite(loss)) or any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError("nonfinite generated objective/gradient")
        gradient_norm = float(torch.stack([g.detach().norm() for g in gradients]).norm())
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in host.parameters()):
            raise RuntimeError("nonfinite updated parameter")
        changed = sum(not torch.equal(value.detach(), initial[key]) for key, value in native.state_dict().items())
        if changed == 0:
            raise RuntimeError("native core did not update")
        if mode == "off":
            native_logits = logits.detach().clone()
            native_rng = after_rng
        if not torch.equal(after_rng, native_rng):
            raise RuntimeError("adapter perturbed native random stream")
        results.append({"condition": name, "mode": mode, "beta": beta, "main_nll": float(main.detach()),
                        "native_auxiliary": float(auxiliary.detach()), "kl_mean_per_dimension": float(kl.detach()),
                        "total_loss": float(loss.detach()), "gradient_l2": gradient_norm,
                        "core_tensors_updated": changed, "optimizer_steps": 1,
                        "logit_max_difference_from_native": float((logits.detach()-native_logits).abs().max()),
                        "native_rng_preserved_relative_to_off": True,
                        "native_forward_consumed_rng": not torch.equal(before_rng, after_rng),
                        "seconds": time.perf_counter()-started})
        del host, native, optimizer
    after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if before != after:
        raise RuntimeError("native source changed")
    if results[0]["logit_max_difference_from_native"] != 0 or results[1]["logit_max_difference_from_native"] != 0:
        raise RuntimeError("off/initial-identity-mean failed native equality")
    return {"schema": "slotspe_vib_generated_training_smoke_v1", "status": "passed",
            "input_scope": "invented_features_not_UNI2h_not_patients", "patient_runs": 0,
            "architecture": {"K": 8, "D": 256, "slot_iters": 10, "patches": 4096,
                             "encoding_dim_generated_only": 8, "invented_pathways": [3, 5]},
            "conditions": results, "native_source_sha256": before,
            "claim_boundary": "one_step_per_condition_proves_execution_not_learning_benefit_or_patient_efficacy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-generated", action="store_true")
    parser.add_argument("--output", default="local_evidence/implementation/20261025_slot_vib/generated_smoke.json")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    private = (ROOT / "local_evidence").resolve()
    if not output.is_relative_to(private) or output == private:
        parser.error("output must be inside ignored local_evidence")
    if not args.run_generated:
        print(json.dumps({"mode": "plan_only", "patient_execution": False,
                          "next": "--run-generated executes invented inputs only", "output": str(output)}, indent=2))
        return
    if output.exists():
        raise FileExistsError("preserve previous receipt; choose a fresh --output")
    result = run_generated()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "conditions": len(result["conditions"]),
                      "patient_runs": 0, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
