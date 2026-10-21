"""Plan by default; --synthetic uses generated tensors only, never patients."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def synthetic_probe():
    import torch
    from survot_rank.research.slotspe_proof.assignment_router import (
        AssignmentConfig, SlotAssignmentRouter,
    )
    from survot_rank.research.slotspe_proof.sparsity_ablation import mean_pair_cosine

    x = torch.zeros(1, 1, 4, 12, dtype=torch.float64)
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(11), dtype=x.dtype)
    rows = []
    before = torch.get_rng_state().clone()
    for mode in ("original", "sinkhorn", "mesh_uniform"):
        r = SlotAssignmentRouter(AssignmentConfig(mode=mode))
        with torch.no_grad():
            a = r(x, noise=noise)
        overlaps = [torch.minimum(a[0, 0, i], a[0, 0, j]).sum().item()
                    for i in range(4) for j in range(i + 1, 4)]
        distinct_values = torch.eye(12, dtype=x.dtype)[None, None]
        constant_values = torch.ones(1, 1, 12, 5, dtype=x.dtype)
        rows.append({
            "mode": mode,
            "attention_overlap_mean": sum(overlaps) / len(overlaps),
            "attention_row_entropy_mean": -(a * a.clamp_min(1e-30).log()).sum(-1).mean().item(),
            "value_update_cosine_distinct_values": mean_pair_cosine((a @ distinct_values)[:, 0]).item(),
            "value_update_cosine_constant_values": mean_pair_cosine((a @ constant_values)[:, 0]).item(),
            "row_sum_max_error": (a.sum(-1) - 1).abs().max().item(),
        })
    return {
        "status": "synthetic_only_not_patient_efficacy",
        "fixture": "four tied slots, twelve tokens, explicit noise seed 11",
        "global_rng_unchanged": torch.equal(before, torch.get_rng_state()),
        "results": rows,
        "interpretation": "Assignment can differ while constant-value updates remain identical. Lower overlap is not sufficient evidence of information or biological specialization.",
        "training_runs": 0, "patient_tasks": 0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    result = synthetic_probe() if args.synthetic else {
        "status": "plan_only", "patient_execution_supported": False,
        "modes": ["original", "sinkhorn", "mesh_uniform"],
        "scope": "Slot Attention internal assignment; same projections, values, GRU/MLP, initial states and slot count.",
        "command": "python -B scripts/check_slotspe_assignment_router.py --synthetic",
        "training_runs": 0, "patient_tasks": 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
