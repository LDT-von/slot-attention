"""Small CPU-only diagnostics of original SlotSPE MoESlotDecoder (no training).

Inputs and weights below are synthetic unless marked as a controlled example.
The original source tree is imported read-only, without creating bytecode.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import copy
import hashlib
import json
import platform

SOURCE = Path(r"C:\Users\栋栋\Desktop\特征解耦论文\SlotSPE-main\SlotSPE-main")
sys.path.insert(0, str(SOURCE))
import torch
from torch import nn
from models.SlotSPE import MoESlotDecoder

torch.set_num_threads(1)
torch.manual_seed(616)


def summary(x):
    x = x.detach().float().flatten()
    return {"min": x.min().item(), "median": x.median().item(),
            "mean": x.mean().item(), "max": x.max().item()}


def diagnose(model, slots, repeats=32):
    model.eval()
    logits, gates, keeps, raw_masses, max_selected = [], [], [], [], []
    with torch.no_grad():
        scores = model.pred_keep_slot(model.map(slots)).squeeze(-1)
        base_soft = torch.softmax(scores / model.temperature, -1)
        for run in range(repeats):
            torch.manual_seed(10000 + run)
            y, gate, keep = model(slots)
            logits.append(y)
            gates.append(gate)
            keeps.append(keep > 0.5)
            raw_masses.append((base_soft * keep).sum(-1))
            max_selected.append(keep.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1) > 0.5)
    ys, gs, ks = torch.stack(logits), torch.stack(gates), torch.stack(keeps)
    gsums = gs.sum(-1)
    p = gs / gsums.clamp_min(1e-30).unsqueeze(-1)
    neff = (-(p * p.clamp_min(1e-30).log()).sum(-1)).exp()
    return {
        "temperature": model.temperature,
        "repeats": repeats,
        "batch": slots.shape[0], "slots": slots.shape[1], "dim": slots.shape[2],
        "configured_k": model.k,
        "random_untrained_weights": True,
        "unique_keep_patterns_per_case": [len(set(tuple(v.tolist()) for v in ks[:, b])) for b in range(slots.shape[0])],
        "fraction_repeat_case_keep_differs_from_first": (ks != ks[0]).any(-1).float().mean().item(),
        "logit_std_across_repeats": summary(ys.std(0, unbiased=False)),
        "logit_abs_difference_from_first": summary((ys - ys[0]).abs()),
        "hard_selected_count": summary(ks.sum(-1)),
        "gate_count_over_0_01": summary((gs > 0.01).sum(-1)),
        "gate_count_over_1e_minus_6": summary((gs > 1e-6).sum(-1)),
        "entropy_effective_gate_count": summary(neff),
        "gate_sum": summary(gsums),
        "gate_sum_under_0_99_count": (gsums < 0.99).sum().item(),
        "gate_sum_under_0_5_count": (gsums < 0.5).sum().item(),
        "raw_masked_softmax_mass": summary(torch.stack(raw_masses)),
        "score_argmax_selected_fraction": torch.stack(max_selected).float().mean().item(),
        "score_range_per_case": summary(scores.max(-1).values - scores.min(-1).values),
    }


base = MoESlotDecoder(dim=32, num_slots=16, num_classes=4,
                      temperature=0.01, topk_ratio=0.25).eval()
slots = torch.randn(8, 16, 32)
temperature_results = []
for temperature in (0.01, 0.1, 1.0):
    model = copy.deepcopy(base)
    model.temperature = temperature
    temperature_results.append(diagnose(model, slots))

# Test the selector path only: the original downstream decoder receives a bool
# mask according to its call site. Boolean conversion removes the ST path.
gradient_results = {}
for use_bool_mask in (True, False):
    model = copy.deepcopy(base)
    model.temperature = 1.0
    torch.manual_seed(123)
    out, gate, keep = model(slots)
    if use_bool_mask:
        # Controlled downstream analog, not the full original model.
        mapped = model.map(slots)
        slot_values = model.decoder(mapped)
        mask = keep.bool()
        prediction = (slot_values * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
    else:
        prediction = out
    loss = prediction.square().mean()
    loss.backward()
    gradients = [p.grad for p in model.pred_keep_slot.parameters()]
    gradient_results["bool_mask_analog" if use_bool_mask else "original_weighted_output"] = {
        "temperature": 1.0,
        "selector_parameter_gradients_all_none": all(g is None for g in gradients),
        "selector_total_gradient_l2": sum(g.double().square().sum().item() for g in gradients if g is not None) ** 0.5,
        "scope": "selector path only; full model gradient must be checked separately",
    }


class FixedScore(nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.register_buffer("scores", torch.tensor(scores, dtype=torch.float32).view(1, -1, 1))
    def forward(self, x):
        return self.scores.expand(x.shape[0], -1, -1)


# Deliberately constructed score-gap example. k=4, S=16. Gumbel can exclude
# the unique score maximum, while the unnoised T=.01 gate has near-zero mass
# at every remaining slot. All other decoder code is original.
controlled = MoESlotDecoder(dim=32, num_slots=16, num_classes=4,
                            temperature=0.01, topk_ratio=0.25).eval()
controlled.load_state_dict(base.state_dict())
controlled.pred_keep_slot = FixedScore([1.0] + [0.0] * 15)
controlled_records = []
with torch.no_grad():
    for run in range(32):
        torch.manual_seed(20000 + run)
        y, gate, keep = controlled(slots[:1])
        controlled_records.append({
            "seed": 20000 + run,
            "argmax_selected": bool(keep[0, 0] > 0.5),
            "selected_indices": (keep[0] > 0.5).nonzero().flatten().tolist(),
            "gate_sum": gate.sum().item(),
            "output_l2": y.norm().item(),
        })

result = {
    "purpose": "CPU synthetic diagnostic of original decoder; no training, checkpoint, or patient performance evaluation",
    "source": str(SOURCE / "models" / "SlotSPE.py"),
    "source_sha256": hashlib.sha256((SOURCE / "models" / "SlotSPE.py").read_bytes()).hexdigest(),
    "torch": torch.__version__, "python": platform.python_version(), "device": "cpu",
    "weights_and_input_seed": 616,
    "temperature_control_note": "same state_dict, same slots, same per-repeat Gumbel RNG seeds at all temperatures",
    "temperature_results": temperature_results,
    "gradient_auxiliary": gradient_results,
    "controlled_counterexample": {
        "constructed": True,
        "keep_scores": [1.0] + [0.0] * 15,
        "temperature": 0.01,
        "trials": 32,
        "argmax_excluded_count": sum(not r["argmax_selected"] for r in controlled_records),
        "gate_sum": summary(torch.tensor([r["gate_sum"] for r in controlled_records])),
        "records": controlled_records,
        "boundary": "constructed score pattern establishes possibility; does not establish its prevalence in trained checkpoints",
    },
}
destination = Path(__file__).with_name("gate_results.json")
destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"saved": str(destination), "temperature_results": temperature_results,
                  "gradient_auxiliary": gradient_results,
                  "counterexample_argmax_excluded_count": result["controlled_counterexample"]["argmax_excluded_count"],
                  "counterexample_gate_sum": result["controlled_counterexample"]["gate_sum"]},
                 ensure_ascii=False, indent=2))
