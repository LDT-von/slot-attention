"""Read-only original SlotSPE audit, invented tensors only, no optimizer."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(r"C:\Users\栋栋\Desktop\特征解耦论文\SlotSPE-main\SlotSPE-main")
OUT = Path(__file__).parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from models.SlotSPE import SlotSPE, MoESlotDecoder
from models.slot_attention import MultiHeadSlotAttention
from models.transformer import Transformer, IterativeCrossAttTransformer
from utils.loss_func import NLLSurvLoss

torch.set_num_threads(1)
FILES = ["models/SlotSPE.py", "models/slot_attention.py", "models/transformer.py",
         "models/omics_encoder.py", "utils/loss_func.py", "utils/core_utils.py",
         "utils/process_args.py", "dataset/dataset_survival.py", "scripts/SlotSPE.sh"]

def hashes():
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}

def scalar(x):
    return float(x.detach().cpu())

def maxdiff(a, b):
    return scalar((a - b).abs().max())

def args(method="parallel_topk_st"):
    return SimpleNamespace(omic_sizes=[5, 7, 3], n_classes=4, encoding_dim=32,
        wsi_projection_dim=32, rna_format="Pathways", slot_num_wsi=8,
        slot_num_omics=8, slot_iters=10, temperature=0.01, topk_ratio=0.25,
        top_k_method=method, bag_loss="nll_surv", alpha_surv=0.0, lambda_recon_loss=0.01)

torch.manual_seed(19003)
sample = dict(x_wsi=torch.randn(2, 16, 32), omic_missing=False,
    x_omic1=torch.randn(2, 5), x_omic2=torch.randn(2, 7), x_omic3=torch.randn(2, 3),
    y=torch.tensor([1, 2]), c=torch.tensor([0.0, 1.0]))

def forward(model, data=None, seed=3):
    torch.manual_seed(seed)
    return model(**(sample if data is None else data))[0]

def capture_initial_noise(module, value):
    original = torch.randn
    def replace(*a, **kw):
        shape = a[0] if len(a) == 1 and isinstance(a[0], (tuple, torch.Size)) else a
        if tuple(shape) == tuple(value.shape):
            return value.to(device=kw.get("device", value.device), dtype=kw.get("dtype", value.dtype)).clone()
        return original(*a, **kw)
    return patch.object(torch, "randn", side_effect=replace)

report = dict(created_utc=datetime.now(timezone.utc).isoformat(), source=str(ROOT),
    scope="synthetic_original_modules_random_weights_only", patient_data_read=False,
    trained_checkpoint_loaded=False, optimizer_steps=0, device="cpu", torch=torch.__version__,
    source_hashes_before=hashes(), model_configuration=vars(args()), checks={}, findings={})

try:
    slot_results = []
    for seed in [3, 17, 29]:
        torch.manual_seed(seed)
        sa = MultiHeadSlotAttention(num_slots=8, dim=32, heads=8, iters=10).eval()
        inp = torch.randn(2, 24, 32)
        noise = torch.randn(2, 8, 32)
        perm = torch.tensor([2, 0, 7, 3, 1, 5, 6, 4])
        with torch.no_grad(), capture_initial_noise(sa, noise):
            original_slots = sa(inp)
        with torch.no_grad(), capture_initial_noise(sa, noise[:, perm]):
            permuted_slots = sa(inp)
        tied_noise = noise[:, :1].expand(-1, 8, -1).clone()
        with torch.no_grad(), capture_initial_noise(sa, tied_noise):
            tied_slots = sa(inp)
        cos = F.normalize(original_slots, dim=-1) @ F.normalize(original_slots, dim=-1).transpose(1, 2)
        offdiag = ~torch.eye(8, dtype=torch.bool)
        row = dict(seed=seed, permutation_equivariance_error=maxdiff(permuted_slots, original_slots[:, perm]),
            tied_initial_states_output_difference=maxdiff(tied_slots, tied_slots[:, :1].expand_as(tied_slots)),
            ordinary_initialization_mean_offdiag_cosine=scalar(cos[:, offdiag].mean()))
        slot_results.append(row)
    report["findings"]["slot_symmetry"] = slot_results
    report["checks"]["slot_permutation_equivariance"] = all(r["permutation_equivariance_error"] < 1e-5 for r in slot_results)
    report["checks"]["identical_slot_states_not_split_by_update"] = all(r["tied_initial_states_output_difference"] < 1e-5 for r in slot_results)

    grad_results = []
    for method in ["parallel_topk_st", "gumbel_topk_st"]:
        torch.manual_seed(3)
        m = SlotSPE(args(method)).train()
        logits, auxiliary = m(**sample)
        named = [(n, p) for n, p in m.named_parameters() if "pred_keep_slot" in n]
        objective = NLLSurvLoss()(logits, sample["y"], None, sample["c"]) / 2
        main_grad = torch.autograd.grad(objective, [p for _, p in named], allow_unused=True, retain_graph=True)
        aux_grad = torch.autograd.grad(auxiliary, [p for _, p in named], allow_unused=True)
        grad_results.append(dict(method=method, main_nll=scalar(objective), auxiliary=scalar(auxiliary),
            main_gradient={n: None if g is None else scalar(g.norm()) for (n, _), g in zip(named, main_grad)},
            auxiliary_gradient={n: None if g is None else scalar(g.norm()) for (n, _), g in zip(named, aux_grad)},
            main_all_none=all(g is None for g in main_grad), auxiliary_all_connected=all(g is not None for g in aux_grad)))
    report["findings"]["selector_gradient"] = grad_results
    report["checks"]["main_selector_gradient_disconnected_both_methods"] = all(r["main_all_none"] for r in grad_results)
    report["checks"]["auxiliary_selector_gradient_connected_both_methods"] = all(r["auxiliary_all_connected"] for r in grad_results)

    torch.manual_seed(17)
    m = SlotSPE(args()).eval()
    with torch.no_grad():
        logits20 = torch.stack([forward(m, seed=1000+i) for i in range(20)])
        replay_a = forward(m, seed=71)
        replay_b = forward(m, seed=71)
    report["findings"]["evaluation_randomness"] = dict(repetitions=20,
        logit_max_range=scalar((logits20.max(0).values-logits20.min(0).values).max()),
        exact_rng_replay_error=maxdiff(replay_a, replay_b), rule="one forward per repetition; no averaging")
    report["checks"]["eval_changes_with_rng"] = report["findings"]["evaluation_randomness"]["logit_max_range"] > 1e-6
    report["checks"]["eval_replay_same_rng"] = maxdiff(replay_a, replay_b) == 0.0

    counts = []
    with torch.no_grad():
        for length in [8, 16, 64]:
            torch.manual_seed(31)
            sw = m.slot_attention_wsi(torch.randn(2, length, 32))
            _, _, keep = m.slot_decoder_wsi(sw)
            counts.append(dict(input_tokens=length, candidate_shape=list(sw.shape),
                selected_counts=keep.detach().sum(-1).tolist(), bool_counts=keep.bool().sum(-1).tolist()))
    report["findings"]["fixed_cardinality"] = counts
    report["checks"]["candidate_and_selected_cardinality_fixed"] = all(r["candidate_shape"][1] == 8 and r["bool_counts"] == [2, 2] for r in counts)

    perm = torch.tensor([2, 0, 7, 3, 1, 5, 6, 4])
    def permute_slots(_module, _args, output):
        return output[:, perm]
    no_gumbel_randomness = lambda t: torch.full_like(t, 0.5)
    with torch.no_grad(), patch.object(torch, "rand_like", side_effect=no_gumbel_randomness):
        ref = forward(m, seed=89)
        hooks = [sa.register_forward_hook(permute_slots) for sa in [m.slot_attention_wsi, m.slot_attention_omic]]
        try:
            relabeled = forward(m, seed=89)
        finally:
            for h in hooks:
                h.remove()
    report["findings"]["full_prediction_slot_relabeling"] = dict(
        max_logit_difference=maxdiff(ref, relabeled), controlled_gumbel_noise="constant 0.5 uniforms in both runs",
        note="Checks arbitrary index labels, not biological semantics or observed cross-patient drift.")
    report["checks"]["prediction_invariant_to_slot_relabeling"] = maxdiff(ref, relabeled) < 1e-5

    with torch.no_grad():
        ref = forward(m, seed=101)
        data = dict(sample)
        data["x_wsi"] = sample["x_wsi"][:, torch.tensor([15, 0, 12, 2, 11, 3, 9, 4, 8, 5, 7, 6, 10, 1, 14, 13])]
        reordered = forward(m, data, seed=101)
    report["findings"]["patch_order_invariance"] = dict(max_logit_difference=maxdiff(ref, reordered),
        coordinates_supplied=False, note="Same feature multiset and RNG. No explicit spatial-layout interface; not evidence that local patch features lack morphology or that multiscale would improve prognosis.")
    report["checks"]["prediction_invariant_to_patch_reordering"] = maxdiff(ref, reordered) < 1e-5

    torch.manual_seed(29)
    transformer = Transformer(dim=32).eval()
    slots = torch.randn(2, 8, 32, requires_grad=True)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :2] = True
    out = transformer(slots, mask=mask)
    gradient = torch.autograd.grad(out.mean(1).square().sum(), slots)[0]
    changed = slots.detach().clone()
    changed[:, 2:] += 4 * torch.randn_like(changed[:, 2:])
    with torch.no_grad():
        out2 = transformer(changed, mask=mask)
        attn = transformer.get_attn(slots.detach(), mask=mask)
    report["findings"]["unselected_self_slots"] = dict(
        kept_output_change=maxdiff(out[:, :2], out2[:, :2]),
        all_slot_mean_change=maxdiff(out.mean(1), out2.mean(1)),
        dropped_slot_gradient_norm=scalar(gradient[:, 2:].norm()),
        dropped_query_first_head_weights=attn[0, 0, 2].tolist(),
        note="Fixed mask, controlled synthetic slot tensors; includes residual route and uniform fully masked query rows.")
    report["checks"]["unselected_slots_change_mean_despite_mask"] = maxdiff(out.mean(1), out2.mean(1)) > 1e-5 and scalar(gradient[:, 2:].norm()) > 1e-5

    captured = {}
    def save_output(name):
        def save(_module, _inputs, output):
            captured[name] = tuple(x.detach().clone() for x in output) if isinstance(output, tuple) else output.detach().clone()
        return save
    hooks = [m.slot_attention_wsi.register_forward_hook(save_output("wsi_slots")),
             m.slot_decoder_wsi.register_forward_hook(save_output("wsi_decoder")),
             m.cross_attention.register_forward_hook(save_output("cross_output"))]
    try:
        with torch.no_grad():
            ref = forward(m, seed=107)
    finally:
        for h in hooks:
            h.remove()
    keep_fixed = captured["wsi_decoder"][2].bool()
    torch.manual_seed(109)
    replacement_slots = captured["wsi_slots"] + (~keep_fixed).unsqueeze(-1) * (4 * torch.randn_like(captured["wsi_slots"]))
    hooks = [m.slot_attention_wsi.register_forward_hook(lambda _m, _i, _o: replacement_slots),
             m.slot_decoder_wsi.register_forward_hook(lambda _m, _i, _o: captured["wsi_decoder"]),
             m.cross_attention.register_forward_hook(lambda _m, _i, _o: captured["cross_output"])]
    try:
        with torch.no_grad():
            altered = forward(m, seed=107)
    finally:
        for h in hooks:
            h.remove()
    risk = lambda logits: -(1 - torch.sigmoid(logits)).cumprod(-1).sum(-1)
    report["findings"]["unselected_self_full_prediction"] = dict(
        selected_slots_unchanged=maxdiff(replacement_slots[keep_fixed], captured["wsi_slots"][keep_fixed]) == 0,
        keep_mask_fixed=True, cross_output_frozen=True, max_logit_difference=maxdiff(ref, altered),
        max_risk_difference=maxdiff(risk(ref), risk(altered)),
        note="Invented inputs, random untrained weights, post-slot intervention; does not replace final-checkpoint patient intervention.")
    report["checks"]["unselected_self_slots_change_full_prediction"] = maxdiff(ref, altered) > 1e-5

    padding = []
    with torch.no_grad():
        ref = forward(m, seed=103)
        for n in [16, 64]:
            data = dict(sample)
            data["x_wsi"] = torch.cat([sample["x_wsi"], torch.zeros(2, n, 32)], dim=1)
            pred = forward(m, data, seed=103)
            padding.append(dict(added_zero_tokens=n, max_logit_difference=maxdiff(ref, pred)))
        zero_projection = m.wsi_mlp(torch.zeros(1, 1, 32))
    report["findings"]["unmasked_zero_tokens"] = dict(rows=padding, zero_raw_feature_projection_norm=scalar(zero_projection.norm()),
        note="Invented zero rows, same weights/RNG. Loader can insert zero rows; observed patient prevalence and harm are not measured.")
    report["checks"]["zero_tokens_change_prediction"] = all(row["max_logit_difference"] > 1e-6 for row in padding)

    # Finite-score numerical counterexample, original decoder, fixed scores only.
    torch.manual_seed(3)
    decoder = MoESlotDecoder(dim=32, num_slots=8, num_classes=4, temperature=0.01, topk_ratio=0.25).eval()
    fake_slots = torch.randn(1, 8, 32)
    fixed_scores = torch.tensor([[[1.0]] + [[0.0]] * 7])
    def scores_hook(_module, _inp, _out):
        return fixed_scores
    score_hook = decoder.pred_keep_slot.register_forward_hook(scores_hook)
    gate_rows = []
    with torch.no_grad():
        try:
            for seed in range(20000, 20032):
                torch.manual_seed(seed)
                prediction, gate, keep = decoder(fake_slots)
                gate_rows.append(dict(seed=seed, top_score_selected=bool(keep[0, 0].bool()),
                    selected_indices=keep[0].bool().nonzero().flatten().tolist(), gate_sum=scalar(gate.sum()),
                    prediction_norm=scalar(prediction.norm()), finite=bool(torch.isfinite(prediction).all())))
        finally:
            score_hook.remove()
    report["findings"]["low_temperature_gate_counterexample"] = dict(scores=fixed_scores.flatten().tolist(),
        temperature=0.01, trials=gate_rows, below_1e6_count=sum(r["gate_sum"] < 1e-6 for r in gate_rows),
        note="Controlled reachable score pattern and original decoder; no claim about trained-model prevalence.")
    report["checks"]["gate_normalization_can_collapse"] = any(r["gate_sum"] < 1e-6 for r in gate_rows)

finally:
    report["source_hashes_after"] = hashes()
    report["source_unchanged"] = report["source_hashes_before"] == report["source_hashes_after"]
    report["checks"]["original_source_unchanged"] = report["source_unchanged"]
    report["all_checks_passed"] = all(report["checks"].values())
    (OUT / "synthetic_checks.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(checks=report["checks"], all_checks_passed=report["all_checks_passed"], report=str(OUT/"synthetic_checks.json")), ensure_ascii=False))

if not report["all_checks_passed"]:
    raise SystemExit(1)
