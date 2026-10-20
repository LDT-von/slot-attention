"""Fixed-selection self interventions on slots produced by a saved model."""


def slot_intervention(model, batch, patient_ids, seed, checkpoint_sha256):
    import torch
    from .model import selected_pool
    model.eval()
    captured, hooks = {}, []
    for key in ("wsi", "omic"):
        def capture(module, inputs, output, name=key):
            captured[name] = (inputs[0].detach().clone(), output[2].detach().clone())
        hooks.append(getattr(model, "slot_decoder_" + key).register_forward_hook(capture))
    try:
        with torch.no_grad():
            model(**batch)
    finally:
        for hook in hooks:
            hook.remove()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    arm = getattr(model, "_proof_arm", "original")
    records = []
    with torch.no_grad():
        cross = model.cross_attention(captured["wsi"][0], captured["omic"][0]).mean(1)
        before, after, selections, changes, selected_changes = {}, {}, {}, {}, {}
        for key in ("wsi", "omic"):
            slots, keep = captured[key]
            hard = keep > 0.5
            noise = torch.randn(slots.shape, generator=generator).to(slots)
            changed = torch.where(hard[..., None], slots, slots + 2 * noise)
            attention = getattr(model, "self_attention_" + key)
            first, second = attention(slots, mask=hard), attention(changed, mask=hard)
            pool = (lambda value: selected_pool(value, keep)) if arm in ("B", "C", "readout_fix") else (lambda value: value.mean(1))
            before[key], after[key] = pool(first), pool(second)
            changes[key] = (before[key] - after[key]).abs().amax(-1).cpu().tolist()
            selected_changes[key] = ((changed - slots).abs() * hard[..., None]).amax((1, 2)).cpu().tolist()
            selections[key] = [row.nonzero().flatten().cpu().tolist() for row in hard]
        # Hold cross fixed: its intentional dense use of dropped slots must not
        # be mistaken for leakage in the self readout being tested here.
        first_logits = model.to_logits(torch.cat([cross, before["wsi"], before["omic"]], dim=1))
        second_logits = model.to_logits(torch.cat([cross, after["wsi"], after["omic"]], dim=1))
        risk = lambda logits: -(1 - logits.sigmoid()).cumprod(-1).sum(-1)
        risk_delta = (risk(first_logits) - risk(second_logits)).abs().cpu().tolist()
        logits_delta = (first_logits - second_logits).abs().amax(-1).cpu().tolist()
        for i, pid in enumerate(patient_ids):
            records.append({"patient_id": pid, "selected_indices": {key: selections[key][i] for key in selections},
                "self_pool_max_abs_change": {key: changes[key][i] for key in changes},
                "selected_slot_max_abs_change": {key: selected_changes[key][i] for key in selected_changes},
                "self_pool_before": {key: before[key][i].cpu().tolist() for key in before},
                "self_pool_after": {key: after[key][i].cpu().tolist() for key in after},
                "cross_frozen_logits_before": first_logits[i].cpu().tolist(),
                "cross_frozen_logits_after": second_logits[i].cpu().tolist(),
                "cross_frozen_risk_before": float(risk(first_logits)[i]),
                "cross_frozen_risk_after": float(risk(second_logits)[i]),
                "cross_frozen_fused_logits_max_abs_change": logits_delta[i],
                "cross_frozen_risk_abs_change": risk_delta[i]})
    maximum = max((max(row["self_pool_max_abs_change"].values()) for row in records), default=0.)
    return {"status": "measured", "input_scope": "slots_from_prepared_patient_inputs_at_saved_final_weights",
        "arm": arm, "patient_ids": list(patient_ids), "checkpoint_sha256": checkpoint_sha256,
        "seed": seed, "fixed_selection": True, "cross_held_fixed": True,
        "unselected_slot_perturbation_scale": 2,
        "max_self_pool_change": maximum,
        "self_interference": "observed" if maximum > 1e-7 else "not_observed", "records": records,
        "interpretation": "Conditional self-path intervention, not global exclusion or generalization; cross remains dense"}
