"""Generated-only contracts for native SlotSPE SCM/SRO training adapters."""
import json
import random

import numpy as np
import pytest
import torch

from slotspe_proof import patient, redundancy_best_engine as best
from slotspe_proof.diversity_engine import (
    ARMS, VALID_MASK_KEY, GraphReadingAllocation, add_real_token_mask,
    arm_contract, execute_diversity_batch, run_diversity_arm,
    validate_diversity_config,
)
from slotspe_proof.native_variants import make_native_adapter
from slotspe_proof.redundancy_engine import (
    _construct, _item, case_seed, execute_logical_batch, make_initial_states, state_hash,
)
from slotspe_proof.slot_regularization import SlotCapture
from tests.test_slotspe_redundancy_best_engine import SOURCE, _base_config, _best_manifest
from tests.test_slotspe_redundancy_engine import MANIFEST, invented_batch, reference_for


@pytest.fixture(autouse=True)
def isolate_generated_runtime():
    previous_threads = torch.get_num_threads()
    py_state, np_state = random.getstate(), np.random.get_state()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            yield
    finally:
        torch.set_num_threads(previous_threads)
        random.setstate(py_state)
        np.random.set_state(np_state)


def _manifest_with_counts(root):
    manifest, reference = _best_manifest(root)
    for item in manifest["patients"].values():
        item["full_patch_count"] = 9
    manifest["patches"] = 9
    return manifest, reference


def _native_pair():
    core, probe = make_initial_states(MANIFEST, SOURCE, 3, projection_dim=16, slot_iters=2)
    model, heads = _construct(MANIFEST, SOURCE, 16, 2, "cpu")
    model.load_state_dict(core)
    heads.load_state_dict(probe)
    model.train()
    heads.train()
    return model, heads


def test_contracts_keep_original_forward_and_content_weight():
    assert ARMS == ("SCM", "SRO", "R4_SCM", "R4_SRO")
    assert validate_diversity_config({"warmup_epochs": 0})["warmup_epochs"] == 0
    for arm in ARMS:
        item = arm_contract(arm)
        assert item["adapter"] == "S0"
        assert item["content_weight"] == (5.0 if arm.startswith("R4_") else 0.0)
        assert item["gram_weight"] == item["slotbert_weight"] == 0
        assert item["no_hazard_variance_or_added_prediction_head"]
    with pytest.raises(ValueError, match="unknown"):
        validate_diversity_config({"hazard_variance": .1})


@pytest.mark.parametrize("settings", [
    {"weight": float("nan")}, {"weight": -.1}, {"cosine_margin": 1},
    {"read_overlap_margin": -.1}, {"warmup_epochs": -1}, {"warmup_epochs": True},
])
def test_invalid_settings_fail_closed(settings):
    with pytest.raises(ValueError):
        validate_diversity_config(settings)


def test_real_token_mask_uses_metadata_not_zero_valued_rows():
    batch = {"x_wsi": torch.zeros(2, 9, 8)}
    manifest = {"patients": {"a": {"full_patch_count": 3}, "b": {"full_patch_count": 15}}}
    copied = add_real_token_mask(batch, manifest, ["a", "b"])
    assert copied[VALID_MASK_KEY].sum(-1).tolist() == [3, 9]
    assert VALID_MASK_KEY not in batch
    assert copied["x_wsi"] is batch["x_wsi"]
    manifest["patients"]["a"].pop("full_patch_count")
    with pytest.raises(ValueError, match="full_patch_count"):
        add_real_token_mask(batch, manifest, ["a", "b"])


def test_graph_reading_capture_matches_native_forward_rng_and_call_zero():
    model, _ = _native_pair()
    batch = invented_batch(1)
    snapshot = patient._source_hashes({"source_root": str(SOURCE)})
    with make_native_adapter(model, "S0", capture=True) as adapter, SlotCapture(model) as capture:
        patient._rng(3289, torch, np)
        expected, auxiliary = model(**batch)
        expected = expected.detach().clone()
        auxiliary = auxiliary.detach().clone()
        rng = torch.random.get_rng_state().clone()
        slot_values = {name: values[0].detach().clone() for name, values in capture.slots.items()}
        expected_reads = {row["modality"]: row["attentions"][-1]
                          for row in adapter.records if row["call_index"] == 0}
    with GraphReadingAllocation(model) as adapter, SlotCapture(model) as capture:
        patient._rng(3289, torch, np)
        prediction, actual_aux = model(**batch)
        assert torch.equal(expected, prediction)
        assert torch.equal(auxiliary, actual_aux)
        assert torch.equal(rng, torch.random.get_rng_state())
        assert {name: len(values) for name, values in capture.slots.items()} == {"wsi": 1, "omic": 2}
        for name in ("wsi", "omic"):
            assert torch.equal(slot_values[name], capture.slots[name][0])
            assert torch.equal(expected_reads[name], adapter.graph_attentions[name])
            assert adapter.graph_attentions[name].requires_grad
        assert len(adapter.records) == 3
        model.eval()
        with torch.no_grad():
            model(**batch)
        assert adapter.graph_attentions == {}
    assert "forward" not in model.slot_attention_wsi.__dict__
    assert "forward" not in model.slot_attention_omic.__dict__
    assert snapshot == patient._source_hashes({"source_root": str(SOURCE)})


@pytest.mark.parametrize("arm", ARMS)
def test_zero_weight_single_update_is_exactly_original_s0_or_r4(arm):
    batch = invented_batch(2)
    reference = reference_for(batch)
    targets = [reference.targets(_item(batch, index)) for index in range(2)]
    seeds = [case_seed(3, 0, index) for index in range(2)]
    original_arm = "R4" if arm.startswith("R4_") else "S0"
    original, original_heads = _native_pair()
    model, heads = _native_pair()
    opt = torch.optim.Adam(original.parameters(), lr=5e-4)
    probe_opt = torch.optim.Adam(original_heads.parameters(), lr=5e-4)
    new_opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    new_probe_opt = torch.optim.Adam(heads.parameters(), lr=5e-4)
    with make_native_adapter(original, "S0", capture=True) as adapter, SlotCapture(original) as capture:
        expected = execute_logical_batch(original, original_heads, adapter, capture, batch, targets,
                                         original_arm, opt, probe_opt, seeds)
    with GraphReadingAllocation(model) as adapter, SlotCapture(model) as capture:
        actual = execute_diversity_batch(model, heads, adapter, capture, batch, targets,
                                         arm, new_opt, new_probe_opt, seeds,
                                         epoch=1, diversity={"weight": 0.0})
    for key in ("total_loss", "main_nll", "native_auxiliary", "content_loss", "probe_loss",
                "core_gradient_norm", "probe_gradient_norm", "added_loss"):
        assert actual[key] == expected[key], key
    assert state_hash(model.state_dict()) == state_hash(original.state_dict())
    assert state_hash(heads.state_dict()) == state_hash(original_heads.state_dict())
    # Adam's moments must also remain paired, not just current parameters.
    for left, right in zip(opt.state.values(), new_opt.state.values()):
        assert left.keys() == right.keys()
        for name in left:
            assert torch.equal(left[name], right[name]), name


@pytest.mark.parametrize("arm", ARMS)
def test_generated_best_smoke_all_new_arms_and_graph_gradients(tmp_path, arm):
    manifest, reference = _manifest_with_counts(tmp_path)
    core, probe = make_initial_states(manifest, SOURCE, 3, projection_dim=16, slot_iters=2)
    config = _base_config(tmp_path, epochs=2, variants=list(ARMS), diversity={"warmup_epochs": 5})
    original_globals = {name: best.run_arm_best.__globals__[name] for name in (
        "ARMS", "ARM_CONTRACTS", "execute_logical_batch", "make_native_adapter", "patient")}
    original_batch = patient._batch
    source_before = patient._source_hashes(config)
    result = run_diversity_arm(config, tmp_path, manifest, arm, core, probe, reference,
                               _generated_test=True)
    assert result["status"] == "completed"
    assert result["protocol"]["checkpoint_selection"] == best.CHECKPOINT_SELECTION
    assert result["protocol"]["contract"]["diversity"]["weight"] == .1
    assert result["loss_adapter"]["new_prediction_parameters"] == 0
    assert result["counters"]["core_optimizer_steps"] == 6
    curve = json.loads((tmp_path / "arms" / arm / "curve.json").read_text(encoding="utf-8"))
    assert {row["diversity_epoch"] for row in curve} == {1, 2}
    assert all(row["diversity_effective_weight"] == pytest.approx(.02 * row["epoch"]) for row in curve)
    assert all(0 <= row["diversity_loss"] <= 1 for row in curve)
    first = curve[0]
    assert first["diversity_metrics"]
    partial = first["component_gradient_norms"]["weighted_diversity"]
    assert any(value["connected"] for value in partial.values())
    if first["diversity_loss"] > 0:
        assert any(value["connected"] and value["norm"] > 0 for value in partial.values())
    else:
        # A margin objective must leave already-separated generated slots alone.
        assert arm.endswith("SCM")
        assert all(value["norm"] in (None, 0.0) for value in partial.values())
    assert "weighted_content_vs_diversity" in first["component_gradient_norms"]["cosines"]
    for name, value in original_globals.items():
        assert best.run_arm_best.__globals__[name] is value
    assert patient._batch is original_batch
    assert patient._source_hashes(config) == source_before
    for name in ("last.pt", "best.pt", "summary.json", "eval_repeats.json", "epoch_validation.json"):
        assert (tmp_path / "arms" / arm / name).exists()


@pytest.mark.parametrize("arm,old_arm", [("SCM", "S0"), ("R4_SRO", "R4")])
def test_zero_weight_full_training_trajectory_and_selected_checkpoint_match(tmp_path, arm, old_arm):
    manifest, reference = _manifest_with_counts(tmp_path)
    core, probe = make_initial_states(manifest, SOURCE, 3, projection_dim=16, slot_iters=2)
    config = _base_config(tmp_path, epochs=2)
    expected = best.run_arm_best(config, tmp_path, manifest, old_arm, core, probe, reference,
                                 output_root=tmp_path / "old", _generated_test=True)
    config.update(variants=list(ARMS), diversity={"weight": 0.0, "warmup_epochs": 0})
    actual = run_diversity_arm(config, tmp_path, manifest, arm, core, probe, reference,
                               output_root=tmp_path / "new", _generated_test=True)
    old_root, new_root = tmp_path / "old/arms" / old_arm, tmp_path / "new/arms" / arm
    old_curve = json.loads((old_root / "curve.json").read_text(encoding="utf-8"))
    new_curve = json.loads((new_root / "curve.json").read_text(encoding="utf-8"))
    assert len(old_curve) == len(new_curve)
    for left, right in zip(old_curve, new_curve):
        for key in ("epoch", "native_seeds", "total_loss", "main_nll", "native_auxiliary", "probe_loss",
                    "content_loss", "core_gradient_norm", "probe_gradient_norm"):
            assert left[key] == right[key], (key, left[key], right[key])
    for filename in ("last.pt", "best.pt"):
        left = torch.load(old_root / filename, map_location="cpu", weights_only=True)
        right = torch.load(new_root / filename, map_location="cpu", weights_only=True)
        for key in ("state_dict", "probe_state_dict"):
            assert state_hash(left[key]) == state_hash(right[key])
    assert actual["best_epoch"] == expected["best_epoch"]
    assert actual["evaluation_summary"] == expected["evaluation_summary"]


def test_failure_restores_lock_and_never_mutates_historical_globals(tmp_path):
    manifest, reference = _manifest_with_counts(tmp_path)
    core, probe = make_initial_states(manifest, SOURCE, 3, projection_dim=16, slot_iters=2)
    config = _base_config(tmp_path, epochs=1, variants=list(ARMS))
    occupied = tmp_path / "arms/SCM"
    occupied.mkdir(parents=True)
    (occupied / "existing.txt").write_text("preserve", encoding="utf-8")
    before = {key: best.run_arm_best.__globals__[key] for key in (
        "ARMS", "ARM_CONTRACTS", "execute_logical_batch", "make_native_adapter", "patient")}
    with pytest.raises(RuntimeError, match="overwrite"):
        run_diversity_arm(config, tmp_path, manifest, "SCM", core, probe, reference,
                          _generated_test=True)
    for key, value in before.items():
        assert best.run_arm_best.__globals__[key] is value
    assert (occupied / "existing.txt").read_text(encoding="utf-8") == "preserve"
    # A subsequent valid run proves the serialization lock was released.
    result = run_diversity_arm(config, tmp_path, manifest, "SRO", core, probe, reference,
                               output_root=tmp_path / "second", _generated_test=True)
    assert result["status"] == "completed"


def test_missing_real_count_refuses_before_output(tmp_path):
    manifest, reference = _best_manifest(tmp_path)
    core, probe = make_initial_states(manifest, SOURCE, 3, projection_dim=16, slot_iters=2)
    config = _base_config(tmp_path, epochs=1, variants=list(ARMS))
    with pytest.raises(ValueError, match="full_patch_count"):
        run_diversity_arm(config, tmp_path, manifest, "SRO", core, probe, reference,
                          _generated_test=True)
    assert not (tmp_path / "arms").exists()


def test_scm_reaches_native_weights_when_generated_slots_exceed_margin():
    model, heads = _native_pair()
    with torch.no_grad():
        # Controlled, invented near-duplicate initial slots; no real weights.
        model.slot_attention_wsi.slots_logsigma.fill_(-3.0)
        model.slot_attention_omic.slots_logsigma.fill_(-3.0)
    batch = invented_batch(2)
    reference = reference_for(batch)
    targets = [reference.targets(_item(batch, i)) for i in range(2)]
    with GraphReadingAllocation(model) as adapter, SlotCapture(model) as capture:
        row = execute_diversity_batch(model, heads, adapter, capture, batch, targets,
            "SCM", torch.optim.Adam(model.parameters(), lr=5e-4),
            torch.optim.Adam(heads.parameters(), lr=5e-4), [case_seed(3, 0, i) for i in range(2)],
            epoch=1, measure_gradient_components=True)
    assert row["scm_loss"] > 0
    gradients = row["component_gradient_norms"]["weighted_diversity"]
    for modality in ("wsi", "omic"):
        assert any(value["connected"] and value["norm"] > 0
                   for name, value in gradients.items() if name.startswith(modality + "."))
