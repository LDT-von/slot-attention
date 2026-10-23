"""Eleven-arm execution contracts using generated tensors only, no patient I/O."""
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from survot_rank.research.slotspe_proof import patient
from survot_rank.research.slotspe_proof.model import build_model, make_args
from survot_rank.research.slotspe_proof.native_variants import make_native_adapter
from survot_rank.research.slotspe_proof.redundancy_content import ContentReference
from survot_rank.research.slotspe_proof.redundancy_engine import (
    ARMS, ARM_CONTRACTS, _construct, angular_metrics, case_seed,
    execute_logical_batch, make_initial_states, matched_angular_distance, run_arm, state_hash,
)
from survot_rank.research.slotspe_proof.slot_regularization import SlotCapture


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/SlotSPE"
MANIFEST = {"omic_sizes": [3, 5, 7], "encoding_dim": 8}


@pytest.fixture(autouse=True)
def isolated_runtime():
    threads = torch.get_num_threads()
    py_state, np_state = random.getstate(), np.random.get_state()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            yield
    finally:
        torch.set_num_threads(threads)
        random.setstate(py_state)
        np.random.set_state(np_state)


def invented_batch(count=2):
    generator = torch.Generator().manual_seed(51091)
    result = {"x_wsi": torch.randn(count, 9, 8, generator=generator),
              "omic_missing": False, "y": torch.arange(count) % 4,
              "c": torch.arange(count).float() % 2}
    for index, dimension in enumerate(MANIFEST["omic_sizes"]):
        result[f"x_omic{index+1}"] = torch.randn(count, dimension, generator=generator)
    return result


def cases_from_batch(batch):
    for index in range(batch["x_wsi"].shape[0]):
        yield {name: value[index:index+1] if torch.is_tensor(value) else value
               for name, value in batch.items()}


def reference_for(batch):
    return ContentReference(MANIFEST["omic_sizes"], MANIFEST["encoding_dim"], target_dim=32,
                            clusters=8, kmeans_steps=2).fit(cases_from_batch(batch))


def test_registry_has_only_requested_eleven_arms_and_fixed_weights():
    assert len(ARMS) == len(set(ARMS)) == 11
    assert set(ARMS) == {"S0", "S1", "S2", "S3", "S4", "S5", "R1", "R2", "R3", "R4", "R5"}
    assert "S6" not in ARM_CONTRACTS and "R0" not in ARM_CONTRACTS
    assert ARM_CONTRACTS["R1"]["gram_weight"] == 10
    assert ARM_CONTRACTS["R5"]["content_weight"] == 5
    assert ARM_CONTRACTS["R2"]["slotbert_temperature"] == .5


def test_pair_initialization_reproducible_and_preserves_all_global_rng():
    random.seed(199)
    np.random.seed(199)
    torch.manual_seed(199)
    py_state, np_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    first = make_initial_states(MANIFEST, SOURCE, 3, projection_dim=16, slot_iters=2)
    assert random.getstate() == py_state
    np.testing.assert_equal(np.random.get_state(), np_state)
    assert torch.equal(torch.get_rng_state(), cpu_state)
    second = make_initial_states(MANIFEST, SOURCE, 3, projection_dim=16, slot_iters=2)
    assert [state_hash(value) for value in first] == [state_hash(value) for value in second]
    third = make_initial_states(MANIFEST, SOURCE, 17, projection_dim=16, slot_iters=2)
    assert state_hash(first[0]) != state_hash(third[0])


def test_case_seed_is_paired_unique_and_evaluation_disjoint():
    train = [case_seed(3, epoch, index) for epoch in range(3) for index in range(379)]
    evaluation = [case_seed(3, repeat, index, evaluation=True) for repeat in range(20) for index in range(77)]
    assert len(set(train)) == len(train)
    assert len(set(evaluation)) == len(evaluation)
    assert not set(train) & set(evaluation)
    with pytest.raises(ValueError):
        case_seed(3, 0, -1)


def test_single_and_zero_slots_are_not_reported_as_diverse():
    one = angular_metrics(torch.tensor([[[1., 0.]]]))
    assert one["cosine"] is None and one["slot_count"] == 1
    assert not one["pair_metrics_defined"]
    zero = angular_metrics(torch.zeros(1, 8, 3))
    assert zero["effective_rank"] is None and zero["cosine"] is None
    assert zero["nonzero_count"] == 0
    opposite = angular_metrics(torch.tensor([[[1., 0.], [-1., 0.]]]))
    assert opposite["cosine"] == -1 and opposite["absolute_cosine"] == 1


def test_angular_matching_covers_ragged_slots_and_rejects_zero_vectors():
    orthogonal = torch.eye(3)
    assert matched_angular_distance(orthogonal, orthogonal[[2, 0, 1]]) == 0
    assert matched_angular_distance(orthogonal[:1], orthogonal) == pytest.approx(2 / 3)
    assert matched_angular_distance(orthogonal[:1], -orthogonal[:1]) == pytest.approx(2)
    assert matched_angular_distance(orthogonal, torch.zeros_like(orthogonal)) is None


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_full_native_dimension_ten_iterations_core_and_probe_backward(arm):
    """All eleven use native 256 dimensions, 8 slots and 10 iterations."""
    before = patient._source_hashes({"source_root": str(SOURCE)})
    batch = invented_batch(1)
    reference = reference_for(invented_batch(3))
    core, probe = make_initial_states(MANIFEST, SOURCE, 3, projection_dim=256, slot_iters=10)
    model, heads = _construct(MANIFEST, SOURCE, 256, 10, "cpu")
    model.load_state_dict(core)
    heads.load_state_dict(probe)
    model.train()
    core_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    probe_optimizer = torch.optim.Adam(heads.parameters(), lr=5e-4)
    with make_native_adapter(model, ARM_CONTRACTS[arm]["adapter"], capture=True) as adapter, SlotCapture(model) as capture:
        result = execute_logical_batch(model, heads, adapter, capture, batch,
                    [reference.targets(batch)], arm, core_optimizer, probe_optimizer, [case_seed(3, 0, 0)])
        assert {name: len(value) for name, value in capture.slots.items()} == {"wsi": 1, "omic": 2}
        assert all(value.isfinite().all() for values in capture.slots.values() for value in values)
        assert result["root_forwards"] == result["core_optimizer_steps"] == result["probe_optimizer_steps"] == 1
        assert result["slot_attention_calls"] == 3
        assert result["core_gradient_norm"] > 0 and result["probe_gradient_norm"] > 0
        for key in ("main_nll", "native_auxiliary", "total_loss", "probe_loss"):
            assert np.isfinite(result[key])
        assert result["gram_loss"] > 0 if arm in {"R1", "R5"} else result["gram_loss"] == 0
        assert result["slotbert_loss"] > 0 if arm in {"R2", "R3"} else result["slotbert_loss"] == 0
        assert result["content_loss"] > 0 if arm in {"R3", "R4", "R5"} else result["content_loss"] == 0
        assert all(parameter.grad is None for parameter in heads.parameters())
    assert state_hash(model.state_dict()) != state_hash(core)
    assert state_hash(heads.state_dict()) != state_hash(probe)
    assert patient._source_hashes({"source_root": str(SOURCE)}) == before
    assert not hasattr(model, "_native_allocation_active")
    assert not model._forward_pre_hooks


def test_logical_batch_rejects_duplicate_seeds_before_any_update():
    model, heads = _construct(MANIFEST, SOURCE, 16, 2, "cpu")
    model.train()
    batch = invented_batch(2)
    reference = reference_for(batch)
    with make_native_adapter(model, "S0", capture=True) as adapter, SlotCapture(model) as capture:
        with pytest.raises(ValueError, match="unique"):
            execute_logical_batch(model, heads, adapter, capture, batch,
                [reference.targets(value) for value in cases_from_batch(batch)], "R1",
                torch.optim.Adam(model.parameters()), torch.optim.Adam(heads.parameters()), [3, 3])
    assert all(parameter.grad is None for parameter in model.parameters())


def generated_manifest(root):
    batch = invented_batch(5)
    patients = {}
    ids = [f"invented_{index}" for index in range(5)]
    for index, name in enumerate(ids):
        path = f"prepared/cache/patients/{name}.pt"
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"patient_id": name, "patches": batch["x_wsi"][index],
                    "omics": [batch[f"x_omic{k+1}"][index] for k in range(3)]}, root / path)
        patients[name] = {"cache_path": path, "censor": float(batch["c"][index]), "time": float(index + 1)}
    fold = {"train": ids[:3], "val": ids[3:], "labels": {name: index % 4 for index, name in enumerate(ids)}}
    manifest = {**MANIFEST, "patients": patients, "folds": {"0": fold},
                "source_hashes": patient._source_hashes({"source_root": str(SOURCE)}), "generated_test_only": True}
    reference = reference_for({key: value[:3] if torch.is_tensor(value) else value for key, value in batch.items()})
    return manifest, reference


def test_generated_end_to_end_receipts_predictions_ragged_content_and_no_resume(tmp_path):
    manifest, reference = generated_manifest(tmp_path)
    core, probe = make_initial_states(manifest, SOURCE, 3, projection_dim=16, slot_iters=2)
    config = {"variants": list(ARMS), "study": "blca", "folds": [0], "seeds": [3],
              "feature_encoder": "UNI2-h",
              "source_root": str(SOURCE), "device": "cpu", "torch_threads": 1,
              "projection_dim": 16, "slot_iters": 2, "lr": 5e-4, "probe_lr": 5e-4,
              "epochs": 1, "eval_repeats": 2, "batch_size": 2, "patches": 9}
    result = run_arm(config, tmp_path, manifest, "R5", core, probe, reference, _generated_test=True)
    assert result["status"] == "completed"
    assert result["counters"] == {"root_forwards": 9, "slot_attention_calls": 21, "slot_attention_iterations": 42,
        "core_backward_calls": 2, "probe_backward_calls": 2,
        "core_optimizer_steps": 2, "probe_optimizer_steps": 2, "component_autograd_grad_calls": 3}
    out = tmp_path / "arms/R5"
    for name in ("last.pt", "curve.json", "summary.json", "eval_repeats.json", "diagnostics.pt", "validation_predictions.csv"):
        assert (out / name).is_file()
    checkpoint = torch.load(out / "last.pt", weights_only=False)
    assert checkpoint["protocol"]["initial_core_hash"] == state_hash(core)
    assert checkpoint["reference_state"]["schema"] == reference.state_dict()["schema"]
    assert len(patient._read(out / "eval_repeats.json")) == 2
    diagnostics = patient._read(out / "patient_diagnostics.json")
    assert len(diagnostics) == 4
    assert {row["patient_id"] for row in diagnostics} == set(manifest["folds"]["0"]["val"])
    assert all(row["donor_patient_id"] != row["patient_id"] for row in diagnostics)
    assert all(row["modalities"]["omic"]["own_mse"] >= 0 for row in diagnostics)
    assert all(row["modalities"]["wsi"]["same_noise_changed_input_distance"] is not None
               for row in diagnostics if row["repeat"] == 0)
    assert all(row["modalities"]["wsi"]["same_input_changed_noise_distance"] is not None
               for row in diagnostics if row["repeat"] == 1)
    curve = patient._read(out / "curve.json")
    assert curve[0]["component_gradient_norms"]["weighted_added_objective"]["wsi"]["norm"] > 0
    assert curve[1]["component_gradient_norms"] is None
    raw = torch.load(out / "raw_validation_repeat_00.pt", weights_only=True)
    assert len(raw) == 2 and set(raw[0]["final_attention"]) == {"wsi", "omic"}
    assert "same_noise_donor_slots" in raw[0]
    with pytest.raises(RuntimeError, match="overwrite"):
        run_arm(config, tmp_path, manifest, "R5", core, probe, reference, _generated_test=True)
    with pytest.raises(ValueError, match="native 256"):
        run_arm(config, tmp_path, manifest, "R5", core, probe, reference)
    with pytest.raises(ValueError, match="UNI2-h"):
        run_arm({**config, "feature_encoder": "other"}, tmp_path, manifest, "R5", core, probe, reference)
    with pytest.raises(ValueError, match="fixed contracts"):
        run_arm({**config, "dias_distance": .1}, tmp_path, manifest, "R5", core, probe, reference)


def test_s0_detached_probe_training_retains_native_core_update():
    """Diagnostic heads and graph captures do not change the reference objective."""
    batch = invented_batch(2)
    reference = reference_for(batch)
    core, probe = make_initial_states(MANIFEST, SOURCE, 3, projection_dim=16, slot_iters=2)
    model, heads = _construct(MANIFEST, SOURCE, 16, 2, "cpu")
    model.load_state_dict(core)
    heads.load_state_dict(probe)
    model.train()
    seeds = [case_seed(3, 0, index) for index in range(2)]
    with make_native_adapter(model, "S0", capture=True) as adapter, SlotCapture(model) as capture:
        execute_logical_batch(model, heads, adapter, capture, batch,
            [reference.targets(value) for value in cases_from_batch(batch)], "S0",
            torch.optim.Adam(model.parameters(), lr=5e-4), torch.optim.Adam(heads.parameters(), lr=5e-4), seeds,
            measure_gradient_components=True)
    native, _ = _construct(MANIFEST, SOURCE, 16, 2, "cpu")
    native.load_state_dict(core)
    native.train()
    optimizer = torch.optim.Adam(native.parameters(), lr=5e-4)
    predictions, auxiliary = [], []
    for seed, case in zip(seeds, cases_from_batch(batch)):
        patient._rng(seed, torch, np)
        logits, extra = native(**case)
        predictions.append(logits)
        auxiliary.append(extra)
    total = native.loss_fn(torch.cat(predictions), y=batch["y"], c=batch["c"], t=None) / 2 + torch.stack(auxiliary).mean()
    total.backward()
    optimizer.step()
    for name, value in native.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=1e-7, rtol=1e-6)


def test_nonfinite_optimizer_result_is_rejected_before_next_batch():
    batch = invented_batch(1)
    reference = reference_for(invented_batch(3))
    model, heads = _construct(MANIFEST, SOURCE, 16, 2, "cpu")
    model.train()
    core_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    parameter = next(model.parameters())

    def broken_step():
        with torch.no_grad():
            parameter.fill_(float("inf"))

    core_optimizer.step = broken_step
    with make_native_adapter(model, "S0", capture=True) as adapter, SlotCapture(model) as capture:
        with pytest.raises(RuntimeError, match="updated core"):
            execute_logical_batch(model, heads, adapter, capture, batch, [reference.targets(batch)], "R1",
                                  core_optimizer, torch.optim.Adam(heads.parameters()), [case_seed(3, 0, 0)])
