"""Invented-input integration of the unified plugin host and native trainer."""
from pathlib import Path

import numpy as np
import pytest
import torch

from slotspe_proof import patient
from slotspe_proof.plugin_engine import (
    _build, _case, _run_arm, best_candidate, logical_groups, logical_update, schedule_seed,
)
from slotspe_proof.plugin_queue import PRESETS, REPO
from slotspe_proof.redundancy_content import ContentReference
from slotspe_proof.redundancy_engine import _cpu_state


@pytest.fixture(autouse=True)
def isolated():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        yield
    torch.set_num_threads(threads)


def cases(count=4):
    generator = torch.Generator().manual_seed(70719)
    return [{"x_wsi": torch.randn(1, 11, 8, generator=generator), "omic_missing": False,
             "x_omic1": torch.randn(1, 3, generator=generator), "x_omic2": torch.randn(1, 5, generator=generator),
             "y": torch.tensor([i % 4]), "c": torch.tensor([float(i % 2)])} for i in range(count)]


def config():
    return {"source_root": str(REPO / "third_party/SlotSPE"), "projection_dim": 16,
            "slot_iters": 2, "device": "cpu", "epochs": 2, "batch_size": 4,
            "patches": 11, "eval_repeats": 2, "lr": .0005}


def test_schedule_namespaces_and_tail_merge_preserve_all_patients():
    groups = logical_groups(list(range(17)))
    assert [len(g) for g in groups] == [8, 9]
    assert [v for g in groups for v in g] == list(range(17))
    seeds = [schedule_seed(p, i, j) for p in ("train", "select", "repeat") for i in range(30) for j in range(400)]
    assert len(seeds) == len(set(seeds))
    assert best_candidate(.5, .5) and not best_candidate(.4, .5)
    for value in (None, float("nan"), -1, 2):
        with pytest.raises(RuntimeError):
            best_candidate(value)
    with pytest.raises(ValueError):
        logical_groups([1, 1])


def test_effective_native_loss_settings_are_verified():
    values = config()
    values["native_alpha"] = .7
    with pytest.raises(RuntimeError, match="effective native"):
        _build(values, {"omic_sizes": [3, 5], "encoding_dim": 8}, "B0_native")


@pytest.mark.parametrize("preset", PRESETS)
def test_all_nine_presets_native_update_and_no_hook_leak(preset):
    batch = cases()
    reference = ContentReference([3, 5], 8).fit(batch)
    targets = [reference.targets(c) for c in batch]
    host = _build(config(), {"omic_sizes": [3, 5], "encoding_dim": 8}, preset)
    before = _cpu_state(host.backbone)
    optimizer = torch.optim.Adam(host.parameters(), lr=.0005)
    result = logical_update(host, batch, targets, list(range(4)),
                            [schedule_seed("train", 0, i) for i in range(4)], optimizer)
    assert result["optimizer_steps"] == 1 and result["root_forwards"] == 4
    assert np.isfinite(list(result.values())).all()
    assert any(not torch.equal(value, before[name]) for name, value in host.backbone.state_dict().items())
    assert not host.backbone.slot_attention_wsi._forward_hooks
    assert not host.backbone.slot_attention_omic._forward_hooks
    if host.parts:
        for module in host.parts.values():
            assert module.private_encoder.weight.grad is not None
            assert module.private_encoder.weight.grad.abs().sum() > 0
    if host.relations:
        assert all(m.g[-1].weight.grad.abs().sum() > 0 for m in host.relations.values())


def test_off_host_matches_native_loss_gradients_and_rng_exactly():
    host = _build(config(), {"omic_sizes": [3, 5], "encoding_dim": 8}, "B0_native")
    case = cases(1)[0]
    state = torch.get_rng_state().clone()
    a, aux_a = host.backbone(**case)
    rng_a = torch.get_rng_state().clone()
    loss_a = host.backbone.loss_fn(a, y=case["y"], c=case["c"], t=None) + aux_a
    loss_a.backward()
    grads_a = {n: None if p.grad is None else p.grad.clone() for n, p in host.backbone.named_parameters()}
    host.zero_grad(set_to_none=True)
    torch.set_rng_state(state)
    b, aux_b = host(**case)
    assert torch.equal(torch.get_rng_state(), rng_a)
    loss_b = host.backbone.loss_fn(b, y=case["y"], c=case["c"], t=None) + aux_b
    loss_b.backward()
    assert torch.equal(a, b) and torch.equal(loss_a, loss_b)
    for name, parameter in host.backbone.named_parameters():
        assert (grads_a[name] is None and parameter.grad is None) or torch.equal(grads_a[name], parameter.grad)
    assert host.added_losses([host.last_records])["added"].item() == 0


def test_v3_zero_initialized_residual_matches_v2_native_forward_and_plugin_base():
    manifest = {"omic_sizes": [3, 5], "encoding_dim": 8}
    baseline = _build(config(), manifest, "V2_separate")
    initial = _cpu_state(baseline.backbone)
    state = torch.get_rng_state().clone()
    a = baseline(**cases(1)[0])[0]
    for preset in ("V3_capacity", "V3_relation"):
        host = _build(config(), manifest, preset, initial)
        torch.set_rng_state(state)
        b = host(**cases(1)[0])[0]
        assert torch.equal(a, b)
        for key, value in baseline.parts.state_dict().items():
            assert torch.equal(value, host.parts.state_dict()[key])


def prepared_fixture(root):
    root.mkdir()
    batch = cases(6)
    patients = {}
    for i, case in enumerate(batch):
        pid = f"invented_{i}"
        # A short training case and a longer validation bag verify both policies.
        patches = case["x_wsi"][0][:3] if i == 0 else torch.cat((case["x_wsi"][0], case["x_wsi"][0][:2]))
        path = root / (pid + ".pt")
        torch.save({"patient_id": pid, "patches": patches,
                    "omics": [case["x_omic1"][0], case["x_omic2"][0]]}, path)
        patients[pid] = {"cache_path": path.name, "time": float(i+1), "censor": float(i%2)}
    manifest = {"generated_test_only": True, "patients": patients, "patches": 11,
                "omic_sizes": [3, 5], "encoding_dim": 8,
                "folds": {"0": {"train": list(patients)[:4], "val": list(patients)[4:],
                          "labels": {pid: i%4 for i,pid in enumerate(patients)}}}}
    return manifest, batch


def test_real_length_validation_zero_padding_training_and_content_target_exclusion(tmp_path):
    root = tmp_path / "invented"
    manifest, batch = prepared_fixture(root)
    reference = ContentReference([3, 5], 8).fit(batch[:4])
    training, targets = _case(root, manifest, "invented_0", "cpu", training=True, seed=3, reference=reference)
    assert training["x_wsi"].shape == (1, 11, 8)
    assert torch.equal(training["x_wsi"][:, 3:], torch.zeros(1, 8, 8))
    assert targets["wsi"].shape == (3, 32)  # padding never becomes a target
    validation, _ = _case(root, manifest, "invented_4", "cpu", training=False)
    assert validation["x_wsi"].shape == (1, 13, 8)  # greater than train budget


def test_complete_generated_arm_best_last_raw_outputs_and_no_overwrite(tmp_path):
    root = tmp_path / "invented"
    manifest, batch = prepared_fixture(root)
    reference = ContentReference([3, 5], 8).fit(batch[:4])
    initial = _cpu_state(_build(config(), manifest, "B0_native").backbone)
    summary = _run_arm(config(), root, manifest, "V2_separate", initial, reference)
    out = root / "arms/V2_separate"
    assert summary["status"] == "completed" and summary["optimizer_steps"] == 2
    assert summary["best_epoch"] in (1, 2)
    for name in ("best.pt", "last.pt", "curve.json", "eval_repeats.json", "raw_repeat_00.pt", "summary.json"):
        assert (out / name).is_file()
    checkpoint = torch.load(out / "best.pt", weights_only=True)
    assert checkpoint["epoch"] == summary["best_epoch"]
    assert checkpoint["protocol"]["validation"].startswith("B1_all_real")
    with pytest.raises(RuntimeError, match="already exists"):
        _run_arm(config(), root, manifest, "V2_separate", initial, reference)


def test_initialization_failure_is_retained_without_overwrite(tmp_path, monkeypatch):
    from slotspe_proof import plugin_engine as engine
    root = tmp_path / "invented"
    root.mkdir()
    def fail(*args, **kwargs):
        raise RuntimeError("injected initialization failure")
    monkeypatch.setattr(engine, "_build", fail)
    with pytest.raises(RuntimeError, match="injected"):
        engine._run_arm(config(), root, {}, "V1_vib", {}, None)
    report = patient._read(root / "arms/V1_vib/failure.json")
    assert report["phase"] == "initialization" and "injected" in report["message"]
    with pytest.raises(RuntimeError, match="already exists"):
        engine._run_arm(config(), root, {}, "V1_vib", {}, None)


def test_logical_batch_rejects_two_patients_disguised_as_one_record():
    host = _build(config(), {"omic_sizes": [3, 5], "encoding_dim": 8}, "B0_native")
    case = cases(1)[0]
    case["x_wsi"] = case["x_wsi"].expand(2, -1, -1)
    optimizer = torch.optim.Adam(host.parameters())
    with pytest.raises(ValueError, match="one native forward"):
        logical_update(host, [case], None, ["invented"], [3], optimizer)
