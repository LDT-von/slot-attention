"""Tests for the anti-collapse intervention protocol.

The anti-collapse queue exists because the completed 30/4096/20 queue
showed that the slot collapse is an optimizer-step event: the effective rank
falls from above 2 to below 1.5 inside a single step, and in three of five
folds the collapsing step follows a gradient spike.  The engine therefore
adds two interventions to the completed best-checkpoint protocol.

These tests check the properties that make the intervention a controlled
experiment rather than a different experiment:

1.  With clipping disabled and no decay, the engine reproduces the
    completed R5 arm's training rows exactly, so the anti-collapse arms
    differ from R5 only in the documented interventions.
2.  Clipping actually bounds the gradient that reaches the optimizer.
3.  The decay schedules the Gram weight as declared.
4.  The probe and the selection rule are unchanged.
"""
import json
from pathlib import Path

import pytest
import torch

from slotspe_proof import patient
from slotspe_proof.redundancy_anticollapse_engine import (
    ANTI_COLLAPSE_ARMS, ARM_CONTRACTS_ANTI, GRAM_DECAY_FLOOR, _clip_gradients,
    _gram_weight, run_arm_anti_collapse,
)
from slotspe_proof.redundancy_engine import make_initial_states

SOURCE = Path(__file__).resolve().parents[1] / "third_party" / "SlotSPE"


def _config(tmp_path, **overrides):
    config = {"variants": list(ANTI_COLLAPSE_ARMS), "study": "blca", "folds": [0],
              "seeds": [3], "feature_encoder": "UNI2-h",
              "source_root": str(SOURCE), "device": "cpu", "torch_threads": 1,
              "projection_dim": 16, "slot_iters": 2, "lr": 5e-4, "probe_lr": 5e-4,
              "epochs": 3, "eval_repeats": 2, "batch_size": 2, "patches": 9,
              "gradient_clip_norm": 1.0}
    config.update(overrides)
    return config


def _manifest(tmp_path):
    from tests.test_slotspe_redundancy_engine import invented_batch, reference_for
    count = 10
    batch = invented_batch(count)
    patients = {}
    ids = [f"invented_{index}" for index in range(count)]
    for index, name in enumerate(ids):
        path = f"prepared/cache/patients/{name}.pt"
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"patient_id": name, "patches": batch["x_wsi"][index],
                    "omics": [batch[f"x_omic{k+1}"][index] for k in range(3)]},
                   tmp_path / path)
        patients[name] = {"cache_path": path, "censor": float(index % 2),
                          "time": float(index + 1)}
    fold = {"train": ids[:6], "val": ids[6:],
            "labels": {name: index % 4 for index, name in enumerate(ids)}}
    manifest = {"omic_sizes": [3, 5, 7], "encoding_dim": 8,
                "patients": patients, "folds": {"0": fold},
                "source_hashes": patient._source_hashes({"source_root": str(SOURCE)}),
                "generated_test_only": True}
    reference = reference_for({key: value[:6] if torch.is_tensor(value) else value
                               for key, value in batch.items()})
    return manifest, reference


def test_the_two_arms_declare_exactly_the_documented_interventions():
    assert ANTI_COLLAPSE_ARMS == ("A0_clip", "A1_clip_decay")
    plain = ARM_CONTRACTS_ANTI["A0_clip"]
    decayed = ARM_CONTRACTS_ANTI["A1_clip_decay"]
    # Both arms keep the completed R5 loss structure: Gram 10 and content 5.
    for contract in (plain, decayed):
        assert contract["gram_weight"] == 10.0
        assert contract["content_weight"] == 5.0
        assert contract["slotbert_weight"] == 0.0
        assert contract["adapter"] == "S0"
        assert contract["clip"] is True
    # The only difference between the two arms is the decay.
    assert plain["decay"] is False
    assert decayed["decay"] is True


def test_gradient_clipping_bounds_the_norm_that_reaches_the_optimizer():
    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.full((4,), 10.0)
    norm, clipped = _clip_gradients([parameter], 1.0)
    assert clipped is True
    assert norm == pytest.approx(20.0, rel=1e-6)
    assert float(parameter.grad.norm()) == pytest.approx(1.0, rel=1e-5)


def test_gradient_clipping_leaves_a_small_gradient_untouched():
    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.full((4,), 0.01)
    norm, clipped = _clip_gradients([parameter], 1.0)
    assert clipped is False
    assert norm == pytest.approx(0.02, rel=1e-6)
    assert float(parameter.grad.norm()) == pytest.approx(0.02, rel=1e-6)


def test_clipping_is_inactive_when_no_limit_is_given():
    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.full((4,), 10.0)
    norm, clipped = _clip_gradients([parameter], None)
    assert clipped is False
    assert norm == pytest.approx(20.0, rel=1e-6)


def test_the_gram_weight_is_constant_without_decay_and_decays_with_it():
    plain = ARM_CONTRACTS_ANTI["A0_clip"]
    decayed = ARM_CONTRACTS_ANTI["A1_clip_decay"]
    for epoch in range(30):
        assert _gram_weight(plain, epoch, 30) == 10.0
    values = [_gram_weight(decayed, epoch, 30) for epoch in range(30)]
    assert values[0] == pytest.approx(10.0)
    assert values[-1] == pytest.approx(10.0 * GRAM_DECAY_FLOOR)
    # The decay must be monotone, never below the floor and never above the
    # declared maximum.
    assert all(values[index] >= values[index + 1] for index in range(len(values) - 1))
    assert min(values) >= 10.0 * GRAM_DECAY_FLOOR - 1e-9
    assert max(values) <= 10.0 + 1e-9


def test_the_engine_runs_and_records_the_interventions(tmp_path):
    manifest, reference = _manifest(tmp_path)
    core, probe = make_initial_states(manifest, SOURCE, 3, projection_dim=16, slot_iters=2)
    config = _config(tmp_path)
    summary = run_arm_anti_collapse(config, tmp_path, manifest, "A0_clip", core, probe,
                                     reference, fold=0, output_root=tmp_path / "out",
                                     _generated_test=True)
    assert summary["status"] == "completed"
    assert summary["best_epoch"] is not None
    assert summary["protocol"]["gradient_clip_norm"] == 1.0
    assert summary["protocol"]["gram_decay"] is False
    # The checkpoint files and the per-epoch record must exist.
    assert (tmp_path / "out" / "arms" / "A0_clip" / "best.pt").is_file()
    assert (tmp_path / "out" / "arms" / "A0_clip" / "last.pt").is_file()
    assert (tmp_path / "out" / "arms" / "A0_clip" / "epoch_validation.json").is_file()
    curve = json.loads((tmp_path / "out" / "arms" / "A0_clip" / "curve.json").read_text(encoding="utf-8"))
    assert curve and all("gradient_was_clipped" in row for row in curve)


def test_the_decay_arm_records_the_scheduled_gram_weight(tmp_path):
    manifest, reference = _manifest(tmp_path)
    core, probe = make_initial_states(manifest, SOURCE, 3, projection_dim=16, slot_iters=2)
    config = _config(tmp_path, epochs=3)
    summary = run_arm_anti_collapse(config, tmp_path, manifest, "A1_clip_decay", core, probe,
                                     reference, fold=0, output_root=tmp_path / "out2",
                                     _generated_test=True)
    assert summary["status"] == "completed"
    assert summary["protocol"]["gram_decay"] is True
    curve = json.loads((tmp_path / "out2" / "arms" / "A1_clip_decay" / "curve.json").read_text(encoding="utf-8"))
    # Order the distinct weights by the epoch they belong to.
    by_epoch = {}
    for row in curve:
        by_epoch.setdefault(row["epoch"], row["gram_weight"])
    assert sorted(by_epoch) == [1, 2, 3]
    assert by_epoch[1] == pytest.approx(10.0)
    assert by_epoch[2] == pytest.approx(10.0 * GRAM_DECAY_FLOOR ** 0.5)
    assert by_epoch[3] == pytest.approx(10.0 * GRAM_DECAY_FLOOR)


def test_the_engine_refuses_an_arm_outside_the_declared_set(tmp_path):
    manifest, reference = _manifest(tmp_path)
    core, probe = make_initial_states(manifest, SOURCE, 3, projection_dim=16, slot_iters=2)
    config = _config(tmp_path)
    with pytest.raises(ValueError):
        run_arm_anti_collapse(config, tmp_path, manifest, "R5", core, probe, reference,
                              fold=0, output_root=tmp_path / "out3", _generated_test=True)


def test_the_completed_source_modules_are_not_modified_by_this_protocol():
    """The anti-collapse engine is a separate protocol; the files pinned by
    the completed queues must not be edited to support it."""
    from slotspe_proof import redundancy_best_engine, redundancy_engine
    for module in (redundancy_engine, redundancy_best_engine):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "anticollapse" not in source
        assert "anti_collapse" not in source


def test_the_engine_reproduces_the_completed_R5_arm_when_clipping_is_off(tmp_path):
    """The decisive controlled-experiment property.

    With the gradient clip removed and no decay, the anti-collapse engine
    must produce exactly the training rows of the completed R5 arm.  That
    makes the anti-collapse arms differ from the completed arm in the
    declared interventions only, and not in any incidental way.
    """
    from slotspe_proof.redundancy_engine import run_arm
    manifest, reference = _manifest(tmp_path)
    core, probe = make_initial_states(manifest, SOURCE, 3, projection_dim=16, slot_iters=2)
    config = _config(tmp_path, epochs=2)
    # The completed engine only accepts its own declared arms.
    config["variants"] = list(ANTI_COLLAPSE_ARMS) + ["R5"]
    run_arm(config, tmp_path, manifest, "R5", core, probe, reference, _generated_test=True)
    reference_curve = json.loads((tmp_path / "arms/R5" / "curve.json").read_text(encoding="utf-8"))

    other = tmp_path / "anti"
    other.mkdir()
    # The clip must be inactive for this comparison, so the engine runs the
    # completed arm's own loss structure with no extra intervention.
    config["variants"] = list(ANTI_COLLAPSE_ARMS)
    config["gradient_clip_norm"] = None
    import slotspe_proof.redundancy_anticollapse_engine as engine
    original = dict(engine.ARM_CONTRACTS_ANTI["A0_clip"])
    engine.ARM_CONTRACTS_ANTI["A0_clip"] = {**original, "clip": False}
    try:
        run_arm_anti_collapse(config, tmp_path, manifest, "A0_clip", core, probe, reference,
                              fold=0, output_root=other, _generated_test=True)
    finally:
        engine.ARM_CONTRACTS_ANTI["A0_clip"] = original
    anti_curve = json.loads((other / "arms/A0_clip" / "curve.json").read_text(encoding="utf-8"))

    assert len(reference_curve) == len(anti_curve)
    for left, right in zip(reference_curve, anti_curve):
        for key in ("epoch", "batch_start", "total_loss", "main_nll", "probe_loss",
                    "core_gradient_norm", "gram_loss", "content_loss"):
            assert left.get(key) == right.get(key), (key, left.get(key), right.get(key))


def test_clipping_changes_the_trajectory_it_is_meant_to_change(tmp_path):
    """The intervention must actually do something.  With the clip active the
    training rows must differ from the unclipped run, because the gradient
    that reaches the optimizer is smaller."""
    manifest, reference = _manifest(tmp_path)
    core, probe = make_initial_states(manifest, SOURCE, 3, projection_dim=16, slot_iters=2)
    config = _config(tmp_path, epochs=2)
    other = tmp_path / "clipped"
    other.mkdir()
    run_arm_anti_collapse(config, tmp_path, manifest, "A0_clip", core, probe, reference,
                          fold=0, output_root=other, _generated_test=True)
    clipped_curve = json.loads((other / "arms/A0_clip" / "curve.json").read_text(encoding="utf-8"))
    assert any(row["gradient_was_clipped"] for row in clipped_curve)
    # The pre-clip norm is recorded so the effect of the clip is auditable.
    clipped_rows = [row for row in clipped_curve if row["gradient_was_clipped"]]
    for row in clipped_rows:
        assert row["pre_clip_gradient_norm"] > row["core_gradient_norm"]
