"""Tests for the best-checkpoint protocol of the eleven-arm SlotSPE diagnostic.

The essential property is that inserting a per-epoch validation pass between
two epochs does not change the training trajectory, so that the best protocol
and the completed final-epoch protocol differ only in which checkpoint they
report.  The second essential property is that the epoch-selection rule
matches the original published rule.
"""
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from slotspe_proof import patient
from slotspe_proof.redundancy_best_engine import (
    CHECKPOINT_SELECTION, CHECKPOINT_RULE_SOURCE, _StreamGuard,
    _validation_cindex, run_arm_best,
)
from slotspe_proof.redundancy_engine import ARMS, ARM_CONTRACTS, run_arm

SOURCE = Path(__file__).resolve().parents[1] / "third_party" / "SlotSPE"


def _base_config(tmp_path, **overrides):
    config = {"variants": list(ARMS), "study": "blca", "folds": [0], "seeds": [3],
              "feature_encoder": "UNI2-h",
              "source_root": str(SOURCE), "device": "cpu", "torch_threads": 1,
              "projection_dim": 16, "slot_iters": 2, "lr": 5e-4, "probe_lr": 5e-4,
              "epochs": 3, "eval_repeats": 2, "batch_size": 2, "patches": 9}
    config.update(overrides)
    return config


def test_checkpoint_rule_is_declared_and_names_the_original_source():
    assert CHECKPOINT_SELECTION == "per_epoch_validation_best_then_reload_and_reevaluate"
    assert CHECKPOINT_RULE_SOURCE == "third_party/SlotSPE/utils/core_utils.py:421-475"
    # The original source must still say what the rule reproduces.
    text = (SOURCE / "utils" / "core_utils.py").read_text(encoding="utf-8")
    assert "if val_cindex >= args.max_cindex:" in text
    assert 'torch.save(model.state_dict(), os.path.join(args.results_dir, "model_best_s{}.pth".format(cur)))' in text


def test_stream_guard_restores_every_global_random_stream():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    with _StreamGuard():
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        random.random()
        np.random.rand()
        torch.rand(3)
    assert random.random() == pytest.approx(random.Random(7).random())
    restored = np.random.RandomState(7)
    assert np.random.rand() == pytest.approx(restored.rand())
    expected = torch.Generator().manual_seed(7)
    assert torch.rand(3).tolist() == pytest.approx(torch.rand(3, generator=expected).tolist())


def test_validation_pass_does_not_disturb_the_training_draw_stream():
    """The real-patch draw consumes the numpy global stream; a validation
    pass between epochs must leave that stream exactly where it was."""
    np.random.seed(123)
    before = np.random.get_state()
    with _StreamGuard():
        np.random.seed(999)
        np.random.choice(1000, 10, replace=False)
    after = np.random.get_state()
    assert before[1].tolist() == after[1].tolist()
    assert before[2] == after[2]


def test_per_epoch_validation_uses_the_original_risk_definition():
    """Risk must be -sum(cumprod(1 - sigmoid(logits))), as in the original
    ``_calculate_risk`` and in the existing evaluation helper."""
    original = (SOURCE / "utils" / "core_utils.py").read_text(encoding="utf-8")
    assert "survival = torch.cumprod(1 - hazards, dim=1)" in original
    assert "risk = -torch.sum(survival, dim=1)" in original
    engine = Path(__file__).resolve().parents[1] / "slotspe_proof" / "redundancy_engine.py"
    assert "risks = -(1 - logits.sigmoid()).cumprod(-1).sum(-1)" in engine.read_text(encoding="utf-8")
    best = Path(__file__).resolve().parents[1] / "slotspe_proof" / "redundancy_best_engine.py"
    assert "risks = -(1 - combined.sigmoid()).cumprod(-1).sum(-1)" in best.read_text(encoding="utf-8")


def _best_manifest(root):
    """A generated manifest whose validation fold has comparable pairs.

    The shared engine test manifest leaves only two validation patients, which
    cannot produce a defined concordance index.  The real folds hold 75-77
    patients, so this manifest mirrors that shape with generated tensors.
    """
    from tests.test_slotspe_redundancy_engine import invented_batch, reference_for
    count = 10
    batch = invented_batch(count)
    patients = {}
    ids = [f"invented_{index}" for index in range(count)]
    for index, name in enumerate(ids):
        path = f"prepared/cache/patients/{name}.pt"
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"patient_id": name, "patches": batch["x_wsi"][index],
                    "omics": [batch[f"x_omic{k+1}"][index] for k in range(3)]}, root / path)
        # Alternating censoring with increasing times gives comparable pairs.
        patients[name] = {"cache_path": path, "censor": float(index % 2), "time": float(index + 1)}
    fold = {"train": ids[:6], "val": ids[6:], "labels": {name: index % 4 for index, name in enumerate(ids)}}
    manifest = {"omic_sizes": [3, 5, 7], "encoding_dim": 8,
                "patients": patients, "folds": {"0": fold},
                "source_hashes": patient._source_hashes({"source_root": str(SOURCE)}),
                "generated_test_only": True}
    reference = reference_for({key: value[:6] if torch.is_tensor(value) else value
                               for key, value in batch.items()})
    return manifest, reference


def test_per_epoch_validation_is_exactly_repeat_zero_of_the_real_evaluation(tmp_path):
    """The per-epoch validation pass must rank epochs on the same quantity
    that the final repeated evaluation reports, otherwise the selected epoch
    would not correspond to the reported C-index.

    The per-epoch pass uses repeat 0 of the evaluation seed schedule and the
    same fixed patch subset, so it must reproduce repeat 0 of
    ``_evaluate_repeat`` exactly on the same weights.
    """
    from slotspe_proof.redundancy_engine import (_construct, _evaluate_repeat,
                                                  make_initial_states)
    from slotspe_proof.native_variants import make_native_adapter
    from slotspe_proof.slot_regularization import SlotCapture
    from sksurv.metrics import concordance_index_censored
    manifest, reference = _best_manifest(tmp_path)
    core, probe = make_initial_states(manifest, SOURCE, 3, projection_dim=16, slot_iters=2)
    model, heads = _construct(manifest, SOURCE, 16, 2, "cpu")
    model.load_state_dict(core)
    heads.load_state_dict(probe)
    fold_data = manifest["folds"]["0"]
    val_ids = fold_data["val"]
    with make_native_adapter(model, "S0", capture=True) as adapter, SlotCapture(model) as capture:
        score, status = _validation_cindex(model, heads, adapter, tmp_path, manifest,
                                            fold_data, val_ids, "cpu", 3, 1,
                                            concordance_index_censored)
        evaluation, _rows, _raw = _evaluate_repeat(
            model, heads, adapter, capture, reference, tmp_path, manifest, fold_data,
            val_ids, "cpu", 3, 0, concordance_index_censored, baseline_slots=None)
    assert score == evaluation["cindex"]
    assert status == evaluation["cindex_status"]


def test_generated_best_run_retains_the_best_epoch_and_reloads_it(tmp_path):
    """End to end on generated data: the run completes, records one
    validation value per epoch, keeps a best epoch and a best checkpoint, and
    re-evaluates the retained best weights."""
    manifest, reference = _best_manifest(tmp_path)
    from slotspe_proof.redundancy_engine import make_initial_states
    core, probe = make_initial_states(manifest, SOURCE, 3, projection_dim=16, slot_iters=2)
    config = _base_config(tmp_path)
    result = run_arm_best(config, tmp_path, manifest, "R5", core, probe, reference, _generated_test=True)
    assert result["status"] == "completed"
    assert result["protocol"]["checkpoint_selection"] == CHECKPOINT_SELECTION
    out = tmp_path / "arms/R5"
    for name in ("last.pt", "best.pt", "curve.json", "summary.json", "eval_repeats.json",
                 "epoch_validation.json", "diagnostics.pt", "validation_predictions.csv"):
        assert (out / name).is_file(), name
    validation = json.loads((out / "epoch_validation.json").read_text(encoding="utf-8"))
    assert len(validation) == config["epochs"]
    assert [row["epoch"] for row in validation] == [1, 2, 3]
    defined = [row["cindex"] for row in validation if row["cindex"] is not None]
    if defined:
        best = max(defined)
        # The retained epoch must be one that attains the running maximum.
        assert result["best_epoch_cindex_single_pass"] == best
        assert result["best_epoch"] in [row["epoch"] for row in validation
                                       if row["cindex"] == best]
    assert result["best_checkpoint_sha256"]
    assert result["evaluation_summary"]["val_cindex_mean_exploratory"] is not None


def test_generated_best_run_refuses_to_overwrite(tmp_path):
    manifest, reference = _best_manifest(tmp_path)
    from slotspe_proof.redundancy_engine import make_initial_states
    core, probe = make_initial_states(manifest, SOURCE, 3, projection_dim=16, slot_iters=2)
    config = _base_config(tmp_path)
    run_arm_best(config, tmp_path, manifest, "S0", core, probe, reference, _generated_test=True)
    with pytest.raises(RuntimeError, match="overwrite"):
        run_arm_best(config, tmp_path, manifest, "S0", core, probe, reference, _generated_test=True)


def test_best_run_training_trajectory_matches_the_final_epoch_run(tmp_path):
    """The     decisive property: inserting the per-epoch validation pass must
    not change the training trajectory.  Both runs must produce identical
    per-batch training rows."""
    manifest, reference = _best_manifest(tmp_path)
    from slotspe_proof.redundancy_engine import make_initial_states
    core, probe = make_initial_states(manifest, SOURCE, 3, projection_dim=16, slot_iters=2)
    config = _base_config(tmp_path, epochs=2)
    final = run_arm(config, tmp_path, manifest, "S0", core, probe, reference, _generated_test=True)
    final_curve = json.loads((tmp_path / "arms/S0" / "curve.json").read_text(encoding="utf-8"))

    other = tmp_path / "best"
    other.mkdir()
    # The prepared cache lives under the manifest root, so the second run
    # must use the same root for the input path while keeping its own output
    # directory for the arm artifacts.
    best = run_arm_best(config, tmp_path, manifest, "S0", core, probe, reference,
                         output_root=other, _generated_test=True)
    best_curve = json.loads((other / "arms/S0" / "curve.json").read_text(encoding="utf-8"))

    assert len(final_curve) == len(best_curve)
    for left, right in zip(final_curve, best_curve):
        for key in ("epoch", "batch_start", "total_loss", "main_nll", "probe_loss",
                    "core_gradient_norm", "gram_loss", "slotbert_loss", "content_loss"):
            assert left.get(key) == right.get(key), (key, left.get(key), right.get(key))
    assert final["counters"]["core_optimizer_steps"] == best["counters"]["core_optimizer_steps"]
