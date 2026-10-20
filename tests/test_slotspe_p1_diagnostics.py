"""Measurement validation with invented inputs; never opens real patient data."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_plan_does_not_import_torch_or_open_tensors(tmp_path):
    code = "import sys; from survot_rank.research.slotspe_proof.p1_diagnostics import plan; r=plan(sys.argv[1]); assert not r['ready_for_readonly_check']; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], cwd=ROOT, check=True, capture_output=True)


def test_observer_preserves_native_prediction_rng_and_state():
    torch = pytest.importorskip("torch")
    import numpy as np
    from survot_rank.research.slotspe_proof.model import build_model, make_args
    from survot_rank.research.slotspe_proof.native_audit import _state_digest, model_purity
    from survot_rank.research.slotspe_proof.p1_diagnostics import NativeSlotObserver, array_statistics
    torch.set_num_threads(2)
    args = make_args([3, 4], encoding_dim=16)
    args.wsi_projection_dim = 32
    args.slot_iters = 3
    model = build_model(ROOT / "third_party/SlotSPE", args, "original").eval()
    sample = {"x_wsi": torch.randn(2, 11, 16), "x_omic1": torch.randn(2, 3), "x_omic2": torch.randn(2, 4), "omic_missing": False}
    before = _state_digest(model)
    torch.manual_seed(321)
    with torch.no_grad():
        baseline = model(**sample)[0]
    after_native_rng = torch.random.get_rng_state().clone()
    observer = NativeSlotObserver(model)
    try:
        torch.manual_seed(321)
        with torch.no_grad():
            observed = model(**sample)[0]
        assert torch.equal(observed, baseline)
        assert torch.equal(after_native_rng, torch.random.get_rng_state())
        assert observer.maximum_update_error() == 0
        assert _state_digest(model) == before
        assert all(model_purity(model, ROOT / "third_party/SlotSPE")["checks"].values())
        for index in range(2):
            arrays = observer.arrays(index)
            for modality in ("wsi", "omic"):
                stats = array_statistics(arrays, modality)
                assert stats["candidate_slots"] == 8 and stats["selected_slots"] == 2
                assert stats["iterations"] == 3
                assert len(stats["pairs"]) == 28
                assert np.isclose(sum(stats["competition_mass"]), 1)
                assert 1 - 1e-6 <= stats["mass_entropy_effective_count"] <= 8 + 1e-6
    finally:
        observer.close()
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_raw_attention_normalizations_are_not_confused():
    import numpy as np
    from survot_rank.research.slotspe_proof.p1_diagnostics import array_statistics
    # Every per-slot attention row sums to one, while token competition assigns
    # all mass to slot zero. Row sums must not be misreported as four active slots.
    attention = np.full((1, 4, 5), .2)
    competition = np.zeros((2, 4, 5))
    competition[:, 0] = 1
    arrays = {"wsi_slots": np.eye(4), "wsi_attention_iterations": attention,
        "wsi_last_attention_by_head": np.repeat(attention, 2, axis=0),
        "wsi_last_competition_by_head": competition,
        "wsi_competition_mass_iterations": np.array([[1., 0., 0., 0.]]), "wsi_keep": np.array([1., 1., 0., 0.]),
        "wsi_attention_epsilon": np.array([1e-8]), "wsi_expected_topk": np.array([2])}
    result = array_statistics(arrays, "wsi")
    assert result["mass_entropy_effective_count"] == 1
    assert result["candidate_slots"] == 4 and result["selected_slots"] == 2
    arrays["wsi_last_competition_by_head"][0, 1] = 1
    with pytest.raises(ValueError, match="competition normalization"):
        array_statistics(arrays, "wsi")


@pytest.mark.parametrize("tamper", ["head_cancellation", "nonbinary_keep"])
def test_raw_validation_rejects_hidden_head_or_gate_tampering(tamper):
    import numpy as np
    from survot_rank.research.slotspe_proof.p1_diagnostics import array_statistics
    arrays = {"wsi_slots": np.eye(4), "wsi_attention_iterations": np.full((1, 4, 5), .2),
        "wsi_last_attention_by_head": np.full((2, 4, 5), .2), "wsi_last_competition_by_head": np.full((2, 4, 5), .25),
        "wsi_competition_mass_iterations": np.full((1, 4), .25), "wsi_keep": np.array([1., 1., 0., 0.]),
        "wsi_attention_epsilon": np.array([1e-8]), "wsi_expected_topk": np.array([2])}
    if tamper == "head_cancellation":
        # Head means and every row sum still agree; native epsilon-normalization
        # relation must catch the inconsistent per-head maps.
        arrays["wsi_last_attention_by_head"][0, 0, :2] += [.01, -.01]
        arrays["wsi_last_attention_by_head"][1, 0, :2] += [-.01, .01]
    else:
        arrays["wsi_keep"] = np.array([2., 1., 0., 0.])
    with pytest.raises(ValueError):
        array_statistics(arrays, "wsi")


@pytest.mark.parametrize("tamper", ["count_summary", "rng_digest", "nan_error"])
def test_offline_audit_rejects_rehashed_false_metadata(tmp_path, tamper):
    """Invented serialization fixture: a valid hash alone is not evidence."""
    import hashlib
    import numpy as np
    from survot_rank.research.slotspe_proof.p1_diagnostics import array_statistics, audit_export
    arrays = {}
    for modality in ("wsi", "omic"):
        arrays.update({modality + "_slots": np.eye(4), modality + "_attention_iterations": np.full((1, 4, 5), .2),
            modality + "_last_attention_by_head": np.full((2, 4, 5), .2), modality + "_last_competition_by_head": np.full((2, 4, 5), .25),
            modality + "_competition_mass_iterations": np.full((1, 4), .25), modality + "_keep": np.array([1., 1., 0., 0.]),
            modality + "_attention_epsilon": np.array([1e-8]), modality + "_expected_topk": np.array([2])})
    np.savez_compressed(tmp_path / "invented.npz", **arrays)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    rng = {"python": "p", "numpy": "n", "torch_cpu": "t", "torch_cuda": []}
    report = {"synthetic_serialization_fixture_not_patient_evidence": True,
        "schema": "slotspe_p1_observations_v2", "input_scope": "historical_local_real_BLCA_cached_validation",
        "server_SPE09_evidence": False, "new_optimizer_steps": 0, "checkpoint_tensors_unchanged": True,
        "original_inputs_unchanged": True, "patient_count": 1,
        "model_purity": {"checks": {k: True for k in ("native_model_class", "native_forward", "native_slot_classes", "native_decoder_classes", "no_repaired_eval_buffers")},
            "model_args": {"slot_num_wsi": 4, "slot_num_omics": 4, "topk_ratio": .5}},
        "historical_replay_max_logit_difference": 0., "observer_max_logit_difference": 0., "reconstructed_native_update_max_error": 0.,
        "replay": {"historical_reference_logits": [[0., 0., 0., 0.]], "native_logits": [[0., 0., 0., 0.]], "observed_logits": [[0., 0., 0., 0.]],
            "native_final_rng_sha256": rng, "observed_final_rng_sha256": dict(rng)},
        "candidate_counts": {"wsi": [4], "omic": [4]}, "selected_counts": {"wsi": [2], "omic": [2]},
        "records": [{"patient_id": "SYNTHETIC_CASE", "raw_path": "invented.npz", "raw_sha256": digest(tmp_path / "invented.npz"), "logits": [0., 0., 0., 0.],
            "wsi": array_statistics(arrays, "wsi"), "omic": array_statistics(arrays, "omic")}]}
    (tmp_path / "observations.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "patient_provenance.json").write_text(json.dumps({"patients": [{"patient_id": "SYNTHETIC_CASE"}]}), encoding="utf-8")
    receipt = {"outputs": {p.name: digest(p) for p in tmp_path.iterdir()}}
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert audit_export(tmp_path)["P1_predictive_harm_proved"] is False
    if tamper == "count_summary":
        report["candidate_counts"]["wsi"] = [2]
    elif tamper == "rng_digest":
        report["replay"]["observed_final_rng_sha256"]["torch_cpu"] = "changed"
    else:
        report["observer_max_logit_difference"] = float("nan")
    (tmp_path / "observations.json").write_text(json.dumps(report), encoding="utf-8")
    receipt["outputs"]["observations.json"] = digest(tmp_path / "observations.json")
    (tmp_path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        audit_export(tmp_path)
