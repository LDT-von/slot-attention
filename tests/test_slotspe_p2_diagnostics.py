"""Invented inputs only: probe integrity, no patient data or training."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from survot_rank.research.slotspe_proof.p2_diagnostics import (
    ARMS, pair_choices, protocol, risks, summarize, transform)

ROOT = Path(__file__).resolve().parents[1]


def invented():
    rng = np.random.default_rng(22)
    result = {}
    for m in ("wsi", "omic"):
        result[m+"_slots"] = rng.normal(size=(8, 32)).astype(np.float32)
        result[m+"_keep"] = np.array([1., 0., 0., 1., 0., 0., 0., 0.], dtype=np.float32)
        attention = rng.uniform(.1, 1., size=(2, 8, 11))
        attention /= attention.sum(-1, keepdims=True)
        attention[:, 4] = attention[:, 2]
        result[m+"_last_attention_by_head"] = attention
    return result


def test_plan_does_not_load_tensors_or_torch(tmp_path):
    code = "import sys; from survot_rank.research.slotspe_proof.p2_diagnostics import plan; assert not plan(sys.argv[1])['ready']; assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], cwd=ROOT, check=True, capture_output=True)


def test_pair_selection_has_no_outcome_or_vector_dependence():
    arrays = invented()
    assert pair_choices(arrays, "wsi")["closest_pair"] == [2, 4]
    arrays["y"] = 99
    arrays["wsi_slots"] *= -111
    assert pair_choices(arrays, "wsi")["closest_pair"] == [2, 4]
    assert pair_choices(arrays, "wsi")["selected_pair"] == [0, 3]


def test_all_probes_preserve_count_mean_and_masks_without_mutating_input():
    arrays = invented()
    saved = {k: v.copy() for k, v in arrays.items()}
    for arm in ARMS:
        altered = transform(arrays, arm)
        for m in ("wsi", "omic"):
            assert altered[m+"_slots"].shape == arrays[m+"_slots"].shape
            np.testing.assert_allclose(altered[m+"_slots"].astype(float).mean(0), arrays[m+"_slots"].astype(float).mean(0), atol=1e-6, rtol=0)
            expected_keep = arrays[m+"_keep"][::-1] if arm in ("permute_"+m, "permute_both") else arrays[m+"_keep"]
            np.testing.assert_array_equal(altered[m+"_keep"], expected_keep)
    for k, v in saved.items():
        np.testing.assert_array_equal(arrays[k], v)
    merged = transform(arrays, "wsi_selected_pair")
    np.testing.assert_array_equal(merged["wsi_slots"][0], merged["wsi_slots"][3])
    np.testing.assert_array_equal(merged["omic_slots"], arrays["omic_slots"])


def test_risk_and_margin_count_are_recomputed_from_logits():
    settings = protocol()
    baseline = np.zeros((3, 4))
    baseline[0] = -1
    baseline[2] = 1
    altered = baseline[::-1].copy()
    result = summarize(baseline, altered, settings)
    assert result["risk_order_reversals"] == 3
    assert result["all_patient_pairs"] == 3
    assert result["patients_logit_difference_above_numerical_threshold"] == 2
    np.testing.assert_allclose(risks(np.zeros((1, 4))), [-.9375])
    altered[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        summarize(baseline, altered, settings)


def test_readout_matches_native_and_permutation_preserves_rng_and_weights():
    import torch
    from survot_rank.research.slotspe_proof.model import build_model, make_args
    from survot_rank.research.slotspe_proof.native_audit import _state_digest
    from survot_rank.research.slotspe_proof.p1_diagnostics import NativeSlotObserver
    from survot_rank.research.slotspe_proof.p2_diagnostics import native_readout
    torch.set_num_threads(2)
    args = make_args([3, 4], encoding_dim=16)
    args.wsi_projection_dim, args.slot_iters = 32, 3
    model = build_model(ROOT / "third_party/SlotSPE", args, "original").eval()
    observer = NativeSlotObserver(model)
    sample = {"x_wsi": torch.randn(2, 11, 16), "x_omic1": torch.randn(2, 3), "x_omic2": torch.randn(2, 4), "omic_missing": False}
    before = _state_digest(model)
    try:
        with torch.no_grad():
            expected = model(**sample)[0]
            arrays = [observer.arrays(i) for i in range(2)]
            rng = torch.random.get_rng_state().clone()
            for arm in ("identity", "permute_wsi", "permute_omic", "permute_both"):
                altered = [transform(a, arm) for a in arrays]
                tensors = [torch.from_numpy(np.stack([a[k] for a in altered])) for k in ("wsi_slots", "omic_slots", "wsi_keep", "omic_keep")]
                result = native_readout(model, *tensors)
                torch.testing.assert_close(result, expected, atol=1e-6, rtol=0)
                assert torch.equal(rng, torch.random.get_rng_state())
        assert _state_digest(model) == before
    finally:
        observer.close()


@pytest.mark.parametrize("tamper", [None, "seed", "rng_empty", "rng_invalid", "input_changed"])
def test_metadata_binds_p1_identity_and_requires_real_rng_digests(tamper):
    from copy import deepcopy
    from survot_rank.research.slotspe_proof.p2_diagnostics import validate_metadata
    p1 = {"fold": 0, "seed": 3, "epochs": 3, "patient_count": 71, "device": "cuda:0"}
    report = {"fold": 0, "seed": 3, "previous_epochs": 3, "patients": 71, "device": "cuda:0",
        "original_inputs_unchanged": True, "rng_before": {"cpu": "a"*64, "cuda": ["b"*64]}}
    report["rng_after"] = deepcopy(report["rng_before"])
    if tamper == "seed":
        report["seed"] = 999
    elif tamper == "rng_empty":
        report["rng_before"] = report["rng_after"] = {}
    elif tamper == "rng_invalid":
        report["rng_before"]["cpu"] = report["rng_after"]["cpu"] = "invalid"
    elif tamper == "input_changed":
        report["original_inputs_unchanged"] = False
    if tamper:
        with pytest.raises(ValueError):
            validate_metadata(report, p1)
    else:
        validate_metadata(report, p1)
