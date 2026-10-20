"""P3 measurement tests: invented inputs/metadata only."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_plan_does_not_load_torch_or_patient_arrays(tmp_path):
    code = "import sys; from survot_rank.research.slotspe_proof.p3_diagnostics import plan; assert not plan(sys.argv[1])['ready']; assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], cwd=ROOT, check=True, capture_output=True)


def make_native():
    import torch
    from survot_rank.research.slotspe_proof.model import build_model, make_args
    torch.set_num_threads(2)
    args = make_args([3, 4], encoding_dim=16)
    args.wsi_projection_dim, args.slot_iters = 32, 3
    return build_model(ROOT / "third_party/SlotSPE", args, "original").eval()


def test_same_initial_draws_have_no_input_conditioning_but_final_slots_do():
    import torch
    from survot_rank.research.slotspe_proof.p3_diagnostics import NativeCapture
    model = make_native()
    capture = NativeCapture(model)
    first = torch.randn(2, 11, 32)
    try:
        torch.manual_seed(56)
        with torch.no_grad():
            model.slot_attention_wsi(first)
        initial = capture.values["wsi_initial"].copy()
        final = capture.values["wsi_final"].copy()
        capture.reset()
        torch.manual_seed(56)
        with torch.no_grad():
            model.slot_attention_wsi(first * 3 + torch.arange(32))
        np.testing.assert_array_equal(initial, capture.values["wsi_initial"])
        assert not np.array_equal(final, capture.values["wsi_final"])
    finally:
        capture.close()


def test_module_state_replay_exactly_matches_native_and_separates_draws():
    import torch
    from survot_rank.research.slotspe_proof.p3_diagnostics import NativeCapture, replay_modules
    from survot_rank.research.slotspe_proof.native_audit import _state_digest
    model = make_native()
    original_state = _state_digest(model)
    capture = NativeCapture(model)
    sample = {"x_wsi": torch.randn(2, 11, 16), "x_omic1": torch.randn(2, 3), "x_omic2": torch.randn(2, 4), "omic_missing": False}
    states, originals = [], []
    try:
        with torch.no_grad():
            for seed in (100, 200):
                torch.manual_seed(seed)
                capture.reset()
                logits = model(**sample)
                states.append([capture.states])
                originals.append((logits[0].clone(), capture.values.copy()))
                encoded = capture.inputs
            before_rng = torch.random.get_rng_state().clone()
            for r in range(2):
                for arm in ("native", "fix_slot_draws", "fix_gumbel_draws", "fix_both_draws"):
                    capture.reset()
                    logits = replay_modules(model, encoded, states, r, 0, arm, "cpu")
                    expected_slots = originals[0 if arm in ("fix_slot_draws", "fix_both_draws") else r][1]
                    for m in ("wsi", "omic"):
                        for suffix in ("_initial", "_final"):
                            np.testing.assert_array_equal(capture.values[m+suffix], expected_slots[m+suffix])
                    if arm == "native":
                        torch.testing.assert_close(logits, originals[r][0], atol=0, rtol=0)
                    if arm == "fix_both_draws":
                        torch.testing.assert_close(logits, originals[0][0], atol=0, rtol=0)
                    assert torch.equal(before_rng, torch.random.get_rng_state())
        assert _state_digest(model) == original_state
    finally:
        capture.close()
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def test_forked_rng_restored_after_exception():
    import torch
    from survot_rank.research.slotspe_proof.p3_diagnostics import replay_rng, rng_state
    state = rng_state()
    with pytest.raises(RuntimeError):
        with replay_rng(state):
            torch.rand(100)
            raise RuntimeError("invented failure")
    assert torch.equal(torch.random.get_rng_state(), state["cpu"])


def test_fixed_gumbel_draw_does_not_mean_fixed_selected_mask():
    import torch
    from survot_rank.research.slotspe_proof.p3_diagnostics import replay_rng, rng_state
    decoder = make_native().slot_decoder_wsi
    with torch.no_grad():
        for layer in (decoder.map[0], decoder.map[2], decoder.pred_keep_slot[0]):
            layer.weight.copy_(torch.eye(32))
            layer.bias.zero_()
        decoder.pred_keep_slot[2].weight.zero_()
        decoder.pred_keep_slot[2].weight[0, 0] = 1
        decoder.pred_keep_slot[2].bias.zero_()
        first, second = torch.zeros(1, 8, 32), torch.zeros(1, 8, 32)
        first[0, 0, 0], second[0, 7, 0] = 10000, 10000
        state = rng_state()
        with replay_rng(state):
            keep1 = decoder(first)[2]
        with replay_rng(state):
            keep2 = decoder(second)[2]
        assert keep1[0, 0] > .5 and keep2[0, 7] > .5
        assert not torch.equal(keep1, keep2)


def test_repeat_summary_validates_raw_slots_and_reports_variation():
    from survot_rank.research.slotspe_proof.p3_diagnostics import array_summary
    arrays = {"logits": np.zeros((20, 3, 4)), "risks": np.full((20, 3), -.9375)}
    for m in ("wsi", "omic"):
        arrays[m+"_initial"] = np.zeros((20, 3, 8, 256))
        arrays[m+"_final"] = np.zeros((20, 3, 8, 256))
        arrays[m+"_keep"] = np.zeros((20, 3, 8))
        arrays[m+"_keep"][:, :, :2] = 1
    provenance = [{"time": float(i+1), "censor": 0} for i in range(3)]
    result = array_summary(arrays, provenance)
    assert result["C_index_validation_exploratory"]["min"] == .5
    assert result["patients_risk_range_above_numerical_threshold"] == 0
    arrays["wsi_keep"][0, 0, 0] = 0
    with pytest.raises(ValueError, match="hard keep"):
        array_summary(arrays, provenance)


def test_exact_risk_audit_preserves_each_historical_pass_kernel_shape():
    import torch
    from survot_rank.research.slotspe_proof.p3_diagnostics import recompute_historical_risks
    # Regresses an audit failure: all 20 repeats in one vectorized sigmoid call
    # can differ from individual 71-patient eval calls by float32 rounding.
    logits = np.random.default_rng(781).normal(size=(20, 71, 4)).astype(np.float32)
    expected = []
    for repeat in logits:
        t = torch.from_numpy(repeat)
        expected.append((-(1-t.sigmoid()).cumprod(-1).sum(-1)).numpy())
    np.testing.assert_array_equal(recompute_historical_risks(logits), np.stack(expected))
