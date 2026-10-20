"""All fixtures are invented tensors; no real patient runs are started."""
from pathlib import Path
import copy
import json

import pytest
import torch

from survot_rank.research.slotspe_proof import model as controls
from survot_rank.research.slotspe_proof.attribution import matched_bc_check, run_attribution_checks
from survot_rank.research.slotspe_proof.interventions import slot_intervention

SOURCE = Path(__file__).resolve().parents[1] / "third_party/SlotSPE"


@pytest.fixture
def fixture():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    args = controls.make_args([3, 4, 5], encoding_dim=16)
    args.wsi_projection_dim, args.slot_iters = 32, 3
    net = controls.build_model(SOURCE, args, "A")
    batch = {"x_wsi": torch.randn(4, 24, 16), "omic_missing": False,
        "y": torch.tensor([0, 1, 2, 3]), "c": torch.tensor([0., 1., 0., 1.])}
    batch.update({f"x_omic{i+1}": torch.randn(4, n) for i, n in enumerate(args.omic_sizes)})
    return args, net.state_dict(), batch


def test_bc_full_forward_main_and_all_auxiliary_gradients(fixture, tmp_path):
    report = run_attribution_checks(SOURCE, tmp_path, [3, 17, 29])
    assert len(report["records"]) == 6
    for row in report["records"]:
        assert row["status"] == "passed"
        assert row["logits_max_abs_difference"] == 0
        assert all(p["exact_equal"] for p in row["auxiliary_all_parameters"].values())
    # Auxiliary training is present in B, even when main->selector is None.
    assert any(p["B"]["nonzero_elements"] > 0 for row in report["records"]
               for name, p in row["auxiliary_all_parameters"].items() if "pred_keep_slot" in name)


def test_a_retains_mean_all_self_and_same_auxiliary_graph(fixture):
    args, state, batch = fixture
    values = {}
    for arm in ("A", "B", "C"):
        net = controls.build_model(SOURCE, args, arm)
        net.load_state_dict(state)
        net.train()
        torch.manual_seed(1003)
        logits, aux = net(**batch)
        named = [(n, p) for n, p in net.named_parameters() if "pred_keep_slot" in n]
        main = net.loss_fn(logits, y=batch["y"], c=batch["c"], t=None)
        main_grads = torch.autograd.grad(main, [p for _, p in named], retain_graph=True, allow_unused=True)
        aux_grads = torch.autograd.grad(aux, [p for _, p in named], allow_unused=True)
        values[arm] = logits, aux, main_grads, aux_grads
    assert all(g is None for g in values["A"][2])
    assert all(g is None for g in values["B"][2])
    assert all(g is not None for g in values["C"][2])
    assert not torch.equal(values["A"][0], values["B"][0])
    for arm in ("B", "C"):
        assert torch.equal(values["A"][1], values[arm][1])
        assert all((a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
                   for a, b in zip(values["A"][3], values[arm][3]))


@pytest.mark.parametrize("arm", ["A", "B", "C", "original"])
def test_parameter_schema_fixed_eval_and_patient_slot_probe(fixture, arm):
    args, state, batch = fixture
    net = controls.build_model(SOURCE, args, arm)
    net.load_state_dict(state, strict=True)
    assert list(net.state_dict()) == list(state)
    assert sum(p.numel() for p in net.parameters()) == sum(p.numel() for p in controls.build_model(SOURCE, args, "original").parameters())
    net.eval()
    if arm != "original":
        with torch.no_grad():
            first = net(**batch)[0]
            second = net(**batch)[0]
            assert torch.equal(first, second)
            # Batch membership does not change the fixed slot initializer.
            single = {key: value[:1] if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            torch.testing.assert_close(first[:1], net(**single)[0], rtol=1e-5, atol=1e-6)
    report = slot_intervention(net, batch, ["SYNTHETIC_0", "SYNTHETIC_1", "SYNTHETIC_2", "SYNTHETIC_3"], 13, "fixture_hash")
    assert len(report["records"]) == 4
    if arm in ("B", "C"):
        assert report["max_self_pool_change"] < 1e-7
        assert all(row["cross_frozen_risk_abs_change"] < 1e-7 for row in report["records"])
    else:
        assert report["max_self_pool_change"] > 1e-7


@pytest.mark.parametrize("magnitude", [0., 1e6, -1e6])
def test_tied_and_extreme_finite_decoder_uses_identical_bc_rule(fixture, magnitude):
    args, state, batch = fixture
    nets = [controls.build_model(SOURCE, args, arm) for arm in ("B", "C")]
    outputs = []
    for net in nets:
        net.load_state_dict(state)
        decoder = net.slot_decoder_wsi
        with torch.no_grad():
            decoder.pred_keep_slot[-1].weight.zero_()
            decoder.pred_keep_slot[-1].bias.fill_(magnitude)
        decoder.train()
        slots = torch.randn(4, 8, 32, generator=torch.Generator().manual_seed(5)).requires_grad_()
        torch.manual_seed(19)
        logits, gates, keep = decoder(slots)
        loss = logits.square().sum() + controls.selected_pool(slots, keep).square().sum()
        loss.backward()
        assert torch.isfinite(logits).all() and torch.isfinite(gates).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in decoder.parameters())
        assert torch.equal(keep.detach().sum(-1), torch.full((4,), 2.))
        torch.testing.assert_close(gates.sum(-1), torch.ones(4))
        outputs.append((logits, gates, keep))
    assert all(torch.equal(b, c) for b, c in zip(*outputs))


def test_bc_check_detects_global_detach_regression(fixture, monkeypatch):
    args, state, batch = fixture
    original_pool = controls.selected_pool
    # Removing the C gradient is caught even though forward stays identical.
    monkeypatch.setattr(controls, "selected_pool", lambda x, keep: original_pool(x, keep.detach()))
    report = matched_bc_check(SOURCE, args, state, batch, 1003)
    assert report["checks"]["forward_exact_equal"]
    assert not report["checks"]["C_main_all_selector_parameters_connected"]
    assert report["status"] == "failed"
