"""Synthetic checks of the DCT prototype bridge, without a DCT model/job."""
import pytest
import torch
from torch.nn import functional as F
from pathlib import Path
import importlib.util
import json

from survot_rank.research.slotspe_proof.prototype_assignment import prototype_coordinates, replay_prototype_pipeline, replay_assignment_variant
from survot_rank.research.slotspe_proof.assignment_router import AssignmentConfig


def fixture():
    g = torch.Generator().manual_seed(41)
    x = torch.randn(2, 5, 7, dtype=torch.float64, generator=g)
    p = torch.randn(3, 7, dtype=torch.float64, generator=g)
    return x, p


def test_soft_pool_matches_dct_expression_exactly():
    x, p = fixture()
    scores = torch.einsum("bnd,kd->bkn", F.normalize(x, dim=-1), F.normalize(p, dim=-1))
    assignments = torch.softmax(scores / .3, dim=1)
    weights = assignments / assignments.sum(-1, keepdim=True).clamp_min(1e-8)
    expected = torch.einsum("bkn,bnd->bkd", weights, x)
    result, actual_weights = prototype_coordinates(x, p)
    assert torch.equal(result, expected) and torch.equal(weights, actual_weights)


@pytest.mark.parametrize("mode", ["dct_soft", "sinkhorn_uniform", "mesh_uniform"])
def test_coordinate_gradients_shapes_and_weight_scale(mode):
    x, p = fixture()
    x.requires_grad_(); p.requires_grad_()
    noise = torch.arange(30, dtype=x.dtype).sin().reshape(2, 1, 3, 5)
    z, a = prototype_coordinates(x, p, mode=mode, noise=noise)
    assert z.shape == (2, 3, 7) and a.shape == (2, 3, 5)
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)))
    z.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_distinct_prototypes_cannot_recover_missing_local_content():
    _, p = fixture()
    x = torch.ones(2, 5, 7, dtype=p.dtype)
    for mode in ("dct_soft", "sinkhorn_uniform", "mesh_uniform"):
        noise = torch.arange(30, dtype=x.dtype).sin().reshape(2, 1, 3, 5)
        z, _ = prototype_coordinates(x, p, mode=mode, noise=noise)
        assert torch.allclose(z, torch.ones_like(z))


def test_identical_prototypes_can_remix_distinct_local_slots_into_duplicates():
    x, _ = fixture()
    p = torch.ones(3, 7, dtype=x.dtype)
    assert not torch.equal(x[:, 0], x[:, 1])
    z, a = prototype_coordinates(x, p)
    assert torch.equal(a[:, 0], a[:, 1])
    assert torch.equal(z[:, 0], z[:, 1])


def test_prototype_pooling_does_not_mutate_parameters_or_consume_rng():
    x, p = fixture()
    old_x, old_p, rng = x.clone(), p.clone(), torch.get_rng_state().clone()
    n = torch.arange(30, dtype=x.dtype).cos().reshape(2, 1, 3, 5)
    prototype_coordinates(x, p, mode="mesh_uniform", noise=n)
    assert torch.equal(x, old_x) and torch.equal(p, old_p)
    assert torch.equal(rng, torch.get_rng_state())


def test_rejects_incompatible_coordinate_configuration():
    x, p = fixture()
    with pytest.raises(ValueError):
        prototype_coordinates(x, p, temperature=0)
    with pytest.raises(ValueError):
        prototype_coordinates(x, p, mode="MetaSlot")
    with pytest.raises(ValueError):
        prototype_coordinates(x, p.float())


def test_two_stage_replay_sanitizes_padding_and_connects_both_stages():
    path = Path(__file__).resolve().parents[1] / "third_party/SlotSPE/models/slot_attention.py"
    spec = importlib.util.spec_from_file_location("_prototype_synthetic_sa", path)
    source = importlib.util.module_from_spec(spec); spec.loader.exec_module(source)
    with torch.random.fork_rng():
        torch.manual_seed(91)
        native = source.MultiHeadSlotAttention(3, 8, heads=2, dim_head=4, iters=2)
        x, initial = torch.randn(2, 6, 8), torch.randn(2, 3, 8)
        prototypes = torch.randn(3, 8, requires_grad=True)
        mask = torch.ones(2, 6, dtype=torch.bool); mask[0, 4:] = False
        x[0, 4:] = float("nan")
        a = replay_prototype_pipeline(native, x, initial, prototypes, token_mask=mask)
        b = replay_prototype_pipeline(native, x.nan_to_num(), initial, prototypes, token_mask=mask)
        assert torch.equal(a["coordinates"], b["coordinates"])
        noises = [torch.randn_like(p) for p in a["local"]["attentions"]]
        result = replay_prototype_pipeline(
            native, x, initial, prototypes, token_mask=mask,
            local_config=AssignmentConfig(mode="mesh_uniform"), local_noises=noises,
            coordinate_mode="mesh_uniform", coordinate_noise=torch.randn(2, 1, 3, 3),
        )
        result["coordinates"].square().sum().backward()
        assert torch.isfinite(prototypes.grad).all() and prototypes.grad.abs().sum() > 0
        assert torch.isfinite(native.to_q.weight.grad).all() and native.to_q.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("name", ["D0_native", "D1_sa_sinkhorn", "D2_sa_mesh", "D3_coordinate_mesh", "D4_both_mesh"])
def test_each_version_spec_runs_its_declared_stages(name):
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "configs/slotspe_redundancy_variants" / (name + ".json")).read_text(encoding="utf-8"))
    source_spec = importlib.util.spec_from_file_location("_variant_synthetic_sa", root / "third_party/SlotSPE/models/slot_attention.py")
    source = importlib.util.module_from_spec(source_spec); source_spec.loader.exec_module(source)
    with torch.random.fork_rng():
        torch.manual_seed(47)
        native = source.MultiHeadSlotAttention(3, 8, heads=2, dim_head=4, iters=2)
        x, initial, p = torch.randn(1, 5, 8), torch.randn(1, 3, 8), torch.randn(3, 8)
        local_noises = [torch.randn(1, 2, 3, 5) for _ in range(2)]
        result = replay_assignment_variant(
            spec, native, x, initial, p, local_noises=local_noises,
            coordinate_noise=torch.randn(1, 1, 3, 3),
        )
        assert result["coordinates"].shape == (1, 3, 8)
        assert torch.isfinite(result["coordinates"]).all()
        assert spec["patient_execution"] is False
