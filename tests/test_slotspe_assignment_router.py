"""Synthetic allocation controls only. No patient inputs or checkpoints."""
from pathlib import Path
import importlib.util
import subprocess
import sys

import pytest
import torch

from survot_rank.research.slotspe_proof.assignment_router import (
    AssignmentConfig, SlotAssignmentRouter, replay_with_assignment,
)
from survot_rank.research.slotspe_proof.problem_trace import replay_slots


def router(mode, **kw):
    return SlotAssignmentRouter(AssignmentConfig(mode=mode, **kw))


def noise_like(x):
    return torch.randn(x.shape, generator=torch.Generator().manual_seed(137), dtype=x.dtype)


def test_original_matches_native_expression_exactly():
    x = noise_like(torch.empty(2, 3, 4, 9))
    expected = torch.nn.functional.normalize(x.softmax(-2) + 1e-8, p=1, dim=-1)
    assert torch.equal(router("original")(x), expected)


def test_cli_default_only_prints_plan_without_importing_torch():
    root = Path(__file__).resolve().parents[1]
    program = ("import runpy,sys; sys.argv=['check_slotspe_assignment_router.py']; "
               "runpy.run_path('scripts/check_slotspe_assignment_router.py',run_name='__main__'); "
               "assert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, "-B", "-c", program], cwd=root,
                            capture_output=True, text=True, check=True)
    assert '"status": "plan_only"' in result.stdout


@pytest.mark.parametrize("mode", ["original", "sinkhorn", "mesh_uniform"])
def test_valid_support_normalization_and_finite_gradients(mode):
    x = noise_like(torch.empty(2, 2, 3, 7, dtype=torch.float64)).requires_grad_()
    mask = torch.tensor([[1, 1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 1, 0, 1]], dtype=torch.bool)
    a = router(mode)(x, token_mask=mask, noise=noise_like(x))
    assert torch.isfinite(a).all()
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)))
    assert (a.masked_select(~mask[:, None, None, :]) == 0).all()
    (a * torch.arange(3, dtype=a.dtype)[None, None, :, None] *
     torch.arange(7, dtype=a.dtype)[None, None, None, :]).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_more_sinkhorn_steps_alone_do_not_break_identical_slot_ties():
    x = torch.tensor([[[[2., 0., -1., 3.], [2., 0., -1., 3.]]]], dtype=torch.float64)
    for mode in ("original", "sinkhorn"):
        a = router(mode, sinkhorn_steps=40)(x)
        assert torch.equal(a[:, :, 0], a[:, :, 1])


def test_mesh_breaks_tie_without_changing_values_or_rng():
    x = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    noise = torch.tensor([[[[1., -1.], [-1., 1.]]]], dtype=x.dtype)
    before = torch.get_rng_state().clone()
    a = router("mesh_uniform", mesh_lr=4)(x, noise=noise)
    assert torch.equal(torch.get_rng_state(), before)
    assert not torch.allclose(a[:, :, 0], a[:, :, 1], atol=0.1)
    values = torch.eye(2, dtype=x.dtype)
    updates = a @ values
    assert not torch.allclose(updates[:, :, 0], updates[:, :, 1], atol=0.1)


def test_different_attention_cannot_create_information_in_constant_values():
    x = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
    noise = torch.tensor([[[[1., -1.], [-1., 1.]]]], dtype=x.dtype)
    a = router("mesh_uniform", mesh_lr=4)(x, noise=noise)
    assert (a[:, :, 0] - a[:, :, 1]).abs().max() > .1
    # A sharper allocation can still produce identical updates. This is why
    # lower overlap alone must not be reported as semantic specialization.
    values = torch.ones(1, 1, 2, 5, dtype=x.dtype)
    update = a @ values
    assert torch.allclose(update[:, :, 0], update[:, :, 1])


def test_jointly_permuting_noise_preserves_slot_and_token_equivariance():
    x = noise_like(torch.empty(1, 2, 3, 7, dtype=torch.float64))
    noise = noise_like(x) * .7
    r = router("mesh_uniform")
    a = r(x, noise=noise)
    si, ti = torch.tensor([2, 0, 1]), torch.tensor([6, 1, 4, 0, 2, 5, 3])
    permuted = r(x[:, :, si][:, :, :, ti], noise=noise[:, :, si][:, :, :, ti])
    assert torch.allclose(permuted, a[:, :, si][:, :, :, ti], atol=1e-10)


def test_mesh_repeats_under_no_grad_and_has_no_parameters():
    x = noise_like(torch.empty(1, 1, 3, 8))
    n = noise_like(x)
    r = router("mesh_uniform")
    with torch.no_grad():
        first, second = r(x, noise=n), r(x, noise=n)
    assert torch.equal(first, second)
    assert not first.requires_grad
    assert not list(r.parameters()) and not r.state_dict()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        r(x, noise=n)


def test_explicit_noise_and_nonempty_mask_required():
    x = torch.zeros(1, 1, 2, 4)
    with pytest.raises(ValueError, match="noise"):
        router("mesh_uniform")(x)
    with pytest.raises(ValueError, match="token_mask"):
        router("original")(x, token_mask=torch.zeros(1, 4, dtype=torch.bool))
    with pytest.raises(ValueError):
        AssignmentConfig(sinkhorn_steps=True)


def test_original_replay_matches_retained_native_replay_and_preserves_state():
    # Load only the original SA class. Small generated tensors, no SlotSPE
    # constructor, patient input, checkpoint, optimizer, or training step.
    root = Path(__file__).resolve().parents[1]
    path = root / "third_party/SlotSPE/models/slot_attention.py"
    spec = importlib.util.spec_from_file_location("_assignment_synthetic_sa", path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    with torch.random.fork_rng():
        torch.manual_seed(43)
        native = source.MultiHeadSlotAttention(3, 8, heads=2, dim_head=4, iters=3)
        x, initial = torch.randn(2, 11, 8), torch.randn(2, 3, 8)
        before = {k: v.clone() for k, v in native.state_dict().items()}
        baseline = replay_slots(native, x, initial)
        actual = replay_with_assignment(native, x, initial, router("original"))
        assert torch.equal(actual["final"], baseline["final"])
        assert all(torch.equal(a, b) for a, b in zip(actual["attentions"], baseline["attentions"]))
        for k, v in native.state_dict().items():
            assert torch.equal(v, before[k])
        mesh = router("mesh_uniform")
        noises = [noise_like(a) for a in baseline["attentions"]]
        output = replay_with_assignment(native, x, initial, mesh, noises=noises)
        output["final"].square().sum().backward()
        assert native.to_q.weight.grad is not None
        assert torch.isfinite(native.to_q.weight.grad).all()
        assert native.to_q.weight.grad.abs().sum() > 0
