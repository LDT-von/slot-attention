import io
import itertools

import pytest
import torch

from slotspe_proof.bounded_native_residual import BoundedNativeResidual
from slotspe_proof.native_preservation_audit import affine_blocks, collision_inputs, _MomentBackbone
from slotspe_proof.plugin_host import SlotPluginHost
from slotspe_proof.slot_disentanglement import SharedPrivateSlots, content_set_loss, separation_terms


@pytest.fixture(autouse=True)
def isolated_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3301)
        yield
    torch.set_num_threads(threads)


def small(bound="composed_frobenius"):
    return BoundedNativeResidual(SharedPrivateSlots(8, 2, 6, 1).double(), bound=bound)


@pytest.mark.parametrize("seed", [731, 105460])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_shape_bound_recovery_and_old_null_direction(seed, dtype):
    source = SharedPrivateSlots(seed=seed).double()
    before, after, _, _ = collision_inputs(source)
    model = BoundedNativeResidual(source).to(dtype=dtype)
    x, y = before.to(dtype), after.to(dtype)
    z0, r0 = model(x)
    z1, _ = model(y)
    mean, local, _ = affine_blocks(source)
    actual = max(float(torch.linalg.matrix_norm(local.double(), 2)),
                 float(torch.linalg.matrix_norm((mean+local).double(), 2)))
    assert float(r0['norm_upper_bound']) >= actual - 1e-6
    assert float(r0['contraction_upper_bound']) < .5
    assert torch.allclose(z1-z0, y-x, atol=8e-7, rtol=8e-7)
    assert torch.allclose(model.inverse(z1), y, atol=8e-7, rtol=8e-7)
    assert z0.shape == x.shape and z0.dtype == dtype


@pytest.mark.parametrize("bound", ["composed_frobenius", "factor_frobenius"])
def test_permutation_and_parameter_only_coefficient(bound):
    model = small(bound)
    x = torch.randn(3, 4, 8, dtype=torch.double)
    perm = torch.tensor([3, 1, 0, 2])
    z, record = model(x)
    zp, rp = model(x[:, perm])
    _, shifted = model(x+100)
    assert torch.allclose(z[:, perm], zp, atol=1e-12)
    assert torch.equal(record['residual_coefficient'], rp['residual_coefficient'])
    assert torch.equal(record['residual_coefficient'], shifted['residual_coefficient'])


def test_task_content_and_coefficient_gradients_are_scoped():
    model = small()
    with torch.no_grad():
        for parameter in model.adapter.parameters():
            parameter.mul_(3)
    x = torch.randn(3, 4, 8, dtype=torch.double, requires_grad=True)
    z, record = model(x)
    task = (z * torch.arange(8, dtype=torch.double)).square().mean()
    gradients = torch.autograd.grad(task, [model.adapter.shared_encoder.weight,
                                         model.adapter.private_encoder.weight,
                                         model.adapter.content_head[0].weight, x], allow_unused=True, retain_graph=True)
    assert all(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in [gradients[0], gradients[1], gradients[3]])
    assert gradients[2] is None
    target = torch.randn(3, 4, 1, dtype=torch.double, requires_grad=True)
    content = content_set_loss(record['content'], target)
    shared, private, target_grad = torch.autograd.grad(content, [model.adapter.shared_encoder.weight,
                                                               model.adapter.private_encoder.weight, target], allow_unused=True)
    assert shared is None and private.norm()>0 and target_grad is None
    assert float(record['norm_upper_bound'])>1
    parameter = model.adapter.private_projection.weight
    analytical = torch.autograd.grad(task, parameter)[0][0, 0]
    original = float(parameter[0, 0])
    losses=[]
    for offset in [-1e-6, 1e-6]:
        with torch.no_grad(): parameter[0, 0] = original+offset
        value, _ = model(x)
        losses.append(float((value*torch.arange(8, dtype=torch.double)).square().mean()))
    with torch.no_grad(): parameter[0, 0] = original
    numerical=(losses[1]-losses[0])/2e-6
    assert float(analytical)==pytest.approx(numerical, rel=2e-5, abs=2e-7)


@pytest.mark.parametrize("bound", ["composed_frobenius", "factor_frobenius"])
def test_zero_weights_and_large_weights_have_finite_bound_and_gradients(bound):
    model=small(bound)
    x=torch.randn(2, 4, 8, dtype=torch.double, requires_grad=True)
    with torch.no_grad():
        for parameter in model.adapter.parameters(): parameter.zero_()
    z, record=model(x)
    assert torch.equal(z,x) and record['norm_upper_bound']==0
    z.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()
    model=small(bound)
    with torch.no_grad():
        for parameter in model.adapter.parameters(): parameter.mul_(100)
    z, record=model(x)
    mean, local, _=affine_blocks(model.adapter)
    coefficient=float(record['residual_coefficient'])
    minimum=min(float(torch.linalg.svdvals(torch.eye(8)+coefficient*local).min()),
                float(torch.linalg.svdvals(torch.eye(8)+coefficient*(mean+local)).min()))
    assert minimum>=1-float(record['contraction_upper_bound'])-1e-10
    assert torch.allclose(model.inverse(z),x,atol=1e-10)


def test_rng_and_state_roundtrip_with_configuration_guard():
    state=torch.random.get_rng_state().clone()
    model=small()
    assert torch.equal(state,torch.random.get_rng_state())
    x=torch.randn(2,4,8,dtype=torch.double)
    buffer=io.BytesIO(); torch.save(model.state_dict(),buffer); buffer.seek(0)
    state_dict=torch.load(buffer,weights_only=True)
    replica=small(); replica.load_state_dict(state_dict)
    assert torch.equal(model(x)[0],replica(x)[0])
    with pytest.raises(RuntimeError,match='configuration mismatch'):
        small('factor_frobenius').load_state_dict(state_dict)


def test_scope_rejects_bad_configuration_low_precision_and_autocast():
    source=SharedPrivateSlots(8,2,6,1)
    for q in [0,1,-1,float('nan'),True]:
        with pytest.raises(ValueError): BoundedNativeResidual(source,contraction=q)
    with pytest.raises(ValueError): BoundedNativeResidual(source,bound='power_iteration')
    with pytest.raises(ValueError): BoundedNativeResidual(source,margin=0)
    model=BoundedNativeResidual(source).half()
    with pytest.raises(ValueError,match='float32 or float64'): model(torch.randn(2,4,8).half())
    model=BoundedNativeResidual(SharedPrivateSlots(8,2,6,1))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        with pytest.raises(ValueError,match='autocast'): model(torch.randn(2,4,8))


def test_existing_hook_can_exercise_candidate_without_new_preset():
    host=SlotPluginHost(_MomentBackbone(256),'V2_content').double().eval()
    for name in ['wsi','omic']:
        host.parts[name]=BoundedNativeResidual(host.parts[name])
    x=torch.randn(2,8,256,dtype=torch.double)
    host(x_wsi=x,x_omics=x,omic_missing=False)
    for name in ['wsi','omic']:
        row=host.last_records[name]
        assert torch.allclose(host.parts[name].inverse(row['latent']),x,atol=1e-12)
    assert len(host.backbone.slot_attention_wsi._forward_hooks)==0


def test_exact_recovery_head_can_bypass_auxiliary_partition():
    model=small()
    x=torch.randn(16,4,8,dtype=torch.double)
    z,_=model(x)
    # Deliberately privileged inverse head: an existence counterexample, not a
    # proposed readout or evidence a finite ordinary head learns this inverse.
    score=model.inverse(z).mean(1)[:,0]
    assert torch.allclose(score,x.mean(1)[:,0],atol=1e-12)
    gradients=torch.autograd.grad(score.square().sum(),list(model.parameters()),allow_unused=True)
    assert max(float(g.abs().max()) for g in gradients if g is not None)<1e-11


def test_independent_nuisance_partition_can_pass_separation_without_task_information():
    model=small()
    with torch.no_grad():
        model.adapter.shared_encoder.weight.zero_(); model.adapter.shared_encoder.bias.zero_()
        model.adapter.private_encoder.weight.zero_(); model.adapter.private_encoder.bias.zero_()
        model.adapter.shared_encoder.weight[:,1]=1
        model.adapter.private_encoder.weight[:,2]=1
    rows=list(itertools.product([-1.,1.],[-1.,1.],[0.,2.]))
    x=torch.zeros(8,4,8,dtype=torch.double)
    for i,(y,n1,n2) in enumerate(rows): x[i,:,0:3]=torch.tensor([y,n1,n2])
    z,record=model(x)
    separation=separation_terms(record['shared'],record['private'])
    assert abs(float(separation['hsic']))<1e-12 and separation['variance']==0
    for nuisance_group in range(4):
        # Row i and i+4 differ only in target; C, U and old F are identical.
        i=nuisance_group
        assert torch.equal(record['shared'][i],record['shared'][i+4])
        assert torch.equal(record['private'][i],record['private'][i+4])
        assert torch.equal(record['base_latent'][i],record['base_latent'][i+4])
    recovered_score=model.inverse(z).mean(1)[:,0]
    assert torch.allclose(recovered_score,x[:,0,0],atol=1e-12)
    # Declared finite population: Y is independent of both nuisance branches.
    # The fixed content-reference gate is deliberately NOT claimed to pass.
