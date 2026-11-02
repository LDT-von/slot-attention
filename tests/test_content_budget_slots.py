import pytest
import torch
from slotspe_proof.content_budget_slots import ContentBudgetSlotSurvival,typed_token_targets,content_budget_terms


@pytest.fixture(autouse=True)
def cpu():
    old=torch.get_num_threads();torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):yield
    torch.set_num_threads(old)


def test_targets_are_frozen_and_missing_groups_unknown():
    x=torch.tensor([[[1.,0.,2.],[1.,0.,-1.]]],requires_grad=True)
    t=typed_token_targets(x,2,2)
    assert t['observed'].tolist()==[[True,False]]
    assert not t['labels'].requires_grad


def test_unknown_targets_do_not_act_as_negative_labels():
    logits=torch.tensor([[0.,3.]],requires_grad=True)
    out={'content_logits':[logits],'normalized_kl_per_slot':[torch.ones(1,2)]}
    t={'labels':torch.zeros(1,2),'observed':torch.tensor([[True,False]])}
    content_budget_terms(out,[t])['content'].backward()
    assert logits.grad[0,0]!=0 and logits.grad[0,1]==0


def test_readout_invariant_when_slots_and_type_ids_permuted_together():
    net=ContentBudgetSlotSurvival(8,8,width=16,slots=4,common_dim=4,private_dim=4)
    with torch.no_grad():
        for e in net.role_offsets:e.normal_()
    g=torch.randn(8,4);us=[torch.randn(8,4,4) for _ in range(2)]
    perm=torch.tensor([2,0,3,1])
    assert torch.allclose(net.readout(g,us),net.readout(g,[u[:,perm] for u in us],[perm,perm]),atol=1e-6)


def test_per_slot_rates_sum_to_joint_and_content_gradient_reaches_encoder():
    net=ContentBudgetSlotSurvival(8,8,width=16,slots=4,common_dim=4,private_dim=4)
    out=net(torch.randn(8,12,8),torch.randn(8,12,8))
    rates=sum(r.sum(1) for r in out['normalized_kl_per_slot'])
    assert torch.allclose(rates,out['normalized_kl_joint'])
    ts=[{'labels':torch.randint(0,2,(8,4)).float(),'observed':torch.ones(8,4,dtype=torch.bool)} for _ in range(2)]
    term=content_budget_terms(out,ts)
    (term['content']+term['budget_violation']).backward()
    assert net.encoders[0].initial.grad is not None
    assert torch.isfinite(net.encoders[0].initial.grad).all()


def test_arbitrary_features_cannot_masquerade_as_type_metadata():
    with pytest.raises(ValueError):typed_token_targets(torch.zeros(1,2,3),2,2)
