import torch
import pytest
from slotspe_proof.content_budget_slots import ContentBudgetSlotSurvival
from slotspe_proof.role_preserving_readout import RolePreservingSlotSurvival


@pytest.fixture(autouse=True)
def cpu():
    old=torch.get_num_threads();torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):yield
    torch.set_num_threads(old)


def test_capacity_is_matched_and_shared_core_initialization_identical():
    for width,k,d in [(16,4,4),(128,8,32)]:
        torch.default_generator.manual_seed(3)
        a=ContentBudgetSlotSurvival(8,8,width=width,slots=k,private_dim=d,common_dim=4)
        torch.default_generator.manual_seed(3)
        b=RolePreservingSlotSurvival(8,8,width=width,slots=k,private_dim=d,common_dim=4)
        pa=sum(p.numel() for p in a.phi.parameters());pb=sum(p.numel() for p in b.phi.parameters())
        assert abs(pa-pb)/pa<.05
        assert all(torch.equal(v,b.state_dict()[name]) for name,v in a.state_dict().items() if not name.startswith('phi.'))


def test_paired_role_slot_permutation_invariant():
    net=RolePreservingSlotSurvival(8,8,width=16,slots=4,private_dim=4,common_dim=4)
    g=torch.randn(8,4);us=[torch.randn(8,4,4) for _ in range(2)]
    p=torch.tensor([2,0,3,1])
    assert torch.allclose(net.readout(g,us),net.readout(g,[u[:,p] for u in us],[p,p]))


def test_no_silent_role_drop_or_duplicate():
    net=RolePreservingSlotSurvival(8,8,width=16,slots=4,private_dim=4,common_dim=4)
    with pytest.raises(ValueError):net.pooled_private([torch.randn(8,3,4)]*2)
    with pytest.raises(ValueError):net.pooled_private([torch.randn(8,4,4)]*2,[torch.zeros(4,dtype=torch.long)]*2)
