"""Selection-free, type-indexed concatenation with matched MLP capacity.

Requires fixed content-role metadata. Canonicalization is by that metadata,
not patient-specific score. Slots and metadata must be permuted together.
"""
import torch
from torch import nn
from .content_budget_slots import ContentBudgetSlotSurvival


class RolePreservingSlotSurvival(ContentBudgetSlotSurvival):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        k,d=self.role_offsets[0].shape
        width=self.phi[0][-1].out_features
        target=sum(p.numel() for p in self.phi[0].parameters())
        hidden=max(1,round((target-width)/(k*d+width+1)))
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(67501)
            self.phi=nn.ModuleList([nn.Sequential(nn.Linear(k*d,hidden,device='cpu'),nn.GELU(),nn.Linear(hidden,width,device='cpu')) for _ in range(2)])

    def pooled_private(self,private,role_indices=None):
        features=[]
        for j,(phi,u) in enumerate(zip(self.phi,private)):
            count=self.role_offsets[j].shape[0]
            if u.shape[1]!=count:raise ValueError('all fixed role slots required')
            ids=torch.arange(count,device=u.device) if role_indices is None else role_indices[j]
            if not torch.equal(ids.sort().values,torch.arange(count,device=u.device)):
                raise ValueError('role IDs must be a complete permutation')
            tagged=u+self.role_offsets[j][ids].unsqueeze(0)
            canonical=tagged[:,ids.argsort()]
            features.append(phi(canonical.flatten(1)))
        return features

    def readout(self,common,private,role_indices=None):
        return self.head(torch.cat([common,*self.pooled_private(private,role_indices)],-1))
