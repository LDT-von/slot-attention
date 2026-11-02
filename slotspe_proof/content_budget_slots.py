"""Fixed-content supervision and soft per-slot rate budgets.

Targets must be frozen input functions, not event labels, student projections
or patient-specific queries. This prototype supplies a synthetic typed-token
adapter only; it is NOT an adapter for clinical WSI/omics data.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .prior_normalized_slots import PriorNormalizedSlotSurvival


def typed_token_targets(x,types,value_column,valid=None):
    """Synthetic type channels are fixed metadata. Missing type is unknown."""
    if x.ndim!=3 or types<1 or not types<=value_column<x.shape[-1]:raise ValueError('typed token layout')
    valid=torch.ones(x.shape[:2],dtype=torch.bool,device=x.device) if valid is None else valid
    if valid.dtype!=torch.bool or valid.shape!=x.shape[:2]:raise ValueError('mask layout')
    metadata=x[...,:types].detach()[valid]
    if not bool(((metadata==0)|(metadata==1)).all()) or not bool((metadata.sum(-1)==1).all()):
        raise ValueError('valid synthetic tokens require exact one-hot type metadata')
    ids=x[...,:types].detach().argmax(-1)
    labels=[];observed=[]
    for k in range(types):
        mask=(ids==k)&valid;count=mask.sum(1)
        average=x[...,value_column].detach().masked_fill(~mask,0).sum(1)/count.clamp_min(1)
        labels.append((average>0).to(x.dtype));observed.append(count>0)
    return {'labels':torch.stack(labels,1),'observed':torch.stack(observed,1)}


class ContentBudgetSlotSurvival(PriorNormalizedSlotSurvival):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        k,d=self.encoders[0].initial.shape[0],self.posterior[0].out_features//2
        self.role_offsets=nn.ParameterList([nn.Parameter(torch.zeros(k,d)) for _ in range(2)])
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(15733)
            self.content_heads=nn.ModuleList([nn.Sequential(nn.Linear(d,d,device='cpu'),nn.SiLU(),nn.Linear(d,1,device='cpu')) for _ in range(2)])

    def readout(self,common,private,role_indices=None):
        pooled=[]
        for j,(phi,u) in enumerate(zip(self.phi,private)):
            ids=torch.arange(u.shape[1],device=u.device) if role_indices is None else role_indices[j]
            # Fixed type identity is carried with slots; no patient-dependent gate.
            pooled.append(phi(u+self.role_offsets[j][ids].unsqueeze(0)).mean(1))
        return self.head(torch.cat([common,*pooled],-1))

    def forward(self,*args,**kwargs):
        out=super().forward(*args,**kwargs)
        # Shared across slots and receives no type ID or common vector.
        out['content_logits']=[head(v).squeeze(-1) for head,v in zip(self.content_heads,out['private'])]
        return out


def content_budget_terms(out,targets,budget=1.):
    if len(targets)!=len(out['content_logits']):raise ValueError('one target per modality required')
    content=[];rates=[]
    for logits,target,rate in zip(out['content_logits'],targets,out['normalized_kl_per_slot']):
        labels=target['labels'].detach();observed=target['observed']
        if logits.shape!=labels.shape or logits.shape!=observed.shape:raise ValueError('target slot shape mismatch')
        losses=F.binary_cross_entropy_with_logits(logits,labels,reduction='none')
        # Average patients equally; no patient exclusion when some targets absent.
        content.append(((losses*observed).sum(1)/observed.sum(1).clamp_min(1)).mean())
        rates.append(torch.relu(rate.mean(0)-budget).mean())
    return {'content':torch.stack(content).mean(),'budget_violation':torch.stack(rates).mean()}
