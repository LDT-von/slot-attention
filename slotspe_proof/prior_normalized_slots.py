"""Prior-normalized private readout for the unchanged common/private host.

V=(U-mu_prior(G))/sigma_prior(G). Pointwise Gaussian KL is unchanged, but
its fixed-reference form bounds information in the ACTUAL private readout V.
This is a conditional Gaussian reparameterization, not an identified semantic
factorization or a new general-purpose MI estimator.
"""
import torch
from .common_private_survival import CommonPrivateSurvival,gaussian_kl


def normalize_gaussian(mu,logvar,prior_mu,prior_logvar):
    return (mu-prior_mu)*torch.exp(-.5*prior_logvar),logvar-prior_logvar


class PriorNormalizedSlotSurvival(CommonPrivateSurvival):
    def forward(self,*args,**kwargs):
        if getattr(self,'_normalizing',False):raise RuntimeError('reentrant normalization forward')
        self._normalizing=True;priors={};posteriors={};handles=[]
        def capture(record,index):
            def hook(module,inputs,output):record[index]=output
            return hook
        try:
            for j,(prior,posterior) in enumerate(zip(self.prior,self.posterior)):
                handles.append(prior.register_forward_hook(capture(priors,j)))
                handles.append(posterior.register_forward_hook(capture(posteriors,j)))
            out=super().forward(*args,**kwargs)
        finally:
            for h in handles:h.remove()
            self._normalizing=False
        raw=out['private'];normalized=[];kl=[]
        for j,u in enumerate(raw):
            pm,plv=priors[j].unsqueeze(1).chunk(2,-1);plv=plv.clamp(-8,4)
            qm,qlv=posteriors[j].chunk(2,-1);qlv=qlv.clamp(-8,4)
            vm,vlv=normalize_gaussian(qm,qlv,pm,plv)
            normalized.append((u-pm)*torch.exp(-.5*plv))
            kl.append(gaussian_kl(vm,vlv,torch.zeros_like(vm),torch.zeros_like(vlv)).sum((1,2)))
        out['raw_private']=raw;out['private']=normalized
        out['normalized_kl_joint']=torch.stack(kl).sum(0)
        out['logits']=self.readout(out['common'],normalized)
        return out
