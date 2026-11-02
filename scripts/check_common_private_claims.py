"""Known-answer mathematical counterexamples; no patient I/O or training."""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slotspe_proof.common_private_diagnostics import partial_probe
from slotspe_proof.common_private_survival import gaussian_kl


def main():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    g = torch.linspace(-2, 2, 401)[:, None]
    p = torch.cos(torch.arange(401.)[:, None])
    zero = torch.zeros_like(g)
    copies = []
    for a in [0., 1., 10., 100.]:
        prior = a*g
        copies.append(float(gaussian_kl(prior+p, zero, prior, zero).mean()))
    assert max(copies)-min(copies) < 1e-12
    gt = torch.linspace(-1.99, 1.99, 400)[:, None]
    probe = partial_probe(g, g.square(), g.square(), gt, gt.square(), gt.square())
    # U=G^2 contains exactly zero conditional information given G, but the
    # linear residual instrument can report nearly perfect 'private increment'.
    probe_value = probe
    theta = torch.tensor(0., requires_grad=True)
    tiny = 1e-4*(2+torch.sin(1e6*theta))
    tiny.backward()
    assert abs(theta.grad.item()-100) < 1e-8
    # Residual E=G*epsilon has E[E|G]=0 yet its variance depends on G.
    cases = torch.cartesian_prod(torch.tensor([1., 2.]), torch.tensor([-1.,1.]))
    residual = cases[:,0]*cases[:,1]
    assert residual[cases[:,0]==1].mean()==0 and residual[cases[:,0]==2].mean()==0
    # XOR: private components each carry no marginal outcome information.
    bits = torch.cartesian_prod(torch.tensor([0,1]),torch.tensor([0,1]))
    y = bits[:,0]^bits[:,1]
    assert all(float(y[bits[:,j]==v].double().mean())==.5 for j in [0,1] for v in [0,1])
    report = {
      'scope':'analytic/fabricated arrays only; zero training, patient, checkpoint or server execution',
      'conditional_kl_common_copy_invariance':{'copy_scales':[0,1,10,100],'kl':copies,'conclusion':'shared predictable copy is free under a matching conditional prior'},
      'partial_probe_false_increment':{'construction':'U=G^2; Y=G^2, so I(Y;U|G)=0 exactly','actual_repository_probe':probe_value},
      'small_loss_large_gradient':{'loss':tiny.item(),'gradient':theta.grad.item()},
      'conditional_mean_residual_not_independent':{'conditional_variances':[1.,4.],'conditional_means':[0.,0.]},
      'xor_synergy':{'marginal_information_bits':0,'joint_information_bits':1},
      'alignment_scaling_escape':{'construction':'c_w=a*p, c_o=a*q; downstream weights /a preserve predictions; alignment scales a^2','qualified':'requires unconstrained downstream rescaling; not a claim of observed patient use'}
    }
    path=Path(__file__).resolve().parents[1]/'docs/research/slotspe_common_private_20261101/claims_counterexamples_receipt.json'
    path.write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
