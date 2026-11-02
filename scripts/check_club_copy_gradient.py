"""Analytic copy-removal construction, not an end-to-end patient/model test."""
import json
from pathlib import Path
import torch


def main():
    # G,P,epsilon are independent unit Gaussians. U=aG+bP+sigma epsilon.
    # Decoder sees G,U. Bayes MSE for P = sigma²/(b²+sigma²), independent of a.
    # True conditional p(U|G) has mean aG and variance b²+sigma².
    # Oracle CLUB(G,U)=a²/(b²+sigma²). The old q(U|G,P) vs r(U|G)
    # KL with matching common mean is independent of a.
    rows=[]
    b=1.;sigma=.3;variance=b*b+sigma*sigma
    for mode in ['no_penalty','old_conditional_kl','oracle_direct_club']:
        a=torch.tensor(2.,dtype=torch.float64,requires_grad=True)
        optimizer=torch.optim.SGD([a],lr=.1)
        for _ in range(60):
            task=a*0+sigma*sigma/variance
            penalty=(a*0+b*b/(2*sigma*sigma)) if mode=='old_conditional_kl' else a*a/variance if mode=='oracle_direct_club' else a*0
            optimizer.zero_grad();(task+penalty).backward();optimizer.step()
        rows.append({'mode':mode,'final_copy_coefficient':float(a.detach()),
                     'bayes_task_mse':sigma*sigma/variance,
                     'true_common_private_mi_nats':float(.5*torch.log1p(a.detach().square()/variance))})
    assert rows[0]['final_copy_coefficient']==2 and rows[1]['final_copy_coefficient']==2
    assert abs(rows[2]['final_copy_coefficient'])<1e-4
    report={'construction':'U=aG+P+0.3epsilon; G,P,epsilon iid standard Gaussian; decoder sees G,U',
            'rows':rows,'scope':'3 scalar optimizations of analytic population objectives, no neural or patient training',
            'limitation':'oracle conditional density, fixed useful coefficient and Bayes decoder; not a guarantee for learned vCLUB or neural readout'}
    path=Path('docs/research/slotspe_common_private_20261101/club_copy_gradient_receipt.json')
    path.write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
