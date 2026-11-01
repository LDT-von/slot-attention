"""CPU-only generated-data acceptance. Never loads patient files or checkpoints."""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slotspe_proof.common_private_survival import CommonPrivateSurvival, gaussian_kl, survival_nll, objective


def generated(seed, n, task):
    g = torch.Generator().manual_seed(seed)
    c, p, q = torch.randn(3, n, generator=g)
    xs = [torch.randn(n, 12, 6, generator=g)*.3 for _ in range(2)]
    for x, u in zip(xs, [p, q]):
        x[:, :, 0] += c[:, None]
        x[:, :, 1] += u[:, None]
    risk = c + p + q if task == 'shared_and_private' else p
    # Independently censored discrete event times, with administrative censoring.
    hazards = torch.sigmoid(risk[:, None] + torch.tensor([-2., -1., 0., 1.]))
    hits = torch.rand(n, 4, generator=g) < hazards
    event_time = torch.where(hits.any(1), hits.int().argmax(1), 4)
    censor = torch.randint(0, 4, (n,), generator=g)
    event = event_time <= censor
    time = torch.minimum(event_time, censor)
    return *xs, time, event.float()


def main():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    m = CommonPrivateSurvival(6, 6, width=16, slots=4, common_dim=4, private_dim=4)
    x, y, t, e = generated(51, 16, 'shared_and_private')
    out = m(x, y, sample=False)
    checks = {}
    perm = torch.randperm(4)
    checks['readout_slot_permutation'] = torch.allclose(out['logits'], m.readout(out['common'], [u[:, perm] for u in out['private']]), atol=1e-6)
    checks['token_permutation'] = torch.allclose(out['logits'], m(x[:, torch.randperm(12)], y, sample=False)['logits'], atol=1e-6)
    padded = torch.cat([x, torch.randn(16, 3, 6)*100], 1)
    mask = torch.cat([torch.ones(16, 12), torch.zeros(16, 3)], 1).bool()
    checks['masked_padding'] = torch.allclose(out['logits'], m(padded, y, mask, sample=False)['logits'], atol=1e-6)
    a,b,c,d = [torch.randn(2,4,3) for _ in range(4)]
    ref = torch.distributions.kl_divergence(torch.distributions.Normal(a,(b/2).exp()), torch.distributions.Normal(c,(d/2).exp()))
    checks['analytic_kl'] = torch.allclose(gaussian_kl(a,b,c,d), ref, atol=1e-5)
    checks['censor_likelihood'] = abs(survival_nll(torch.zeros(2,4),torch.tensor([1,2]),torch.tensor([1.,0.])).item()-2.5*torch.log(torch.tensor(2.)).item()) < 1e-6
    objective(out,t,e).backward()
    checks['finite_gradients'] = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in m.parameters())
    assert all(checks.values()), checks
    runs=[]
    for task in ['shared_and_private','private_only']:
        train=generated(100,256,task)
        val=generated(101,256,task)
        for seed in [3,13]:
            for beta in [0., .001, 1.]:
                torch.manual_seed(seed)
                net=CommonPrivateSurvival(6,6,width=16,slots=4,common_dim=4,private_dim=4)
                optimizer=torch.optim.Adam(net.parameters(),lr=.003)
                initial=survival_nll(net(*train[:2],sample=False)['logits'],*train[2:]).item()
                for step in range(80):
                    idx=torch.randperm(256)[:32]
                    o=net(train[0][idx],train[1][idx])
                    loss=objective(o,train[2][idx],train[3][idx],beta=beta)
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                net.eval()
                with torch.no_grad():
                    torch.manual_seed(2000)
                    draws=[net(*val[:2]) for _ in range(5)]
                    nll=sum(survival_nll(o['logits'],*val[2:]).item() for o in draws)/5
                    final=survival_nll(net(*train[:2],sample=False)['logits'],*train[2:]).item()
                runs.append(dict(task=task,seed=seed,beta=beta,initial_train_meanmode_nll=initial,final_train_meanmode_nll=final,holdout_expected_sample_nll=nll,kl_joint=draws[0]['kl_joint'].mean().item()))
    result={'checks':checks,'runs':runs,'patient_execution':False,'steps_per_run':80,'note':'12 toy trainings, not patient evidence; five sampling repeats are not independent trainings. No architecture search or best-epoch selection.'}
    dest=Path('docs/research/slotspe_common_private_20261101/synthetic_receipt.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
