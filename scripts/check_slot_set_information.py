"""Mean-matched bags: distinguish set-information capacity from redundancy.

K1 cannot distinguish any bags here with this linear token-value encoder.
K4 may recover set dispersion via competitive attention. A win for K4 does
not establish that its final slots carry unique complementary information.
"""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from slotspe_proof.common_private_survival import CommonPrivateSurvival,survival_nll


def data(seed,n):
    g=torch.Generator().manual_seed(seed)
    a=.5+torch.randint(0,2,(n,),generator=g).float()
    b=.5+torch.randint(0,2,(n,),generator=g).float()
    pattern=torch.tensor([1.,-1.]*6)
    x=torch.zeros(n,12,6);y=torch.zeros_like(x)
    x[:,:,1]=a[:,None]*pattern;y[:,:,2]=b[:,None]*pattern
    risk=4*(a.square()+b.square()-2.5)
    events=(torch.rand(n,generator=g)<risk.sigmoid()).float()
    return x,y,torch.zeros(n,dtype=torch.long),events


def main():
    torch.set_num_threads(2);rows=[]
    for seed in [3,13,23]:
        torch.manual_seed(seed)
        template=CommonPrivateSurvival(6,6,width=16,slots=4,common_dim=4,private_dim=4,bins=1)
        initial={k:v.detach().clone() for k,v in template.state_dict().items()}
        train=data(10100+seed,512);test=data(11100+seed,1024)
        assert train[0].mean(1).abs().max()==0 and train[1].mean(1).abs().max()==0
        for k in [1,4]:
            net=CommonPrivateSurvival(6,6,width=16,slots=k,common_dim=4,private_dim=4,bins=1)
            state={name:(v[:1] if name.endswith('.initial') and k==1 else v) for name,v in initial.items()}
            net.load_state_dict(state)
            opt=torch.optim.Adam(net.parameters(),lr=.003)
            schedule=torch.Generator().manual_seed(22000+seed)
            noise_rng=torch.Generator().manual_seed(32000+seed)
            for _ in range(500):
                idx=torch.randperm(512,generator=schedule)[:32]
                noise=[torch.randn(32,4,4,generator=noise_rng)[:,:k] for _ in range(2)]
                out=net(train[0][idx],train[1][idx],noise=noise)
                loss=survival_nll(out['logits'],train[2][idx],train[3][idx])+.1*out['cross']+.1*out['alignment']
                opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                out=net(*test[:2],sample=False)
                logits=out['logits'];nll=float(survival_nll(logits,*test[2:]))
                spread=float(logits.std(unbiased=False))
            rows.append({'seed':seed,'slots_per_modality':k,'meanmode_test_nll':nll,'logit_std':spread,
                         'parameter_count':sum(p.numel() for p in net.parameters()),'core_updates':500})
            print(json.dumps(rows[-1]),flush=True)
            if k==1:assert spread<1e-6
    report={'status':'completed','rows':rows,'scope':'6 generated CPU neural trainings; no patient or checkpoint reads',
            'theory':'K1 with linear values and uniform token average sees identical means for all bags',
            'limitation':'meanmode endpoint; K4 win demonstrates set-aggregation capacity, not private/common semantics or per-slot unique contribution'}
    Path('docs/research/slotspe_common_private_20261101/slot_set_information_receipt.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())


if __name__=='__main__':main()
