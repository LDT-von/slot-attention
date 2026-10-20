"""Falsifiable original-vs-repair mechanism experiments, no optimizer steps."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,time,copy
import torch
from torch import nn
from fixed_model import SOURCE,make_args,build_model,selected_pool
from dataset.dataset_survival import SurvivalDatasetFactory,SurvivalDataset

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'model'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS=[3,17,29,43,61]

def load_inputs():
    previous=json.loads((ROOT.parent/'slotspe_problem_audit_20261016/diagnostic_results.json').read_text(encoding='utf-8'))
    factory=SurvivalDatasetFactory('blca',str(SOURCE/'dataset_csv'),'Pathways',
        'survival_months_dss',signature='combine',num_patches=256)
    dataset=SurvivalDataset(factory,r'E:\TCGA-huggingface\blca\uni\pt_files','train',0,1024)
    bags=[];omics=[];labels=[];cs=[]
    for record in previous['data']['patients']:
        raw=torch.cat([torch.load(p,map_location='cpu',weights_only=True) for p in record['files']]).float()
        bags.append(raw[record['sampled_indices']])
        omics.append(dataset.load_genes(record['patient']))
        y,_,c=dataset.get_label(record['patient'])
        labels.append(y);cs.append(c)
    data={'x_wsi':torch.stack(bags).to(DEVICE),'omic_missing':False,
          'y':torch.stack(labels).to(DEVICE),'c':torch.stack(cs).to(DEVICE)}
    for i in range(len(factory.omic_sizes)):
        data[f'x_omic{i+1}']=torch.stack([o[i] for o in omics]).to(DEVICE)
    return data,make_args(factory.omic_sizes),[p['patient'] for p in previous['data']['patients']]

def grad_row(net,inputs,seed):
    net.train()
    torch.manual_seed(seed)
    pred,aux=net(**inputs)
    loss=net.loss_fn(pred,y=inputs['y'],c=inputs['c'],t=None)/len(inputs['y'])
    ps=[(n,p) for n,p in net.named_parameters() if 'pred_keep_slot' in n]
    grad=torch.autograd.grad(loss,[p for _,p in ps],allow_unused=True)
    return {'main_nll':float(loss.detach()),'auxiliary_loss':float(aux.detach()),
            'selector_gradients':{n:None if g is None else float(g.norm()) for (n,_),g in zip(ps,grad)},
            'connected_tensor_count':sum(g is not None for g in grad),
            'nonzero_tensor_count':sum(g is not None and float(g.norm())>1e-12 for g in grad)}

def main():
    started=time.time()
    data,args,patients=load_inputs()
    results={'scope':'mechanism_proofs_random_weights_not_patient_performance',
             'patients':patients,'seeds':SEEDS,'source':str(SOURCE),
             'torch':torch.__version__,'device':str(DEVICE),'records':[],
             'fix_scope':'No added trainable parameters; cross branch remains dense.',
             'source_hashes':{str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [SOURCE/'models/SlotSPE.py',SOURCE/'models/slot_attention.py',SOURCE/'models/transformer.py']}}
    for seed in SEEDS:
        torch.manual_seed(seed)
        original=build_model(args,'original').to(DEVICE)
        fixed=build_model(args,'readout_fix').to(DEVICE)
        fixed.load_state_dict(original.state_dict(),strict=True)
        assert set(original.state_dict())==set(fixed.state_dict())
        assert sum(p.numel() for p in original.parameters())==sum(p.numel() for p in fixed.parameters())
        row={'seed':seed,'parameter_count':sum(p.numel() for p in original.parameters()),
             'same_initial_state':all(torch.equal(v,fixed.state_dict()[k]) for k,v in original.state_dict().items())}
        row['original_gradient']=grad_row(original,data,seed+1000)
        row['fixed_gradient']=grad_row(fixed,data,seed+1000)
        (OUT/f'seed_{seed}_observation.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        assert row['original_gradient']['connected_tensor_count']==0
        assert row['fixed_gradient']['connected_tensor_count']==8
        assert row['fixed_gradient']['nonzero_tensor_count']>=4
        for name,net in [('original',original),('fixed',fixed)]:
            net.eval()
            with torch.no_grad():
                pred=torch.stack([net(**data)[0] for _ in range(16)])
            row[name+'_eval_max_range']=float((pred.max(0).values-pred.min(0).values).max())
        assert row['original_eval_max_range']>1e-5
        assert row['fixed_eval_max_range']<1e-7
        order=torch.tensor([3,1,0,2],device=DEVICE)
        reordered={k:(v[order] if torch.is_tensor(v) else v) for k,v in data.items()}
        with torch.no_grad():
            baseline=fixed(**data)[0]
            shuffled=fixed(**reordered)[0]
        row['fixed_batch_reorder_max_abs']=float((baseline[order]-shuffled).abs().max())
        assert row['fixed_batch_reorder_max_abs']<2e-6
        # Hold the selected set fixed; perturb only the discarded slots.
        torch.manual_seed(seed+2000)
        slots=torch.randn(4,8,256,device=DEVICE,requires_grad=True)
        keep=torch.zeros(4,8,device=DEVICE);keep[:,:2]=1
        mask=keep.bool()
        changed=torch.where(mask[...,None],slots,slots+torch.randn_like(slots)*2)
        a=original.self_attention_wsi(slots,mask=mask)
        b=original.self_attention_wsi(changed,mask=mask)
        original_pool=a.mean(1)
        fixed_pool=selected_pool(a,keep)
        probe=torch.randn_like(original_pool)
        go,=torch.autograd.grad((original_pool*probe).sum(),slots,retain_graph=True)
        gf,=torch.autograd.grad((fixed_pool*probe).sum(),slots)
        row['fixed_set_intervention']={
            'original_pool_max_change':float((a.mean(1)-b.mean(1)).abs().max()),
            'fixed_pool_max_change':float((selected_pool(a,keep)-selected_pool(b,keep)).abs().max()),
            'original_dropped_gradient_norm':float(go[~mask].norm()),
            'fixed_dropped_gradient_norm':float(gf[~mask].norm()),
            'scope':'self branch only; dense cross branch is intentionally not claimed to exclude slots'}
        assert row['fixed_set_intervention']['original_pool_max_change']>0.1
        assert row['fixed_set_intervention']['fixed_pool_max_change']<1e-7
        assert row['fixed_set_intervention']['fixed_dropped_gradient_norm']<1e-7
        results['records'].append(row)
        (OUT/'proof_progress.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        del original,fixed
        if DEVICE.type=='cuda':torch.cuda.empty_cache()
        print(json.dumps({'seed':seed,'assertions':'passed','original_eval_range':row['original_eval_max_range'],
                          'fixed_gradient_nonzero':row['fixed_gradient']['nonzero_tensor_count']}),flush=True)
    results['elapsed_seconds']=time.time()-started
    results['all_assertions_passed']=True
    (OUT/'proof_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print('saved',OUT/'proof_results.json',flush=True)

if __name__=='__main__':main()
