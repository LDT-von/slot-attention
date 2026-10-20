"""Same-weight limiting-case and extreme-gate controls, no training."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,copy
import torch
from fixed_model import make_args,build_model

torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parent
rows=[]
for seed in [3,17,29,43,61]:
    torch.manual_seed(seed)
    args=make_args([3,4,5],encoding_dim=16)
    args.wsi_projection_dim=32
    args.slot_iters=3
    args.topk_ratio=1.
    a=build_model(args,'original').train()
    b=build_model(args,'readout_fix').train()
    b.load_state_dict(a.state_dict(),strict=True)
    inputs={'x_wsi':torch.randn(4,24,16),'omic_missing':False,
            'y':torch.tensor([0,1,2,3]),'c':torch.tensor([0.,1.,0.,1.])}
    for i,n in enumerate([3,4,5]):inputs[f'x_omic{i+1}']=torch.randn(4,n)
    torch.manual_seed(seed+100)
    original,_=a(**inputs)
    torch.manual_seed(seed+100)
    fixed,_=b(**inputs)
    difference=float((original-fixed).abs().max())
    assert difference<1e-6
    rows.append({'seed':seed,'keep_all_main_logits_max_abs':difference})

args=make_args([3,4,5],encoding_dim=16)
args.wsi_projection_dim=32
original=build_model(args,'original').slot_decoder_wsi.train()
fixed=build_model(args,'readout_fix').slot_decoder_wsi.train()
fixed.load_state_dict(original.state_dict(),strict=True)
scores=torch.tensor([1.]+[0.]*7).reshape(1,8,1)
hooks=[d.pred_keep_slot.register_forward_hook(lambda m,i,o:scores) for d in [original,fixed]]
slots=torch.randn(1,8,32)
gate_rows=[]
for seed in range(20000,20032):
    torch.manual_seed(seed)
    _,go,ko=original(slots)
    torch.manual_seed(seed)
    _,gf,kf=fixed(slots)
    assert torch.equal(ko.detach()>0.5,kf.detach()>0.5)
    assert abs(float(gf.sum())-1)<1e-6
    gate_rows.append({'seed':seed,'top_score_selected':bool(ko[0,0]>0.5),
                      'original_gate_sum':float(go.sum()),'fixed_gate_sum':float(gf.sum())})
for h in hooks:h.remove()
assert any(r['original_gate_sum']<1e-10 for r in gate_rows)
result={'scope':'controlled_tensor_limit_tests_not_patient_performance',
        'same_forward_hard_selection':True,'keep_all_control':rows,
        'extreme_gate_control':gate_rows,'all_assertions_passed':True,
        'surrogate_temperature':1.0,
        'note':'Straight-through gradients are surrogate, not an unbiased discrete derivative.'}
(ROOT/'model/numeric_controls.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'controls':'passed','keep_all_max_error':max(r['keep_all_main_logits_max_abs'] for r in rows),
                  'original_tiny_gate_count':sum(r['original_gate_sum']<1e-10 for r in gate_rows),
                  'fixed_tiny_gate_count':sum(r['fixed_gate_sum']<1e-10 for r in gate_rows)}),flush=True)
