"""Run the unchanged desktop SlotSPE on paired BLCA inputs and controlled probes.

No fitted checkpoint is available. All observations concern implementation
properties, not trained patient accuracy. No source/data file is modified.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import time
import copy
import torch
import torch.nn.functional as F

SOURCE = Path(r'C:\Users\栋栋\Desktop\特征解耦论文\SlotSPE-main\SlotSPE-main')
OUT = Path(__file__).resolve().parent
FEATURES = Path(r'E:\TCGA-huggingface\blca\uni\pt_files')
sys.path.insert(0, str(SOURCE))
from models.SlotSPE import SlotSPE
from dataset.dataset_survival import SurvivalDatasetFactory, SurvivalDataset
from utils.process_args import _process_args

torch.set_num_threads(2)
torch.manual_seed(1603)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
started = time.time()
files = ['models/SlotSPE.py', 'models/slot_attention.py',
         'models/transformer.py', 'models/omics_encoder.py',
         'utils/loss_func.py', 'utils/core_utils.py',
         'utils/process_args.py', 'dataset/dataset_survival.py']
def source_hashes():
    return {p: hashlib.sha256((SOURCE / p).read_bytes()).hexdigest() for p in files}

result = {
    'scope': 'unchanged_original_code_random_weights_not_trained_performance',
    'source': str(SOURCE), 'seed': 1603, 'torch': torch.__version__,
    'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
    'trained_checkpoint_loaded': False, 'optimizer_steps': 0,
    'source_hashes_before': source_hashes(), 'experiments': {},
}
factory = SurvivalDatasetFactory('blca', str(SOURCE / 'dataset_csv'),
                                'Pathways', 'survival_months_dss',
                                signature='combine', n_bins=4, num_patches=256)
dataset = SurvivalDataset(factory, str(FEATURES), split_key='train', fold=0, encoding_dim=1024)
bags, omic_cases, labels, censor, patients = [], [], [], [], []
for _, row in dataset.label_df.iterrows():
    pid = row['case id']
    if pid not in factory.gene_data_df.columns:
        continue
    paths = [FEATURES / (s.rstrip('.svs') + '.pt') for s in row['wsi'].split(', ')]
    if not paths or not all(p.exists() for p in paths):
        continue
    chunks = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    bag = torch.cat(chunks).float()
    assert bag.ndim == 2 and bag.shape[1] == 1024 and torch.isfinite(bag).all()
    indices = torch.linspace(0, len(bag) - 1, 256).round().long()
    bags.append(bag[indices])
    omic_cases.append(dataset.load_genes(pid))
    labels.append(int(row['label']))
    censor.append(float(row[factory.censorship_var]))
    patients.append({'patient': pid, 'files': [str(p) for p in paths],
                     'full_patch_count': len(bag), 'sampled_indices': indices.tolist(),
                     'file_sha256': [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]})
    if len(bags) == 4:
        break
assert len(bags) == 4
args = _process_args()
args.omic_sizes = factory.omic_sizes
args.encoding_dim = 1024
inputs = {'x_wsi': torch.stack(bags).to(device), 'omic_missing': False,
          'y': torch.tensor(labels, device=device), 'c': torch.tensor(censor, device=device)}
for i in range(len(factory.omic_sizes)):
    inputs[f'x_omic{i+1}'] = torch.stack([p[i] for p in omic_cases]).to(device)
result['data'] = {
    'cohort': 'BLCA', 'origin_split': 'fold_0_train_subset', 'patients': patients,
    'pathway_count': len(args.omic_sizes), 'pathway_sizes': args.omic_sizes,
    'patches_per_patient': 256, 'encoding_dim': args.encoding_dim,
    'sampling': 'evenly_spaced_fixed_indices',
    'rna_transform': 'unchanged original loader (no extra scaling)',
    'rna_sha256': hashlib.sha256((SOURCE/'dataset_csv/raw_rna_data_inter/blca_rna_inter.csv').read_bytes()).hexdigest(),
}
net = SlotSPE(args).to(device).eval()
result['model'] = {k: getattr(args, k) for k in ['wsi_projection_dim','slot_num_wsi','slot_num_omics',
                 'slot_iters','temperature','topk_ratio','top_k_method','lambda_recon_loss']}
result['model']['parameter_count'] = sum(p.numel() for p in net.parameters())

# A. Original eval repeatedly re-samples both initial slots and Gumbel gates.
keeps = {'wsi': [], 'omic': []}
hooks = []
for name in keeps:
    module = getattr(net, 'slot_decoder_' + name)
    def capture(module, inp, out, key=name):
        keeps[key].append(out[2].detach().cpu())
    hooks.append(module.register_forward_hook(capture))
with torch.no_grad():
    logits = torch.stack([net(**inputs)[0].cpu() for _ in range(32)])
    risks = -(1-logits.sigmoid()).cumprod(-1).sum(-1)
    torch.manual_seed(777)
    a = net(**inputs)[0]
    torch.manual_seed(777)
    b = net(**inputs)[0]
for h in hooks:
    h.remove()
result['experiments']['eval_repeat_32_real_inputs'] = {
    'logit_max_range': float((logits.max(0).values-logits.min(0).values).max()),
    'risk_std_per_patient': risks.std(0).tolist(),
    'risk_range_per_patient': (risks.max(0).values-risks.min(0).values).tolist(),
    'risk_mean_per_patient': risks.mean(0).tolist(),
    'identical_rng_reset_logit_max_abs': float((a-b).abs().max()),
    'unique_keep_masks_per_patient': {k: [len(torch.unique(torch.stack(v[:32])[:,i], dim=0))
                                           for i in range(4)] for k,v in keeps.items()},
    'note': 'Not a C-index or trained-patient stability estimate.',
}

# B. Final survival NLL cannot train bool-cast selection scores through masks.
net.eval()
torch.manual_seed(888)
pred, _ = net(**inputs)
main_loss = net.loss_fn(pred, y=inputs['y'], c=inputs['c'], t=None)/4
named = [(n,p) for n,p in net.named_parameters() if 'pred_keep_slot' in n]
grads = torch.autograd.grad(main_loss, [p for _,p in named], allow_unused=True)
main_grad = {n: None if g is None else float(g.norm()) for (n,_),g in zip(named,grads)}
net.train()
torch.manual_seed(888)
pred, aux = net(**inputs)
aux_grads = torch.autograd.grad(aux, [p for _,p in named], allow_unused=True)
result['experiments']['selection_gradient_real_inputs'] = {
    'main_nll': float(main_loss.detach()),
    'main_nll_to_selector': main_grad,
    'auxiliary_to_selector': {n: None if g is None else float(g.norm()) for (n,_),g in zip(named,aux_grads)},
    'note': 'Auxiliary unimodal supervision remains; absence of main gradient is not absence of all selector training.',
}
net.eval()

# C. A fixed hard mask does not exclude dropped tokens from pooled self output.
torch.manual_seed(999)
slots = torch.randn(2,8,256,device=device,requires_grad=True)
mask = torch.zeros(2,8,device=device,dtype=torch.bool)
mask[:,:2] = True
delta = torch.randn_like(slots)*3
changed = torch.where(mask[...,None], slots, slots+delta)
with torch.no_grad():
    before = net.self_attention_wsi(slots, mask=mask)
    after = net.self_attention_wsi(changed, mask=mask)
    attn = net.self_attention_wsi.get_attn(slots,mask=mask)
probe_weight = torch.randn(2,256,device=device)
pooled = net.self_attention_wsi(slots,mask=mask).mean(1)
g, = torch.autograd.grad((pooled*probe_weight).sum(), slots)
result['experiments']['fixed_mask_dropped_slot_intervention'] = {
    'input_type': 'controlled synthetic slot tensors through unchanged real module',
    'selected_output_max_abs_change': float((before[mask]-after[mask]).abs().max()),
    'dropped_output_max_abs_change': float((before[~mask]-after[~mask]).abs().max()),
    'pooled_output_max_abs_change': float((before.mean(1)-after.mean(1)).abs().max()),
    'pooled_gradient_norm_selected': float(g[mask].norm()),
    'pooled_gradient_norm_dropped': float(g[~mask].norm()),
    'dropped_query_attention_first_head': attn[0,0,2].tolist(),
}
cross = net.cross_attention(slots, slots.flip(1))
cross_weight = torch.randn_like(cross)
cg, = torch.autograd.grad((cross*cross_weight).sum(), slots)
result['experiments']['cross_branch_all_slots'] = {
    'gradient_norm_nominally_dropped_slots': float(cg[~mask].norm()),
    'note': 'Original full forward supplies no keep masks to cross_attention; the mask is only a reference subset here.',
}

# D. Reconstruction has a constant-latent global minimum despite distinct inputs.
# This is a constructive counterexample, not an observed trained collapse.
small_args = copy.copy(args)
small_args.omic_sizes = [3,4,5]
small_args.encoding_dim = 16
small_args.wsi_projection_dim = 32
small_args.slot_iters = 3
constant = SlotSPE(small_args).to(device).eval()
with torch.no_grad():
    for p in constant.parameters():
        p.zero_()
    constant.wsi_mlp[-1].bias[0] = 1
    for encoder in constant.sig_networks:
        encoder[-1][0].bias[0] = 1
    for head_name in ['reconstruction_head_omic','reconstruction_head_joint','reconstruction_head_wsi']:
        getattr(constant,head_name).cross_attn.out_proj.bias[0] = 1
cx = torch.randn(4,20,16,device=device)
co = [torch.randn(4,n,device=device) for n in [3,4,5]]
wp = constant.wsi_mlp(cx)
op = torch.stack([e(x) for e,x in zip(constant.sig_networks,co)],1)
ws = constant.slot_attention_wsi(wp)
os = constant.slot_attention_omic(op)
ks = torch.ones(4,8,device=device,dtype=torch.bool)
ro = constant.reconstruction_head_omic(op,os,mask=ks)
rj = constant.reconstruction_head_joint(op,constant.slot_attention_omic(wp))
rw = constant.reconstruction_head_wsi(wp,ws,mask=ks)
lom = F.mse_loss(ro,op.detach())
lcross = F.mse_loss(rj,op.detach())
lwsi = 1-F.cosine_similarity(F.normalize(rw,dim=-1),F.normalize(wp,dim=-1),dim=-1).mean()
result['experiments']['constant_latent_reconstruction_counterexample'] = {
    'input_type': 'constructed distinct random raw inputs; controlled weights in separate model instance',
    'omics_self_mse': float(lom.detach()), 'cross_mse': float(lcross.detach()),
    'wsi_cosine_loss': float(lwsi.detach()),
    'raw_wsi_patient_variance': float(cx.var(0).mean()),
    'encoded_wsi_patient_variance': float(wp.var(0).mean()),
    'encoded_omics_patient_variance': float(op.var(0).mean()),
    'note': 'Only reconstruction objective degeneracy. Survival losses are still positive and oppose losing prognostic information.',
}

# E. WSI reconstruction target is not detached; omics reconstruction target is.
torch.manual_seed(1777)
raw = torch.randn(2,24,1024,device=device)
encoded = net.wsi_mlp(raw)
real_slots = net.slot_attention_wsi(encoded)
reconstructed = net.reconstruction_head_wsi(encoded,real_slots,mask=torch.ones(2,8,device=device,dtype=torch.bool))
ln = 1-F.cosine_similarity(F.normalize(reconstructed.detach(),dim=-1),
                           F.normalize(encoded,dim=-1),dim=-1).mean()
target_grad = torch.autograd.grad(ln,net.wsi_mlp[-1].weight)[0]
result['experiments']['wsi_target_gradient'] = {
    'controlled_synthetic_input': True, 'decoder_output_detached_for_isolation': True,
    'gradient_norm_encoder_from_target_side_alone': float(target_grad.norm()),
}
result['elapsed_seconds'] = time.time()-started
result['source_hashes_after'] = source_hashes()
result['source_unchanged'] = result['source_hashes_before'] == result['source_hashes_after']
(OUT/'diagnostic_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'saved':str(OUT/'diagnostic_results.json'),
                  'elapsed_seconds':result['elapsed_seconds'],
                  'source_unchanged':result['source_unchanged'],
                  'pathway_count':len(args.omic_sizes),
                  'eval':result['experiments']['eval_repeat_32_real_inputs'],
                  'main_selector_gradients_all_none':all(v is None for v in main_grad.values()),
                  'dropped_slot_probe':result['experiments']['fixed_mask_dropped_slot_intervention'],
                  'reconstruction_counterexample':result['experiments']['constant_latent_reconstruction_counterexample']},
                 ensure_ascii=False,indent=2))
