"""Train-fitted empirical role adapter and two-phase survival engine.

No experiment is executed on import. This finite-target decomposition makes
no claim to identify biological sources or remove measurement noise.
"""
import copy
import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from .components.role_model import RoleTargetModel, RoleAllocation, BoundedLinear
from .components.role_reference import FrozenRoleReference, ReferenceProvenance, role_content_losses
from .components.losses import discrete_hazard_nll
from .patient_pilot import write, sha, ROOT


def tensor_hash(named):
    h = hashlib.sha256()
    for name, value in sorted(named):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ContentAdapter:
    """Train-only scaling + truncated SVD; explicit pathway identities/positions."""
    def __init__(self, p, g, sizes, target_dim):
        if len(p) != len(g) or len(p) <= target_dim or sum(sizes) != g.shape[1]:
            raise ValueError('paired training targets and sufficient train rows required')
        self.sizes = list(sizes)
        self.stats = {}
        for name, x in [('p', p), ('g', g)]:
            if not torch.isfinite(x).all(): raise ValueError('nonfinite train target')
            mu = x.mean(0); sd = x.std(0, unbiased=False).clamp_min(1e-4)
            z = (x-mu)/sd
            _, singular, vh = torch.linalg.svd(z, full_matrices=False)
            if len(singular) < target_dim or singular[target_dim-1] <= singular[0]*1e-8:
                raise ValueError('requested target dimension exceeds varying train rank; do not silently change protocol')
            basis = vh[:target_dim].T.contiguous()
            # Unit train variance in retained score coordinates. This is an
            # explicit finite metric, not a guarantee about all input content.
            score_scale = (singular[:target_dim]/math.sqrt(len(x))).clamp_min(1e-8)
            self.stats[name] = {'mean': mu, 'sd': sd, 'basis': basis, 'score_scale': score_scale}

    def target(self, x, modality):
        s = self.stats[modality]
        return (((x-s['mean'])/s['sd']) @ s['basis'])/s['score_scale']

    def token_inputs(self, patches, omics):
        if [len(x) for x in omics] != self.sizes: raise ValueError('pathway/gene order schema changed')
        p = (patches-self.stats['p']['mean'])/self.stats['p']['sd']
        g = (torch.cat(omics)-self.stats['g']['mean'])/self.stats['g']['sd']
        width, groups = max(self.sizes), len(self.sizes)
        tokens = torch.zeros(groups, width+groups, dtype=torch.float64)
        at = 0
        for i, n in enumerate(self.sizes):
            tokens[i, :n] = g[at:at+n]
            tokens[i, width+i] = 1.
            at += n
        return p, tokens

    def state(self):
        return {'sizes': self.sizes, 'stats': self.stats}


class SurvivalRoleModel(RoleTargetModel):
    def __init__(self, reference, adapter, anchors, c):
        super().__init__(reference, p_input_dim=len(adapter.stats['p']['mean']),
            g_input_dim=max(adapter.sizes)+len(adapter.sizes),
            p_allocation=RoleAllocation(*c['allocation']['p']), g_allocation=RoleAllocation(*c['allocation']['g']),
            width=c['width'], iterations=c['iterations'], readout_hidden=c['hidden'],
            readout_bound=c['readout_bound'], interaction=True, train_role_anchors=anchors)
        # Retain the bounded additive + centered interaction function; produce
        # K hazard logits instead of the synthetic prototype's one regressor.
        for name in ('p_head', 'g_head'):
            getattr(self.readout, name)[2] = BoundedLinear(c['hidden'], c['bins'], c['readout_bound'])
        self.readout.cross = BoundedLinear(c['hidden'], c['bins'], c['readout_bound'])


def fit_boundaries(times, events, bins):
    t = times[events]
    if len(t) < bins or not torch.isfinite(times).all() or (times < 0).any():
        raise ValueError('finite nonnegative times and enough uncensored train events required')
    edges = torch.quantile(t, torch.arange(1, bins, dtype=t.dtype)/bins)
    if (edges <= 0).any() or (edges[1:] <= edges[:-1]).any():
        raise ValueError('distinct positive training event quantiles required')
    return edges


def label_times(times, events, boundaries):
    # A time on an internal boundary belongs to the next event interval; a
    # censoring time on it has completed exactly the preceding intervals.
    return torch.searchsorted(boundaries.contiguous(), times.contiguous(), right=True).long()


class PreparedStore:
    def __init__(self, root, manifest):
        self.root, self.manifest = Path(root), manifest

    def load(self, pid):
        path = (self.root/self.manifest['patients'][pid]['cache_path']).resolve()
        if self.root.resolve() not in path.parents: raise ValueError('cache escapes prepared root')
        item = torch.load(path, weights_only=True, map_location='cpu')
        if item['patient_id'] != pid: raise ValueError('cache identity mismatch')
        patches = item['patches'].double()
        omics = [x.double() for x in item['omics']]
        if patches.ndim != 2 or len(patches) == 0 or not torch.isfinite(patches).all():
            raise ValueError('finite nonempty real patches required')
        if not all(torch.isfinite(x).all() for x in omics): raise ValueError('nonfinite omics')
        return patches, omics


def collect_targets(store, ids):
    p, g = [], []
    for pid in ids:
        patches, omics = store.load(pid)
        p.append(patches.mean(0)); g.append(torch.cat(omics))
    return torch.stack(p), torch.stack(g)


def batch(store, ids, adapter, boundaries, c, device, *, generator):
    ps, gs, pm, pt, gt, times, events = [], [], [], [], [], [], []
    for pid in ids:
        patches, omics = store.load(pid)
        # Targets use all real patches; training bags sample without replacement
        # at every access and pad. The model explicitly masks padding.
        pt.append(adapter.target(patches.mean(0), 'p'))
        gt.append(adapter.target(torch.cat(omics), 'g'))
        count = min(len(patches), c['patches'])
        indices = torch.randperm(len(patches), generator=generator)[:count].sort().values
        p, g = adapter.token_inputs(patches[indices], omics)
        padded = torch.zeros(c['patches'], p.shape[1], dtype=p.dtype); padded[:count] = p
        ps.append(padded); gs.append(g); pm.append(torch.arange(c['patches']) < count)
        row = store.manifest['patients'][pid]
        times.append(row['time']); events.append(row['censor'] < 1)
    times = torch.tensor(times, dtype=torch.float64)
    events = torch.tensor(events, dtype=torch.bool)
    result = {'p': torch.stack(ps), 'g': torch.stack(gs), 'p_mask': torch.stack(pm),
        'g_mask': torch.ones(len(ids), len(adapter.sizes), dtype=torch.bool),
        'p_target': torch.stack(pt), 'g_target': torch.stack(gt), 'time': times,
        'event': events, 'completed': label_times(times, events, boundaries)}
    return {k: v.to(device) for k, v in result.items()}


def model_forward(model, b):
    return model(b['p'], b['g'], b['p_mask'], b['g_mask'])


def run_prepared(root, out, manifest, c, code_hashes, *, split_guard=None, synthetic=False):
    if split_guard is None and not synthetic:
        raise ValueError('verified split admission required before patient execution')
    identity = split_guard() if split_guard is not None else {'schema':'invented_smoke_only'}
    manifest_hash = sha(Path(root)/'prepared/manifest.json')
    torch.set_num_threads(c['torch_threads']); torch.manual_seed(c['seed'])
    device = torch.device(c['device'])
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('requested CUDA unavailable; no silent CPU fallback')
    if Path(out).exists(): raise FileExistsError('fresh run required')
    train = manifest['folds'][str(c['fold'])]['train']; val = manifest['folds'][str(c['fold'])]['val']
    if len(train) != len(set(train)) or len(val) != len(set(val)) or set(train) & set(val):
        raise ValueError('unique disjoint train/validation patients required')
    if not train or not val: raise ValueError('nonempty splits required')
    store = PreparedStore(root, manifest)
    p, g = collect_targets(store, train)
    adapter = ContentAdapter(p, g, manifest['omic_sizes'], c['target_dim'])
    tp, tg = adapter.target(p, 'p'), adapter.target(g, 'g')
    reference = FrozenRoleReference.estimated(tp, tg, common_rank=c['common_rank'],
        coordinate_system='target', provenance=ReferenceProvenance('train',
        'standardized PCA patch-mean and pathway-expression targets; linear predictable/residual subspaces',
        'training fold only; empirical roles, no biological source certification'))
    anchors = reference.targets(tp[:64], tg[:64])
    times = torch.tensor([manifest['patients'][i]['time'] for i in train], dtype=torch.float64)
    events = torch.tensor([manifest['patients'][i]['censor'] < 1 for i in train], dtype=torch.bool)
    boundaries = fit_boundaries(times, events, c['bins'])
    model = SurvivalRoleModel(reference, adapter, anchors, c).double().to(device)
    out = Path(out); out.mkdir(parents=True)
    initial_source_hashes = {p: sha(ROOT/p) for p in code_hashes}
    if initial_source_hashes != code_hashes:
        raise RuntimeError('source changed after execution entry')
    receipt = {'schema':'empirical_role_patient_pilot_v1','status':'running','config':c,
        'target_definition':reference.metadata,'phase_policy':'fixed final role epoch then fixed final survival epoch; no validation selection',
        'boundaries_fit_on':'uncensored training only','boundaries':boundaries.tolist(),
        'train_count':len(train),'validation_count':len(val),'test_used':False,
        'prepared_manifest_SHA256':sha(Path(root)/'prepared/manifest.json'),
        'split_identity':identity,
        'source_SHA256':initial_source_hashes,'epoch_logs':[], 'optimizer_steps':{'role':0,'task':0},
        'semantic_scope':'finite empirical linear target roles; no identified sources, known noise or PID claims',
        'evaluation_scope':'single fold/seed short engineering validation; no matched baseline or patient efficacy claim'}
    write(out/'receipt.json', receipt)
    rng = torch.Generator().manual_seed(c['seed']+9000)
    optim = torch.optim.Adam(list(model.encoders.parameters())+list(model.pools.parameters())+list(model.decoders.parameters()), lr=c['role_lr'])
    for phase in ('role','task'):
        if phase == 'task':
            model.freeze_roles()
            frozen = tensor_hash([(n,p) for n,p in model.named_parameters() if not n.startswith('readout.')])
            reference_before = tensor_hash(list(model.reference.named_buffers()))
            optim = torch.optim.Adam(model.readout.parameters(), lr=c['task_lr'])
        for epoch in range(c[phase+'_epochs']):
            model.train(); order = torch.randperm(len(train), generator=rng).tolist(); total = 0.
            for start in range(0,len(train),c['batch_size']):
                ids = [train[i] for i in order[start:start+c['batch_size']]]
                b = batch(store, ids, adapter, boundaries, c, device, generator=rng)
                output = model_forward(model,b)
                if phase == 'role':
                    target = model.reference.targets(b['p_target'],b['g_target'])
                    loss = sum(role_content_losses(output['roles'],target).values())/4
                else:
                    loss = discrete_hazard_nll(output['prediction'], b['completed'], b['event'])
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
                optim.zero_grad(set_to_none=True); loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('nonfinite gradient')
                optim.step(); total += float(loss.detach())*len(ids); receipt['optimizer_steps'][phase] += 1
            receipt['epoch_logs'].append({'phase':phase,'epoch':epoch+1,'train_loss':total/len(train)})
            write(out/'receipt.json',receipt)
            print(f'phase={phase} epoch={epoch+1} train_loss={total/len(train):.6f}',flush=True)
        if phase == 'role':
            torch.save({'state_dict':model.state_dict(),'adapter':adapter.state(),'config':c,
                        'boundaries':boundaries,'reference_metadata':reference.metadata},out/'role_final.pt')
    assert frozen == tensor_hash([(n,p) for n,p in model.named_parameters() if not n.startswith('readout.')])
    assert reference_before == tensor_hash(list(model.reference.named_buffers()))
    if sha(Path(root)/'prepared/manifest.json') != manifest_hash:
        raise RuntimeError('prepared manifest changed before evaluation')
    if split_guard is not None and split_guard() != identity:
        raise RuntimeError('split identity changed before evaluation')
    model.eval(); losses, logits, targets_mse = [], [], []
    eval_rng = torch.Generator().manual_seed(c['seed']+19000)
    with torch.no_grad():
        for start in range(0,len(val),c['batch_size']):
            ids=val[start:start+c['batch_size']]
            b=batch(store,ids,adapter,boundaries,c,device,generator=eval_rng)
            output=model_forward(model,b)
            losses.append(discrete_hazard_nll(output['prediction'],b['completed'],b['event'],reduction='none').cpu())
            logits.append(output['prediction'].cpu())
            targets_mse.append(torch.stack(list(role_content_losses(output['roles'],model.reference.targets(b['p_target'],b['g_target'])).values()))*len(ids))
    logits=torch.cat(logits); risks=-(1-logits.sigmoid()).cumprod(-1).sum(-1)
    from sksurv.metrics import concordance_index_censored
    import numpy as np
    try:
        concordance=float(concordance_index_censored(np.array([manifest['patients'][i]['censor']<1 for i in val],dtype=bool),
            np.array([manifest['patients'][i]['time'] for i in val]),risks.numpy())[0])
        if not math.isfinite(concordance): raise ValueError('nonfinite C-index')
        metric_status='computed'
    except ValueError as error:
        concordance=None; metric_status='undefined: '+str(error)
    checkpoint={'state_dict':model.state_dict(),'adapter':adapter.state(),'config':c,'boundaries':boundaries,
                'reference_metadata':reference.metadata}
    torch.save(checkpoint,out/'survival_final.pt')
    reloaded, reloaded_adapter, reloaded_boundaries=load_checkpoint(out/'survival_final.pt',device)
    assert tensor_hash(list(reloaded.named_parameters())) == tensor_hash(list(model.named_parameters()))
    assert torch.equal(boundaries,reloaded_boundaries)
    with torch.no_grad():
        assert torch.equal(model_forward(model,b)['prediction'],model_forward(reloaded,b)['prediction'])
    receipt.update(status='completed',roles_frozen_during_task=True,reference_frozen=True,
        validation={'NLL':float(torch.cat(losses).mean()),'C_index':concordance,'C_index_status':metric_status,
                    'role_target_MSE':(torch.stack(targets_mse).sum(0)/len(val)).tolist(),
                    'patient_performance_conclusion':False},checkpoint_reload_parameters_equal=True)
    # Per-patient arrays and identifiers remain exclusively under local_evidence.
    write(out/'validation_predictions.json',{'patient_ids':val,'logits':logits.tolist(),'risks':risks.tolist()})
    assert initial_source_hashes == {p:sha(ROOT/p) for p in code_hashes}
    receipt['artifacts_SHA256']={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='receipt.json'}
    write(out/'receipt.json',receipt)
    return {'status':'completed','output_root':str(out),'validation':receipt['validation'],
            'optimizer_steps':receipt['optimizer_steps'],'scope':receipt['evaluation_scope']}


def load_checkpoint(path, device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    state=saved['state_dict'];meta=saved['reference_metadata']
    reference=FrozenRoleReference(
        {v:state['reference.'+v+'_mean'] for v in ('p','g')},
        {v:state['reference.'+v+'_transform'] for v in ('p','g')},
        {v:state['reference.'+v+'_projector'] for v in ('p','g')},
        provenance=ReferenceProvenance(meta['fit_split'],meta['target_definition'],meta['source']),
        mode=meta['mode'],singular_values=state['reference.singular_values'])
    adapter=ContentAdapter.__new__(ContentAdapter)
    adapter.sizes=saved['adapter']['sizes'];adapter.stats=saved['adapter']['stats']
    anchors={}
    for v in ('p','g'):
        anchors[v+'_common'],anchors[v+'_residual']=state['readout.'+v+'_anchors'].split(reference.dimension(v),dim=1)
    model=SurvivalRoleModel(reference,adapter,anchors,saved['config']).double()
    model.load_state_dict(state);model.to(device);model.eval();model.requires_grad_(False)
    return model,adapter,saved['boundaries']


def synthetic_smoke(out):
    """Complete invented-cache round trip, NOT patient execution."""
    from .patient_pilot import config
    c=config(); c.update(patches=12,batch_size=4,target_dim=3,common_rank=1,width=8,hidden=6,
                         iterations=2,role_epochs=2,task_epochs=2,device='cpu',torch_threads=1)
    out=Path(out); out.mkdir(parents=True)
    prepared=out/'fake_prepared'; (prepared/'prepared/cache').mkdir(parents=True)
    rng=torch.Generator().manual_seed(908110)
    manifest={'omic_sizes':[4,5], 'patients':{},'folds':{'0':{'train':[],'val':[]}}}
    for i in range(32):
        pid=f'INVENTED_{i:03d}'; rel=f'prepared/cache/{pid}.pt'
        patches=torch.randn(5+i%15,6,generator=rng)
        omics=[torch.randn(n,generator=rng) for n in manifest['omic_sizes']]
        torch.save({'patient_id':pid,'patches':patches,'omics':omics},prepared/rel)
        manifest['patients'][pid]={'cache_path':rel,'cache_sha256':sha(prepared/rel),'time':float(i+1),'censor':float(i%3==0)}
        manifest['folds']['0']['train' if i<24 else 'val'].append(pid)
    write(prepared/'prepared/manifest.json',manifest)
    result=run_prepared(prepared,out/'run',manifest,c,{},synthetic=True)
    result.update(synthetic_only=True,patient_values_read=False,patient_training=0,server_execution=False)
    write(out/'synthetic_smoke_summary.json',result)
    return result
