"""Arm: the scaled 16-slot role architecture on the archived matched35 schedule.

This is a NEW module.  Nothing in the archived matched35 / spread sources is
edited, so the archived receipts keep their provenance.

What this arm changes relative to the archived spread arm, and nothing else
--------------------------------------------------------------------------
  1. slot allocation `{"p": [4, 4], "g": [4, 4]}` -> 16 slots in total
  2. slot width 256 (was 64)
  3. omics encoding by 329 INDEPENDENT pathway MLPs
  4. omics encoder input: raw per-pathway expression, standardized per
     pathway on train rows only
  5. reconstruction heads: DEFERRED by instruction; there is no reconstruction
     loss and no reconstruction head in this model
  6. batch size 8 (was 2)

Everything else is the archived protocol: the split, the bags, the 4096 real
patches, the without-replacement draw and the zero-padding rule, the role
target statistic, the train-fold SVD target reduction, the frozen role
reference, the bounded readout, the four-bin discrete hazard survival
objective, the optimiser, the learning rates, the epoch budget, the selection
rule and the evaluation.

Honest notes
------------
* The p ROLE TARGET statistic and the target width / common rank are
  configuration.  The measured probe chose `bagstd` with `target_dim 16` and
  `common_rank 6` (fold-mean 0.7071, a LINEAR PROBE, not a trained score).
* The reconstruction head is absent, so this arm is NOT comparable with the
  original SlotSPE reconstruction-aware baseline without saying so.
* More slots of the same kind raise capacity AND raise the risk of slot
  redundancy.  The redundancy must be measured with the existing slot relation
  tables, not assumed away.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..components.losses import discrete_hazard_nll
from ..components.role_model import BoundedLinear
from ..components.role_reference import (
    FrozenRoleReference, ReferenceProvenance, role_content_losses,
)
from ..components.spread_target import bag_spread
from ..full_slot_diagnostics import (
    ablate_each_slot, read_diagnostics_summary, read_mass_summary, role_summary,
    slot_summary,
)
from ..patient_pilot import ROOT, sha
from ..patient_pilot_engine import (
    PreparedStore, fit_boundaries, label_times, tensor_hash,
)
from .adapter import ScaledContentAdapter, fit_pathway_moments
from .role_model import ScaledRoleTargetModel


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_receipt(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def _save_checkpoint(path, checkpoint):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def validate_protocol(c):
    """Reject any drift from the locked scaled protocol.

    The protocol is locked because the architecture is being scaled on
    purpose: a silent change of the slot budget, the width, the omics encoder
    or the batch size would make the run uninterpretable.
    """
    expected = dict(schema='scaled_role16_v1', study='blca', fold=0, seed=3,
                    patches=4096, batch_size=8,
                    allocation={'p': [4, 4], 'g': [4, 4]}, width=256, iterations=3,
                    hidden=32, bins=4, role_epochs=12, task_epochs=35, patience=35,
                    role_lr=.003, task_lr=.001, role_weight=1., task_mode='joint',
                    readout_bound=2., fixed_task_budget=True, read_diagnostics=True,
                    ablation_nll=True, torch_threads=4, device='cuda:0',
                    target_dim=16, common_rank=6, p_target_mode='spread',
                    omics_dropout=0.25, reconstruction_heads='deferred')
    for key, value in expected.items():
        if key == 'device':
            # The protocol pins the device family, not the physical index, so
            # that two arms may run concurrently on two GPUs without any other
            # parameter changing.
            if c.get(key) not in ('cuda:0', 'cuda:1'):
                raise ValueError('the scaled protocol requires a cuda device')
            continue
        if c.get(key) != value or isinstance(value, (int, bool)) and type(c[key]) is not type(value):
            raise ValueError('locked scaled protocol differs: ' + key)
    if c['common_rank'] >= c['target_dim']:
        raise ValueError('a nonempty residual target requires common_rank < target_dim')
    if c['task_mode'] not in ('joint', 'continued_role_only'):
        raise ValueError('invalid task mode')
    if c['p_target_mode'] not in ('mean', 'spread'):
        raise ValueError('p_target_mode must be mean or spread')
    for modality in ('p', 'g'):
        if c['allocation'][modality] != [4, 4]:
            raise ValueError('the scaled protocol fixes 4 common and 4 residual slots per modality')
    if sum(c['allocation'][m][0] + c['allocation'][m][1] for m in ('p', 'g')) != 16:
        raise ValueError('the scaled protocol fixes 16 slots in total')


class ScaledSurvivalRoleModel(ScaledRoleTargetModel):
    """The scaled role model with the archived hazard-logit readout heads."""

    def __init__(self, reference, adapter, anchors, c):
        super().__init__(reference, p_input_dim=len(adapter.stats['p']['mean']),
                         omic_sizes=adapter.sizes,
                         p_allocation=_allocation(c['allocation']['p']),
                         g_allocation=_allocation(c['allocation']['g']),
                         width=c['width'], iterations=c['iterations'],
                         readout_hidden=c['hidden'], readout_bound=c['readout_bound'],
                         interaction=True, train_role_anchors=anchors,
                         omics_dropout=c.get('omics_dropout', 0.25))
        for name in ('p_head', 'g_head'):
            getattr(self.readout, name)[2] = BoundedLinear(c['hidden'], c['bins'], c['readout_bound'])
        self.readout.cross = BoundedLinear(c['hidden'], c['bins'], c['readout_bound'])


def _allocation(pair):
    from ..components.role_model import RoleAllocation
    common, residual = pair
    return RoleAllocation(int(common), int(residual))


def collect_target_matrices(store, ids, p_target_mode):
    """Per-patient target matrix, the shared mean matrix and the omics matrix.

    The mean matrix is always returned because its moments normalize the p
    ENCODER INPUT, in every arm.
    """
    if p_target_mode not in ('mean', 'spread'):
        raise ValueError('p_target_mode must be mean or spread')
    targets, means, omics = [], [], []
    for pid in ids:
        patches, blocks = store.load(pid)
        means.append(patches.mean(0))
        if p_target_mode == 'mean':
            targets.append(patches.mean(0))
        else:
            targets.append(bag_spread(patches))
        omics.append(torch.cat(blocks))
    return torch.stack(targets), torch.stack(means), torch.stack(omics)


def scaled_batch(store, ids, adapter, boundaries, c, device, *, generator):
    """The archived batch, with the scaled omics encoder input.

    The sampling rule, the without-replacement draw, the zero-padding rule and
    the mask are the archived ones: every real patch is cached, a fresh subset
    is drawn without replacement on each access, and bags shorter than the
    budget are zero-padded rather than excluded.
    """
    ps, gs, pm, pt, gt, times, events = [], [], [], [], [], [], []
    per_patient_omics = []
    for pid in ids:
        patches, omics = store.load(pid)
        pt.append(adapter.p_target(patches))
        gt.append(adapter.target(torch.cat(omics), 'g'))
        count = min(len(patches), c['patches'])
        indices = torch.randperm(len(patches), generator=generator)[:count].sort().values
        p = adapter.patches_input(patches[indices])
        g = adapter.omics_input(omics)
        padded = torch.zeros(c['patches'], p.shape[1], dtype=p.dtype)
        padded[:count] = p
        ps.append(padded)
        per_patient_omics.append(g)
        pm.append(torch.arange(c['patches']) < count)
        row = store.manifest['patients'][pid]
        times.append(row['time'])
        events.append(row['censor'] < 1)
    # The pathways own DIFFERENT gene counts, so the omics batch cannot be one
    # rectangular tensor.  It is a LIST of per-pathway [patients, genes_i]
    # tensors, which is exactly what the 329 independent pathway MLPs consume.
    gs = [torch.stack([patient_omics[i] for patient_omics in per_patient_omics])
          for i in range(len(adapter.sizes))]
    times = torch.tensor(times, dtype=torch.float64)
    events = torch.tensor(events, dtype=torch.bool)
    result = {'p': torch.stack(ps), 'g': gs, 'p_mask': torch.stack(pm),
              'g_mask': torch.ones(len(ids), len(adapter.sizes), dtype=torch.bool),
              'p_target': torch.stack(pt), 'g_target': torch.stack(gt), 'time': times,
              'event': events, 'completed': label_times(times, events, boundaries)}
    return {k: (v if isinstance(v, list) else v.to(device)) for k, v in result.items()}


def scaled_batch_hash(batch):
    """Hash a scaled batch, including the LIST of per-pathway omics tensors."""
    h = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        if isinstance(value, list):
            h.update(key.encode())
            for index, item in enumerate(value):
                h.update(('%d:' % index).encode())
                h.update(item.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def scaled_evaluate_epoch(model, store, ids, adapter, boundaries, config, device, *, seed,
                          survival):
    """The archived evaluation, using the scaled batch."""
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('nonempty unique ordered validation ids required')
    model.eval()
    observe = bool(config.get('read_diagnostics', False))
    with_nll = bool(config.get('ablation_nll', observe))
    for encoder in model.encoders.values():
        encoder.observe_read_diagnostics = observe
    rng = torch.Generator().manual_seed(seed)
    roles, slots, attention, ablations, read_mass = {}, {}, {}, {}, {}
    losses, logits = [], []
    read_observations, deleted_nll, delta_nll = {}, {}, {}
    role_sums = {}
    for start in range(0, len(ids), config['batch_size']):
        batch_ids = ids[start:start + config['batch_size']]
        b = scaled_batch(store, batch_ids, adapter, boundaries, config, device, generator=rng)
        out = model(b['p'], b['g'], b['p_mask'], b['g_mask'])
        target = model.reference.targets(b['p_target'], b['g_target'])
        for key, value in role_content_losses(out['roles'], target).items():
            role_sums[key] = role_sums.get(key, 0.) + float(value) * len(batch_ids)
        if survival:
            losses.append(discrete_hazard_nll(out['prediction'], b['completed'], b['event'],
                                              reduction='none').cpu())
            logits.append(out['prediction'].cpu())
            if with_nll:
                ablated = ablate_each_slot(model, out, b['completed'], b['event'])
                for destination, metric in ((deleted_nll, 'NLL'), (delta_nll, 'delta_NLL')):
                    for key, values in ablated[metric].items():
                        previous = destination.setdefault(key, [0.] * len(values))
                        destination[key] = [old + new * len(batch_ids)
                                            for old, new in zip(previous, values)]
                changes = ablated['mean_abs_logit_change']
            else:
                changes = ablate_each_slot(model, out)
            for key, values in changes.items():
                if key not in ablations:
                    ablations[key] = [0.] * len(values)
                ablations[key] = [old + new * len(batch_ids)
                                  for old, new in zip(ablations[key], values)]
        for modality in ('p', 'g'):
            part = out['parts'][modality]
            cut = model.allocations[modality].common
            for role, part_key, section in (('common', 'common_slots', slice(None, cut)),
                                            ('residual', 'private_slots', slice(cut, None))):
                key = modality + '_' + role
                roles.setdefault(key, []).append(out['roles'][key].cpu())
                slots.setdefault(key, []).append(part[part_key].cpu())
                attention.setdefault(key, []).append(part['read_attention'][:, section].cpu())
                read_mass.setdefault(key, []).append(part['raw_read_mass'][:, :, section].cpu())
                if observe:
                    destination = read_observations.setdefault(key, {})
                    for name, value in part['read_diagnostics'].items():
                        destination.setdefault(name, []).append(value[:, :, section].cpu())
    result = {
        'role_target_MSE': {k: v / len(ids) for k, v in role_sums.items()},
        'roles': {k: role_summary(torch.cat(v)) for k, v in roles.items()},
        'slots': {k: slot_summary(torch.cat(v), torch.cat(attention[k])) for k, v in slots.items()},
        'raw_reading_mass': {k: read_mass_summary(torch.cat(v)) for k, v in read_mass.items()},
        'validation_sampling_seed': seed,
        'ablation_mean_abs_logit_change': {k: [v / len(ids) for v in values]
                                            for k, values in ablations.items()},
        'ablation_scope': 'slot removal in role pool; not conditional CPI',
    }
    if observe:
        result['read_diagnostics'] = {
            role: read_diagnostics_summary({name: torch.cat(values) for name, values in measurements.items()})
            for role, measurements in read_observations.items()}
    if with_nll:
        result['ablation_NLL'] = {k: [v / len(ids) for v in values]
                                  for k, values in deleted_nll.items()}
        result['ablation_delta_NLL'] = {k: [v / len(ids) for v in values]
                                        for k, values in delta_nll.items()}
        result['ablation_delta_NLL_definition'] = ('deleted NLL minus full NLL; positive means '
                                                   'deletion worsens fitted-model NLL')
    if survival:
        from sksurv.metrics import concordance_index_censored
        logits = torch.cat(logits)
        risks = -(1 - logits.sigmoid()).cumprod(-1).sum(-1).numpy()
        events = np.array([store.manifest['patients'][i]['censor'] < 1 for i in ids], dtype=bool)
        times = np.array([store.manifest['patients'][i]['time'] for i in ids])
        try:
            cindex = float(concordance_index_censored(events, times, risks)[0])
            if not math.isfinite(cindex):
                raise ValueError('nonfinite C-index')
            status = 'computed'
        except ValueError as error:
            cindex, status = None, 'undefined: ' + str(error)
        result.update(NLL=float(torch.cat(losses).mean()), C_index=cindex, C_index_status=status)
    return result


def load_scaled_checkpoint(path, device='cpu'):
    """Reload a scaled-arm checkpoint into the scaled model class."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = saved['state_dict']
    meta = saved['reference_metadata']
    reference = FrozenRoleReference(
        {v: state['reference.' + v + '_mean'] for v in ('p', 'g')},
        {v: state['reference.' + v + '_transform'] for v in ('p', 'g')},
        {v: state['reference.' + v + '_projector'] for v in ('p', 'g')},
        provenance=ReferenceProvenance(meta['fit_split'], meta['target_definition'], meta['source']),
        mode=meta['mode'], singular_values=state['reference.singular_values'])
    adapter = ScaledContentAdapter.__new__(ScaledContentAdapter)
    adapter.sizes = saved['adapter']['sizes']
    adapter.stats = saved['adapter']['stats']
    adapter.p_target_mode = saved['adapter']['p_target_mode']
    adapter.pathway_mean = saved['adapter']['pathway_mean']
    adapter.pathway_sd = saved['adapter']['pathway_sd']
    anchors = {}
    for v in ('p', 'g'):
        anchors[v + '_common'], anchors[v + '_residual'] = state['readout.' + v + '_anchors'].split(
            reference.dimension(v), dim=1)
    model = ScaledSurvivalRoleModel(reference, adapter, anchors, saved['config']).double()
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, adapter, saved['boundaries']


def run_scaled_arm(root, out, manifest, c, source_hashes, *, split_guard=None, synthetic=False):
    """Train the scaled 16-slot role architecture on the archived schedule."""
    if not synthetic and split_guard is None:
        raise ValueError('verified patient split admission required')
    root, out = Path(root), Path(out)
    if out.exists():
        raise FileExistsError('fresh run required')
    if source_hashes != {name: sha(ROOT / name) for name in source_hashes}:
        raise RuntimeError('source changed before execution')
    identity = split_guard() if split_guard else {'schema': 'invented_smoke_only'}
    manifest_hash = sha(root / 'prepared/manifest.json')
    train, val = (manifest['folds'][str(c['fold'])][key] for key in ('train', 'val'))
    if not train or not val or len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val):
        raise ValueError('nonempty unique disjoint train and validation required')
    torch.set_num_threads(c['torch_threads'])
    torch.manual_seed(c['seed'])
    device = torch.device(c['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')

    store = PreparedStore(root, manifest)
    p_target_mode = c['p_target_mode']
    target_matrix, mean_matrix, g_matrix = collect_target_matrices(store, train, p_target_mode)
    input_mean = mean_matrix.mean(0)
    input_sd = mean_matrix.std(0, unbiased=False).clamp_min(1e-4)
    pathway_mean, pathway_sd = fit_pathway_moments(store, train, manifest['omic_sizes'])
    adapter = ScaledContentAdapter(target_matrix, g_matrix, manifest['omic_sizes'],
                                   c['target_dim'], input_mean=input_mean, input_sd=input_sd,
                                   pathway_mean=pathway_mean, pathway_sd=pathway_sd,
                                   p_target_mode=p_target_mode)
    tp, tg = adapter.target(target_matrix, 'p'), adapter.target(g_matrix, 'g')
    target_definition = ('standardized PCA patch-spread and pathway-expression targets'
                         if p_target_mode == 'spread' else
                         'standardized PCA patch-mean and pathway-expression targets')
    reference = FrozenRoleReference.estimated(
        tp, tg, common_rank=c['common_rank'], coordinate_system='target',
        provenance=ReferenceProvenance(
            'train', target_definition,
            'training fold only; empirical roles, no biological source certification'))
    anchors = reference.targets(tp[:64], tg[:64])
    times = torch.tensor([manifest['patients'][i]['time'] for i in train], dtype=torch.float64)
    events = torch.tensor([manifest['patients'][i]['censor'] < 1 for i in train], dtype=torch.bool)
    boundaries = fit_boundaries(times, events, c['bins'])
    model = ScaledSurvivalRoleModel(reference, adapter, anchors, c).double().to(device)

    out.mkdir(parents=True)
    omics_encoder_parameters = model.omics_encoder.parameter_count()
    receipt = {'schema': 'scaled_role16_v1', 'status': 'running', 'config': c,
               'started_at': _now(), 'source_SHA256': source_hashes,
               'split_identity': identity, 'prepared_manifest_SHA256': manifest_hash,
               'train_count': len(train), 'validation_count': len(val), 'test_used': False,
               'model_parameters': sum(x.numel() for x in model.parameters()),
               'omics_encoder_parameters': omics_encoder_parameters,
               'omics_encoder_independent_networks': model.omics_encoder.num_pathways,
               'slot_budget': model.slot_budget(),
               'total_slot_count': model.total_slot_count(),
               'slot_width': model.width,
               'reconstruction_heads': 'deferred by instruction; no reconstruction loss',
               'role_target_statistic': p_target_mode,
               'role_target_statistic_definition': (
                   'per-feature standard deviation of the patient\'s real patch bag, '
                   'n-1 convention, over all cached real patches'),
               'omics_encoder_input': ('raw per-pathway expression, standardized per pathway '
                                       'by train-fold moments, then 329 independent pathway MLPs'),
               'mechanism': ('the slot budget is 4 common + 4 residual slots per modality at '
                             'width 256, and the omics side enters through 329 independent '
                             'pathway MLPs instead of a shared SVD adapter'),
               'survival_objective': 'archived four-bin discrete hazard NLL',
               'difference_from_archived_spread_arm': ('the slot budget, the slot width, the omics '
                                                       'encoder and the batch size; the split, the bags, '
                                                       'the patch budget, the omics target, the reference, '
                                                       'the readout, the optimiser, the learning rates, '
                                                       'the epoch budget and the selection rule are the '
                                                       'archived ones'),
               'selection': 'maximum C-index, exact tie lower NLL, then earlier epoch',
               'epoch_logs': [], 'best': None, 'last': None,
               'optimizer_steps': {'role': 0, 'task': 0},
               'scope': 'development validation only; no independent test or causal interpretation'}
    _write_receipt(out / 'receipt.json', receipt)

    rng = torch.Generator().manual_seed(c['seed'] + 9000)
    eval_seed = c['seed'] + 19000
    reference_hash = tensor_hash(list(model.reference.named_buffers()))
    optim = torch.optim.Adam(list(model.encoders.parameters()) + list(model.pools.parameters()) +
                             list(model.decoders.parameters()) + list(model.omics_encoder.parameters()),
                             lr=c['role_lr'])
    best, stale = None, 0
    try:
        for phase in ('role', 'task'):
            if phase == 'task':
                optim = torch.optim.Adam(model.parameters(), lr=c['task_lr'])
                receipt['task_optimizer_initial_state_empty'] = not bool(optim.state)
                receipt['task_optimizer_initial_groups'] = optim.state_dict()['param_groups']
            for epoch in range(1, c[phase + '_epochs'] + 1):
                model.train()
                order = torch.randperm(len(train), generator=rng).tolist()
                total, role_total, task_total = 0., 0., 0.
                draw_hash = hashlib.sha256()
                draw_hash.update(bytes(rng.get_state().tolist()))
                draw_hash.update(str(order).encode())
                for start in range(0, len(train), c['batch_size']):
                    ids = [train[i] for i in order[start:start + c['batch_size']]]
                    b = scaled_batch(store, ids, adapter, boundaries, c, device, generator=rng)
                    draw_hash.update(scaled_batch_hash(b).encode())
                    output = model(b['p'], b['g'], b['p_mask'], b['g_mask'])
                    role_loss = sum(role_content_losses(
                        output['roles'], model.reference.targets(b['p_target'], b['g_target'])).values()) / 4
                    if phase == 'role':
                        loss = role_loss
                    else:
                        task_loss = discrete_hazard_nll(output['prediction'], b['completed'], b['event'])
                        loss = task_loss + c['role_weight'] * role_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite training loss')
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError('nonfinite gradient')
                    optim.step()
                    receipt['optimizer_steps'][phase] += 1
                    total += float(loss.detach()) * len(ids)
                    role_total += float(role_loss.detach()) * len(ids)
                    if phase == 'task':
                        task_total += float(task_loss.detach()) * len(ids)
                if sha(root / 'prepared/manifest.json') != manifest_hash or (split_guard and split_guard() != identity):
                    raise RuntimeError('split identity or manifest changed before evaluation')
                metrics = scaled_evaluate_epoch(model, store, val, adapter, boundaries, c, device,
                                                 seed=eval_seed, survival=phase == 'task')
                log = {'phase': phase, 'epoch': epoch, 'completed_at': _now(),
                       'train_loss': total / len(train),
                       'train_role_loss': role_total / len(train),
                       'train_survival_NLL': task_total / len(train) if phase == 'task' else None,
                       'validation': metrics,
                       'input_draw_SHA256': draw_hash.hexdigest()}
                receipt['epoch_logs'].append(log)
                checkpoint = {'state_dict': copy.deepcopy(model.state_dict()),
                              'adapter': adapter.state(), 'config': c,
                              'boundaries': boundaries, 'reference_metadata': reference.metadata,
                              'optimizer_state': optim.state_dict(), 'phase': phase, 'epoch': epoch,
                              'train_generator_state': rng.get_state(),
                              'torch_rng_state': torch.get_rng_state(),
                              'cuda_rng_state': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                              'split_identity': identity, 'manifest_SHA256': manifest_hash,
                              'source_SHA256': source_hashes}
                improved = False
                if phase == 'task':
                    row = {'epoch': epoch, 'C_index': metrics['C_index'], 'NLL': metrics['NLL']}
                    key = lambda r: (r['C_index'], -r['NLL'], -r['epoch'])
                    if best is None or key(row) > key(best):
                        best, stale = row, 0
                        improved = True
                    else:
                        stale += 1
                    receipt['best'], receipt['last'] = best, row
                    checkpoint['selection_state'] = {'best': best, 'stale_epochs': stale}
                _save_checkpoint(out / (phase + '_last.pt'), checkpoint)
                if improved:
                    _save_checkpoint(out / 'task_best.pt', checkpoint)
                _write_receipt(out / 'receipt.json', receipt)
                print(f"phase={phase} epoch={epoch} train_loss={total/len(train):.6f} "
                      f"cindex={metrics.get('C_index')}", flush=True)
        if reference_hash != tensor_hash(list(model.reference.named_buffers())):
            raise RuntimeError('training reference changed')
        if source_hashes != {name: sha(ROOT / name) for name in source_hashes}:
            raise RuntimeError('source changed during execution')
        if sha(root / 'prepared/manifest.json') != manifest_hash or (split_guard and split_guard() != identity):
            raise RuntimeError('split identity or manifest changed before checkpoint verification')
        for label in ('best', 'last'):
            reloaded, reloaded_adapter, reloaded_boundaries = load_scaled_checkpoint(
                out / ('task_' + label + '.pt'), device)
            verified = scaled_evaluate_epoch(reloaded, store, val, reloaded_adapter,
                                             reloaded_boundaries, c, device, seed=eval_seed,
                                             survival=True)
            expected = receipt[label]
            if verified['C_index'] != expected['C_index'] or not math.isclose(
                    verified['NLL'], expected['NLL'], rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(label + ' checkpoint evaluation mismatch')
        receipt['checkpoint_reload_verified'] = True
        receipt['reference_frozen'] = True
        receipt['artifacts_SHA256'] = {p.name: sha(p) for p in out.iterdir()
                                       if p.is_file() and p.name != 'receipt.json'}
        receipt['status'] = 'completed'
        receipt['completed_at'] = _now()
        _write_receipt(out / 'receipt.json', receipt)
        return receipt
    except Exception as error:
        receipt.update(status='failed', failed_at=_now(),
                       error=type(error).__name__ + ': ' + str(error))
        _write_receipt(out / 'receipt.json', receipt)
        raise
