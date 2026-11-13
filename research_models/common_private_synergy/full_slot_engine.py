"""Finite-budget empirical-role training; no execution occurs on import.

The caller is responsible for patient-cache verification and scope admission.
This implementation is experimental and must pass synthetic integration checks
before any patient use. Checkpoints and receipts belong under local_evidence.
"""
import copy
from datetime import datetime, timezone
import json
import math
import platform
from pathlib import Path

import torch

from .components.losses import discrete_hazard_nll
from .components.role_reference import FrozenRoleReference, ReferenceProvenance, role_content_losses
from .full_slot_diagnostics import better_validation, read_mass_summary
from .components.slots import READ_EPS, READ_NORMALIZATION
from .full_slot_epoch import evaluate_epoch
from .patient_pilot import ROOT, sha
from .patient_pilot_engine import (
    ContentAdapter, PreparedStore, SurvivalRoleModel, batch, collect_targets,
    fit_boundaries, load_checkpoint, model_forward, tensor_hash,
)


def _write_receipt(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _save_checkpoint(path, checkpoint):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _now():
    return datetime.now(timezone.utc).isoformat()


def run_full(root, out, manifest, c, source_hashes, *, split_guard=None, synthetic=False):
    if not synthetic and split_guard is None:
        raise ValueError('verified patient split admission required')
    if c['task_mode'] not in ('frozen', 'joint', 'continued_role_only'):
        raise ValueError('task_mode must be frozen, joint or continued_role_only')
    if c['role_weight'] <= 0 or c['patience'] < 1:
        raise ValueError('positive role weight and patience required')
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
    p, g = collect_targets(store, train)
    adapter = ContentAdapter(p, g, manifest['omic_sizes'], c['target_dim'])
    tp, tg = adapter.target(p, 'p'), adapter.target(g, 'g')
    reference = FrozenRoleReference.estimated(tp, tg, common_rank=c['common_rank'], coordinate_system='target',
        provenance=ReferenceProvenance('train', 'standardized PCA patch-mean and pathway-expression targets',
                                       'training fold only; empirical roles, no biological source certification'))
    anchors = reference.targets(tp[:64], tg[:64])
    times = torch.tensor([manifest['patients'][i]['time'] for i in train], dtype=torch.float64)
    events = torch.tensor([manifest['patients'][i]['censor'] < 1 for i in train], dtype=torch.bool)
    boundaries = fit_boundaries(times, events, c['bins'])
    model = SurvivalRoleModel(reference, adapter, anchors, c).double().to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    out.mkdir(parents=True)
    receipt = {'schema': 'full_slot_research_v1', 'status': 'running', 'config': c,
               'started_at': _now(),
               'source_SHA256': source_hashes, 'split_identity': identity,
               'prepared_manifest_SHA256': manifest_hash, 'train_count': len(train), 'validation_count': len(val),
               'test_used': False, 'model_parameters': sum(x.numel() for x in model.parameters()),
               'selection': 'maximum C-index, exact tie lower NLL, then earlier epoch',
               'read_normalization': {'policy': READ_NORMALIZATION, 'epsilon': READ_EPS, 'epsilon_on': 'real_tokens_only'},
               'epoch_logs': [], 'best': None, 'last': None,
               'optimizer_steps': {'role': 0, 'task': 0},
               'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                               'cuda_runtime': torch.version.cuda,
                               'device': str(device),
                               'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'},
               'scope': 'development validation only; no independent test or causal interpretation'}
    _write_receipt(out / 'receipt.json', receipt)
    rng = torch.Generator().manual_seed(c['seed'] + 9000)
    eval_seed = c['seed'] + 19000
    reference_hash = tensor_hash(list(model.reference.named_buffers()))
    optim = torch.optim.Adam(list(model.encoders.parameters()) + list(model.pools.parameters()) +
                             list(model.decoders.parameters()), lr=c['role_lr'])
    best, stale = None, 0
    try:
        for phase in ('role', 'task'):
            if phase == 'task':
                role_before = tensor_hash([(n, p) for n, p in model.named_parameters() if n.startswith(('encoders.', 'pools.', 'decoders.'))])
                if c['task_mode'] == 'frozen':
                    model.freeze_roles()
                    params = model.readout.parameters()
                elif c['task_mode'] == 'continued_role_only':
                    # Role slots keep fitting; the readout is held fixed.
                    model.freeze_readout()
                    params = [parameter for name, parameter in model.named_parameters()
                              if name.startswith(('encoders.', 'pools.', 'decoders.'))]
                else:
                    params = model.parameters()
                optim = torch.optim.Adam(params, lr=c['task_lr'])
            for epoch in range(1, c[phase + '_epochs'] + 1):
                model.train()
                order = torch.randperm(len(train), generator=rng).tolist()
                total, role_total, task_total = 0., 0., 0.
                grads = {name: {'p': None, 'g': None} for name in ('combined', 'survival', 'role')}
                train_mass = {'p': [], 'g': []}
                for start in range(0, len(train), c['batch_size']):
                    ids = [train[i] for i in order[start:start + c['batch_size']]]
                    b = batch(store, ids, adapter, boundaries, c, device, generator=rng)
                    output = model_forward(model, b)
                    for modality in ('p', 'g'):
                        train_mass[modality].append(output['parts'][modality]['raw_read_mass'].cpu())
                    for modality in ('p', 'g'):
                        raw = output['parts'][modality]['slots']
                        if raw.requires_grad:
                            raw.retain_grad()
                    role_loss = sum(role_content_losses(output['roles'], model.reference.targets(b['p_target'], b['g_target'])).values()) / 4
                    task_loss = None if phase == 'role' else discrete_hazard_nll(output['prediction'], b['completed'], b['event'])
                    loss = role_loss if phase == 'role' else task_loss
                    if phase == 'task' and c['task_mode'] == 'joint':
                        loss = loss + c['role_weight'] * role_loss
                    if phase == 'task' and c['task_mode'] == 'continued_role_only':
                        # Role slots keep fitting the role objective at the
                        # same role_weight and task_lr used by the joint arm.
                        # The readout is frozen and the task gradient is
                        # measured but not applied, so the difference between
                        # this arm's applied slot gradient and the joint arm's
                        # isolates the task gradient contribution.
                        loss = c['role_weight'] * role_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite training loss')
                    optim.zero_grad(set_to_none=True)
                    raw_slots = [output['parts'][m]['slots'] for m in ('p', 'g')]
                    if all(value.requires_grad for value in raw_slots):
                        components = {'role': role_loss}
                        if task_loss is not None and c['task_mode'] in ('joint', 'continued_role_only'):
                            components['survival'] = task_loss
                        for name, component in components.items():
                            for modality, grad in zip(('p', 'g'), torch.autograd.grad(component, raw_slots, retain_graph=True, allow_unused=True)):
                                if grad is not None:
                                    value = grad.detach().norm(dim=-1).sum(dim=0).cpu() * len(ids)
                                    previous = grads[name][modality]
                                    grads[name][modality] = value if previous is None else previous + value
                    loss.backward()
                    if any(x.grad is not None and not torch.isfinite(x.grad).all() for x in model.parameters()):
                        raise FloatingPointError('nonfinite gradient')
                    for modality in ('p', 'g'):
                        grad = output['parts'][modality]['slots'].grad
                        if grad is not None:
                            value = grad.detach().norm(dim=-1).sum(dim=0).cpu() * len(ids)
                            previous = grads['combined'][modality]
                            grads['combined'][modality] = value if previous is None else previous + value
                    optim.step()
                    receipt['optimizer_steps'][phase] += 1
                    total += float(loss.detach()) * len(ids)
                    role_total += float(role_loss.detach()) * len(ids)
                    if task_loss is not None:
                        task_total += float(task_loss.detach()) * len(ids)
                if sha(root / 'prepared/manifest.json') != manifest_hash or (split_guard and split_guard() != identity):
                    raise RuntimeError('split identity or manifest changed before evaluation')
                metrics = evaluate_epoch(model, store, val, adapter, boundaries, c, device,
                                         seed=eval_seed, survival=phase == 'task')
                log = {'phase': phase, 'epoch': epoch, 'completed_at': _now(), 'train_loss': total / len(train), 'validation': metrics,
                       'train_role_loss': role_total / len(train),
                       'train_survival_NLL': task_total / len(train) if phase == 'task' else None,
                       'train_raw_reading_mass': {k: read_mass_summary(torch.cat(v)) for k, v in train_mass.items()},
                       'raw_slot_gradient_mean': {name: {k: (v / len(train)).tolist() if v is not None else None
                                                        for k, v in values.items()} for name, values in grads.items()},
                       'gradient_scope': 'mean raw-slot gradient norms for batch-mean losses; role term shown unweighted'}
                receipt['epoch_logs'].append(log)
                checkpoint = {'state_dict': copy.deepcopy(model.state_dict()), 'adapter': adapter.state(), 'config': c,
                              'boundaries': boundaries, 'reference_metadata': reference.metadata,
                              'optimizer_state': optim.state_dict(), 'phase': phase, 'epoch': epoch,
                              'train_generator_state': rng.get_state(), 'torch_rng_state': torch.get_rng_state(),
                              'cuda_rng_state': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                              'split_identity': identity, 'manifest_SHA256': manifest_hash,
                              'source_SHA256': source_hashes}
                improved = False
                if phase == 'task':
                    row = {'epoch': epoch, 'C_index': metrics['C_index'], 'NLL': metrics['NLL']}
                    if better_validation(row, best):
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
                print(f'phase={phase} epoch={epoch} train_loss={total/len(train):.6f}', flush=True)
                if phase == 'task' and stale >= c['patience']:
                    receipt['early_stop'] = 'patience'
                    break
        if reference_hash != tensor_hash(list(model.reference.named_buffers())):
            raise RuntimeError('training reference changed')
        role_after = tensor_hash([(n, p) for n, p in model.named_parameters() if n.startswith(('encoders.', 'pools.', 'decoders.'))])
        receipt['role_parameters_changed_during_task'] = role_before != role_after
        if c['task_mode'] == 'frozen' and role_before != role_after:
            raise RuntimeError('frozen role parameters changed during task')
        if c['task_mode'] == 'continued_role_only' and role_before == role_after:
            raise RuntimeError('continued_role_only role parameters did not change during task')
        if sha(root / 'prepared/manifest.json') != manifest_hash or (split_guard and split_guard() != identity):
            raise RuntimeError('split identity or manifest changed before checkpoint verification')
        if source_hashes != {name: sha(ROOT / name) for name in source_hashes}:
            raise RuntimeError('source changed during execution')
        # Reload both full model snapshots and verify they reproduce evaluation.
        for label in ('best', 'last'):
            reloaded, reloaded_adapter, reloaded_boundaries = load_checkpoint(out / ('task_' + label + '.pt'), device)
            verified = evaluate_epoch(reloaded, store, val, reloaded_adapter, reloaded_boundaries, c, device,
                                      seed=eval_seed, survival=True)
            expected = receipt[label]
            if verified['C_index'] != expected['C_index'] or not math.isclose(verified['NLL'], expected['NLL'], rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(label + ' checkpoint evaluation mismatch')
            saved = torch.load(out / ('task_' + label + '.pt'), map_location='cpu', weights_only=True)
            if c['task_mode'] == 'joint':
                reloaded.requires_grad_(True)
                parameters = reloaded.parameters()
            elif c['task_mode'] == 'continued_role_only':
                reloaded.requires_grad_(True)
                reloaded.readout.requires_grad_(False)
                parameters = [parameter for name, parameter in reloaded.named_parameters()
                              if name.startswith(('encoders.', 'pools.', 'decoders.'))]
            else:
                reloaded.readout.requires_grad_(True)
                parameters = reloaded.readout.parameters()
            restored_optimizer = torch.optim.Adam(parameters, lr=c['task_lr'])
            restored_optimizer.load_state_dict(saved['optimizer_state'])
            restored_generator = torch.Generator()
            restored_generator.set_state(saved['train_generator_state'])
            if not torch.equal(restored_generator.get_state(), saved['train_generator_state']):
                raise RuntimeError(label + ' generator state reload mismatch')
            if restored_optimizer.state_dict()['param_groups'] != saved['optimizer_state']['param_groups']:
                raise RuntimeError(label + ' optimizer state reload mismatch')
        receipt['checkpoint_reload_verified'] = True
        receipt['checkpoint_optimizer_and_generator_reload_verified'] = True
        receipt['reference_frozen'] = True
        if device.type == 'cuda':
            receipt['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        receipt['artifacts_SHA256'] = {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'receipt.json'}
        receipt['status'] = 'completed'
        receipt['completed_at'] = _now()
        _write_receipt(out / 'receipt.json', receipt)
        return receipt
    except Exception as error:
        receipt['status'] = 'failed'
        receipt['failed_at'] = _now()
        receipt['error'] = type(error).__name__ + ': ' + str(error)
        _write_receipt(out / 'receipt.json', receipt)
        raise
