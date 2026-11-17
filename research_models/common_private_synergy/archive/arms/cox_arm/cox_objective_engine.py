"""Cox-objective arm: one common start, two survival objectives.

Added as a NEW module. The archived matched35 sources stay byte for byte so
that the historical receipts keep their provenance and their hashes.

What this protocol changes relative to the archived matched35 protocol, and
nothing else:

  * the survival objective is a Cox partial likelihood over a batch of
    patients instead of a four-bin discrete hazard negative log-likelihood;
  * the survival batch is 32 patients instead of 2, because a partial
    likelihood over two patients is uninformative and, on this training fold,
    43 percent of the two-patient batches contain no observed event at all;
  * the role warmup is reused from the archived matched35 shared_role
    checkpoint rather than rerun, so the arms below start from exactly the
    role representation that the archived run started from.

What this protocol keeps identical to the archived matched35 protocol:

  * the split identity, the cohort, the 4096 real patches per bag, the
    sampling rule, the zero padding rule, the role targets, the frozen role
    reference, the slot allocation, the encoder and decoder, the readout, the
    optimiser, the learning rates, the task epoch budget and the selection
    rule.

The discrete hazard arm is retained inside this protocol as the paired
control, so that the objective comparison happens inside one common start and
does not have to be compared against the archived numbers, which were produced
with a two-patient survival batch that the Cox objective cannot use.

Gradient policy and update schedule, stated explicitly because it is the part
that the survival batch size changes:

  * The role content loss updates the encoders exactly as the archived
    continued_role_only arm did: once per two-patient micro-batch, with the
    same mean-over-batch scaling, so the role representation keeps fitting
    on the same schedule as the archived run.
  * The survival gradient reaches only the readout, because the readout
    consumes detached roles, exactly as the archived continued_role_only arm
    did.
  * The readout is stepped once per survival batch of 32 patients, because a
    Cox partial likelihood needs the whole batch before it can be
    differentiated. The archived arm stepped the readout once per two
    patients. Both arms of THIS protocol use the 32-patient schedule, so the
    objective remains the only difference between them, but the readout
    receives sixteen times fewer updates than in the archived run. This is
    recorded in the receipt and is a limitation of the comparison, not a
    hidden difference between the two arms.

Why the survival batch is a separate number: the role encoder consumes one
bag of 4096 real patches per patient, which dominates memory. The Cox
objective needs many patients in one partial likelihood, but it only needs
their risk scores, not their bags. The encoder therefore runs in micro-batches
of two, the encoder graph is released after each micro-batch, and the readout
graphs, which are small, are accumulated until the whole survival batch is
available.
"""
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .components.cox_surv import cox_partial_likelihood, cox_risk_from_logits
from .components.losses import discrete_hazard_nll
from .components.role_reference import role_content_losses
from .patient_pilot import ROOT, sha
from .patient_pilot_engine import batch, load_checkpoint, tensor_hash


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
    """Reject any drift from the locked Cox-objective protocol."""
    expected = dict(schema='full_slot_cox35_v1', study='blca', fold=0, seed=3,
                    patches=4096, batch_size=2, survival_batch_size=32,
                    target_dim=8, common_rank=2,
                    allocation={'p': [2, 2], 'g': [2, 2]}, width=64, iterations=3,
                    hidden=32, bins=4, role_epochs=12, task_epochs=35, patience=35,
                    role_lr=.003, task_lr=.001, role_weight=1.,
                    task_mode='continued_role_only', readout_bound=2.,
                    fixed_task_budget=True, read_diagnostics=True, ablation_nll=False,
                    torch_threads=2, device='cuda:0')
    for key, value in expected.items():
        if c.get(key) != value or isinstance(value, (int, bool)) and type(c[key]) is not type(value):
            raise ValueError('locked Cox-objective protocol differs: ' + key)
    if c['survival_batch_size'] < c['batch_size']:
        raise ValueError('the survival batch must be at least the encoder batch')
    if c['survival_batch_size'] % c['batch_size']:
        raise ValueError('the survival batch must be a whole number of encoder batches')
    # The objective is selected per arm by the pair driver, so it is absent
    # from the shared protocol configuration and is set on the arm copy.
    objective = c.get('survival_objective')
    if objective is not None and objective not in ('cox', 'discrete_hazard'):
        raise ValueError('the survival objective must be cox or discrete_hazard')


def _survival_objective(name, logits, labels):
    """Return the survival loss for the requested objective.

    Both objectives consume the SAME prediction tensor and the SAME labels, so
    the two arms differ only in the loss and not in the forward pass, the
    readout, the data draws or the role gradient path.
    """
    if name == 'cox':
        risk = cox_risk_from_logits(logits)
        return cox_partial_likelihood(risk, labels['time'], labels['event'])
    if name == 'discrete_hazard':
        return discrete_hazard_nll(logits, labels['completed'], labels['event'])
    raise ValueError('unknown survival objective: ' + name)


def _risk_for_cindex(logits, objective):
    """Risk score for the concordance index; larger means a larger event risk.

    sksurv's concordance_index_censored expects a value whose increase means an
    increase in the risk of the event. The Cox objective optimises a linear risk
    score with exactly that meaning, so its raw score is used directly. The
    discrete hazard objective is scored the way the archived protocol scores
    it, as the negative cumulative survival probability, so that the two arms
    remain directly comparable.
    """
    if objective == 'cox':
        return cox_risk_from_logits(logits)
    return -(1 - logits.sigmoid()).cumprod(-1).sum(-1)


def _better(candidate, best):
    """Selection rule: maximum C-index, then lower survival loss, then earlier epoch."""
    def key(row):
        c, loss, epoch = row['C_index'], row['loss'], row['epoch']
        if not isinstance(epoch, int) or epoch < 1 or c is None or loss is None:
            raise ValueError('finite validation metrics and a positive epoch are required')
        if not math.isfinite(c) or not math.isfinite(loss):
            raise ValueError('finite validation metrics are required')
        return (c, -loss, -epoch)
    return best is None or key(candidate) > key(best)


@torch.no_grad()
def evaluate_epoch_cox(model, store, ids, adapter, boundaries, config, device, *, seed, objective):
    """Fixed-seed validation evaluation for the Cox-objective protocol.

    The bags, the draw order and the role targets follow the archived protocol
    exactly. Only the survival metric changes, and the metric is computed from
    the same prediction tensor in both arms.
    """
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('nonempty unique ordered validation ids required')
    model.eval()
    observe = bool(config.get('read_diagnostics', False))
    for encoder in model.encoders.values():
        encoder.observe_read_diagnostics = observe
    rng = torch.Generator().manual_seed(seed)
    role_sums, losses, logits = {}, [], []
    for start in range(0, len(ids), config['batch_size']):
        batch_ids = ids[start:start + config['batch_size']]
        b = batch(store, batch_ids, adapter, boundaries, config, device, generator=rng)
        roles, _ = model.encode(b['p'], b['g'], b['p_mask'], b['g_mask'])
        target = model.reference.targets(b['p_target'], b['g_target'])
        for key, value in role_content_losses(roles, target).items():
            role_sums[key] = role_sums.get(key, 0.) + float(value) * len(batch_ids)
        prediction = model.predict_roles(roles)['prediction']
        losses.append(float(_survival_objective(objective, prediction, {
            'time': b['time'], 'event': b['event'], 'completed': b['completed']})))
        logits.append(prediction.cpu())
    logits = torch.cat(logits)
    risks = _risk_for_cindex(logits, objective).numpy()
    events = np.array([store.manifest['patients'][i]['censor'] < 1 for i in ids], dtype=bool)
    times = np.array([store.manifest['patients'][i]['time'] for i in ids])
    from sksurv.metrics import concordance_index_censored
    try:
        cindex = float(concordance_index_censored(events, times, risks)[0])
        if not math.isfinite(cindex):
            raise ValueError('nonfinite C-index')
        status = 'computed'
    except ValueError as error:
        cindex, status = None, 'undefined: ' + str(error)
    result = {'role_target_MSE': {k: v / len(ids) for k, v in role_sums.items()},
              'validation_sampling_seed': seed,
              'survival_objective': objective}
    value = float(sum(losses) / len(losses))
    if objective == 'cox':
        result['NLL'] = None
        result['Cox_partial_NLL'] = value
    else:
        result['NLL'] = value
        result['Cox_partial_NLL'] = None
    result.update(C_index=cindex, C_index_status=status)
    return result


def run_cox_objective_arm(root, out, manifest, c, source_hashes, *, split_guard=None,
                          synthetic=False, initial_checkpoint=None):
    """Run one survival objective arm from a shared role checkpoint.

    The survival gradient trains the readout only. The role encoders keep
    fitting from the role content loss on the archived schedule, so the two
    arms of this protocol differ in the survival objective and in nothing else.
    """
    if not synthetic and split_guard is None:
        raise ValueError('verified patient split admission required')
    if c['task_mode'] != 'continued_role_only':
        raise ValueError('the Cox-objective protocol uses the continued_role_only route')
    if initial_checkpoint is None:
        raise ValueError('a completed shared role checkpoint is required')
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

    model, adapter, boundaries = load_checkpoint(initial_checkpoint, device)
    model.requires_grad_(True)
    reference = model.reference
    from .patient_pilot_engine import PreparedStore
    store = PreparedStore(root, manifest)
    out.mkdir(parents=True)
    receipt = {'schema': 'full_slot_cox_objective_v1', 'status': 'running', 'config': c,
               'started_at': _now(), 'source_SHA256': source_hashes,
               'split_identity': identity, 'prepared_manifest_SHA256': manifest_hash,
               'train_count': len(train), 'validation_count': len(val), 'test_used': False,
               'shared_checkpoint_SHA256': sha(initial_checkpoint),
               'shared_checkpoint_role_epochs': c['role_epochs'],
               'model_parameters': sum(x.numel() for x in model.parameters()),
               'gradient_policy': ('role content loss updates the encoders per two-patient micro-batch; '
                                   'the survival gradient reaches only the readout, because the readout '
                                   'consumes detached roles'),
               'update_schedule': ('the readout is stepped once per survival batch of '
                                   f"{c['survival_batch_size']} patients, so it receives sixteen times "
                                   'fewer updates than the archived two-patient arm; both arms of this '
                                   'protocol share that schedule'),
               'selection': 'maximum C-index, then lower survival loss, then earlier epoch',
               'epoch_logs': [], 'best': None, 'last': None,
               'optimizer_steps': {'role': 0, 'task': 0},
               'scope': 'development validation only; no independent test or causal interpretation'}
    _write_receipt(out / 'receipt.json', receipt)

    rng = torch.Generator().manual_seed(c['seed'] + 9000)
    saved = torch.load(initial_checkpoint, map_location='cpu', weights_only=True)
    rng.set_state(saved['train_generator_state'])
    torch.set_rng_state(saved['torch_rng_state'])
    if device.type == 'cuda' and saved['cuda_rng_state'] is not None:
        torch.cuda.set_rng_state(saved['cuda_rng_state'], device)
    eval_seed = c['seed'] + 19000

    # One optimiser over every parameter, at the archived task learning rate.
    # The archived continued_role_only arm used exactly this policy.
    optim = torch.optim.Adam(model.parameters(), lr=c['task_lr'])
    receipt['task_optimizer_initial_state_empty'] = not bool(optim.state)
    receipt['task_optimizer_initial_groups'] = optim.state_dict()['param_groups']

    best, stale = None, 0
    try:
        for epoch in range(1, c['task_epochs'] + 1):
            model.train()
            order = torch.randperm(len(train), generator=rng).tolist()
            total, role_total = 0., 0.
            draw_hash = hashlib.sha256()
            draw_hash.update(bytes(rng.get_state().tolist()))
            draw_hash.update(str(order).encode())
            for start in range(0, len(order), c['survival_batch_size']):
                group = [train[i] for i in order[start:start + c['survival_batch_size']]]
                # The partial likelihood needs the whole survival batch, but the
                # bags are the memory-dominant part. The encoder runs in
                # micro-batches; its graph is released after each micro-batch,
                # while the readout graph, which is small, is accumulated.
                predictions, labels = [], {'time': [], 'event': [], 'completed': []}
                for micro in range(0, len(group), c['batch_size']):
                    ids = group[micro:micro + c['batch_size']]
                    b = batch(store, ids, adapter, boundaries, c, device, generator=rng)
                    draw_hash.update(tensor_hash(list(b.items())).encode())
                    roles, _ = model.encode(b['p'], b['g'], b['p_mask'], b['g_mask'])
                    target = model.reference.targets(b['p_target'], b['g_target'])
                    role_loss = sum(role_content_losses(roles, target).values()) / 4
                    optim.zero_grad(set_to_none=True)
                    role_loss.backward()
                    _step(optim, model)
                    receipt['optimizer_steps']['role'] += 1
                    role_total += float(role_loss.detach()) * len(ids)
                    # The readout sees detached roles, so the survival gradient
                    # cannot reach the encoders.
                    detached = {k: v.detach() for k, v in roles.items()}
                    predictions.append(model.predict_roles(detached)['prediction'])
                    labels['time'].append(b['time'])
                    labels['event'].append(b['event'])
                    labels['completed'].append(b['completed'])
                combined = {k: torch.cat(v) for k, v in labels.items()}
                logits = torch.cat(predictions, 0)
                task_loss = _survival_objective(c['survival_objective'], logits, combined)
                if not torch.isfinite(task_loss):
                    raise FloatingPointError('nonfinite survival loss')
                optim.zero_grad(set_to_none=True)
                task_loss.backward()
                _step(optim, model)
                receipt['optimizer_steps']['task'] += 1
                total += float(task_loss.detach()) * len(group)
            if sha(root / 'prepared/manifest.json') != manifest_hash or (split_guard and split_guard() != identity):
                raise RuntimeError('split identity or manifest changed before evaluation')
            metrics = evaluate_epoch_cox(model, store, val, adapter, boundaries, c, device,
                                         seed=eval_seed, objective=c['survival_objective'])
            log = {'epoch': epoch, 'completed_at': _now(),
                   'train_survival_loss': total / len(train),
                   'train_role_loss': role_total / len(train),
                   'validation': metrics,
                   'input_draw_SHA256': draw_hash.hexdigest()}
            receipt['epoch_logs'].append(log)
            loss_value = metrics['NLL'] if metrics['NLL'] is not None else metrics['Cox_partial_NLL']
            row = {'epoch': epoch, 'C_index': metrics['C_index'], 'loss': loss_value}
            improved = _better(row, best)
            if improved:
                best, stale = row, 0
            else:
                stale += 1
            receipt['best'], receipt['last'] = best, row
            checkpoint = {'state_dict': copy.deepcopy(model.state_dict()),
                          'adapter': adapter.state(), 'config': c,
                          'boundaries': boundaries, 'reference_metadata': reference.metadata,
                          'optimizer_state': optim.state_dict(), 'phase': 'task', 'epoch': epoch,
                          'train_generator_state': rng.get_state(),
                          'torch_rng_state': torch.get_rng_state(),
                          'cuda_rng_state': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                          'split_identity': identity, 'manifest_SHA256': manifest_hash,
                          'source_SHA256': source_hashes,
                          'selection_state': {'best': best, 'stale_epochs': stale}}
            _save_checkpoint(out / 'task_last.pt', checkpoint)
            if improved:
                _save_checkpoint(out / 'task_best.pt', checkpoint)
            _write_receipt(out / 'receipt.json', receipt)
            print(f"objective={c['survival_objective']} epoch={epoch} "
                  f"cindex={metrics['C_index']} loss={loss_value}", flush=True)
        receipt['status'] = 'completed'
        receipt['completed_at'] = _now()
        receipt['artifacts_SHA256'] = {p.name: sha(p) for p in out.iterdir()
                                       if p.is_file() and p.name != 'receipt.json'}
        _write_receipt(out / 'receipt.json', receipt)
        return receipt
    except Exception as error:
        receipt.update(status='failed', failed_at=_now(),
                       error=type(error).__name__ + ': ' + str(error))
        _write_receipt(out / 'receipt.json', receipt)
        raise


def _step(optim, model):
    """Step the optimiser after checking that every live gradient is finite."""
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError('nonfinite gradient')
    optim.step()
