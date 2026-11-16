"""Arm B: the g slot symmetry-breaking variant of the matched35 protocol.

Added as a NEW module. Nothing in the archived matched35 sources is edited, so
the archived receipts keep their provenance.

What this arm changes relative to the archived matched35 `joint` /
`continued_role_only` arms, and nothing else:

  The slot encoder gains one learnable per-slot query bias, initialised to
  exactly zero. At initialisation the forward pass is therefore bit-for-bit
  identical to the archived encoder. Everything else -- the split, the bags,
  the 4096 real patches, the sampling and zero-padding rule, the role targets,
  the frozen role reference, the slot allocation, the readout, the optimiser,
  the learning rates, the epoch budget, the selection rule and the four-bin
  discrete hazard survival objective -- is the archived matched35 protocol.

  The bias is a per-slot degree of freedom, not an identifier for a biological
  factor. Whether it changes the survival score is measured, not assumed.

The arm keeps the archived discrete hazard objective on purpose. Arm A changes
the survival objective and Arm B changes the slot mechanism, so keeping Arm B
on the archived objective means each arm has exactly one changed variable with
respect to the archived baseline.
"""
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from .components.losses import discrete_hazard_nll
from .components.role_reference import FrozenRoleReference, ReferenceProvenance, role_content_losses
from .components.symmetry_slots import SymmetryBrokenRoleSlotEncoder
from .full_slot_epoch import evaluate_epoch
from .patient_pilot import ROOT, sha
from .patient_pilot_engine import (
    ContentAdapter, PreparedStore, SurvivalRoleModel, batch, collect_targets,
    fit_boundaries, label_times, tensor_hash,
)


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
    """Reject any drift from the locked symmetry protocol."""
    expected = dict(schema='full_slot_symmetry35_v1', study='blca', fold=0, seed=3,
                    patches=4096, batch_size=2, target_dim=8, common_rank=2,
                    allocation={'p': [2, 2], 'g': [2, 2]}, width=64, iterations=3,
                    hidden=32, bins=4, role_epochs=12, task_epochs=35, patience=35,
                    role_lr=.003, task_lr=.001, role_weight=1., task_mode='joint',
                    readout_bound=2., fixed_task_budget=True, read_diagnostics=True,
                    ablation_nll=True, torch_threads=2, device='cuda:0')
    for key, value in expected.items():
        if key == 'device':
            # The protocol pins the device family, not the physical index, so
            # that two arms may run concurrently on two different GPUs without
            # any other parameter changing. The chosen index is recorded in
            # the receipt through the configuration.
            if c.get(key) not in ('cuda:0', 'cuda:1'):
                raise ValueError('the symmetry protocol requires a cuda device')
            continue
        if c.get(key) != value or isinstance(value, (int, bool)) and type(c[key]) is not type(value):
            raise ValueError('locked symmetry protocol differs: ' + key)


class SymmetrySurvivalRoleModel(SurvivalRoleModel):
    """The archived survival role model with symmetry-broken slot encoders.

    The archived encoder instances are replaced by symmetry-broken ones that
    receive the archived initial weights verbatim, so the two protocols start
    from the same initialisation under the same seed, and the only difference
    is the extra zero-initialised per-slot query bias.
    """

    def __init__(self, reference, adapter, anchors, c):
        super().__init__(reference, adapter, anchors, c)
        for modality, encoder in list(self.encoders.items()):
            broken = SymmetryBrokenRoleSlotEncoder(encoder.project.in_features, encoder.config)
            missing, unexpected = broken.load_state_dict(encoder.state_dict(), strict=False)
            if list(missing) != ['slot_query_bias'] or unexpected:
                raise ValueError('only the per-slot query bias may be missing')
            self.encoders[modality] = broken


def load_symmetry_checkpoint(path, device='cpu'):
    """Reload a symmetry-arm checkpoint into the symmetry model class."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = saved['state_dict']
    meta = saved['reference_metadata']
    reference = FrozenRoleReference(
        {v: state['reference.' + v + '_mean'] for v in ('p', 'g')},
        {v: state['reference.' + v + '_transform'] for v in ('p', 'g')},
        {v: state['reference.' + v + '_projector'] for v in ('p', 'g')},
        provenance=ReferenceProvenance(meta['fit_split'], meta['target_definition'], meta['source']),
        mode=meta['mode'], singular_values=state['reference.singular_values'])
    adapter = ContentAdapter.__new__(ContentAdapter)
    adapter.sizes = saved['adapter']['sizes']
    adapter.stats = saved['adapter']['stats']
    anchors = {}
    for v in ('p', 'g'):
        anchors[v + '_common'], anchors[v + '_residual'] = state['readout.' + v + '_anchors'].split(
            reference.dimension(v), dim=1)
    model = SymmetrySurvivalRoleModel(reference, adapter, anchors, saved['config']).double()
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, adapter, saved['boundaries']


def run_symmetry_arm(root, out, manifest, c, source_hashes, *, split_guard=None, synthetic=False):
    """Train the symmetry-broken variant on the archived matched35 schedule.

    The schedule, the losses, the batching and the selection rule are the
    archived matched35 ones. The only difference is the encoder class.
    """
    if not synthetic and split_guard is None:
        raise ValueError('verified patient split admission required')
    if c['task_mode'] != 'joint':
        raise ValueError('the symmetry arm uses the archived joint route')
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
    reference = FrozenRoleReference.estimated(
        tp, tg, common_rank=c['common_rank'], coordinate_system='target',
        provenance=ReferenceProvenance(
            'train', 'standardized PCA patch-mean and pathway-expression targets',
            'training fold only; empirical roles, no biological source certification'))
    anchors = reference.targets(tp[:64], tg[:64])
    times = torch.tensor([manifest['patients'][i]['time'] for i in train], dtype=torch.float64)
    events = torch.tensor([manifest['patients'][i]['censor'] < 1 for i in train], dtype=torch.bool)
    boundaries = fit_boundaries(times, events, c['bins'])
    model = SymmetrySurvivalRoleModel(reference, adapter, anchors, c).double().to(device)

    out.mkdir(parents=True)
    receipt = {'schema': 'full_slot_symmetry_v1', 'status': 'running', 'config': c,
               'started_at': _now(), 'source_SHA256': source_hashes,
               'split_identity': identity, 'prepared_manifest_SHA256': manifest_hash,
               'train_count': len(train), 'validation_count': len(val), 'test_used': False,
               'model_parameters': sum(x.numel() for x in model.parameters()),
               'mechanism': ('one learnable per-slot query bias per encoder, initialised to exactly '
                             'zero; at initialisation the forward pass equals the archived encoder'),
               'survival_objective': 'archived four-bin discrete hazard NLL',
               'difference_from_archived_matched35': ('the slot encoder class only; the split, bags, '
                                                      'patch budget, targets, reference, allocation, readout, '
                                                      'optimiser, learning rates, epoch budget and selection '
                                                      'rule are the archived ones'),
               'selection': 'maximum C-index, exact tie lower NLL, then earlier epoch',
               'epoch_logs': [], 'best': None, 'last': None,
               'optimizer_steps': {'role': 0, 'task': 0},
               'scope': 'development validation only; no independent test or causal interpretation'}
    _write_receipt(out / 'receipt.json', receipt)

    rng = torch.Generator().manual_seed(c['seed'] + 9000)
    eval_seed = c['seed'] + 19000
    reference_hash = tensor_hash(list(model.reference.named_buffers()))
    bias_before = tensor_hash([(n, p) for n, p in model.named_parameters() if n.endswith('slot_query_bias')])
    optim = torch.optim.Adam(list(model.encoders.parameters()) + list(model.pools.parameters()) +
                             list(model.decoders.parameters()), lr=c['role_lr'])
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
                    b = batch(store, ids, adapter, boundaries, c, device, generator=rng)
                    draw_hash.update(tensor_hash(list(b.items())).encode())
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
                metrics = evaluate_epoch(model, store, val, adapter, boundaries, c, device,
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
        bias_after = tensor_hash([(n, p) for n, p in model.named_parameters() if n.endswith('slot_query_bias')])
        receipt['slot_query_bias_changed_during_training'] = bias_before != bias_after
        if bias_before == bias_after:
            raise RuntimeError('the per-slot query bias did not move; the arm would be a no-op')
        if sha(root / 'prepared/manifest.json') != manifest_hash or (split_guard and split_guard() != identity):
            raise RuntimeError('split identity or manifest changed before checkpoint verification')
        for label in ('best', 'last'):
            reloaded, reloaded_adapter, reloaded_boundaries = load_symmetry_checkpoint(
                out / ('task_' + label + '.pt'), device)
            verified = evaluate_epoch(reloaded, store, val, reloaded_adapter, reloaded_boundaries,
                                      c, device, seed=eval_seed, survival=True)
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
