"""Gated one-condition gradient-strength protocol; import never reads data."""
from pathlib import Path
import json
import math

from .patient_pilot import sha


SHARED_CHECKPOINT_SHA256 = 'efae2d2b6c7fc72d157813f97b8e484c8f2a81dcb7c85bb6e5f1e4ed611abd64'
START_KEYS = ('initial_model_tensor_SHA256', 'initial_train_rng_SHA256',
              'initial_torch_rng_SHA256', 'initial_cuda_rng_SHA256',
              'task_optimizer_initial_groups', 'task_start_role_parameter_SHA256',
              'split_identity', 'prepared_manifest_SHA256')


def validate_protocol(c):
    from .matched35 import validate_protocol as validate_endpoint
    original = dict(c)
    original['schema'] = 'full_slot_matched35_v1'
    original.pop('role_survival_gradient_scale', None)
    original.pop('head_survival_gradient_scale', None)
    validate_endpoint(original)
    if c.get('schema') != 'full_slot_lambda05_v1':
        raise ValueError('lambda05 schema required')
    for name, expected in (('role_survival_gradient_scale', .5),
                           ('head_survival_gradient_scale', 1.)):
        value = c.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value != expected:
            raise ValueError('locked lambda05 gradient coefficient differs: ' + name)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run_lambda05(root, out, manifest, c, hashes, *, run_root,
                 original_exporter=None, aggregates=None, split_guard=None,
                 synthetic=False):
    """Gate historical chain before importing the model or opening a checkpoint.

    synthetic=True is a Python-only invented-fixture integration path. The CLI
    never exposes it and cannot bypass the pinned historical-chain verifier.
    """
    if c.get('schema') != 'full_slot_lambda05_v1':
        raise ValueError('lambda05 schema required')
    for key, expected in (('role_survival_gradient_scale', .5),
                          ('head_survival_gradient_scale', 1.)):
        if type(c.get(key)) not in (int, float) or not math.isfinite(c[key]) or c[key] != expected:
            raise ValueError('locked lambda05 gradient coefficient differs: ' + key)
    if not c.get('fixed_task_budget') or not c.get('read_diagnostics') or not c.get('ablation_nll'):
        raise ValueError('fixed task budget and complete diagnostics required')
    if not synthetic:
        validate_protocol(c)
        if split_guard is None:
            raise ValueError('verified patient split admission required')
        fold = manifest['folds']['0']
        if len(fold['train']) != 304 or len(fold['val']) != 76:
            raise ValueError('lambda05 requires the original 304/76 split')
        from scripts.verify_matched35_chain import verify_chain
        chain = verify_chain(run_root, original_exporter, aggregates)
        if chain.get('status') != 'verified':
            raise ValueError('original matched35 chain not verified')
    else:
        chain = {'schema': 'invented_chain_only', 'status': 'verified',
                 'input_SHA256': {}}
    run_root, out = Path(run_root), Path(out)
    if out.exists():
        raise FileExistsError('preserve prior output; fresh lambda05 output required')
    checkpoint = run_root / 'shared_role/role_last.pt'
    checkpoint_hash = sha(checkpoint)
    if not synthetic and checkpoint_hash != SHARED_CHECKPOINT_SHA256:
        raise ValueError('original common-start checkpoint hash differs')
    endpoints = {arm: _read(run_root / arm / 'receipt.json')
                 for arm in ('joint', 'continued_role_only')}
    left, right = endpoints.values()
    if any(left[key] != right[key] for key in START_KEYS):
        raise ValueError('historical endpoint initial states differ')
    for endpoint in endpoints.values():
        if (endpoint['status'] != 'completed' or not endpoint['task_optimizer_initial_state_empty']
                or endpoint['shared_checkpoint_SHA256'] != checkpoint_hash):
            raise ValueError('historical endpoint is not the admitted common-start arm')
        rows = endpoint['epoch_logs']
        if (len(rows) != c['task_epochs'] or endpoint['last']['epoch'] != c['task_epochs']
                or any(row['phase'] != 'task' or row['epoch'] != i
                       for i, row in enumerate(rows, 1))):
            raise ValueError('historical endpoint task budget differs')
    draws = [row['input_draw_SHA256'] for row in left['epoch_logs']]
    if draws != [row['input_draw_SHA256'] for row in right['epoch_logs']]:
        raise ValueError('historical endpoint input draws differ')
    # Model imports and checkpoint loading occur only after the read-only gate.
    from .full_slot_engine import run_full, _write_receipt
    result = run_full(root, out, manifest, c, hashes, split_guard=split_guard,
                      synthetic=synthetic, initial_checkpoint=checkpoint,
                      origin_receipt=run_root / 'shared_role/receipt.json',
                      expected_start={key:left[key] for key in START_KEYS}, expected_draws=draws)
    try:
        if any(json.loads(json.dumps(result[key])) != left[key] for key in START_KEYS):
            raise RuntimeError('lambda05 common-start state differs from endpoints')
        if not result['task_optimizer_initial_state_empty']:
            raise RuntimeError('lambda05 task optimizer did not start empty')
        if (len(result['epoch_logs']) != c['task_epochs'] or result['last']['epoch'] != c['task_epochs']
                or result['optimizer_steps']['role'] != 0 or 'early_stop' in result):
            raise RuntimeError('lambda05 fixed task-only budget incomplete')
        if [row['input_draw_SHA256'] for row in result['epoch_logs']] != draws:
            raise RuntimeError('lambda05 input draws differ from endpoints')
        if sha(checkpoint) != checkpoint_hash:
            raise RuntimeError('original shared checkpoint changed')
        if not synthetic:
            verify_chain(run_root, original_exporter, aggregates)
        result.update(protocol='lambda05_fixed35_v1', synthetic_only=synthetic,
                      historical_chain_audit=chain,
                      endpoint_receipts_SHA256={arm: sha(run_root / arm / 'receipt.json')
                                               for arm in endpoints},
                      endpoint_start_and_draws_verified=True,
                      new_role_epochs=0, new_task_epochs=len(result['epoch_logs']),
                      total_new_epochs=len(result['epoch_logs']),
                      role_survival_gradient_scale=.5,
                      head_survival_gradient_scale=1.,
                      endpoint_reuse_scope='same common start, fixed inputs and original optimizer policy; not independent repetitions')
        _write_receipt(out / 'receipt.json', result)
        return result
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__,
                      error='lambda05 endpoint comparison failed; inspect private artifacts')
        _write_receipt(out / 'receipt.json', result)
        raise
