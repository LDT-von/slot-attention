"""One role warmup, two fixed-budget gradient-route arms; no import execution."""
from pathlib import Path

from .patient_pilot import sha


def validate_protocol(c):
    expected = dict(schema='full_slot_matched35_v1', study='blca', fold=0, seed=3,
                    patches=4096, batch_size=2, target_dim=8, common_rank=2,
                    allocation={'p': [2, 2], 'g': [2, 2]}, width=64, iterations=3,
                    hidden=32, bins=4, role_epochs=12, task_epochs=35, patience=35,
                    role_lr=.003, task_lr=.001, role_weight=1., task_mode='joint',
                    readout_bound=2., fixed_task_budget=True, read_diagnostics=True, ablation_nll=True,
                    torch_threads=2, device='cuda:0')
    for key, value in expected.items():
        if c.get(key) != value or isinstance(value, (int, bool)) and type(c[key]) is not type(value):
            raise ValueError('locked matched35 protocol differs: ' + key)


def run_matched(root, out, manifest, c, hashes, *, split_guard=None, synthetic=False):
    from .full_slot_engine import run_full, _write_receipt, _now
    if not synthetic:
        validate_protocol(c)
        fold = manifest['folds']['0']
        if len(fold['train']) != 304 or len(fold['val']) != 76:
            raise ValueError('matched35 requires the admitted 304/76 development split')
    if c.get('schema') != 'full_slot_matched35_v1':
        raise ValueError('matched35 schema required even for synthetic runs')
    if not c.get('fixed_task_budget') or not c.get('read_diagnostics'):
        raise ValueError('fixed budget and read diagnostics required')
    if split_guard is None and not synthetic:
        raise ValueError('verified patient split admission required')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema='matched35_pair_v1', status='running', started_at=_now(),
                   config=c, source_SHA256=hashes, test_used=False, synthetic_only=synthetic,
                   expected_total_epochs=c['role_epochs'] + 2*c['task_epochs'], arms={},
                   scope='development mechanism comparison; no independent test',
                   optimizer_policy='fresh identical Adam groups/state in both task arms; role Adam discarded')
    path = out / 'paired_receipt.json'
    _write_receipt(path, receipt)
    try:
        common = run_full(root, out/'shared_role', manifest, c, hashes,
                          split_guard=split_guard, synthetic=synthetic, warmup_only=True)
        checkpoint = out/'shared_role/role_last.pt'
        shared_sha = sha(checkpoint)
        receipt['shared_checkpoint_SHA256'] = shared_sha
        receipt['shared_role_epochs'] = len(common['epoch_logs'])
        _write_receipt(path, receipt)
        results = {}
        for mode in ('joint', 'continued_role_only'):
            if sha(checkpoint) != shared_sha:
                raise RuntimeError('shared checkpoint changed')
            result = run_full(root, out/mode, manifest, {**c, 'task_mode': mode}, hashes,
                              split_guard=split_guard, synthetic=synthetic, initial_checkpoint=checkpoint)
            results[mode] = result
            receipt['arms'][mode] = dict(status=result['status'], best=result['best'], last=result['last'],
                                        epochs=len(result['epoch_logs']), receipt_SHA256=sha(out/mode/'receipt.json'))
            _write_receipt(path, receipt)
        a, b = results.values()
        keys = ('shared_checkpoint_SHA256', 'initial_model_tensor_SHA256', 'initial_train_rng_SHA256',
                'initial_torch_rng_SHA256', 'initial_cuda_rng_SHA256', 'task_optimizer_initial_groups',
                'task_start_role_parameter_SHA256', 'split_identity', 'prepared_manifest_SHA256')
        if any(a[k] != b[k] for k in keys):
            raise RuntimeError('arm starting state mismatch')
        if not all(r['task_optimizer_initial_state_empty'] for r in results.values()):
            raise RuntimeError('task optimizer did not start empty')
        for r in results.values():
            if len(r['epoch_logs']) != c['task_epochs'] or r['last']['epoch'] != c['task_epochs']:
                raise RuntimeError('fixed task budget incomplete')
        if [e['input_draw_SHA256'] for e in a['epoch_logs']] != [e['input_draw_SHA256'] for e in b['epoch_logs']]:
            raise RuntimeError('arm data draws mismatch')
        if sha(checkpoint) != shared_sha:
            raise RuntimeError('shared checkpoint changed during arms')
        receipt.update(status='completed', completed_at=_now(), common_start_verified=True,
                       matched_input_draws_verified=True, task_optimizer_reset_verified=True,
                       total_epochs=len(common['epoch_logs']) + sum(len(r['epoch_logs']) for r in results.values()))
        _write_receipt(path, receipt)
        return receipt
    except Exception as error:
        receipt.update(status='failed', failed_at=_now(), error_type=type(error).__name__)
        _write_receipt(path, receipt)
        raise
