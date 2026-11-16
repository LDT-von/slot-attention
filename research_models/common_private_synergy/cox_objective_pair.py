"""Run the Cox-objective protocol: one shared role start, two survival objectives.

Added as a NEW module. The archived matched35 sources stay byte for byte.

The two arms are:

  cox              the survival loss is a Cox partial likelihood over 32 patients
  discrete_hazard  the survival loss is the archived four-bin discrete hazard NLL

Both arms start from the SAME archived shared_role checkpoint, use the SAME
split, the SAME bags, the SAME draw order and the SAME update schedule, and
differ only in the survival loss. The discrete arm is the paired control, so
the objective comparison does not depend on the archived numbers, which were
produced with a two-patient survival batch that the Cox objective cannot use.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from .cox_objective_engine import run_cox_objective_arm, validate_protocol
from .patient_pilot import sha


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_receipt(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def run_cox_pair(root, out, manifest, c, hashes, *, split_guard=None, synthetic=False,
                 shared_checkpoint=None):
    """Run both survival objectives from one shared role checkpoint."""
    if not synthetic:
        validate_protocol(c)
        fold = manifest['folds']['0']
        if len(fold['train']) != 304 or len(fold['val']) != 76:
            raise ValueError('the Cox-objective protocol requires the admitted 304/76 development split')
    if split_guard is None and not synthetic:
        raise ValueError('verified patient split admission required')
    if shared_checkpoint is None:
        raise ValueError('a completed shared role checkpoint is required')
    out = Path(out)
    if out.exists():
        raise FileExistsError('fresh run required')
    out.mkdir(parents=True)
    receipt = {'schema': 'cox_objective_pair_v1', 'status': 'running', 'started_at': _now(),
               'config': c, 'source_SHA256': hashes, 'test_used': False,
               'synthetic_only': synthetic, 'arms': {},
               'shared_checkpoint_SHA256': sha(shared_checkpoint),
               'expected_total_epochs': 2 * c['task_epochs'],
               'scope': 'development mechanism comparison; no independent test',
               'objective_difference': ('the only difference between the arms is the survival loss; '
                                       'the split, bags, draw order, readout, gradient policy and '
                                       'update schedule are shared')}
    path = out / 'paired_receipt.json'
    _write_receipt(path, receipt)
    try:
        results = {}
        for objective in ('discrete_hazard', 'cox'):
            result = run_cox_objective_arm(
                root, out / objective, manifest, {**c, 'survival_objective': objective},
                hashes, split_guard=split_guard, synthetic=synthetic,
                initial_checkpoint=shared_checkpoint)
            results[objective] = result
            receipt['arms'][objective] = {
                'status': result['status'], 'best': result['best'], 'last': result['last'],
                'epochs': len(result['epoch_logs']),
                'receipt_SHA256': sha(out / objective / 'receipt.json')}
            _write_receipt(path, receipt)
        # The two arms must have seen the same patients in the same order and
        # the same bags, otherwise the objective comparison is not controlled.
        a = results['discrete_hazard']['epoch_logs']
        b = results['cox']['epoch_logs']
        if len(a) != len(b):
            raise RuntimeError('the arms ran a different number of epochs')
        for left, right in zip(a, b):
            if left['input_draw_SHA256'] != right['input_draw_SHA256']:
                raise RuntimeError('the arms did not draw the same inputs')
        receipt.update(status='completed', completed_at=_now(),
                       matched_input_draws_verified=True,
                       total_epochs=sum(len(r['epoch_logs']) for r in results.values()))
        _write_receipt(path, receipt)
        return receipt
    except Exception as error:
        receipt.update(status='failed', failed_at=_now(), error_type=type(error).__name__)
        _write_receipt(path, receipt)
        raise
