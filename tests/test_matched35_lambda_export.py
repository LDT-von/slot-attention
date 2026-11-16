"""Invented fixed-shape JSON confirms the new export has a privacy allowlist."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.export_lambda05_aggregates import export_receipt
from test_matched35_chain import row, ROLES

ROOT = Path(__file__).resolve().parents[1]


def invented_receipt():
    rows = [row('task', i, 'joint') for i in range(1, 36)]
    for entry in rows:
        entry['validation']['roles'] = {r: dict(patients=76, mean_norm=1., across_patient_std_rms=.1,
            centered_patient_effective_rank=1., patient_rank_upper_bound=8) for r in ROLES}
        entry['validation']['slots'] = {r: dict(patients=76, slots=2, width=64, mean_norm=1., zero_slot_fraction=0.,
            across_patient_std_rms=.1, patient_rank_upper_bound=64, pair_cosine_mean=.1,
            pair_cosine_undefined_count=0, read_overlap_mean=.1,
            mean_norm_by_slot=[1., 1.], zero_fraction_by_slot=[0., 0.], across_patient_std_rms_by_slot=[.1, .1],
            centered_patient_effective_rank_by_slot=[1., 1.]) for r in ROLES}
    selected = dict(epoch=35, C_index=rows[-1]['validation']['C_index'], NLL=.5)
    return dict(status='completed', test_used=False, synthetic_only=False, protocol='lambda05_fixed35_v1',
        config=json.loads((ROOT/'configs/common_private_lambda05_v1.json').read_text(encoding='utf-8')) | {'configuration_SHA256':'a'*64},
        new_role_epochs=0, new_task_epochs=35, total_new_epochs=35, epoch_logs=rows, best=selected, last=selected,
        shared_checkpoint_SHA256='b'*64, endpoint_start_verified_before_updates=True,
        endpoint_start_and_draws_verified=True, checkpoint_reload_verified=True,
        checkpoint_optimizer_and_generator_reload_verified=True, reference_frozen=True,
        role_survival_gradient_scale=.5, head_survival_gradient_scale=1.,
        historical_chain_audit={'aggregate_numeric_crosscheck_complete':True})


class LambdaExportTest(unittest.TestCase):
    def test_allowlist_drops_private_fields_at_every_nesting_level(self):
        source = invented_receipt()
        source['patient_id'] = 'INVENTED_PRIVATE'
        first = source['epoch_logs'][0]
        first['validation']['roles']['p_common']['patient_id'] = 'INVENTED_PRIVATE'
        first['validation']['read_diagnostics']['p_common']['patient_id'] = 'INVENTED_PRIVATE'
        first['validation']['read_diagnostics']['p_common']['metrics']['query_norm']['patient_id'] = 'INVENTED_PRIVATE'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'receipt.json'; p.write_text(json.dumps(source), encoding='utf-8')
            result=export_receipt(p,hashlib.sha256(p.read_bytes()).hexdigest())
        self.assertEqual(len(result['epoch_rows']),35)
        self.assertNotIn('INVENTED_PRIVATE',json.dumps(result))
        self.assertEqual(result['epoch_rows'][0]['validation']['ablation_delta_NLL']['p_common'],[.05,-.05])

    def test_wrong_hash_budget_and_deletion_sign_refused(self):
        for mutation in ('hash','budget','deletion','head'):
            source=invented_receipt()
            if mutation=='budget':source['epoch_logs'].pop()
            if mutation=='deletion':source['epoch_logs'][0]['validation']['ablation_delta_NLL']['p_common'][0]=.9
            if mutation=='head':source['head_survival_gradient_scale']=.5
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'receipt.json';p.write_text(json.dumps(source),encoding='utf-8')
                wanted='0'*64 if mutation=='hash' else hashlib.sha256(p.read_bytes()).hexdigest()
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):export_receipt(p,wanted)


if __name__=='__main__':unittest.main()
