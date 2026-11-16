"""Published aggregate and invented JSON checks; no patient/model execution."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_matched35_aggregates import _audit_exported, audit_aggregates
from scripts.verify_matched35_chain import _json_bytes, _scan_aggregate
from scripts.reproduce_matched35_export import reproduce

ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT/'research_models/common_private_synergy/exports/matched35_82epoch_aggregates.json'
spec = importlib.util.spec_from_file_location('original_matched35_exporter', ROOT/'scripts/export_matched35_aggregates.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


class FullExportAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.value = json.loads(AGGREGATE.read_text(encoding='utf-8'))

    def test_full_published82_crosschecks(self):
        result = audit_aggregates(AGGREGATE)
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['total_epochs'], 82)
        self.assertEqual(result['control_C_index_higher_epochs'], 33)
        self.assertAlmostEqual(result['paired_C_index_mean'], .05367156208277705)
        self.assertEqual(result['arms']['joint']['p_common_late_count'], 767)
        self.assertEqual(result['arms']['continued_role_only']['p_common_late_count'], 325)
        self.assertFalse(result['model_executed'])

    def test_epoch_sequence_selection_and_start_tampering(self):
        for mutation in ('sequence', 'best', 'start', 'budget'):
            value = copy.deepcopy(self.value)
            arm = value['arms']['joint']
            if mutation == 'sequence': arm['epoch_rows'][3]['epoch'] = 3
            if mutation == 'best': arm['best']['epoch'] = 24
            if mutation == 'start': arm['initial_model_tensor_SHA256'] = 'f'*64
            if mutation == 'budget': arm['epoch_rows'].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                _audit_exported(value)

    def test_gradients_axes_denominators_and_signed_deletion(self):
        for mutation in ('gradient', 'axis', 'denominator', 'deletion', 'rank'):
            value = copy.deepcopy(self.value)
            row = value['arms']['continued_role_only']['epoch_rows'][0]
            if mutation == 'gradient': row['raw_slot_gradient_mean']['survival']['p'][1] = .01
            if mutation == 'axis': row['read_diagnostics']['p_common']['metrics']['query_norm']['mean_by_iteration_slot'][0].append(1.)
            if mutation == 'denominator': row['raw_reading_mass']['p_common']['observations'] = 76
            if mutation == 'deletion': row['ablation_delta_NLL']['p_common'][0] += .1
            if mutation == 'rank': row['roles']['p_common']['centered_patient_effective_rank'] = 3.
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                _audit_exported(value)

    def test_nonfinite_duplicate_and_private_data_rejected(self):
        for raw in (b'{"x":NaN}', b'{"x":1,"x":2}'):
            with self.assertRaises(ValueError): _json_bytes(raw)
        value = copy.deepcopy(self.value)
        value['arms']['joint']['epoch_rows'][0]['read_diagnostics']['patient_id'] = 'INVENTED_PRIVATE'
        with self.assertRaises(ValueError): _audit_exported(value)

    def test_original_projection_is_deterministic_and_preserved(self):
        rows = self.value['arms']['joint']['epoch_rows']
        receipt_rows = []
        for row in rows:
            validation = {k: row[k] for k in ('C_index', 'NLL', 'role_target_MSE', 'read_diagnostics',
                'ablation_NLL', 'ablation_delta_NLL', 'ablation_delta_NLL_definition',
                'raw_reading_mass', 'roles', 'slots')}
            receipt_rows.append({**row, 'validation': validation})
        self.assertEqual(original.arm_epochs({'epoch_logs': receipt_rows}), rows)
        self.assertNotIn('ablation_mean_abs_logit_change', rows[0])

    def test_original_nested_copy_is_not_a_general_privacy_allowlist(self):
        # Demonstrate why the preserved historical tool needs the pinned wrapper.
        row = self.value['arms']['joint']['epoch_rows'][0]
        validation = {k: copy.deepcopy(row[k]) for k in ('C_index', 'NLL', 'role_target_MSE', 'read_diagnostics')}
        validation['read_diagnostics']['patient_id'] = 'INVENTED_PRIVATE'
        projected = original.arm_epochs({'epoch_logs': [{**row, 'validation': validation}]})
        self.assertEqual(projected[0]['read_diagnostics']['patient_id'], 'INVENTED_PRIVATE')
        with self.assertRaises(ValueError): _scan_aggregate(projected)

    def test_reproduction_fails_before_execution_for_unpinned_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'run').mkdir()
            with self.assertRaises((ValueError, FileNotFoundError)):
                reproduce(root/'run', ROOT/'scripts/export_matched35_aggregates.py',
                          root/'new.json', root/'audit.json')
            self.assertFalse((root/'new.json').exists())


if __name__ == '__main__':
    unittest.main()
