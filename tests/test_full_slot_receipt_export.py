"""Privacy, provenance and sequence checks; invented JSON only, no model runs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.export_full_slot_receipt_review import export_receipt, main


def mass(patients, slots):
    return {'threshold': 1e-8, 'observations': patients * 3 * slots, 'minimum': .1,
            'below_threshold_count': 0, 'below_threshold_fraction': 0.,
            'below_threshold_count_by_iteration': [0, 0, 0],
            'below_threshold_count_by_slot': [0] * slots, 'minimum_by_iteration': [.1] * 3}


def fixture(mode='joint'):
    config = {'fold': 0, 'seed': 3, 'patches': 4096, 'batch_size': 2, 'target_dim': 8,
              'common_rank': 2, 'width': 64, 'iterations': 3, 'hidden': 32, 'bins': 4,
              'role_epochs': 1, 'task_epochs': 2, 'patience': 10, 'role_lr': .003,
              'task_lr': .001, 'role_weight': 1., 'readout_bound': 2.,
              'task_mode': mode, 'allocation': {'p': [2, 2], 'g': [2, 2]}}
    role = {'patients': 4, 'mean_norm': 1., 'across_patient_std_rms': .2,
            'centered_patient_effective_rank': 1.9, 'patient_rank_upper_bound': 3}
    slot = {**role, 'slots': 2, 'width': 64, 'zero_slot_fraction': 0., 'pair_cosine_mean': .9,
            'pair_cosine_undefined_count': 0, 'read_overlap_mean': .8,
            'mean_norm_by_slot': [1., 1.], 'zero_fraction_by_slot': [0., 0.],
            'across_patient_std_rms_by_slot': [.2, .2], 'centered_patient_effective_rank_by_slot': [2., 2.]}
    keys = [m + '_' + r for m in ('p', 'g') for r in ('common', 'residual')]
    validation = {'role_target_MSE': dict.fromkeys(keys, .2), 'validation_sampling_seed': 19003,
                  'roles': {k: copy.deepcopy(role) for k in keys},
                  'slots': {k: copy.deepcopy(slot) for k in keys},
                  'raw_reading_mass': {k: mass(4, 2) for k in keys},
                  'ablation_mean_abs_logit_change': {k: [.01, .02] for k in keys}, 'C_index': .6, 'NLL': .8}
    rows = []
    for phase, epoch in [('role', 1), ('task', 1), ('task', 2)]:
        rows.append({'phase': phase, 'epoch': epoch, 'train_loss': .4, 'train_role_loss': .2,
                     'train_survival_NLL': .3 if phase == 'task' else None,
                     'validation': copy.deepcopy(validation), 'train_raw_reading_mass': {m: mass(8, 4) for m in ('p', 'g')},
                     'raw_slot_gradient_mean': {t: {m: [.1] * 4 for m in ('p', 'g')}
                                                for t in ('combined', 'survival', 'role')}})
    # Exact C tie: lower NLL chooses epoch 2, not the earlier epoch.
    rows[-1]['validation']['NLL'] = .7
    return {'schema': 'full_slot_research_v1', 'status': 'completed', 'test_used': False,
            'config': config, 'train_count': 8, 'validation_count': 4, 'model_parameters': 114220,
            'read_normalization': {'policy': 'masked_add_eps_l1_v2', 'epsilon': 1e-8, 'epsilon_on': 'real_tokens_only'},
            'prepared_manifest_SHA256': 'a' * 64, 'optimizer_steps': {'role': 4, 'task': 8},
            'best': {'epoch': 2, 'C_index': .6, 'NLL': .7}, 'last': {'epoch': 2, 'C_index': .6, 'NLL': .7},
            'checkpoint_reload_verified': True, 'checkpoint_optimizer_and_generator_reload_verified': True,
            'reference_frozen': True, 'role_parameters_changed_during_task': mode == 'joint',
            'artifacts_SHA256': {n: 'b' * 64 for n in ('role_last.pt', 'task_best.pt', 'task_last.pt')},
            'epoch_logs': rows}


class ReceiptExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'run' / 'receipt.json'
        self.path.parent.mkdir()

    def save(self, value):
        self.path.write_text(json.dumps(value), encoding='utf-8')
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def test_complete_aggregate_export_and_rank_bound(self):
        sha = self.save(fixture())
        result = export_receipt(self.path, sha, 'joint')
        self.assertEqual(result['audit']['epoch_counts'], {'role': 1, 'task': 2})
        self.assertEqual(result['best']['epoch'], 2)
        r = result['epoch_logs'][0]['validation']['roles']['p_common']
        self.assertEqual(r['patient_rank_upper_bound'], 3)
        self.assertEqual(r['architectural_rank_upper_bound'], 2)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), sha)

    def test_private_unknown_fields_never_copied(self):
        source = fixture()
        for where in (source, source['config'], source['epoch_logs'][0],
                      source['epoch_logs'][0]['validation'], source['epoch_logs'][0]['validation']['roles']['p_common']):
            where['patient_ids'] = ['PRIVATE_SENTINEL']
            where['predictions'] = [123456789.123]
        source['split_identity'] = {'train': ['PRIVATE_SENTINEL']}
        sha = self.save(source)
        result = json.dumps(export_receipt(self.path, sha, 'joint'))
        for forbidden in ('PRIVATE_SENTINEL', 'patient_ids', 'predictions', '123456789.123', 'split_identity'):
            self.assertNotIn(forbidden, result)

    def test_hash_mismatch_rejected(self):
        self.save(fixture())
        with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
            export_receipt(self.path, '0' * 64, 'joint')

    def test_wrong_arm_or_old_protocol_rejected(self):
        sha = self.save(fixture())
        with self.assertRaises(ValueError):
            export_receipt(self.path, sha, 'frozen')
        for key, value in [('status', 'running'), ('schema', 'old'), ('read_normalization', {}), ('test_used', True)]:
            source = fixture(); source[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                export_receipt(self.path, self.save(source), 'joint')

    def test_nonfinite_or_private_text_in_known_field_rejected(self):
        for bad in (float('nan'), float('inf'), 'PRIVATE_SENTINEL', True):
            source = fixture(); source['epoch_logs'][0]['train_loss'] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                export_receipt(self.path, self.save(source), 'joint')

    def test_patient_length_vector_and_false_denominator_rejected(self):
        for key, value in [('below_threshold_count_by_slot', [0] * 76), ('observations', 76)]:
            source = fixture(); source['epoch_logs'][0]['validation']['raw_reading_mass']['p_common'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                export_receipt(self.path, self.save(source), 'joint')

    def test_missing_epoch_and_false_best_rejected(self):
        source = fixture(); del source['epoch_logs'][1]
        with self.assertRaises(ValueError):
            export_receipt(self.path, self.save(source), 'joint')
        source = fixture(); source['best']['NLL'] = .2
        with self.assertRaises(ValueError):
            export_receipt(self.path, self.save(source), 'joint')

    def test_output_cannot_change_original_run_or_existing_file(self):
        sha = self.save(fixture())
        for output in (self.path, self.path.parent / 'new.json'):
            with self.subTest(output=output), self.assertRaises((ValueError, FileExistsError)):
                main(['--receipt', 'joint', str(self.path), sha, '--output', str(output)])
        output = self.root / 'keep.json'; output.write_text('KEEP')
        with self.assertRaises(FileExistsError):
            main(['--receipt', 'joint', str(self.path), sha, '--output', str(output)])
        self.assertEqual(output.read_text(), 'KEEP')
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), sha)

    def test_cli_exports_both_arms_without_checkpoint_files(self):
        joint_sha = self.save(fixture())
        frozen_path = self.root / 'frozen' / 'receipt.json'; frozen_path.parent.mkdir()
        frozen_path.write_text(json.dumps(fixture('frozen')), encoding='utf-8')
        frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
        output = self.root / 'new_review' / 'aggregates.json'
        main(['--receipt', 'joint', str(self.path), joint_sha, '--receipt', 'frozen', str(frozen_path), frozen_sha,
              '--output', str(output)])
        result = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(set(result['runs']), {'joint', 'frozen'})
        self.assertEqual(len(result['runs']['joint']['epoch_logs']), 3)
        self.assertFalse(result['runs']['frozen']['audit']['checkpoints_loaded_or_rehashed'])


if __name__ == '__main__':
    unittest.main()
