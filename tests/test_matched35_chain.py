"""Invented JSON only: fixed pins, complete curves, axes and safe audit output."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.verify_matched35_chain import (
    CONFIG_SHA256, EXPECTED_SHA256, METRICS, ROLES, SHARED_CHECKPOINT_SHA256,
    _json_bytes, _verify_chain, main,
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def config(mode):
    return dict(schema='full_slot_matched35_v1', study='blca', fold=0, seed=3,
                patches=4096, batch_size=2, target_dim=8, common_rank=2,
                allocation={'p': [2, 2], 'g': [2, 2]}, width=64, iterations=3,
                hidden=32, bins=4, role_epochs=12, task_epochs=35, patience=35,
                role_lr=.003, task_lr=.001, role_weight=1., task_mode=mode,
                readout_bound=2., fixed_task_budget=True, read_diagnostics=True,
                ablation_nll=True, torch_threads=2, device='cuda:0',
                configuration_SHA256=CONFIG_SHA256)


def mass(bags, slots):
    return dict(observations=bags*3*slots, threshold=1e-8, minimum=.1,
                below_threshold_count=0, below_threshold_fraction=0.,
                below_threshold_count_by_iteration=[0]*3,
                below_threshold_count_by_slot=[0]*slots, minimum_by_iteration=[.1]*3)


def diagnostics(bags):
    result = dict(axes=['iteration', 'slot'], bag_observations=bags,
                  observation_count=bags*3*2, raw_mass_threshold=1e-8,
                  below_threshold_count_by_iteration_slot=[[0, 0] for _ in range(3)],
                  below_threshold_fraction_by_iteration_slot=[[0., 0.] for _ in range(3)], metrics={})
    for name in METRICS:
        value = -2. if name.startswith(('log_pre', 'logit_')) else .1
        result['metrics'][name] = dict(mean=value, minimum=value, maximum=value,
            **{key: [[value, value] for _ in range(3)] for key in
               ('mean_by_iteration_slot', 'minimum_by_iteration_slot', 'maximum_by_iteration_slot')})
    return result


def row(phase, epoch, mode):
    value = dict(phase=phase, epoch=epoch, input_draw_SHA256=digest(str(epoch).encode()),
                 train_loss=.8, train_role_loss=.3, train_survival_NLL=.5 if phase == 'task' else None,
                 train_raw_reading_mass={m: mass(304, 4) for m in ('p', 'g')},
                 train_read_diagnostics={r: diagnostics(304) for r in ROLES},
                 raw_slot_gradient_mean={t: {m: [.1]*4 for m in ('p', 'g')}
                                         for t in ('role', 'survival', 'combined')},
                 validation=dict(validation_sampling_seed=19003,
                    role_target_MSE={r: .3 for r in ROLES},
                    raw_reading_mass={r: mass(76, 2) for r in ROLES},
                    read_diagnostics={r: diagnostics(76) for r in ROLES}))
    if phase == 'role':
        value['raw_slot_gradient_mean']['survival'] = {'p': None, 'g': None}
    else:
        value['validation'].update(C_index=.5+epoch/10000+(mode == 'continued_role_only')*.01,
            NLL=.5, ablation_NLL={r: [.55, .45] for r in ROLES},
            ablation_delta_NLL={r: [.05, -.05] for r in ROLES},
            ablation_mean_abs_logit_change={r: [.01, .02] for r in ROLES})
        if mode == 'continued_role_only':
            value['raw_slot_gradient_mean']['survival'] = {'p': [0.]*4, 'g': [0.]*4}
    return value


def receipt(mode):
    phase = 'role' if mode == 'shared_role' else 'task'
    count = 12 if phase == 'role' else 35
    result = dict(schema='full_slot_research_v1', status='completed', test_used=False,
                  train_count=304, validation_count=76, config=config('joint' if phase == 'role' else mode),
                  source_SHA256={'invented_source.py': 'a'*64},
                  split_identity={'schema': 'invented identity', 'identity_sha256': 'b'*64},
                  prepared_manifest_SHA256='c'*64, model_parameters=245164,
                  initial_model_tensor_SHA256='1'*64, initial_train_rng_SHA256='2'*64,
                  initial_torch_rng_SHA256='3'*64, initial_cuda_rng_SHA256='4'*64,
                  read_normalization=dict(policy='masked_add_eps_l1_v2', epsilon=1e-8, epsilon_on='real_tokens_only'),
                  reference_frozen=True, epoch_logs=[row(phase, i, mode) for i in range(1, count+1)],
                  optimizer_steps={'role': 1824 if phase == 'role' else 0, 'task': 5320 if phase == 'task' else 0},
                  artifacts_SHA256={'role_last.pt': SHARED_CHECKPOINT_SHA256})
    if phase == 'task':
        last = result['epoch_logs'][-1]
        best = dict(epoch=35, C_index=last['validation']['C_index'], NLL=.5)
        result.update(task_gradient_policy='joint' if mode == 'joint' else 'role_only_trainable_head',
                      checkpoint_reload_verified=True, checkpoint_optimizer_and_generator_reload_verified=True,
                      role_parameters_changed_during_task=True, task_optimizer_initial_state_empty=True,
                      task_start_role_parameter_SHA256='5'*64,
                      task_optimizer_initial_groups=[dict(lr=.001, params=[0, 1, 2])],
                      shared_checkpoint_SHA256=SHARED_CHECKPOINT_SHA256, best=best, last=dict(best))
    return result


class ChainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root/'run'
        self.exporter = self.root/'exporter.py'
        self.exporter.write_bytes(b'# invented exporter, never executed\n')
        self.aggregate = self.root/'aggregates.json'
        self.aggregate.write_text(json.dumps(dict(model_executed=False, patient_rows_exported=False,
                                                 patients=76, known_aggregate=[.1, .2])), encoding='utf-8')
        self.documents = {mode+'/receipt.json': receipt(mode)
                          for mode in ('shared_role', 'joint', 'continued_role_only')}
        self.documents['paired_receipt.json'] = dict(schema='matched35_pair_v1', status='completed',
            test_used=False, synthetic_only=False, total_epochs=82, expected_total_epochs=82,
            shared_role_epochs=12, common_start_verified=True, matched_input_draws_verified=True,
            task_optimizer_reset_verified=True, config=config('joint'),
            source_SHA256={'invented_source.py': 'a'*64}, shared_checkpoint_SHA256=SHARED_CHECKPOINT_SHA256,
            arms={})

    def save(self):
        hashes = dict(original_exporter=digest(self.exporter.read_bytes()), aggregates=digest(self.aggregate.read_bytes()))
        for mode in ('shared_role', 'joint', 'continued_role_only'):
            name = mode+'/receipt.json'
            path = self.run/name
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = (json.dumps(self.documents[name], allow_nan=True)+'\n').encode()
            path.write_bytes(raw)
            hashes[name] = digest(raw)
        pair = self.documents['paired_receipt.json']
        for mode in ('joint', 'continued_role_only'):
            doc = self.documents[mode+'/receipt.json']
            pair['arms'][mode] = dict(status='completed', epochs=35, receipt_SHA256=hashes[mode+'/receipt.json'],
                                       best=copy.deepcopy(doc['best']), last=copy.deepcopy(doc['last']))
        path = self.run/'paired_receipt.json'
        path.write_bytes((json.dumps(pair)+'\n').encode())
        hashes['paired_receipt.json'] = digest(path.read_bytes())
        return hashes

    def verify(self):
        hashes = self.save()
        return _verify_chain(self.run, self.exporter, self.aggregate, hashes)

    def test_complete82_and_safe_recomputed_output(self):
        self.documents['joint/receipt.json']['unknown_private_extension'] = ['INVENTED_PRIVATE_SENTINEL']
        before = self.save()
        result = _verify_chain(self.run, self.exporter, self.aggregate, before)
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['phase_epochs'], {'shared_role': 12, 'joint': 35, 'continued_role_only': 35})
        self.assertEqual(result['input_SHA256'], before)
        self.assertAlmostEqual(result['paired_C_index_mean'], .01)
        self.assertEqual(result['p_common_observation_denominator_per_epoch'], 456)
        self.assertNotIn('INVENTED_PRIVATE_SENTINEL', json.dumps(result))
        self.assertNotIn('split_identity', json.dumps(result))
        for name, sha in before.items():
            path = self.exporter if name == 'original_exporter' else self.aggregate if name == 'aggregates' else self.run/name
            self.assertEqual(digest(path.read_bytes()), sha)

    def test_fixed_public_pins_reject_unrelated_receipts(self):
        self.save()
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            _verify_chain(self.run, self.exporter, self.aggregate, EXPECTED_SHA256)

    def test_exporter_or_aggregate_byte_change_rejected(self):
        for name in ('original_exporter', 'aggregates'):
            with self.subTest(name=name):
                pins = self.save()
                path = self.exporter if name == 'original_exporter' else self.aggregate
                previous = path.read_bytes()
                path.write_bytes(previous+b'\n')
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    _verify_chain(self.run, self.exporter, self.aggregate, pins)
                path.write_bytes(previous)

    def test_missing_duplicate_and_reordered_epoch_rejected(self):
        for mutation in ('missing', 'duplicate', 'reordered'):
            original = copy.deepcopy(self.documents['joint/receipt.json']['epoch_logs'])
            rows = self.documents['joint/receipt.json']['epoch_logs']
            if mutation == 'missing':
                rows.pop(2)
            elif mutation == 'duplicate':
                rows[2]['epoch'] = 2
            else:
                rows[2], rows[3] = rows[3], rows[2]
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'epoch'):
                self.verify()
            self.documents['joint/receipt.json']['epoch_logs'] = original

    def test_false_best_last_or_wrong_budget_rejected(self):
        for field in ('best', 'last', 'optimizer_steps'):
            doc = self.documents['joint/receipt.json']
            before = copy.deepcopy(doc[field])
            if field == 'optimizer_steps':
                doc[field]['task'] = 5319
            else:
                doc[field]['C_index'] = .99
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify()
            doc[field] = before

    def test_false_denominator_axis_and_fraction_rejected(self):
        val = self.documents['joint/receipt.json']['epoch_logs'][0]['validation']
        original = copy.deepcopy(val)
        mutations = (
            lambda: val['raw_reading_mass']['p_common'].update(observations=76),
            lambda: val['read_diagnostics']['p_common']['metrics']['query_norm'].update(mean_by_iteration_slot=[[.1]*76]*3),
            lambda: val['raw_reading_mass']['p_common'].update(below_threshold_fraction=.5),
            lambda: val['ablation_delta_NLL'].update(p_common=[0.]*76),
            lambda: val['ablation_delta_NLL'].update(p_common=[.15, .15]),
        )
        for index, mutate in enumerate(mutations):
            mutate()
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.verify()
            val.clear(); val.update(copy.deepcopy(original))

    def test_nan_inf_boolean_and_text_metric_rejected(self):
        for value in (float('nan'), float('inf'), True, 'INVENTED_PRIVATE_SENTINEL'):
            self.documents['joint/receipt.json']['epoch_logs'][0]['train_loss'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.verify()

    def test_common_start_or_input_draw_mismatch_rejected(self):
        doc = self.documents['continued_role_only/receipt.json']
        for field in ('initial_model_tensor_SHA256', 'initial_cuda_rng_SHA256', 'split_identity', 'task_optimizer_initial_groups'):
            before = doc[field]
            doc[field] = {'different': True} if isinstance(before, dict) else [{'lr': .001, 'params': [999]}] if isinstance(before, list) else '9'*64
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify()
            doc[field] = before
        doc['epoch_logs'][1]['input_draw_SHA256'] = '9'*64
        with self.assertRaisesRegex(ValueError, 'input draw trajectories'):
            self.verify()

    def test_control_gradient_semantics_rejected(self):
        term = self.documents['continued_role_only/receipt.json']['epoch_logs'][0]['raw_slot_gradient_mean']
        term['survival']['p'][0] = .1
        with self.assertRaisesRegex(ValueError, 'survival-to-role edge'):
            self.verify()
        term['survival']['p'][0] = 0.
        term['combined']['g'][1] = .2
        with self.assertRaisesRegex(ValueError, 'combined gradient'):
            self.verify()

    def test_unknown_private_aggregate_fields_and_patient_axis_rejected(self):
        cases = ({'patient_id': 'INVENTED_PRIVATE_SENTINEL'}, {'by_patient_loss': [1.]},
                 {'safe_name': [1.]*76}, {'note': 'TCGA-00-0000'}, {'prediction': [1., 2.]})
        for value in cases:
            self.aggregate.write_text(json.dumps(value), encoding='utf-8')
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.verify()

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate object key'):
            _json_bytes(b'{"same":1,"same":2}')

    def test_cli_never_overwrites_input_or_existing_file(self):
        self.save()
        for output in (self.run/'new.json', self.aggregate, self.exporter):
            with self.subTest(output=output), self.assertRaises(ValueError):
                main(['--run-root', str(self.run), '--original-exporter', str(self.exporter),
                      '--aggregates', str(self.aggregate), '--output', str(output)])
        output = self.root/'existing.json'; output.write_text('KEEP')
        with self.assertRaises(FileExistsError):
            main(['--run-root', str(self.run), '--original-exporter', str(self.exporter),
                  '--aggregates', str(self.aggregate), '--output', str(output)])
        self.assertEqual(output.read_text(), 'KEEP')

    def test_help_needs_no_site_packages(self):
        script = Path(__file__).resolve().parents[1]/'scripts/verify_matched35_chain.py'
        result = subprocess.run([sys.executable, '-S', '-B', str(script), '--help'],
                                capture_output=True, text=True, check=True)
        self.assertIn('--original-exporter', result.stdout)


if __name__ == '__main__':
    unittest.main()
