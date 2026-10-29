"""Four one-step runs on invented tensors only; no real patient inputs."""
import json
from pathlib import Path

import torch

from slotspe_proof import crossview_engine as engine
from slotspe_proof.crossview_queue import ARMS
from slotspe_proof.redundancy_content import ContentReference


def test_generated_four_arm_step_checkpoint_and_evaluation(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = torch.Generator().manual_seed(941)
        ids = [f'invented-{i}' for i in range(5)]
        patients, fit_cases = {}, []
        for i, pid in enumerate(ids):
            patches = torch.randn(10, 8, generator=rng)
            omics = [torch.randn(3, generator=rng), torch.randn(5, generator=rng)]
            torch.save({'patient_id': pid, 'patches': patches, 'omics': omics}, tmp_path / f'{pid}.pt')
            patients[pid] = {'cache_path': f'{pid}.pt', 'censor': 0., 'time': float(i + 1)}
            if i < 3:
                fit_cases.append({'x_wsi': patches.unsqueeze(0), 'x_omic1': omics[0].unsqueeze(0),
                                  'x_omic2': omics[1].unsqueeze(0)})
        manifest = {'patients': patients, 'patches': 8, 'omic_sizes': [3, 5], 'encoding_dim': 8,
                    'folds': {'0': {'train': ids[:3], 'val': ids[3:],
                                    'labels': {pid: i % 4 for i, pid in enumerate(ids)}}}}
        config = {'source_root': str(Path(__file__).resolve().parents[1] / 'third_party/SlotSPE'),
                  'projection_dim': 16, 'slot_iters': 2, 'device': 'cpu', 'native_alpha': .5,
                  'native_reconstruction_weight': .01, 'epochs': 1, 'patches': 8,
                  'batch_size': 3, 'lr': .0005, 'eval_repeats': 1}
        reference = ContentReference([3, 5], 8, target_dim=32, clusters=4).fit(fit_cases)
        native = engine.existing._build(config, manifest, 'B0_native')
        initial = engine._cpu_state(native.backbone)
        v2 = engine.existing._build(config, manifest, 'V2_separate', initial)
        v2_hash = engine.state_hash(engine._v2_state(v2))
        for arm in ARMS:
            result = engine._run_arm(config, tmp_path, manifest, arm, initial, reference, v2_hash, lambda: None)
            assert result['status'] == 'completed' and result['optimizer_steps'] == 1
            assert result['train_patients'] == 3 and result['validation_patients'] == 2
            assert (tmp_path / 'arms' / arm / 'raw_repeat_00.pt').is_file()
        base = torch.load(tmp_path / 'arms' / ARMS[1] / 'last.pt', weights_only=True)['state_dict']
        compute = torch.load(tmp_path / 'arms' / ARMS[2] / 'last.pt', weights_only=True)['state_dict']
        for name, value in base.items():
            assert torch.equal(value, compute['host.' + name]), name
        for arm in ARMS[2:]:
            result = json.loads((tmp_path / 'arms' / arm / 'crossview_eval.json').read_text())
            assert result['training_patients_for_constant'] == 3
            assert result['eligible_validation_patients'] == 2
            assert result['views_per_patient'] == 1 and not result['donor_pairs_independent']
            for component in ('mass', 'moment'):
                scores = result[component]
                assert scores['own_mse'] >= 0 and scores['training_constant_mse'] >= 0
                assert scores['donor_mse'] >= 0
                assert abs(scores['constant_minus_own'] - (scores['training_constant_mse'] - scores['own_mse'])) < 1e-12
        assert not (tmp_path / 'arms' / ARMS[1] / 'crossview_eval.json').exists()
    finally:
        torch.set_num_threads(old_threads)
