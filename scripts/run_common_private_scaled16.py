"""Runner for the scaled 16-slot role architecture. Defaults to a nonexecuting plan.

Added as a NEW script.  The archived matched35 / spread scripts and sources
stay byte for byte so that the historical receipts keep their provenance.

The architecture scaled here follows the user's specification:

  1. slot allocation 4 / 4 / 4 / 4, i.e. 16 slots in total
  2. slot width 256
  3. omics encoding by 329 INDEPENDENT networks
  4. omics input: RAW pathway expression -> 329 MLPs
  5. reconstruction heads: DEFERRED, so this arm has no reconstruction loss
  6. batch size 8

Everything else is the archived matched35 / spread protocol: the split, the
bags, the 4096 real patches, the without-replacement draw and the zero-padding
rule, the role target statistic, the train-fold target reduction, the frozen
role reference, the bounded readout, the four-bin discrete hazard survival
objective, the optimiser, the learning rates, the epoch budget, the selection
rule and the evaluation.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_models.common_private_synergy.patient_pilot import ROOT, private_path, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=ROOT / 'configs/common_private_scaled16_v1.json')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='continue an interrupted run from its last checkpoint')
    parser.add_argument('--prepared-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()

    from research_models.common_private_synergy.patient_pilot import config as load_config
    from research_models.common_private_synergy.scaled.engine import validate_protocol
    c = load_config(args.config)
    validate_protocol(c)
    c['configuration_SHA256'] = sha(args.config)
    if not args.run:
        print(json.dumps({
            'config': c,
            'execution': False,
            'architecture': {
                'slot_allocation': c['allocation'],
                'total_slots': sum(c['allocation'][m][0] + c['allocation'][m][1] for m in ('p', 'g')),
                'slot_width': c['width'],
                'omics_encoder': '329 independent per-pathway MLPs (raw pathway expression -> 329 tokens)',
                'omics_encoder_parameter_reference': (
                    'the original SlotSPE builds one two-layer MLP per pathway in '
                    'init_per_path_model; that shape is reproduced here, the original is not modified'),
                'reconstruction_heads': 'deferred by instruction; no reconstruction loss',
                'batch_size': c['batch_size'],
            },
            'total_epochs': c['role_epochs'] + c['task_epochs'],
            'expected_train_count': 304, 'expected_validation_count': 76,
            'validation_seed': c['seed'] + 19000, 'test_used': False,
            'measured_probe_ceiling': {
                'bagstd_td16_rank6': 0.7071,
                'note': ('fold mean over six cohort/fold cells of a linear probe of the '
                         'architecture own reference geometry; NOT a trained-model score '
                         'and NOT a patient-performance claim'),
            },
            'notice': ('Protocol candidate only: patient execution requires complete '
                       'synthetic and environment verification plus explicit execution '
                       'authorisation.'),
        }, indent=2))
        return

    if not args.prepared_root or not args.output_root:
        parser.error('--run requires --prepared-root and --output-root')
    out = private_path(args.output_root)
    if out.exists() and not args.resume:
        raise FileExistsError('preserve prior output; choose a fresh output root '
                              'or pass --resume to continue it')

    from slotspe_proof.patient import _config, _manifest, _source_hashes
    from research_models.common_private_synergy.pilot_split import validate_split
    from research_models.common_private_synergy.scaled.engine import run_scaled_arm

    root = args.prepared_root.resolve()
    data = _config(json.loads((root / 'resolved_config.json').read_text(encoding='utf-8')))
    if (data['study'] != c['study'] or data['folds'] != [0] or data['patches'] != c['patches']
            or data['splits_dir'] != '5fold_uni2h' or data['label_col'] != 'survival_months_dss'):
        raise ValueError('prepared data scope mismatch')
    local_data = {**data, 'source_root': str(ROOT / 'third_party/SlotSPE')}
    if _source_hashes(data) != _source_hashes(local_data):
        raise RuntimeError('prepared original source differs from this checkout')

    manifest = json.loads((root / 'prepared/manifest.json').read_text(encoding='utf-8'))
    identity = validate_split(data, manifest)
    try:
        manifest = _manifest(data, root, verify_cache=True)
    except Exception:
        raise RuntimeError('prepared-cache integrity verification failed') from None
    if not manifest.get('patch_policy', '').startswith('Every real patch is cached'):
        raise ValueError('full-real-patch cache required')

    def recheck():
        current = json.loads((root / 'prepared/manifest.json').read_text(encoding='utf-8'))
        verified = validate_split(data, current)
        if current != manifest or verified != identity:
            raise RuntimeError('prepared split changed after admission')
        return verified

    hashes = {
        'research_models/common_private_synergy/scaled/engine.py': sha(
            ROOT / 'research_models/common_private_synergy/scaled/engine.py'),
        'research_models/common_private_synergy/scaled/role_model.py': sha(
            ROOT / 'research_models/common_private_synergy/scaled/role_model.py'),
        'research_models/common_private_synergy/scaled/omics_encoder.py': sha(
            ROOT / 'research_models/common_private_synergy/scaled/omics_encoder.py'),
        'research_models/common_private_synergy/scaled/adapter.py': sha(
            ROOT / 'research_models/common_private_synergy/scaled/adapter.py'),
        'research_models/common_private_synergy/components/spread_target.py': sha(
            ROOT / 'research_models/common_private_synergy/components/spread_target.py'),
        'research_models/common_private_synergy/components/contracts.py': sha(
            ROOT / 'research_models/common_private_synergy/components/contracts.py'),
        'research_models/common_private_synergy/components/role_model.py': sha(
            ROOT / 'research_models/common_private_synergy/components/role_model.py'),
        'research_models/common_private_synergy/components/role_reference.py': sha(
            ROOT / 'research_models/common_private_synergy/components/role_reference.py'),
        'research_models/common_private_synergy/components/slots.py': sha(
            ROOT / 'research_models/common_private_synergy/components/slots.py'),
        'research_models/common_private_synergy/components/losses.py': sha(
            ROOT / 'research_models/common_private_synergy/components/losses.py'),
        'research_models/common_private_synergy/full_slot_diagnostics.py': sha(
            ROOT / 'research_models/common_private_synergy/full_slot_diagnostics.py'),
        'research_models/common_private_synergy/patient_pilot.py': sha(
            ROOT / 'research_models/common_private_synergy/patient_pilot.py'),
        'research_models/common_private_synergy/patient_pilot_engine.py': sha(
            ROOT / 'research_models/common_private_synergy/patient_pilot_engine.py'),
        'research_models/common_private_synergy/pilot_split.py': sha(
            ROOT / 'research_models/common_private_synergy/pilot_split.py'),
        'scripts/run_common_private_scaled16.py': sha(
            ROOT / 'scripts/run_common_private_scaled16.py'),
        'slotspe_proof/patient.py': sha(ROOT / 'slotspe_proof/patient.py'),
    }
    try:
        result = run_scaled_arm(root, out, manifest, c, hashes, split_guard=recheck,
                                resume=bool(args.resume))
    except Exception:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'exception_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise SystemExit('Run failed; inspect the private run receipt and traceback.') from None
    print(json.dumps({'status': result['status'], 'best': result['best'], 'last': result['last'],
                      'total_epochs': len(result['epoch_logs']),
                      'model_parameters': result['model_parameters'],
                      'omics_encoder_parameters': result['omics_encoder_parameters'],
                      'slot_budget': result['slot_budget'],
                      'scope': result['scope']}, indent=2))


if __name__ == '__main__':
    main()
