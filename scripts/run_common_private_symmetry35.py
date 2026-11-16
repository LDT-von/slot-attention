"""Symmetry-breaking arm runner. Defaults to a nonexecuting plan.

Added as a NEW script. The archived matched35 script and sources stay byte for
byte so that the historical receipts keep their provenance.

The arm trains the archived matched35 schedule with one changed variable: the
slot encoder gains a zero-initialised learnable per-slot query bias.
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
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/common_private_symmetry35_v1.json')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--prepared-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()

    from research_models.common_private_synergy.patient_pilot import config as load_config
    from research_models.common_private_synergy.symmetry_arm import validate_protocol
    c = load_config(args.config)
    validate_protocol(c)
    c['configuration_SHA256'] = sha(args.config)
    if not args.run:
        print(json.dumps({
            'config': c, 'execution': False,
            'arm': 'symmetry_broken_per_slot_query_bias',
            'mechanism': ('one learnable per-slot query bias per encoder, initialised to exactly zero, '
                          'so the initial forward pass equals the archived encoder bit for bit'),
            'survival_objective': 'archived four-bin discrete hazard NLL',
            'difference_from_archived_matched35': 'the slot encoder class only',
            'total_epochs': c['role_epochs'] + c['task_epochs'],
            'expected_train_count': 304, 'expected_validation_count': 76,
            'validation_seed': c['seed'] + 19000, 'test_used': False,
            'notice': ('Protocol candidate only: patient execution requires complete synthetic and '
                       'environment verification plus explicit execution authorisation.'),
        }, indent=2))
        return

    if not args.prepared_root or not args.output_root:
        parser.error('--run requires --prepared-root and --output-root')
    out = private_path(args.output_root)
    if out.exists():
        raise FileExistsError('preserve prior output; choose a fresh output root')

    from slotspe_proof.patient import _config, _manifest, _source_hashes
    from research_models.common_private_synergy.pilot_split import validate_split
    from research_models.common_private_synergy.symmetry_arm import run_symmetry_arm

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
        'research_models/common_private_synergy/components/symmetry_slots.py': sha(
            ROOT / 'research_models/common_private_synergy/components/symmetry_slots.py'),
        'research_models/common_private_synergy/symmetry_arm.py': sha(
            ROOT / 'research_models/common_private_synergy/symmetry_arm.py'),
        'research_models/common_private_synergy/patient_pilot.py': sha(
            ROOT / 'research_models/common_private_synergy/patient_pilot.py'),
        'research_models/common_private_synergy/patient_pilot_engine.py': sha(
            ROOT / 'research_models/common_private_synergy/patient_pilot_engine.py'),
        'research_models/common_private_synergy/pilot_split.py': sha(
            ROOT / 'research_models/common_private_synergy/pilot_split.py'),
        'research_models/common_private_synergy/components/role_model.py': sha(
            ROOT / 'research_models/common_private_synergy/components/role_model.py'),
        'research_models/common_private_synergy/components/role_reference.py': sha(
            ROOT / 'research_models/common_private_synergy/components/role_reference.py'),
        'research_models/common_private_synergy/components/slots.py': sha(
            ROOT / 'research_models/common_private_synergy/components/slots.py'),
        'research_models/common_private_synergy/components/losses.py': sha(
            ROOT / 'research_models/common_private_synergy/components/losses.py'),
        'research_models/common_private_synergy/components/contracts.py': sha(
            ROOT / 'research_models/common_private_synergy/components/contracts.py'),
        'research_models/common_private_synergy/full_slot_diagnostics.py': sha(
            ROOT / 'research_models/common_private_synergy/full_slot_diagnostics.py'),
        'research_models/common_private_synergy/full_slot_epoch.py': sha(
            ROOT / 'research_models/common_private_synergy/full_slot_epoch.py'),
        'scripts/run_common_private_symmetry35.py': sha(
            ROOT / 'scripts/run_common_private_symmetry35.py'),
        'slotspe_proof/patient.py': sha(ROOT / 'slotspe_proof/patient.py'),
    }
    try:
        result = run_symmetry_arm(root, out, manifest, c, hashes, split_guard=recheck)
    except Exception:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'exception_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise SystemExit('Run failed; inspect the private run receipt and traceback.') from None
    print(json.dumps({'status': result['status'], 'best': result['best'], 'last': result['last'],
                      'scope': result['scope']}, indent=2))


if __name__ == '__main__':
    main()
