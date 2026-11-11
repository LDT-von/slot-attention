"""Isolated full-slot research entry. Defaults to a nonexecuting plan."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_models.common_private_synergy.patient_pilot import ROOT, config, private_path, sha, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/common_private_full_joint_v1.json')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--prepared-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    c = config(args.config)
    if c['schema'] != 'full_slot_research_v1' or c['task_mode'] not in ('frozen', 'joint'):
        raise ValueError('invalid full-slot schema or task mode')
    if type(c['patience']) is not int or c['patience'] < 1 or not math.isfinite(c['role_weight']) or c['role_weight'] <= 0:
        raise ValueError('positive patience and finite role weight required')
    c['configuration_SHA256'] = sha(args.config)
    if not args.run:
        print(json.dumps({'config': c, 'execution': False, 'selection': 'C-index; exact tie lower NLL, then earlier epoch',
                          'validation_seed': c['seed'] + 19000, 'test_used': False,
                          'notice': 'Protocol candidate only: patient execution requires complete synthetic and environment verification.'}, indent=2))
        return
    if not args.prepared_root or not args.output_root:
        parser.error('--run requires --prepared-root and --output-root')
    out = private_path(args.output_root)
    if out.exists():
        raise FileExistsError('preserve prior output; choose a fresh output root')
    from slotspe_proof.patient import _config, _manifest, _source_hashes
    from research_models.common_private_synergy.pilot_split import validate_split
    from research_models.common_private_synergy.full_slot_engine import run_full
    root = args.prepared_root.resolve()
    data = _config(json.loads((root / 'resolved_config.json').read_text(encoding='utf-8')))
    if (data['study'] != c['study'] or data['folds'] != [0] or data['patches'] != c['patches']
            or data['splits_dir'] != '5fold_uni2h' or data['label_col'] != 'survival_months_dss'):
        raise ValueError('prepared data scope mismatch')
    # Keep the old preparation provenance intact while verifying that this
    # isolated checkout has identical original source files.
    local_data = {**data, 'source_root': str(ROOT / 'third_party/SlotSPE')}
    if _source_hashes(data) != _source_hashes(local_data):
        raise RuntimeError('prepared original source differs from this checkout')
    manifest = json.loads((root / 'prepared/manifest.json').read_text(encoding='utf-8'))
    identity = validate_split(data, manifest)
    try:
        manifest = _manifest(data, root, verify_cache=True)
    except Exception:
        # Cache verification errors can contain sensitive patient filenames.
        raise RuntimeError('prepared-cache integrity verification failed') from None
    if not manifest.get('patch_policy', '').startswith('Every real patch is cached'):
        raise ValueError('full-real-patch cache required')
    def recheck():
        current = json.loads((root / 'prepared/manifest.json').read_text(encoding='utf-8'))
        verified = validate_split(data, current)
        if current != manifest or verified != identity:
            raise RuntimeError('prepared split changed after admission')
        return verified
    hashes = source_hashes()
    for name in ('scripts/run_common_private_full_slot.py',
                 'research_models/common_private_synergy/components/contracts.py',
                 'research_models/common_private_synergy/full_slot_diagnostics.py',
                 'research_models/common_private_synergy/full_slot_epoch.py',
                 'research_models/common_private_synergy/full_slot_engine.py'):
        hashes[name] = sha(ROOT / name)
    try:
        result = run_full(root, out, manifest, c, hashes, split_guard=recheck)
    except Exception:
        # The traceback can contain private cache paths; retain it only in
        # the ignored run directory, and keep terminal output generic.
        out.mkdir(parents=True, exist_ok=True)
        (out / 'exception_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise SystemExit('Run failed; inspect the private run receipt and traceback.') from None
    print(json.dumps({'status': result['status'], 'best': result['best'], 'last': result['last'],
                      'scope': result['scope']}, indent=2))


if __name__ == '__main__':
    main()
