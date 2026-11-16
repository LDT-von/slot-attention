"""Gated lambda05 mechanism study; 35 new task epochs, no new role warmup. Defaults to a nonexecuting plan."""
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
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/common_private_lambda05_v1.json')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--prepared-root', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--original-exporter', type=Path)
    parser.add_argument('--aggregates', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    c = config(args.config)
    from research_models.common_private_synergy.lambda05 import validate_protocol
    validate_protocol(c)
    c['configuration_SHA256'] = sha(args.config)
    if not args.run:
        print(json.dumps({'config': c, 'execution': False, 'selection': 'C-index; exact tie lower NLL, then earlier epoch',
                          'total_new_epochs': 35, 'new_role_epochs': 0, 'reused_endpoint_scales': [0, 1],
                          'role_survival_gradient_scale': .5, 'head_survival_gradient_scale': 1.,
                          'expected_train_count': 304, 'expected_validation_count': 76,
                          'validation_seed': c['seed'] + 19000, 'test_used': False,
                          'notice': 'Protocol candidate only: patient execution requires complete synthetic and environment verification.'}, indent=2))
        return
    if not all((args.prepared_root, args.output_root, args.run_root, args.original_exporter, args.aggregates)):
        parser.error('--run requires --prepared-root, --output-root, --run-root, --original-exporter and --aggregates')
    out = private_path(args.output_root)
    if out.exists():
        raise FileExistsError('preserve prior output; choose a fresh output root')
    # The pinned read-only historical chain must pass before any model/data imports.
    from scripts.verify_matched35_chain import verify_chain
    chain = verify_chain(args.run_root, args.original_exporter, args.aggregates)
    if chain.get('status') != 'verified':
        raise ValueError('historical chain gate did not verify')
    from slotspe_proof.patient import _config, _manifest, _source_hashes
    from research_models.common_private_synergy.pilot_split import validate_split
    from research_models.common_private_synergy.lambda05 import run_lambda05
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
    for name in ('scripts/run_common_private_lambda05.py',
                 'scripts/verify_matched35_chain.py',
                 'scripts/audit_matched35_aggregates.py',
                 'research_models/common_private_synergy/lambda05.py',
                 'research_models/common_private_synergy/matched35.py',
                 'research_models/common_private_synergy/components/contracts.py',
                 'research_models/common_private_synergy/full_slot_diagnostics.py',
                 'research_models/common_private_synergy/full_slot_epoch.py',
                 'research_models/common_private_synergy/full_slot_engine.py'):
        hashes[name] = sha(ROOT / name)
    try:
        result = run_lambda05(root, out, manifest, c, hashes, split_guard=recheck,
                              run_root=args.run_root, original_exporter=args.original_exporter,
                              aggregates=args.aggregates)
    except Exception:
        # The traceback can contain private cache paths; retain it only in
        # the ignored run directory, and keep terminal output generic.
        out.mkdir(parents=True, exist_ok=True)
        (out / 'exception_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise SystemExit('Run failed; inspect the private run receipt and traceback.') from None
    print(json.dumps({'status': result['status'], 'total_new_epochs': result['total_new_epochs'],
                      'best': result['best'], 'last': result['last'],
                      'scope': result['scope']}, indent=2))


if __name__ == '__main__':
    main()
