"""Isolated empirical-role survival pilot. Import/--plan never reads patients.

The pilot uses TRAIN-fitted finite content targets and ordinary linear CCA.
It does not use synthetic source labels, known noise, or biological purity claims.
Prepared inputs reuse the original-matched SlotSPE cache format, not its model.
Only --prepare and --run access patient values. --smoke uses invented patients.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'configs/common_private_patient_pilot.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8'))


def private_path(path):
    p = Path(path).expanduser().resolve()
    base = (ROOT/'local_evidence').resolve()
    if p == base or base not in p.parents:
        raise ValueError('patient outputs must be in a child of ignored repository local_evidence/')
    return p


def config(path=DEFAULT):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    if c['study'] != 'blca' or c['fold'] != 0 or c['seed'] != 3:
        raise ValueError('pilot scope is BLCA fold0 seed3; no automatic expansion')
    for k in ('patches', 'batch_size', 'target_dim', 'common_rank', 'width', 'iterations',
              'hidden', 'bins', 'role_epochs', 'task_epochs', 'torch_threads'):
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError('positive integer required: '+k)
    if c['bins'] != 4 or c['common_rank'] >= c['target_dim']:
        raise ValueError('four time bins and a nonempty residual target required')
    for v in ('p', 'g'):
        if len(c['allocation'][v]) != 2 or any(type(n) is not int or n < 1 for n in c['allocation'][v]):
            raise ValueError('explicit positive common/residual slot counts required')
    for k in ('role_lr', 'task_lr', 'readout_bound'):
        if isinstance(c[k], bool) or not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError('positive finite value required: '+k)
    return c


def plan(c):
    return {'scope': 'exploratory engineering patient pilot, not validated patient efficacy', 'config': c,
        'targets': 'PCA of full real-patch mean and concatenated pathway expression; train-only standardization/SVD',
        'reference': 'train-only linear CCA in finite standardized PCA target coordinates; no source/noise certification',
        'input': 'WSI features and real mask; pathway padded gene-position vector plus pathway one-hot identity',
        'time_labels': 'train uncensored quantile boundaries; completed-interval right-censor NLL',
        'checkpoint': 'fixed final epoch in each phase; validation evaluated once after training; test unused',
        'mask_policy': 'research model masks padded tokens; original SlotSPE source and loader unmodified',
        'comparison_limit': 'no matched S0 arm yet; historical C-index has different protocol and cannot establish improvement',
        'execution': {'patient_values_read': False, 'training': False, 'server_execution': False}}


def source_hashes():
    names = ['research_models/common_private_synergy/patient_pilot.py',
             'research_models/common_private_synergy/patient_pilot_engine.py',
             'research_models/common_private_synergy/components/role_model.py',
             'research_models/common_private_synergy/components/role_reference.py',
             'research_models/common_private_synergy/components/slots.py',
             'research_models/common_private_synergy/components/losses.py',
             'slotspe_proof/patient.py']
    return {name: sha(ROOT/name) for name in names}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group()
    for name in ('plan', 'doctor', 'smoke', 'prepare', 'run'):
        modes.add_argument('--'+name, action='store_true')
    p.add_argument('--config', type=Path, default=DEFAULT)
    p.add_argument('--data-config', type=Path)
    p.add_argument('--prepared-root', type=Path)
    p.add_argument('--output-root', type=Path)
    a = p.parse_args(argv); c = config(a.config)
    if not any((a.doctor, a.smoke, a.prepare, a.run)):
        print(json.dumps(plan(c), ensure_ascii=False, indent=2)); return
    if a.doctor:
        import importlib.util
        dependencies = {k: importlib.util.find_spec(k) is not None for k in ('torch','numpy','pandas','h5py','sksurv')}
        result = {'mode': 'metadata-only doctor', 'dependencies': dependencies,
                  'source_hashes': source_hashes(), 'patient_values_read': False, 'training': False,
                  'prepared_manifest_exists': bool(a.prepared_root and (a.prepared_root/'prepared/manifest.json').is_file()),
                  'next': 'prepared manifest/config and patient inputs are fully verified only in explicit --run'}
        result['server_data_environment_present'] = {k:bool(os.environ.get(k)) for k in
            ('SLOTSPE_METADATA_ROOT','SLOTSPE_FEATURES_ROOT')}
        print(json.dumps(result, indent=2));
        if not all(dependencies.values()): raise SystemExit(2)
        return
    if not a.output_root: p.error('fresh --output-root required for execution')
    out = private_path(a.output_root)
    if out.exists(): raise FileExistsError('preserve existing run; select a fresh output-root')
    if a.smoke:
        from .patient_pilot_engine import synthetic_smoke
        result = synthetic_smoke(out)
        print(json.dumps(result, indent=2)); return
    from slotspe_proof.patient import _config, _manifest, prepare
    if a.prepare:
        if a.data_config:
            supplied = json.loads(a.data_config.read_text(encoding='utf-8'))
        else:
            if not all(os.environ.get(k) for k in ('SLOTSPE_METADATA_ROOT','SLOTSPE_FEATURES_ROOT')):
                p.error('--prepare needs --data-config or both existing SLOTSPE_METADATA_ROOT / SLOTSPE_FEATURES_ROOT')
            supplied = {'source_root':str(ROOT/'third_party/SlotSPE'),
                'metadata_root':os.environ['SLOTSPE_METADATA_ROOT'],
                'features_root':os.environ['SLOTSPE_FEATURES_ROOT'],'study':'blca',
                'label_col':'survival_months_dss','signature':'combine','folds':[0],'seeds':[3],
                'patches':c['patches'],'splits_dir':'5fold_uni2h'}
        data_config = _config(supplied)
    else:
        if not a.prepared_root: p.error('--run needs --prepared-root')
        data_config = _config(json.loads((a.prepared_root/'resolved_config.json').read_text(encoding='utf-8')))
    if (data_config['study'] != c['study'] or data_config['folds'] != [0]
            or data_config['patches'] != c['patches'] or data_config['splits_dir'] != '5fold_uni2h'
            or data_config['label_col'] != 'survival_months_dss'
            or Path(data_config['source_root']).resolve() != (ROOT/'third_party/SlotSPE').resolve()):
        raise ValueError('data scope must match BLCA DSS fold0 / 5fold_uni2h / patch budget / local frozen original')
    if a.prepare:
        out.mkdir(parents=True)
        write(out/'resolved_config.json', data_config)
        prepare(data_config, out)
        print('Prepared inputs only; no model trained. Next use --run --prepared-root '+str(out)); return
    manifest = _manifest(data_config, a.prepared_root, verify_cache=True)
    if not manifest.get('patch_policy','').startswith('Every real patch is cached'):
        raise ValueError('full-real-patch cache required; old truncated caches are refused')
    from .patient_pilot_engine import run_prepared
    result = run_prepared(a.prepared_root, out, manifest, c, source_hashes())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
