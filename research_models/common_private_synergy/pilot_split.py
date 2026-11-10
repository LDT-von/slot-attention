"""Explicit split selection and admission for the isolated patient pilot.

Import is standard-library only. Legacy queues and third_party are untouched.
Old manifests without a complete split ledger are refused, never rewritten.
"""
import contextlib
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

SCHEMA = 'pilot_explicit_split_v1'
ROLES = ('train', 'val')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def id_summary(ids):
    ids = list(ids)
    if any(not isinstance(x, str) or not x or x != x.strip() for x in ids):
        raise ValueError('nonempty exact string identities required')
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate identities')
    return {'count': len(ids), 'set_sha256': digest(sorted(ids)), 'order_sha256': digest(ids)}


def read_split(config, fold):
    name = config.get('splits_dir', '5fold')
    if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('split directory must be one explicit directory name')
    path = Path(config['metadata_root'])/'splits'/name/config['study']/f'fold_{fold}.csv'
    info = {'path': str(path.absolute()), 'realpath': str(path.resolve()), 'sha256': file_hash(path)}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not all(r in reader.fieldnames for r in ROLES):
            raise ValueError('split CSV needs train and val columns')
        ids = {r: [] for r in ROLES}
        for row in reader:
            for r in ROLES:
                value = row[r]
                if value is not None and value.strip():
                    ids[r].append(value)
    for r in ROLES:
        id_summary(ids[r])
        if not ids[r]: raise ValueError('empty requested '+r+' split')
    if set(ids['train']) & set(ids['val']):
        raise ValueError('requested train/val overlap')
    if file_hash(path) != info['sha256']:
        raise RuntimeError('split changed while reading')
    return info, ids


def split_dataset_class(native_class, config, observations):
    """Override only the new pilot's dataset class, never the native class."""
    class ExplicitSplitDataset(native_class):
        def _load_split(self):
            info, ids = read_split(config, self.fold)
            key = (str(self.fold), self.split_key)
            observations[key] = info
            clinical = self.dataset_factory.clinical_df
            selected = clinical[clinical['case id'].isin(ids[self.split_key])].copy()
            selected = selected.reset_index(drop=True)
            id_summary(selected['case id'].astype(str).tolist())
            return selected
    return ExplicitSplitDataset


def fold_identity(requested, actual, ids, retained, excluded):
    if requested != actual:
        raise ValueError('requested and actually opened split differ')
    summaries = {}
    for r in ROLES:
        asked, kept = set(ids[r]), set(retained[r])
        entries = [e for e in excluded if e['split'] == r]
        removed = [e['patient_id'] for e in entries]
        id_summary(removed); id_summary(retained[r])
        if kept & set(removed) or asked != kept | set(removed):
            raise ValueError('requested must equal disjoint retained plus excluded for '+r)
        if any(not e.get('reason') or not e.get('source') for e in entries):
            raise ValueError('every exclusion needs reason and source')
        summaries[r] = {'requested': id_summary(ids[r]), 'retained': id_summary(retained[r]),
                        'excluded': id_summary(removed),
                        'excluded_reasons': dict(sorted(Counter(e['reason'] for e in entries).items()))}
    if set(retained['train']) & set(retained['val']):
        raise ValueError('retained train/val overlap')
    return {'requested': requested, 'actual': actual, 'groups': summaries}


def validate_split(config, manifest):
    """Read split metadata only; no tensors, clinical values or cache loading."""
    identity = manifest.get('split_identity')
    if not isinstance(identity, dict) or identity.get('schema') != SCHEMA:
        raise ValueError('legacy/unverified split identity: retain this cache as historical; prepare a fresh cache')
    expected_folds = {str(f) for f in config['folds']}
    if set(manifest['folds']) != expected_folds or set(identity.get('folds', {})) != expected_folds:
        raise ValueError('split identity fold scope mismatch')
    checked = {}
    for f in config['folds']:
        info, ids = read_split(config, f)
        row = manifest['folds'][str(f)]; entry = identity['folds'][str(f)]
        if entry['requested'] != info or entry['actual'] != info:
            raise ValueError('split file identity changed or requested/actual mismatch')
        if row['split_csv'] != info['path'] or manifest['input_csv_sha256'].get(info['path']) != info['sha256']:
            raise ValueError('actual split file absent from input file hash ledger')
        if any(e.get('split') not in ROLES for e in row['excluded']):
            raise ValueError('invalid exclusion split')
        rebuilt = fold_identity(info, info, ids, row, row['excluded'])
        if rebuilt != entry:
            raise ValueError('cohort membership/order/exclusion ledger differs from split identity')
        expected_counts = {'train': len(row['train']), 'val': len(row['val']),
                           'excluded': len(row['excluded']), 'overlap': 0}
        if row['counts'] != expected_counts or row['original_split_counts'] != {r: len(ids[r]) for r in ROLES}:
            raise ValueError('split counts mismatch')
        if any(pid not in manifest['patients'] for r in ROLES for pid in row[r]):
            raise ValueError('retained patient missing from cache index')
        checked[str(f)] = rebuilt
    return {'schema': SCHEMA, 'folds': checked, 'identity_sha256': digest(checked)}


def prepare(config, output_root):
    """New explicit preparation, using native gene loader and unchanged helpers."""
    import numpy as np
    import pandas as pd
    import torch
    from slotspe_proof import patient as legacy

    config = legacy._config(config)
    root = Path(output_root); prepared = root/'prepared'
    if prepared.exists(): raise FileExistsError('preserve existing/partial preparation; use a fresh root')
    specs = {str(f): read_split(config, f) for f in config['folds']}
    csv_hashes = {str(p): file_hash(p) for p in legacy._csv_paths(config)}
    source_hashes = legacy._source_hashes(config)
    native = legacy._dataset_module(config)
    observed = {}
    Dataset = split_dataset_class(native.SurvivalDataset, config, observed)
    censor = legacy._CENSORS[config['label_col']]

    class TrainFactory(native.SurvivalDatasetFactory):
        def _setup_clinical_data(self):
            path = Path(self.data_path)/'clinical/all'/f'{self.study}.csv'
            table = pd.read_csv(path, dtype={'case id': str})
            if table['case id'].isna().any() or table['case id'].duplicated().any():
                raise ValueError('clinical identities must be present and unique')
            columns = ['case id', self.label_col, self.censorship_var, 'wsi']
            self.raw_clinical = table[columns].copy()
            self.clinical_df = table[columns].dropna().reset_index(drop=True)
            self.clinical_df.insert(2, 'label', -1)
            self.bins = None

    prepared.mkdir(parents=True); (prepared/'cache/patients').mkdir(parents=True)
    with (prepared/'factory_prepare.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        factory = TrainFactory(config['study'], config['metadata_root'], 'Pathways', config['label_col'],
                               signature=config['signature'], n_bins=4, num_patches=config['patches'])
    if not factory.omic_sizes or any(n < 1 for n in factory.omic_sizes):
        raise ValueError('nonempty original pathway schema required')
    if factory.gene_data_df.columns.duplicated().any() or factory.gene_data_df.index.duplicated().any():
        raise ValueError('duplicate RNA identities')
    raw = factory.raw_clinical.set_index('case id')
    patients, failures, folds, identities = {}, {}, {}, {}
    encoding_dim = config['encoding_dim']

    def cache(row, dataset):
        nonlocal encoding_dim
        pid = str(row['case id'])
        if pid in patients: return None
        if pid in failures: return failures[pid]
        if pid not in factory.gene_data_df.columns: return 'missing_rna_patient_column'
        t, c = float(row[config['label_col']]), float(row[censor])
        if not math.isfinite(t) or t < 0 or c not in (0., 1.): return 'invalid_survival_endpoint'
        slides = [s.strip() for s in str(row['wsi']).split(',') if s.strip()]
        if not slides: return 'missing_slide_identifiers'
        paths = [Path(config['features_root'])/(s.rstrip('.svs')+'.pt') for s in slides]
        if not all(p.is_file() for p in paths):
            paths = [p.with_suffix('.h5') for p in paths]
            if not all(p.is_file() for p in paths): return 'at_least_one_missing_slide_feature'
        chunks, slide_info = [], []
        for path in paths:
            before = path.stat()
            if path.suffix == '.h5':
                import h5py
                with h5py.File(path, 'r') as hf: tensor = torch.from_numpy(legacy._h5_feature_array(hf))
            else:
                try: tensor = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                except (RuntimeError, EOFError): tensor = torch.load(path, map_location='cpu', weights_only=True)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or min(tensor.shape) < 1:
                raise ValueError('feature tensor must be nonempty [patch,channel]')
            if encoding_dim is None: encoding_dim = int(tensor.shape[1])
            if tensor.shape[1] != encoding_dim: raise ValueError('feature width changed')
            if not torch.isfinite(tensor).all(): return 'nonfinite_selected_inputs'
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('feature changed during preparation')
            chunks.append(tensor.float())
            slide_info.append({'path': str(path.resolve()), 'patch_count': len(tensor), 'channels': tensor.shape[1],
                               'size_bytes': before.st_size, 'mtime_ns': before.st_mtime_ns})
        patches = torch.cat(chunks)
        omics = dataset.load_genes(pid)
        if len(omics) != len(factory.omic_sizes) or any(x.ndim != 1 or len(x) != n for x, n in zip(omics, factory.omic_sizes)):
            raise ValueError('original pathway schema changed')
        if not all(torch.isfinite(x).all() for x in omics): return 'nonfinite_selected_inputs'
        target = prepared/'cache/patients'/(hashlib.sha256(pid.encode()).hexdigest()+'.pt')
        torch.save({'patient_id': pid, 'patches': patches, 'omics': [x.cpu().float() for x in omics]}, target)
        patients[pid] = {'patient_id': pid, 'time': t, 'censor': c, 'cache_path': target.relative_to(root).as_posix(),
            'cache_sha256': file_hash(target), 'full_patch_count': len(patches), 'slides': slide_info,
            'full_patch_tensor_sha256': hashlib.sha256(patches.contiguous().numpy().tobytes()).hexdigest()}
        return None

    for f in config['folds']:
        requested, ids = specs[str(f)]; retained, excluded = {}, []
        for r in ROLES:
            ds = Dataset(factory, config['features_root'], r, f, encoding_dim or 1)
            if observed[(str(f), r)] != requested: raise RuntimeError('split changed before selection')
            rows = {str(row['case id']): row for _, row in ds.label_df.iterrows()}
            if set(rows)-set(ids[r]): raise ValueError('loader selected patients outside requested split')
            retained[r] = []
            for pid in ids[r]:
                if pid not in rows:
                    if pid not in raw.index: reason = 'missing_clinical_record'
                    elif pd.isna(raw.loc[pid, config['label_col']]): reason = 'missing_clinical_endpoint'
                    elif pd.isna(raw.loc[pid, censor]): reason = 'missing_censor_indicator'
                    elif pd.isna(raw.loc[pid, 'wsi']): reason = 'missing_slide_identifiers'
                    else: raise ValueError('eligible requested patient silently absent from loader')
                    source = 'clinical_table'
                else:
                    reason = cache(rows[pid], ds); source = 'feature_cache'
                if reason:
                    failures[pid] = reason
                    excluded.append({'patient_id': pid, 'split': r, 'reason': reason, 'source': source})
                else: retained[r].append(pid)
            if not retained[r]: raise ValueError('no retained '+r+' patients')
        event_times = [patients[p]['time'] for p in retained['train'] if patients[p]['censor'] < 1]
        if len(event_times) < 4: raise ValueError('at least four train events required')
        _, edges = pd.qcut(event_times, q=4, labels=False, retbins=True)
        if not np.all(np.diff(edges) > 0): raise ValueError('non-distinct train quantiles')
        edges[0], edges[-1] = -np.inf, np.inf
        labels = {p: int(np.searchsorted(edges[1:-1], patients[p]['time'], side='right')) for r in ROLES for p in retained[r]}
        identities[str(f)] = fold_identity(requested, observed[(str(f), 'train')], ids, retained, excluded)
        folds[str(f)] = {'fold': f, 'split_csv': requested['path'], **retained, 'labels': labels, 'excluded': excluded,
            'original_split_counts': {r: len(ids[r]) for r in ROLES},
            'counts': {'train': len(retained['train']), 'val': len(retained['val']), 'excluded': len(excluded), 'overlap': 0},
            'binning': {'fit_subset': 'complete_case_train_uncensored_only', 'q': 4,
                       'edges': ['-inf', *[float(x) for x in edges[1:-1]], 'inf'],
                       'boundary_rule': 'left_closed_right_open; searchsorted internal boundaries side=right',
                       'validation_outcomes_used_for_training_bins': False}}
    if source_hashes != legacy._source_hashes(config) or csv_hashes != {str(p): file_hash(p) for p in legacy._csv_paths(config)}:
        raise RuntimeError('source or metadata changed during preparation')
    manifest = {'schema_version': legacy.SCHEMA_VERSION, 'prepared_utc': legacy._now(),
        'preparation_config': legacy._preparation_config(config), 'source_root': config['source_root'],
        'source_hashes': source_hashes, 'input_csv_sha256': csv_hashes,
        'pilot_prepare_source_SHA256': {'pilot_split.py': file_hash(__file__),
                                      'slotspe_proof/patient.py': file_hash(legacy.__file__)},
        'dataset_module_path': str(Path(config['source_root'])/'dataset/dataset_survival.py'),
        'omic_sizes': [int(x) for x in factory.omic_sizes], 'encoding_dim': encoding_dim,
        'patches': config['patches'], 'folds': folds, 'patients': patients,
        'split_identity': {'schema': SCHEMA, 'folds': identities},
        'counts': {'unique_complete_case_patients': len(patients), 'per_fold': {f: r['counts'] for f, r in folds.items()}},
        'patch_policy': 'Every real patch is cached; each access resamples without replacement and pads short bags.',
        'rna_transform': 'Original pathway load_genes; no added preparation transform',
        'scope': 'explicit-split pilot preparation; not historical baseline reproduction'}
    validate_split(config, manifest)
    legacy._dump(prepared/'manifest.json', manifest)
    return manifest
