"""Cross-modality and within-modality slot relation tables.

Read-only diagnostic. It reloads existing checkpoints and recomputes
validation relations on the admitted development split. Every artifact goes
to the ignored local_evidence tree; no patient array is written into the
public tree.

WHAT IS COMPARABLE
------------------
The pathology (p) slots and the omics (g) slots live in two DIFFERENT
learned 64-dimensional spaces. A cosine between a raw p slot and a raw g
slot is NOT a quantity of this architecture: the two spaces have no shared
basis, no shared metric and no shared units. Such a number is therefore
NOT reported here.

The architecture does define exactly one cross-modality correspondence: the
frozen role reference. It whitens the two train-fold target matrices,
takes the SVD of their cross-covariance, and uses the leading common_rank
canonical directions. The k-th pathology canonical direction and the k-th
omics canonical direction are a MATCHED PAIR by construction, and that
pairing is the cross-modality correspondence.

Recovering that pairing needs the canonical directions, which the stored
projector does not contain: a projector is q q^T and loses the column
ORDER of q, and the ORDER IS THE PAIRING. So the reference geometry is
RECOMPUTED on the training fold with the architecture's own code path
(`FrozenRoleReference.estimated` semantics), and the recomputation is
VERIFIED against the stored checkpoint projectors and against the stored
singular values. If the verification fails, nothing is reported.

REPORTED QUANTITIES
-------------------
Within a modality:
  * the raw slot-pair cosine matrix of each role group, which is the
    archived slot_summary definition;
  * the cosine of a slot with ITSELF across DIFFERENT patients, which is
    the informative identity-stability number (a slot with itself is 1 by
    definition and carries no information).

Across modalities:
  * for every (pathology slot, omics slot) pair, the Pearson correlation
    across validation patients of their loadings on the MATCHED canonical
    direction k. This is the number that answers whether a pathology slot
    and an omics slot carry the same cross-modal factor;
  * the maximum absolute correlation over all canonical directions, which
    is basis-order independent.

These are geometric and correlational properties on a development
validation slice. They are not a patient-performance claim, not a
correction-efficacy claim and not a biological identification claim.
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_models.common_private_synergy.spread_arm import (
    load_spread_checkpoint, spread_batch)
from research_models.common_private_synergy.patient_pilot_engine import PreparedStore
from research_models.common_private_synergy.components.role_reference import (
    FrozenRoleReference, ReferenceProvenance, _whiten)

RESEARCH = Path(__file__).resolve().parents[1]
PREPARED = Path(os.environ.get('SLOTSPE_PREPARED_ROOT',
                              '/data1/slotspe_split_guard_0934bb6/local_evidence/'
                              'common_private_pilot_prepare_splitfixed_20261110_01'))
CACHE = Path(os.environ.get('SLOTSPE_RELATION_CACHE',
                             RESEARCH / 'local_evidence/slot_relation_cache_20261117_01'))
GROUPS = ('p_common', 'p_residual', 'g_common', 'g_residual')


def build_cache(manifest):
    """Load every needed patient once and cache the raw per-patient statistics."""
    CACHE.mkdir(parents=True, exist_ok=True)
    store = PreparedStore(PREPARED, manifest)
    wanted = {}
    for fold in ('0',):
        for key in ('train', 'val'):
            for pid in manifest['folds'][fold][key]:
                wanted[pid] = True
    stats = {}
    for index, pid in enumerate(sorted(wanted)):
        path = CACHE / ('patient_%s.pt' % pid)
        if path.exists():
            stats[pid] = torch.load(path, weights_only=True)
            continue
        patches, blocks = store.load(pid)
        entry = {'mean': patches.mean(0).double(),
                 'spread': patches.double().std(0),
                 'omics': torch.cat(blocks).double()}
        torch.save(entry, path)
        stats[pid] = entry
        if index % 20 == 0:
            print('  cached %d/%d patients' % (index, len(wanted)), flush=True)
    return stats


def reference_geometry(adapter, train_ids, stats, common_rank):
    """Recompute the reference geometry and return the canonical directions."""
    p = torch.stack([adapter.target(stats[i]['spread'], 'p') for i in train_ids])
    g = torch.stack([adapter.target(stats[i]['omics'], 'g') for i in train_ids])
    pm, pw, pz = _whiten(p, 1e-10)
    gm, gw, gz = _whiten(g, 1e-10)
    u, s, vh = torch.linalg.svd(pz.T @ gz / len(p), full_matrices=True)
    qp = torch.linalg.qr(pw @ u[:, :common_rank], mode='reduced').Q
    qg = torch.linalg.qr(gw @ vh[:common_rank].T, mode='reduced').Q
    return {'p_mean': pm, 'g_mean': gm, 'p_whiten': pw, 'g_whiten': gw,
            'u': u, 'vh': vh, 'singular': s,
            'p_projector': qp @ qp.T, 'g_projector': qg @ qg.T,
            'p_target': p, 'g_target': g}


def verify_geometry(geometry, model, tolerance=1e-6):
    """The recomputation must reproduce the stored reference exactly."""
    checks = {}
    stored_p = model.reference.p_projector.double().cpu()
    stored_g = model.reference.g_projector.double().cpu()
    checks['p_projector_max_abs_difference'] = float((geometry['p_projector'] - stored_p).abs().max())
    checks['g_projector_max_abs_difference'] = float((geometry['g_projector'] - stored_g).abs().max())
    checks['p_mean_max_abs_difference'] = float((geometry['p_mean'] - model.reference.p_mean.double().cpu()).abs().max())
    checks['g_mean_max_abs_difference'] = float((geometry['g_mean'] - model.reference.g_mean.double().cpu()).abs().max())
    stored_s = model.reference.singular_values.double().cpu()
    n = min(len(stored_s), len(geometry['singular']))
    checks['singular_value_max_abs_difference'] = float((geometry['singular'][:n] - stored_s[:n]).abs().max())
    checks['tolerance'] = tolerance
    checks['passed'] = (checks['p_projector_max_abs_difference'] < tolerance
                        and checks['g_projector_max_abs_difference'] < tolerance
                        and checks['singular_value_max_abs_difference'] < tolerance)
    return checks


def canonical_loadings(decoded, geometry, modality):
    """Loadings of a decoded slot content on the canonical directions."""
    if modality == 'p':
        z = (decoded - geometry['p_mean']) @ geometry['p_whiten']
        return z @ geometry['u']
    z = (decoded - geometry['g_mean']) @ geometry['g_whiten']
    return z @ geometry['vh'].T


def pearson(x, y):
    x = x.double() - x.double().mean()
    y = y.double() - y.double().mean()
    denom = x.norm() * y.norm()
    if float(denom) < 1e-12:
        return None
    return float((x @ y) / denom)


def collect(arm_name, config_path, checkpoint_path, stats, manifest, device):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    model, adapter, boundaries = load_spread_checkpoint(checkpoint_path, device)
    train_ids = manifest['folds']['0']['train']
    val_ids = manifest['folds']['0']['val']

    geometry = reference_geometry(adapter, train_ids, stats, config['common_rank'])
    checks = verify_geometry(geometry, model)
    if not checks['passed']:
        raise RuntimeError('recomputed reference geometry does not match the checkpoint: '
                           + json.dumps(checks))

    store = PreparedStore(PREPARED, manifest)
    rng = torch.Generator().manual_seed(config['seed'] + 19000)
    raw_slots = {g: [] for g in GROUPS}
    decoded = {g: [] for g in GROUPS}
    with torch.no_grad():
        for start in range(0, len(val_ids), config['batch_size']):
            ids = val_ids[start:start + config['batch_size']]
            batch = spread_batch(store, ids, adapter, boundaries, config, device, generator=rng)
            out = model(batch['p'], batch['g'], batch['p_mask'], batch['g_mask'])
            for modality in ('p', 'g'):
                part = out['parts'][modality]
                for role, key in (('common', 'common_slots'), ('residual', 'private_slots')):
                    group = modality + '_' + role
                    group_slots = part[key].double()
                    raw_slots[group].append(group_slots.cpu())
                    per_slot = decoded[group]
                    while len(per_slot) < group_slots.shape[1]:
                        per_slot.append([])
                    for index in range(group_slots.shape[1]):
                        pooled = model.pools[group](group_slots[:, index:index + 1, :])
                        per_slot[index].append(model.decoders[group](pooled).cpu())

    raw_slots = {k: torch.cat(v) for k, v in raw_slots.items()}
    # One matrix per slot: [patients, target_dim].
    decoded = {k: [torch.cat(batches) for batches in slot_list]
               for k, slot_list in decoded.items()}

    result = {'arm': arm_name, 'config': config_path, 'checkpoint': checkpoint_path,
              'patients': len(val_ids), 'common_rank': config['common_rank'],
              'target_dim': config['target_dim'],
              'geometry_verification': checks}

    # ---------- within-modality ----------
    within = {}
    for group in GROUPS:
        x = raw_slots[group]
        count = x.shape[1]
        matrix = [[1.0] * count for _ in range(count)]
        pair_stats = {}
        for i in range(count):
            for j in range(i + 1, count):
                cos = torch.nn.functional.cosine_similarity(x[:, i], x[:, j], dim=-1)
                matrix[i][j] = matrix[j][i] = float(cos.mean())
                pair_stats['%d_vs_%d' % (i, j)] = {'mean': float(cos.mean()),
                                                    'std': float(cos.std())}
        stability = {}
        for i in range(count):
            v = x[:, i]
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            gram = v @ v.T
            n = len(v)
            off = gram[~torch.eye(n, dtype=torch.bool)]
            stability['slot_%d' % i] = {'mean': float(off.mean()), 'std': float(off.std()),
                                        'max': float(off.max()), 'min': float(off.min())}
        within[group] = {'slots': count,
                         'cosine_matrix': matrix,
                         'pair_cosine_statistics': pair_stats,
                         'cosine_with_itself_other_patients': stability}
    result['within_modality'] = within

    # ---------- cross-modality ----------
    loadings = {}
    for group in GROUPS:
        modality = group.split('_')[0]
        loadings[group] = [canonical_loadings(matrix, geometry, modality)
                           for matrix in decoded[group]]
    cross = {}
    for p_group in ('p_common', 'p_residual'):
        for g_group in ('g_common', 'g_residual'):
            rows = []
            for i in range(raw_slots[p_group].shape[1]):
                row = []
                for j in range(raw_slots[g_group].shape[1]):
                    lp, lg = loadings[p_group][i], loadings[g_group][j]
                    per_direction = [pearson(lp[:, k], lg[:, k])
                                     for k in range(min(lp.shape[1], lg.shape[1]))]
                    valid = [v for v in per_direction if v is not None]
                    row.append({'matched_direction_correlations': per_direction,
                                'max_abs_correlation': max((abs(v) for v in valid), default=None),
                                'max_correlation_direction': (max(
                                    range(len(valid)), key=lambda k: abs(valid[k]))
                                    if valid else None)})
                rows.append(row)
            cross['%s__vs__%s' % (p_group, g_group)] = rows
    result['cross_modality'] = cross
    result['cross_modality_note'] = (
        'Pearson correlation across validation patients between the loading of a pathology '
        'slot and the loading of an omics slot on the MATCHED canonical direction k of the '
        'frozen role reference. The matching is the architecture own construction: the k-th '
        'pathology canonical direction and the k-th omics canonical direction come from the '
        'same SVD of the whitened cross-covariance. A raw cross-modality cosine is not '
        'reported because the two modalities live in separate learned spaces.')
    return result


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    manifest = json.loads((PREPARED / 'prepared/manifest.json').read_text(encoding='utf-8'))
    print('caching patient statistics...', flush=True)
    stats = build_cache(manifest)
    print('cache ready', flush=True)

    out = RESEARCH / 'local_evidence/slot_relation_tables_20261117_01'
    out.mkdir(parents=True, exist_ok=True)
    arms = [('spread_td8_rank2', RESEARCH / 'configs/common_private_spread35_v1.json',
             RESEARCH / 'local_evidence/spread35_td8_rank2_20261117_01/task_best.pt'),
            ('spread_td16_rank6', RESEARCH / 'configs/common_private_spread35_td16_v1.json',
             RESEARCH / 'local_evidence/spread35_td16_rank6_20261117_01/task_best.pt')]
    for name, config_path, checkpoint_path in arms:
        result = collect(name, str(config_path), str(checkpoint_path), stats, manifest, device)
        path = out / (name + '.json')
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('=== ' + name, flush=True)
        print('  geometry verification:', json.dumps(result['geometry_verification']), flush=True)
        for group in GROUPS:
            w = result['within_modality'][group]
            print('  within %-11s cosine matrix %s' % (group, w['cosine_matrix']))
            print('       self across other patients %s' % (
                {k: round(v['mean'], 4) for k, v in w['cosine_with_itself_other_patients'].items()}))
        for key, rows in result['cross_modality'].items():
            print('  cross ' + key)
            for row in rows:
                print('      ', [None if c['max_abs_correlation'] is None
                                  else round(c['max_abs_correlation'], 4) for c in row])
        print('  written ' + str(path), flush=True)


if __name__ == '__main__':
    main()
