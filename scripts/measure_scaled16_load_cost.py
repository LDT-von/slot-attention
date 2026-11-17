"""Measure the REAL data-loading cost of the scaled arm. Reads patient caches.

This script measures the CPU COST of the scaled batch builder on the real
prepared cache: how long it takes to load one patient's real patches, take the
4096-patch subset, standardize it, and build the per-pathway omics inputs.

It is a LOAD-COST measurement, not a training run and not a performance
measurement.  It reads patient feature values (which live only in the ignored
local_evidence tree) and writes no patient identifiers, no per-patient arrays
and no model outputs.

The reason this measurement exists: the archived arms spent roughly 290
SECONDS PER EPOCH on CPU work with batch size 2, and the scaled arm raises
both the batch size and the patch width.  If the loader dominates, the scaled
arm's schedule is not affordable and that must be known BEFORE a long run.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from research_models.common_private_synergy.patient_pilot import ROOT, sha
from research_models.common_private_synergy.patient_pilot_engine import PreparedStore
from research_models.common_private_synergy.scaled.adapter import (
    ScaledContentAdapter, fit_pathway_moments)
from research_models.common_private_synergy.scaled.engine import collect_target_matrices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--patients', type=int, default=24)
    parser.add_argument('--patches', type=int, default=4096)
    args = parser.parse_args()

    root = args.prepared_root.resolve()
    manifest = json.loads((root / 'prepared/manifest.json').read_text(encoding='utf-8'))
    store = PreparedStore(root, manifest)
    train = manifest['folds']['0']['train']
    ids = train[:args.patients]

    print('=== 1 the cost of the per-pathway moment fit on train rows ===')
    start = time.time()
    pathway_mean, pathway_sd = fit_pathway_moments(store, ids, manifest['omic_sizes'])
    fit_seconds = time.time() - start
    print('  %d patients in %.2f s -> %.3f s/patient'
          % (len(ids), fit_seconds, fit_seconds / len(ids)))

    print('\n=== 2 the cost of one scaled batch of 8 patients ===')
    target_matrix, mean_matrix, g_matrix = collect_target_matrices(store, ids, 'spread')
    adapter = ScaledContentAdapter(
        target_matrix, g_matrix, manifest['omic_sizes'], 16,
        input_mean=mean_matrix.mean(0), input_sd=mean_matrix.std(0, unbiased=False).clamp_min(1e-4),
        pathway_mean=pathway_mean, pathway_sd=pathway_sd, p_target_mode='spread')

    generator = torch.Generator().manual_seed(3)
    per_patient, per_batch = [], []
    for start_index in range(0, len(ids) - 7, 8):
        batch_ids = ids[start_index:start_index + 8]
        t0 = time.time()
        ps, per_patient_omics = [], []
        for pid in batch_ids:
            t1 = time.time()
            patches, omics = store.load(pid)
            t2 = time.time()
            adapter.p_target(patches)
            t3 = time.time()
            count = min(len(patches), args.patches)
            indices = torch.randperm(len(patches), generator=generator)[:count].sort().values
            adapter.patches_input(patches[indices])
            t4 = time.time()
            adapter.omics_input(omics)
            t5 = time.time()
            per_patient.append({'load': t2 - t1, 'p_target': t3 - t2,
                                'patch_subset': t4 - t3, 'omics_input': t5 - t4})
            ps.append(None)
            per_patient_omics.append(None)
        per_batch.append(time.time() - t0)

    def mean_of(key):
        return statistics.fmean(x[key] for x in per_patient)

    print('  per patient: load %.3f s | p_target %.3f s | patch subset %.3f s | omics input %.3f s'
          % (mean_of('load'), mean_of('p_target'), mean_of('patch_subset'), mean_of('omics_input')))
    print('  per batch of 8: %.3f s (median %.3f s)'
          % (statistics.fmean(per_batch), statistics.median(per_batch)))

    print('\n=== 3 the projected cost of the archived schedule ===')
    steps_per_epoch = (304 + 7) // 8
    per_epoch = statistics.fmean(per_batch) * steps_per_epoch
    print('  %d steps/epoch x %.3f s = %.1f s/epoch of CPU loading'
          % (steps_per_epoch, statistics.fmean(per_batch), per_epoch))
    print('  47 epochs (12 role + 35 task) -> %.1f min of CPU loading alone'
          % (per_epoch * 47 / 60))

    print('\n=== 4 what the same measurement says about the ARCHIVED batch size 2 ===')
    per_batch2 = statistics.fmean(per_batch) * 2 / 8
    steps2 = (304 + 1) // 2
    print('  %d steps/epoch x %.3f s = %.1f s/epoch' % (steps2, per_batch2, per_batch2 * steps2))
    print('  47 epochs -> %.1f min' % (per_batch2 * steps2 * 47 / 60))

    print('\nNOTE: this measures the CPU LOAD only. It says nothing about the '
          'model score.')


if __name__ == '__main__':
    main()
