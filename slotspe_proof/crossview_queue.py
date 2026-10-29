"""Independent matched cross-view queue. Only an explicit --run reads patients.

This queue retains the established original sampling and train-fold binning.
Its fixed-final checkpoint policy intentionally differs from the existing
best-validation plugin queue; results must not be combined across policies.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from . import plugin_queue

REPO = plugin_queue.REPO
DEFAULT_CONFIG = plugin_queue.DEFAULT_CONFIG
DEFAULT_OUTPUT = REPO / 'local_evidence/crossview_blca_uni2h_fold0_seed3_run01'
QUICK_OUTPUT = REPO / 'local_evidence/crossview_quick_blca_uni2h_fold0_seed3_run01'
QUICK_PROFILE = 'quick_3epochs_256patches_3repeats_v1'
QUICK_VALUES = {'epochs': 3, 'patches': 256, 'eval_repeats': 3}
ARMS = ('B0_native', 'V2_separate', 'V2_crossview_compute', 'V2_crossview_train')
CHECKPOINT_POLICY = 'fixed_final_epoch_no_validation_selection'
PLUGIN_POLICY = 'official_best_validation_latest_tie'
VIEW_WEIGHT = 1.0
VIEW_TEMPERATURE = 1.0
VIEW_DECODER_SEED = 731
VIEW_SEED_OFFSET = 800_000_000_000


def normalize_config(values, *, metadata_root=None, features_root=None, device=None,
                     require_inputs=False):
    """Validate plugin defaults but explicitly replace only checkpoint selection."""
    if not isinstance(values, dict):
        raise ValueError('configuration must be an object')
    raw = dict(values)
    profile = raw.pop('crossview_profile', None)
    if profile not in (None, QUICK_PROFILE):
        raise ValueError('unknown crossview profile')
    if profile == QUICK_PROFILE:
        for key, value in QUICK_VALUES.items():
            if type(raw.get(key)) is not int or raw[key] != value:
                raise ValueError('quick profile requires its exact locked value: ' + key)
            raw[key] = plugin_queue._LOCKED[key]
    # Permit re-normalizing our own effective configuration while still using
    # the existing strict plugin scope and validation code.
    if raw.get('checkpoint_selection') == CHECKPOINT_POLICY:
        raw['checkpoint_selection'] = PLUGIN_POLICY
    runtime = plugin_queue.normalize_config(raw, metadata_root=metadata_root,
        features_root=features_root, device=device, require_inputs=require_inputs)
    runtime['checkpoint_selection'] = CHECKPOINT_POLICY
    if profile == QUICK_PROFILE:
        runtime.update(QUICK_VALUES)
        runtime['crossview_profile'] = QUICK_PROFILE
    return runtime


def make_plan(values, output_root=DEFAULT_OUTPUT, **overrides):
    runtime = normalize_config(values, **overrides)
    output = plugin_queue.output_path(output_root)
    return {
        'mode': 'plan', 'config': runtime, 'output_root': str(output),
        'arms': [
            {'name': ARMS[0], 'preset': 'B0_native', 'auxiliary_view': 'off'},
            {'name': ARMS[1], 'preset': 'V2_separate', 'auxiliary_view': 'off'},
            {'name': ARMS[2], 'preset': 'V2_separate', 'auxiliary_view': 'compute', 'view_loss_weight': 0.0},
            {'name': ARMS[3], 'preset': 'V2_separate', 'auxiliary_view': 'train', 'view_loss_weight': VIEW_WEIGHT},
        ],
        'training_runs': len(ARMS), 'epoch_budget': len(ARMS) * runtime['epochs'],
        'view_spec': {'temperature': VIEW_TEMPERATURE, 'decoder_seed': VIEW_DECODER_SEED,
                      'auxiliary_seed_offset': VIEW_SEED_OFFSET,
                      'main_forward': 'complete_original_sampled_bag',
                      'view_a_and_b': 'disjoint_true_rows_only',
                      'true_length_source': 'len(selected) returned by original _sample_bag',
                      'short_bag_policy': 'retain_in_main_loss_skip_view_when_fewer_than_2_real_rows',
                      'loss_aggregation': 'mean_over_logical_batch_with_skipped_cases_zero',
                      'target_fit': 'training_fold_only',
                      'view_objective': 'all_slots_query_free_anchor_mass_and_first_moment_mse',
                      'moment_weight': 1.0,
                      'target_semantics': 'fixed_feature_proxy_not_biological_ground_truth'},
        'evaluation': {'checkpoint_policy': CHECKPOINT_POLICY,
                       'patch_policy': runtime['validation_patch_policy'],
                       'repeats': runtime['eval_repeats'],
                       'crossview_diagnostic': 'one_fixed_split_per_validation_case_train_fold_constant_all_other_donors',
                       'plugin_queue_policy_for_comparison': PLUGIN_POLICY,
                       'cross_policy_scores_comparable': False},
        'patient_work_executed': False,
        'unresolved_environment_variables': sorted({token for key in ('metadata_root','features_root')
            for token in plugin_queue._ENV_TOKEN.findall(runtime[key])}),
        'limitations': ['single_fold_single_seed_exploratory',
                        'no_claim_of_patient_efficacy_or_clinical_semantics'],
    }


def doctor(values, **overrides):
    """Use the existing read-only metadata/runtime doctor without model I/O."""
    runtime = normalize_config(values, require_inputs=True, **overrides)
    # The shared doctor validates its own original checkpoint lock. Its output
    # is renamed to prevent it being mistaken for this queue's policy.
    plugin_runtime = dict(runtime, checkpoint_selection=PLUGIN_POLICY)
    if plugin_runtime.pop('crossview_profile', None) == QUICK_PROFILE:
        plugin_runtime.update({key: plugin_queue._LOCKED[key] for key in QUICK_VALUES})
    result = plugin_queue.doctor(plugin_runtime)
    # Fail before any training if the shared trainer's evaluation/runtime
    # dependencies are absent or an obsolete einops installation is active.
    for name in ('numpy', 'pandas', 'scipy', 'h5py', 'einops', 'sksurv'):
        importlib.import_module(name)
    einops = importlib.import_module('einops')
    if not hasattr(einops, 'pack') or not hasattr(einops, 'unpack'):
        raise RuntimeError('einops.pack and einops.unpack are required')
    importlib.import_module('sksurv.metrics').concordance_index_censored
    result['plugin_doctor_config'] = result.pop('config')
    result['effective_config'] = runtime
    result['checkpoint_policy'] = CHECKPOINT_POLICY
    result['scope'] = 'read_only_paths_hashes_runtime_no_patient_arrays_or_model'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--plan', action='store_true', help='Print plan without accessing patient inputs (default).')
    group.add_argument('--doctor', action='store_true', help='Read-only metadata paths/hashes and runtime checks.')
    group.add_argument('--run', action='store_true', help='Explicitly prepare data and run this four-arm queue.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--quick', action='store_true', help='Exploratory four-arm pilot: 3 epochs, 256 training patches, 3 evaluation repeats.')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--metadata-root')
    parser.add_argument('--features-root')
    parser.add_argument('--device')
    args = parser.parse_args(argv)
    try:
        raw = json.loads(args.config.read_text(encoding='utf-8'))
        overrides = {'metadata_root': args.metadata_root, 'features_root': args.features_root, 'device': args.device}
        if args.quick:
            raw = normalize_config(raw, **overrides)
            raw.update(QUICK_VALUES, crossview_profile=QUICK_PROFILE)
        output = plugin_queue.output_path(args.output_root or (QUICK_OUTPUT if args.quick else DEFAULT_OUTPUT))
        if args.run:
            runtime = normalize_config(raw, require_inputs=True, **overrides)
            doctor(raw, **overrides)
            result = importlib.import_module('slotspe_proof.crossview_engine').execute(runtime, output)
        elif args.doctor:
            result = doctor(raw, **overrides)
        else:
            result = make_plan(raw, output, **overrides)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=True))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
