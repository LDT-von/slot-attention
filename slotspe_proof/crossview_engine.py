"""Four matched cross-view patient arms; called only from explicit --run.

The queue's output and all patient artifacts are restricted to local_evidence.
Original source, the prior plugin queue and its checkpoints are not modified.
"""
from __future__ import annotations

import importlib.metadata
import copy
import math
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch

from . import patient
from . import plugin_engine as existing
from .crossview_queue import (ARMS, CHECKPOINT_POLICY, REPO, VIEW_DECODER_SEED,
    VIEW_SEED_OFFSET, VIEW_TEMPERATURE, VIEW_WEIGHT, normalize_config)
from .plugin_queue import output_path
from .redundancy_content import ContentReference
from .redundancy_engine import _cpu_state, state_hash


def _dump(path, obj):
    patient._dump(Path(path), obj)


def implementation_hashes():
    """Snapshot the exact code used, including the separate queue entrypoint."""
    result = existing.implementation_hashes()
    entry = REPO / 'scripts/run_slotspe_crossview.py'
    result[entry.relative_to(REPO).as_posix()] = patient._sha(entry)
    return result


def _snapshot_code(root, hashes):
    """Copy only explicitly hashed public source paths into the new output."""
    base = Path(root) / 'code_snapshot'
    base.mkdir()
    for rel, digest in sorted(hashes.items()):
        src = (REPO / rel).resolve()
        if not src.is_relative_to(REPO.resolve()) or not src.is_file():
            raise RuntimeError('source snapshot path escaped repository or disappeared: ' + rel)
        dest = base / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        if patient._sha(dest) != digest:
            raise RuntimeError('source changed during snapshot: ' + rel)


def _case(root, manifest, pid, device, *, training, seed=0, reference=None):
    """The plugin case contract plus the actual selected real-row count.

    A real all-zero feature remains real. Padding is never inferred by value.
    Sampling is performed exactly once using the existing patient helper.
    """
    item = torch.load(Path(root) / manifest['patients'][pid]['cache_path'],
                      map_location='cpu', weights_only=True)
    if item['patient_id'] != pid:
        raise RuntimeError('prepared cache identity mismatch')
    patches = item['patches']
    if training:
        bag, selected = patient._sample_bag(patches, manifest['patches'], np, torch,
                                            generator_seed=seed + 600_000_000_000)
        real_count = len(selected)
        real = bag[:real_count]
    else:
        bag = real = patches
        real_count = len(patches)
    fold = manifest['folds']['0']
    case = {'x_wsi': bag.unsqueeze(0).to(device), 'omic_missing': False,
            'y': torch.tensor([fold['labels'][pid]], dtype=torch.long, device=device),
            'c': torch.tensor([manifest['patients'][pid]['censor']], dtype=torch.float32, device=device)}
    for i, genes in enumerate(item['omics'], 1):
        case[f'x_omic{i}'] = genes.unsqueeze(0).to(device)
    targets = None
    if reference is not None:
        values = {'x_wsi': real.unsqueeze(0),
                  **{f'x_omic{i+1}': genes.unsqueeze(0) for i, genes in enumerate(item['omics'])}}
        targets = {name: value.to(device) for name, value in reference.targets(values).items()}
    return case, targets, real_count


def _build_arm(config, manifest, arm, initial, reference_state):
    if arm not in ARMS:
        raise ValueError('unknown cross-view arm')
    preset = 'B0_native' if arm == ARMS[0] else 'V2_separate'
    base = existing._build(config, manifest, preset, initial)
    if arm in ARMS[2:]:
        from .crossview_reconstruction import CrossViewV2
        host = CrossViewV2(base, reference_state, temperature=VIEW_TEMPERATURE,
                           decoder_seed=VIEW_DECODER_SEED).to(torch.device(config['device']))
    else:
        host = base
    return base, host


def _v2_state(base):
    return {name: value for name, value in _cpu_state(base).items() if name.startswith('parts.')}


def _checkpoint(host, optimizer, scheduler, protocol, epoch, steps):
    """Preserve tensor state and the wrapper's reference-signature dict."""
    state = {name: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
             for name, value in host.state_dict().items()}
    return {'state_dict':state, 'optimizer_state':optimizer.state_dict(),
            'scheduler_state':scheduler.state_dict(), 'protocol':protocol,
            'epoch':epoch, 'optimizer_steps':steps, 'cpu_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'rng_protocol':'original per-case seeds; separate isolated auxiliary seed'}


def logical_update(base, host, arm, cases, targets, counts, ids, seeds, optimizer):
    """One original logical batch plus an isolated A/B auxiliary objective."""
    n = len(cases)
    if not n or len(set(ids)) != n or any(len(x) != n for x in (targets, counts, ids, seeds)):
        raise ValueError('one distinct patient, target, real count and seed per case required')
    if len(set(seeds)) != n:
        raise ValueError('native seeds must be unique within a logical batch')
    if arm not in ARMS:
        raise ValueError('unknown arm')
    optimizer.zero_grad(set_to_none=True)
    host.train()
    main, native_aux, records, view_terms, observed_terms, view_meta = [], [], [], [], [], []
    for case, count, seed in zip(cases, counts, seeds):
        if (case['x_wsi'].ndim != 3 or case['x_wsi'].shape[0] != 1 or
                case['y'].shape != (1,) or case['c'].shape != (1,)):
            raise ValueError('one native forward and one label per patient required')
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= case['x_wsi'].shape[1]:
            raise ValueError('real_count must come from selected indices of this bag')
        patient._rng(seed, torch, np)
        if arm in ARMS[2:]:
            mode = 'compute' if arm == ARMS[2] else 'train'
            logits, aux = host(**case, plugin_seed=seed + 400_000_000_000,
                               real_count=count, aux_seed=seed + VIEW_SEED_OFFSET,
                               mode=mode, run_aux=True)
            value = host.crossview_loss
            if not torch.is_tensor(value) or value.ndim != 0 or not bool(torch.isfinite(value)):
                raise RuntimeError('crossview loss must be a finite scalar')
            view_terms.append(value)
            observed = host.crossview_raw_loss
            if not torch.is_tensor(observed) or observed.ndim != 0 or not bool(torch.isfinite(observed)):
                raise RuntimeError('observed crossview loss must be a finite scalar')
            observed_terms.append(observed.detach())
            meta = host.last_crossview
            if not isinstance(meta, dict):
                raise RuntimeError('crossview metadata is missing')
            # The plugin must not expose target features through public metrics.
            forbidden = {'target', 'prediction', 'features', 'slots', 'patient_id'}
            if forbidden.intersection(meta):
                raise RuntimeError('crossview metadata contains a private/raw tensor field')
            view_meta.append(meta)
        else:
            logits, aux = base(**case, plugin_seed=seed + 400_000_000_000)
        main.append(base.backbone.loss_fn(logits, y=case['y'], c=case['c'], t=None))
        native_aux.append(torch.as_tensor(aux, device=logits.device))
        records.append(base.last_records)
    main_loss = torch.stack(main).mean()
    auxiliary = torch.stack(native_aux).mean()
    terms = base.added_losses(records, targets if base.parts else None)
    raw_view = torch.stack(view_terms).mean() if view_terms else main_loss.new_zeros(())
    observed_view = torch.stack(observed_terms).mean() if observed_terms else main_loss.new_zeros(())
    used_view = raw_view * VIEW_WEIGHT if arm == ARMS[3] else raw_view * 0.0
    total = main_loss + auxiliary + terms['added'] + used_view
    if not bool(torch.isfinite(total)):
        raise RuntimeError('nonfinite crossview objective')
    total.backward()
    gradients = [p.grad for p in host.parameters() if p.grad is not None]
    if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise RuntimeError('missing or nonfinite gradients')
    norm = torch.stack([g.detach().norm() for g in gradients]).norm()
    if not bool(torch.isfinite(norm)):
        raise RuntimeError('nonfinite gradient norm')
    optimizer.step()
    existing._finite_parameters(host)
    skipped = sum(bool(item.get('skipped')) for item in view_meta)
    return {'main_nll': float(main_loss.detach()), 'native_auxiliary': float(auxiliary.detach()),
            'total_loss': float(total.detach()), 'gradient_l2': float(norm),
            **{name: float(value.detach()) for name, value in terms.items()},
            'view_loss_observed': float(observed_view), 'view_loss_used': float(used_view.detach()),
            'view_cases': len(view_meta), 'view_skipped': skipped,
            'patients': n, 'optimizer_steps': 1, 'root_forwards': n}


def _crossview_scores(pred, target, constant):
    """Equal-patient own/constant MSE and mean of all off-diagonal donors."""
    n = pred.shape[0]
    own = (pred - target).square().mean().item()
    baseline = (target - constant.reshape(1, -1)).square().mean().item()
    donor = None
    if n >= 2:
        distances = ((pred.square().sum(1)[:, None] + target.square().sum(1)[None, :]
                      - 2 * (pred @ target.T)) / pred.shape[1]).clamp_min_(0)
        donor = ((distances.sum() - distances.diagonal().sum()) / (n * (n - 1))).item()
    return {'own_mse': own, 'training_constant_mse': baseline,
            'donor_mse': donor, 'constant_minus_own': baseline - own,
            'donor_minus_own': donor - own if donor is not None else None}


def _crossview_evaluate(host, root, manifest, config):
    """One prespecified held-out split per case; only aggregate statistics saved.

    The constant is an equal-patient mean of full-bag fixed targets from the
    training fold. Donors are every other eligible held-out case; the resulting
    pair count is not an independent sample count.
    """
    train = manifest['folds']['0']['train']
    valid = manifest['folds']['0']['val']
    masses, moments = [], []
    with torch.no_grad():
        for pid in train:
            item = torch.load(Path(root) / manifest['patients'][pid]['cache_path'],
                              map_location='cpu', weights_only=True)
            if item['patient_id'] != pid or len(item['patches']) < 1:
                raise RuntimeError('invalid training cache for crossview constant')
            mass, moment = host._statistics(item['patches'].unsqueeze(0))
            masses.append(mass.cpu().double())
            moments.append(moment.cpu().double())
        if not masses:
            raise RuntimeError('crossview constant requires training patients')
        constants = (torch.stack(masses).mean(0), torch.stack(moments).mean(0))
        records, skipped = [], 0
        host.eval()
        for pos, pid in enumerate(valid):
            case, _, count = _case(root, manifest, pid, config['device'], training=False)
            seed = existing.schedule_seed('repeat', 0, pos)
            patient._rng(seed, torch, np)
            host(**case, plugin_seed=seed + 400_000_000_000, real_count=count,
                 aux_seed=seed + VIEW_SEED_OFFSET, mode='train', run_aux=True, capture_aux=True)
            record = host.last_crossview_tensors
            if record is None:
                skipped += 1
            else:
                records.append(tuple(v.squeeze(0).double().reshape(-1) for v in record))
        host.last_crossview_tensors = None
    n = len(records)
    result = {'eligible_validation_patients': n, 'skipped_validation_patients': skipped,
              'training_patients_for_constant': len(train), 'views_per_patient': 1,
              'constant': 'equal_patient_mean_of_training_full_bag_fixed_statistics',
              'donors': 'all_other_eligible_validation_cases',
              'donor_pairs_independent': False,
              'limitations': 'finite_bag_views_are_dependent; proxies_not_biological_labels; no_clinical_effect_claim'}
    if not n:
        result['mass'] = result['moment'] = None
        return result
    for name, pred_idx, target_idx, constant in (
            ('mass', 0, 2, constants[0]), ('moment', 1, 3, constants[1])):
        pred = torch.stack([row[pred_idx] for row in records])
        target = torch.stack([row[target_idx] for row in records])
        result[name] = _crossview_scores(pred, target, constant)
    return result


def _run_arm(config, root, manifest, arm, initial, reference, v2_initial_hash, verify_live):
    out = Path(root) / 'arms' / arm
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    steps = 0
    try:
        base, host = _build_arm(config, manifest, arm, initial, reference.state_dict())
        core_hash = state_hash(_cpu_state(base.backbone))
        if core_hash != state_hash(initial):
            raise RuntimeError('native initialization differs across arms')
        v2_hash = state_hash(_v2_state(base)) if base.parts else None
        if v2_hash is not None and v2_hash != v2_initial_hash:
            raise RuntimeError('V2 initial state differs across V2 arms')
        optimizer = torch.optim.Adam(host.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=1e-6)
        protocol = {'arm': arm, 'config': config, 'initial_native_hash': core_hash,
            'initial_v2_hash': v2_hash, 'parameter_count': sum(p.numel() for p in host.parameters()),
            'checkpoint_policy': CHECKPOINT_POLICY, 'validation': 'all_real_patches_no_padding_no_subsampling',
            'view_mode': 'off' if arm in ARMS[:2] else 'compute' if arm == ARMS[2] else 'train',
            'view_loss_weight': VIEW_WEIGHT if arm == ARMS[3] else 0.0,
            'view_target_policy': 'fixed_anchor_mass_and_first_moment_v1',
            'view_moment_weight': 1.0,
            'view_aux_seed_offset': VIEW_SEED_OFFSET, 'view_temperature': VIEW_TEMPERATURE,
            'claim_boundary': 'single_fold_single_seed_exploratory_no_patient_efficacy_established',
            'started_utc': existing._now()}
        _dump(out / 'protocol.json', protocol)
        train_ids = manifest['folds']['0']['train']
        positions = {pid: i for i, pid in enumerate(train_ids)}
        curve = []
        if config['device'].startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(torch.device(config['device']))
        for epoch in range(config['epochs']):
            verify_live()
            order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(3000 + epoch)).tolist()
            batches = []
            for group in existing.logical_groups(order, config['batch_size']):
                ids = [train_ids[position] for position in group]
                seeds = [existing.schedule_seed('train', epoch, positions[pid]) for pid in ids]
                loaded = [_case(root, manifest, pid, config['device'], training=True, seed=seed,
                                reference=reference if base.parts else None) for pid, seed in zip(ids, seeds)]
                result = logical_update(base, host, arm, [x[0] for x in loaded], [x[1] for x in loaded],
                                        [x[2] for x in loaded], ids, seeds, optimizer)
                steps += 1
                batches.append({'patient_ids': ids, 'native_seeds': seeds, **result})
            scheduler.step()
            checkpoint = _checkpoint(host, optimizer, scheduler, protocol, epoch+1, steps)
            torch.save(checkpoint, out / 'last.pt')
            curve.append({'epoch': epoch+1, 'batches': batches, 'optimizer_steps': steps})
            _dump(out / 'curve.json', curve)
            print(f'arm={arm} epoch={epoch+1}/{config["epochs"]} steps={steps}', flush=True)
        if config['epochs'] < 1 or not (out / 'last.pt').exists():
            raise RuntimeError('no final checkpoint')
        host.load_state_dict(torch.load(out / 'last.pt', map_location='cpu', weights_only=True)['state_dict'], strict=True)
        evaluations = []
        for repeat in range(config['eval_repeats']):
            # Evaluate the identical unmodified full-bag prediction branch;
            # no auxiliary view loss is run or used at inference.
            result = existing._evaluate(base, root, manifest, config, 'repeat', repeat, reference,
                                        raw_path=out / f'raw_repeat_{repeat:02d}.pt')
            evaluations.append(result)
            _dump(out / 'eval_repeats.json', evaluations)
        if arm in ARMS[2:]:
            view_eval = _crossview_evaluate(host, root, manifest, config)
            _dump(out / 'crossview_eval.json', view_eval)
        scores = [item['cindex'] for item in evaluations]
        summary = {'status':'completed', 'arm':arm, 'checkpoint_policy':CHECKPOINT_POLICY,
            'final_epoch':config['epochs'], 'epochs':config['epochs'], 'patches':config['patches'],
            'optimizer_steps':steps, 'train_patients':len(train_ids),
            'validation_patients':len(manifest['folds']['0']['val']),
            'repeat_cindex_mean':float(np.mean(scores)), 'repeat_cindex_std_population':float(np.std(scores)),
            'independent_runs':1, 'random_repeats_are_not_independent_patients':True,
            'last_sha256':patient._sha(out / 'last.pt'), 'elapsed_seconds':time.perf_counter()-started,
            'cuda_peak_bytes':torch.cuda.max_memory_allocated(config['device']) if config['device'].startswith('cuda') else None,
            'claim_boundary':'single_fold_single_seed_exploratory_no_patient_efficacy_established'}
        verify_live()
        _dump(out / 'summary.json', summary)
        return summary
    except BaseException as exc:
        _dump(out / 'failure.json', {'status':'failed', 'type':type(exc).__name__, 'message':str(exc),
                                     'optimizer_steps_completed':steps, 'elapsed_seconds':time.perf_counter()-started})
        raise


def execute(config, output_root):
    """Prepare, fit reference on training patients only, then run all four arms."""
    config = normalize_config(config, require_inputs=True)
    root = output_path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    queue_path = root / 'crossview_queue_manifest.json'
    queue = {'schema':'slotspe_crossview_queue_v1', 'status':'preparing',
             'started_utc':existing._now(), 'arms':list(ARMS), 'completed_arms':[]}
    _dump(queue_path, queue)
    try:
        if config['device'].startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no automatic CPU fallback')
        torch.set_num_threads(config['torch_threads'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        code = implementation_hashes()
        metadata = {str(path): patient._sha(path) for path in patient._csv_paths(config)}
        identity = {'config':config, 'implementation_sha256':code, 'metadata_sha256':metadata,
                    'runtime':{'python':sys.version, 'executable':sys.executable,
                               'torch':str(torch.__version__), 'numpy':np.__version__,
                               'platform':platform.platform(),
                               'libraries':{n:importlib.metadata.version(n) for n in
                                  ('scipy','scikit-survival','scikit-learn','pandas','h5py','einops')}}}
        _snapshot_code(root, code)
        _dump(root / 'resolved_config.json', config)
        _dump(root / 'identity.json', identity)
        queue['identity_sha256'] = patient._sha(root / 'identity.json')
        _dump(queue_path, queue)
        patient.prepare(config, root)
        manifest = patient._manifest(config, root, verify_cache=True)
        reference = ContentReference(manifest['omic_sizes'], manifest['encoding_dim'])
        def train_inputs():
            for pid in manifest['folds']['0']['train']:
                item = torch.load(root / manifest['patients'][pid]['cache_path'], weights_only=True, map_location='cpu')
                if item['patient_id'] != pid:
                    raise RuntimeError('reference cache identity mismatch')
                yield {'x_wsi':item['patches'].unsqueeze(0),
                       **{f'x_omic{i+1}': genes.unsqueeze(0) for i, genes in enumerate(item['omics'])}}
        reference.fit(train_inputs())
        native = existing._build(config, manifest, 'B0_native')
        native_initial = _cpu_state(native.backbone)
        v2 = existing._build(config, manifest, 'V2_separate', native_initial)
        v2_initial_hash = state_hash(_v2_state(v2))
        torch.save({'native':native_initial, 'reference':reference.state_dict(),
                    'v2_initial':_v2_state(v2)}, root / 'initial.pt')
        queue.update(status='running', prepared_manifest_sha256=patient._sha(root/'prepared/manifest.json'),
                     initial_sha256=patient._sha(root/'initial.pt'), initial_v2_hash=v2_initial_hash)
        _dump(queue_path, queue)
        del native, v2
        def verify_live():
            if (implementation_hashes()!=code or any(patient._sha(Path(path))!=digest for path,digest in metadata.items())
                    or patient._sha(root/'prepared/manifest.json')!=queue['prepared_manifest_sha256']
                    or patient._sha(root/'initial.pt')!=queue['initial_sha256']):
                raise RuntimeError('live source, metadata, preparation or initialization changed')
        summaries=[]
        for arm in ARMS:
            verify_live()
            summary=_run_arm(config,root,manifest,arm,native_initial,reference,v2_initial_hash,verify_live)
            summaries.append(summary)
            queue['completed_arms'].append(arm)
            _dump(queue_path,queue)
        patient._manifest(config,root,verify_cache=True)
        verify_live()
        _dump(root/'summary.json', {'status':'completed','arms':summaries,
                                   'claim_boundary':'paired_single_fold_exploratory_not_generalization'})
        queue.update(status='completed', completed_utc=existing._now(),
                     summary_sha256=patient._sha(root/'summary.json'))
        _dump(queue_path,queue)
        return {'status':'completed','arms':list(ARMS),'output_root':str(root)}
    except BaseException as exc:
        queue.update(status='failed', exception=type(exc).__name__, message=str(exc), failed_utc=existing._now())
        _dump(queue_path,queue)
        _dump(root/'failure.json', {'status':'failed','type':type(exc).__name__,'message':str(exc),
                                   'completed_arms':queue['completed_arms']})
        raise
