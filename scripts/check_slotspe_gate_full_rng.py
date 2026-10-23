"""Plan-only by default. --run observes full native patient evaluation RNG.

Weights and original model files are read-only. Only hooks and scoped diagnostic
controls are used. Patient rows remain in ignored local_evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from check_slotspe_gate_collapse import read, sha, write, distribution


def run(args):
    import numpy as np
    import torch
    from torch.torch_version import TorchVersion
    from survot_rank.research.slotspe_proof.model import build_model, load_source
    from survot_rank.research.slotspe_proof.native_audit import model_purity, _state_digest

    out, patient_root, observed = (Path(x).resolve() for x in
                                 (args.output_dir, args.patient_root, args.observations_root))
    if not out.is_relative_to((REPO / 'local_evidence').resolve()) or out.exists():
        raise ValueError('Fresh ignored local_evidence output required')
    o = read(observed / 'observations.json')
    source_receipt = read(args.source_verification)
    comparison = source_receipt['comparison']
    if comparison['runtime_difference_count'] != 0 or comparison['runtime_file_equal_count'] != 10:
        raise ValueError('Official source gate did not pass')
    cache = patient_root / 'cache/paired_blca_fold0.pt'
    checkpoint = patient_root / 'seed_3/original/last.pt'
    historical_eval = patient_root / 'seed_3/original/eval_repeats.json'
    manifest = patient_root / 'manifest.json'
    inputs = {observed / 'observations.json': sha(observed / 'observations.json')}
    for path in (cache, checkpoint, historical_eval, manifest):
        expected = [d for old, d in o['source_inputs_sha256'].items()
                    if Path(old.replace('\\', '/')).name == path.name]
        if len(expected) != 1 or sha(path) != expected[0]:
            raise ValueError('Historical patient input hash mismatch: ' + str(path))
        inputs[path] = expected[0]
    source_root = REPO / 'third_party/SlotSPE'
    for rel, digest in o['model_purity']['source_sha256'].items():
        inputs[source_root / rel] = digest
    for entry in source_receipt['file_comparisons']:
        if entry['classification'] != 'documentation':
            path = source_root / entry['path']
            if sha(path) != entry['official_sha256']:
                raise ValueError('Source no longer matches official bytes')
            inputs[path] = entry['official_sha256']
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError('Source or input changed')
    data = torch.load(cache, map_location='cpu', mmap=True, weights_only=True)
    ids = [r['patient_id'] for r in o['records']]
    indices = [i for i, split in enumerate(data['split']) if split == 'val']
    if [data['patient_id'][i] for i in indices] != ids or len(ids) != len(set(ids)) or len(ids) != 71:
        raise ValueError('Historical cohort order mismatch')
    historical = read(historical_eval)
    seeds = [r['eval_seed'] for r in historical]
    if len(seeds) != 20 or seeds != list(range(3080000, 3080020)):
        raise ValueError('Expected the prespecified historical twenty eval seeds')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for exact historical replay')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    devices = list(range(torch.cuda.device_count()))
    source = load_source(source_root)
    sampler = source.modules['models.slot_attention']
    with torch.serialization.safe_globals([TorchVersion]):
        saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    if saved['arm'] != 'original' or saved['seed'] != 3 or saved['optimizer_steps'] != 111:
        raise ValueError('Checkpoint protocol mismatch')
    rows = [{'patient_id': pid, 'modalities': {m: {'repeats': []} for m in ('wsi', 'omic')}} for pid in ids]
    captures, handles, errors = {}, [], Counter()
    enabled = True
    full_calls = control_calls = 0
    out.mkdir(parents=True)
    write(out / 'execution_status.json', {'status': 'started', 'patients': len(ids), 'repeats': 20,
          'training_runs': 0, 'checkpoint_sha256': inputs[checkpoint], 'runner_sha256': sha(__file__)})

    def state():
        return torch.random.get_rng_state(), [torch.cuda.get_rng_state(d) for d in devices]

    def restore(value):
        torch.random.set_rng_state(value[0])
        for d, s in zip(devices, value[1]):
            torch.cuda.set_rng_state(s, d)

    with torch.random.fork_rng(devices=devices), torch.no_grad():
        model = build_model(str(source_root), SimpleNamespace(**o['model_purity']['model_args']), 'original').to(args.device).eval()
        model.load_state_dict(saved['state_dict'], strict=True)
        del saved
        purity = model_purity(model, source_root)
        before = _state_digest(model)
        for modality in ('wsi', 'omic'):
            decoder = getattr(model, 'slot_decoder_' + modality)
            if (decoder.k, decoder.num_slots, decoder.temperature, decoder.top_k_method) != (2, 8, .01, 'parallel_topk_st'):
                raise ValueError('Original decoder configuration differs')
            def pre(_module, args, name=modality):
                if enabled:
                    captures[name] = {'slots': args[0].detach().clone(), 'pre_rng': state()}
            def score(_module, _inputs, output, name=modality):
                if enabled:
                    captures[name]['scores'] = output.detach().squeeze(-1).clone()
            def post(_module, _inputs, output, name=modality):
                if enabled:
                    captures[name].update({'gate': output[1].detach().clone(), 'keep': output[2].detach().clone()})
            handles.extend([decoder.register_forward_pre_hook(pre),
                            decoder.pred_keep_slot.register_forward_hook(score),
                            decoder.register_forward_hook(post)])
        try:
            for repeat, seed in enumerate(seeds):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                for start in range(0, len(ids), args.batch_size):
                    ix = indices[start:start + args.batch_size]
                    batch = {'x_wsi': data['wsi'][ix].to(args.device), 'omic_missing': False}
                    batch.update({f'x_omic{k+1}': value[ix].to(args.device) for k, value in enumerate(data['omics'])})
                    logits, _ = model(**batch)
                    full_calls += 1
                    expected_logits = torch.tensor(historical[repeat]['logits'][start:start + len(ix)], device=args.device)
                    errors['historical_full_logits_max_abs'] = max(errors['historical_full_logits_max_abs'], float((logits - expected_logits).abs().max()))
                    for modality in ('wsi', 'omic'):
                        c = captures[modality]
                        scores, gate, keep = (c[k] for k in ('scores', 'gate', 'keep'))
                        mask = keep > .5
                        if not bool((mask.sum(-1) == 2).all()) or not all(bool(torch.isfinite(c[k]).all()) for k in ('slots', 'scores', 'gate', 'keep')):
                            raise ValueError('Invalid actual native gate tensors')
                        decoder = getattr(model, 'slot_decoder_' + modality)
                        enabled = False
                        try:
                            with torch.random.fork_rng(devices=devices):
                                restore(c['pre_rng'])
                                try:
                                    decoder.temperature = 1.
                                    _, warm_gate, warm_keep = decoder(c['slots'])
                                    control_calls += 1
                                finally:
                                    decoder.temperature = .01
                                restore(c['pre_rng'])
                                with patch.object(sampler, 'gumbel_noise', lambda x: torch.zeros_like(x)):
                                    _, quiet_gate, quiet_keep = decoder(c['slots'])
                                    control_calls += 1
                        finally:
                            enabled = True
                        if not torch.equal(mask, warm_keep > .5):
                            raise RuntimeError('Same-noise temperature control changed hard selection')
                        top2 = scores.topk(2, -1).indices
                        if not torch.equal((quiet_keep > .5).sum(-1), torch.full((len(ix),), 2, device=args.device)) or not bool((quiet_keep > .5).gather(1, top2).all()):
                            raise RuntimeError('Zero-noise control did not select deterministic Top2')
                        p32 = torch.softmax(scores / .01, -1)
                        mass32 = (p32 * keep).sum(-1)
                        formula = p32 * keep / (mass32.unsqueeze(-1) + 1e-8)
                        errors['native_gate_formula_max_abs'] = max(errors['native_gate_formula_max_abs'], float((formula - gate).abs().max()))
                        effective = scores.double() / .01
                        logmass = torch.logsumexp(effective.masked_fill(~mask, -torch.inf), -1) - torch.logsumexp(effective, -1)
                        rawmass = (torch.softmax(scores.double(), -1) * mask).sum(-1)
                        stable_gate = torch.softmax((scores / .01).masked_fill(~mask, -torch.inf), -1)
                        overlaps = mask.gather(1, top2).sum(-1)
                        for i in range(len(ix)):
                            rows[start+i]['modalities'][modality]['repeats'].append({
                                'repeat': repeat, 'eval_seed': seed, 'keep_count': int(mask[i].sum()),
                                'gate_sum': float(gate[i].sum()), 'scores': scores[i].tolist(),
                                'gate_weights': gate[i].tolist(), 'selected_indices': mask[i].nonzero().flatten().tolist(),
                                'deterministic_top2_indices': top2[i].tolist(), 'deterministic_top2_overlap': int(overlaps[i]),
                                'top1_retained': bool(mask[i, scores[i].argmax()]),
                                'selected_temperature_mass_float32': float(mass32[i]),
                                'selected_temperature_mass_float64': float(logmass[i].exp()),
                                'log_selected_temperature_mass_float64': float(logmass[i]),
                                'selected_raw_score_mass_float64': float(rawmass[i]),
                                'temperature_1_same_noise_gate_sum': float(warm_gate[i].sum()),
                                'no_noise_temperature_0_01_gate_sum': float(quiet_gate[i].sum()),
                                'masked_softmax_same_mask_gate_sum': float(stable_gate[i].sum())})
                print(f'Completed full-model RNG repeat {repeat+1}/20; cumulative full batch forwards={full_calls}', flush=True)
        finally:
            for handle in handles:
                handle.remove()
        after = _state_digest(model)
        if before != after or any(getattr(model, 'slot_decoder_' + m).temperature != .01 for m in ('wsi', 'omic')):
            raise RuntimeError('Model state or settings changed')
    if max(errors.values(), default=0) > 1e-6:
        raise RuntimeError('Historical exact replay failed: ' + str(errors))
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise RuntimeError('Read-only input or source changed')
    summary = {}
    for modality in ('wsi', 'omic'):
        per = [row['modalities'][modality] for row in rows]
        for patient in per:
            reps = patient['repeats']
            patient['unique_retained_sets'] = len({tuple(r['selected_indices']) for r in reps})
            patient['gate_sum_across_rng'] = distribution([r['gate_sum'] for r in reps])
        pooled = [r for patient in per for r in patient['repeats']]
        summary[modality] = {
            'primary_first_rng_unique_patients': distribution([patient['repeats'][0]['gate_sum'] for patient in per]),
            'all_rng_patient_draws_not_independent_patients': distribution([r['gate_sum'] for r in pooled]),
            'unique_patients_ever_below_0_5': sum(any(r['gate_sum'] < .5 for r in p['repeats']) for p in per),
            'unique_patients_ever_below_0_01': sum(any(r['gate_sum'] < .01 for r in p['repeats']) for p in per),
            'selected_temperature_mass': distribution([r['selected_temperature_mass_float64'] for r in pooled]),
            'selected_raw_score_mass': distribution([r['selected_raw_score_mass_float64'] for r in pooled]),
            'top2_overlap_histogram': dict(Counter(str(r['deterministic_top2_overlap']) for r in pooled)),
            'top2_overlap_mean': float(np.mean([r['deterministic_top2_overlap'] for r in pooled])),
            'top1_missed_fraction': float(np.mean([not r['top1_retained'] for r in pooled])),
            'unique_retained_sets': distribution([p['unique_retained_sets'] for p in per]),
            'controls': {key: distribution([r[key] for r in pooled]) for key in
                ('temperature_1_same_noise_gate_sum', 'no_noise_temperature_0_01_gate_sum', 'masked_softmax_same_mask_gate_sum')}}
    result = {'schema': 'slotspe_readonly_full_model_gate_rng_v1', 'status': 'completed',
        'recorded_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'protocol': {'cohort': 'historical local BLCA validation', 'patients': 71, 'fold': 0, 'training_seed': 3,
            'epochs': 3, 'training_patients': 289, 'validation_patients': 71, 'excluded': 21,
            'patches': 256, 'slots': 8, 'k': 2, 'temperature': .01, 'epsilon': 1e-8,
            'repeats': 20, 'eval_seeds': seeds, 'batch_size': args.batch_size, 'device': args.device,
            'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(args.device),
            'scope': 'Full original forward; fixed raw patient inputs and weights; regenerate Gaussian slots and Gumbel gates each repeat.',
            'controls': 'Native decoder on each freshly generated slot tensor; restore pre-decoder RNG for paired temperature=1; scoped zero Gumbel noise control.',
            'set_stability_limit': 'Slot identities are reinitialized per full-model repeat; index-set diversity combines Gaussian slot and Gumbel selection variation.',
            'gate_sum': 'Sum of actual returned weights AFTER epsilon normalization.',
            'temperature_mass': 'Softmax(score/.01) mass inside actual hard mask; float64 logsumexp reference.',
            'raw_mass': 'Softmax(unscaled score) mass inside actual hard mask; separate from temperature mass.',
            'primary_unit': 'Unique patient at repeat=0; all draws and ever-affected patient counts reported separately.'},
        'source_verification': source_receipt,
        'input_sha256': {str(p.relative_to(REPO)).replace('\\', '/'): digest for p, digest in inputs.items()},
        'summary': summary, 'per_patient': rows,
        'checks': {'error_maxima': dict(errors), 'historical_full_logits_replay_passed': True,
            'model_purity': purity['checks'], 'model_state_sha256_before': before, 'model_state_sha256_after': after,
            'sources_weights_inputs_unchanged': True, 'hooks_removed_and_settings_restored': True,
            'training_runs': 0, 'optimizer_steps': 0, 'full_model_batch_forwards': full_calls,
            'full_model_patient_forwards': len(ids)*20, 'new_slot_generation_patient_pairs': len(ids)*20,
            'paired_control_decoder_batch_forwards': control_calls, 'server_execution': False,
            'new_server_raw_receipt': False, 'runner_sha256': sha(__file__)},
        'limitations': ['One short-trained local checkpoint; not the 77-patient protocol or mature weights.',
            'Final fused forward does not consume slot_gate soft weights; gate decay is an auxiliary-head observation.',
            'No survival-metric scoring, outcome analysis, retraining, or correction-efficacy claim.']}
    write(out / 'gate_diagnostic.json', result)
    write(out / 'execution_status.json', {'status': 'completed', 'result_sha256': sha(out / 'gate_diagnostic.json'),
          'full_model_batch_forwards': full_calls, 'paired_control_decoder_batch_forwards': control_calls,
          'training_runs': 0, 'error_maxima': dict(errors)})
    print({'status': 'completed', 'patients': len(rows), 'summary': summary}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patient-root', type=Path, required=True)
    p.add_argument('--observations-root', type=Path, required=True)
    p.add_argument('--source-verification', type=Path, required=True)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    if not args.run:
        print({'action': 'plan_only', 'patients_expected': 71, 'repeats': 20, 'training': False})
        return
    if args.output_dir is None:
        p.error('--run requires --output-dir')
    run(args)


if __name__ == '__main__':
    main()
