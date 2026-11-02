"""Accept synthetic receipts and known Gaussian-channel calculations only."""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_crossmodal_shared_bottleneck import paired, explicit_noise
from slotspe_proof.common_private_survival import survival_nll
from slotspe_proof.slot_pooling_diagnostics import code_inputs
from slotspe_proof.gaussian_pooling_oracle import channel_snrs, scalar_channel_mi, raw_mean_bayes_logits

DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    source = ROOT/'local_evidence/slot_pooling_rate_20261102/receipt.json'
    receipt = json.loads(source.read_text(encoding='utf-8'))
    assert receipt['status'] == 'completed' and len(receipt['rows']) == 30
    assert all(digest(ROOT/p) == h for p, h in receipt['source_sha256_before_execution'].items())
    old_path = DOC/'xor_readout_capacity_receipt.json'
    assert digest(old_path) == receipt['historical_partial_receipt_sha256']
    old = json.loads(old_path.read_text(encoding='utf-8'))
    modes = {'clean_single': 'before_phi_clean_single', 'single_total1': 'before_phi_rate1_single',
             'repeated_total4': 'before_phi_rate1_repeated', 'noise_only': 'before_phi_noise_only'}
    exact = 0
    for row in receipt['rows']:
        if row['readout'] == 'existing_mean' and row['condition'] in modes:
            previous = next(r for r in old['oracle_rows'] if (r['seed'], r['mode']) == (row['seed'], modes[row['condition']]))
            assert row['expected_nll'] == previous['expected_nll']
            exact += 1
    assert exact == 12
    summary = []
    for condition in receipt['plan']['conditions']:
        group = [r for r in receipt['rows'] if r['condition'] == condition]
        raw_scores = []
        for seed in receipt['plan']['seeds']:
            test = paired(360000+seed, 384, 'private_xor')
            random = torch.Generator().manual_seed(370000+seed)
            for n in [192, 128, 384]: explicit_noise(random, n)
            for _ in range(5):
                noise = explicit_noise(random, 384)
                g, v = code_inputs(test['shared_factor'], test['private_bits'], noise['private_noise'], condition)
                raw_scores.append(float(survival_nll(raw_mean_bayes_logits(g, v, condition), test['time'], test['event'])))
        row = {'condition': condition, 'raw_mean_bayes_nll': sum(raw_scores)/len(raw_scores)}
        for readout in receipt['plan']['readouts']:
            subset = [r for r in group if r['readout'] == readout]
            row[readout+'_mean_nll'] = sum(r['expected_nll'] for r in subset)/3
        row['full_code_bayes_mean_nll'] = sum(r['conditional_code_bayes_nll'] for r in group)/6
        row['paired_bank_minus_mean_nll'] = [next(r['expected_nll'] for r in group if (r['seed'], r['readout']) == (s, 'role_bank'))-next(r['expected_nll'] for r in group if (r['seed'], r['readout']) == (s, 'existing_mean')) for s in receipt['plan']['seeds']]
        summary.append(row)
    channels = []
    for name, snr in channel_snrs().items():
        q128, q256 = scalar_channel_mi(snr), scalar_channel_mi(snr, 256)
        channels.append({'channel': name, 'snr': snr, **q256,
                         'quadrature_128_vs_256_max_difference': max(abs(q128[k]-q256[k]) for k in q128)})
    complete_refits = old['decoder_refit_rows']+[receipt['missing_refit']]
    result = {'schema': 'slotspe_pooling_rate_stage_summary_v1', 'status': 'completed',
              'raw_receipt_sha256': digest(source), 'summary': summary, 'known_gaussian_channels': channels,
              'old_readout_deterministic_repeats_verified': exact,
              'new_readout_arm_fits': 18, 'old_refit_completion_executions': 1,
              'total_optimizer_updates_this_stage': 15500,
              'refit_nll_by_seed': {str(r['seed']): r['expected_nll'] for r in complete_refits},
              'refit_mean_nll': sum(r['expected_nll'] for r in complete_refits)/3,
              'limitations': ['Exploratory attribution on old synthetic data, not independent confirmation.',
                             'Role bank changes phi shape and pooling scale as well as role access; near-equal parameters do not isolate those factors.',
                             'Raw-mean analytic loss is not an impossibility theorem for nonlinear Deep Sets.',
                             'Quadrature is for known Gaussian channels and planted bits, not estimated neural-slot mutual information.',
                             'Single-total1 role bank fails at seed419; no stable superiority or patient benefit established.'],
              'source_sha256': {p: digest(ROOT/p) for p in ['scripts/summarize_slot_pooling_rate.py', 'slotspe_proof/gaussian_pooling_oracle.py']},
              'patient_execution': False, 'server_execution': False}
    for path, payload in [(DOC/'slot_pooling_rate_neural_receipt.json', receipt), (DOC/'slot_pooling_rate_stage_summary.json', result)]:
        if path.exists(): raise RuntimeError('preserve earlier receipts')
        path.write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
