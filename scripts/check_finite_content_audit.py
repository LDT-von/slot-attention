"""Exact known-answer audit; standard-library-only, zero model training."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slotspe_proof.finite_content_audit import CODINGS, SLOTS, four_factor_law, content_row, noisy_shared_law

DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output-root', default='local_evidence/finite_content_audit_20261102')
    args = parser.parse_args()
    plan_path = DOC/'finite_content_plan_20261102.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if not args.run:
        print(json.dumps(plan, indent=2)); return
    destination = (ROOT/args.output_root).resolve()
    if not destination.is_relative_to(ROOT/'local_evidence') or destination == ROOT/'local_evidence' or destination.exists():
        raise ValueError('fresh ignored output required')
    files = ['scripts/check_finite_content_audit.py', 'slotspe_proof/finite_content_audit.py',
             'slotspe_proof/crossmodal_shared_bottleneck.py']
    pins = {p: sha(ROOT/p) for p in files}
    rows = [content_row(coding) for coding in CODINGS]
    assert max(abs(row['identity_residual']) for row in rows) < 1e-10
    by_name = {r['coding']: r for r in rows}
    assert by_name['factorwise']['whole_minus_oracle_best_single_nll'] < -.01
    assert abs(by_name['duplicated_patient']['whole_minus_oracle_best_single_nll']) < 1e-10
    assert by_name['within_modality_synergy']['per_slot_conditional_content_mse'] > 0
    law = four_factor_law('factorwise')
    parity = {'joint_information': law.mi(['parity'], ['v0', 'v1'], ['g']),
              'sum_leave_one_out': sum(law.mi(['parity'], [v], ['g', *[u for u in ['v0', 'v1'] if u != v]]) for v in ['v0', 'v1']),
              'tc_given_common': law.tc(['v0', 'v1'], ['g']),
              'tc_given_common_and_task': law.tc(['v0', 'v1'], ['g', 'parity'])}
    assert abs(parity['sum_leave_one_out']-2*parity['joint_information']) < 1e-10
    shared = []
    for correction in [False, True]:
        distribution = noisy_shared_law(correction)
        shared.append({'noise_correction_code': correction,
                       'shared_target_entropy_given_g': distribution.entropy(['s'], ['g']),
                       'shared_target_entropy_given_g_v': distribution.entropy(['s'], ['g', 'v']),
                       'mi_g_v': distribution.mi(['g'], ['v']), 'mi_s_v': distribution.mi(['s'], ['v'])})
    sabotage = [{'omitted_decoder_nll': m, 'constant_full_decoder_nll': math.log(2),
                 'objective_full_minus_point1_delta': 1.1*math.log(2)-.1*m} for m in [2., 10., 100.]]
    hinge = [{'active_same_factor_copies': copies, 'per_active_slot_gaussian_rate': .5,
              'capacity': 1., 'sum_rate': .5*copies, 'mean_hinge': 0.,
              'rate_gradient_of_hinge_strictly_below_capacity': 0.} for copies in [1, 2]]
    assert all(sha(ROOT/p) == h for p, h in pins.items())
    result = {'schema': 'slotspe_finite_content_audit_receipt_v1', 'status': 'completed',
              'plan': plan, 'plan_sha256': sha(plan_path), 'source_sha256': pins,
              'rows': rows, 'parity_synergy_control': parity, 'noisy_shared_control': shared,
              'leave_one_out_decoder_sabotage': sabotage, 'flat_budget_control': hinge,
              'probability_table_arrays_published': False,
              'max_rate_identity_residual': max(abs(row['identity_residual']) for row in rows),
              'neural_training_runs': 0, 'optimizer_updates': 0, 'patient_execution': False, 'server_execution': False,
              'decision': 'Do not add per-slot content, target-conditioned TC, or naive leave-one-out reward. Existing rate already mixes conditional TC with useful information and marginal mismatch. Next minimal ablation should compare flat versus continuous private rate on the multi-factor task; this has not been run or implemented.',
              'limitations': ['All entropies concern declared finite population, not learned neural representations.',
                             'Oracle matched-marginal rate is not current Gaussian-prior KL; their marginal mismatch differs.',
                             'Known-population best subset is an oracle bound, not validated finite-probe selection.',
                             'All-slot sufficiency does not identify individual slot semantics or force every slot to be used.']}
    destination.mkdir(parents=True)
    (destination/'receipt.json').write_bytes((json.dumps(result, indent=2)+'\n').encode())
    public = DOC/'finite_content_audit_receipt.json'
    if public.exists(): raise RuntimeError('preserve historical public receipts')
    public.write_bytes((json.dumps(result, indent=2)+'\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
