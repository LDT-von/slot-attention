"""Source-locked V2 boundary collisions and residual counter-controls only."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slotspe_proof.native_preservation_audit import preservation_case, summary_dependence_counterexample, actual_hook_collision, cancellation_case

DOC = ROOT/'docs/research/slotspe_common_private_20261101'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def gradient_and_geometry_controls():
    gate_rows = []
    for branch_value in [0., 2.]:
        gain = torch.tensor(0., dtype=torch.double, requires_grad=True)
        branch = torch.tensor(branch_value, dtype=torch.double, requires_grad=True)
        ((2+gain*branch)-1).square().backward()
        gate_rows.append({'branch_at_zero_gain': branch_value, 'gain_gradient': float(gain.grad), 'branch_gradient': float(branch.grad)})
    angle = torch.tensor(.7, dtype=torch.double, requires_grad=True)
    rotation = torch.stack([torch.stack([angle.cos(), -angle.sin()]), torch.stack([angle.sin(), angle.cos()])])
    x = torch.tensor([1., 2.], dtype=torch.double)
    (rotation.T@(rotation@x)).square().sum().backward()
    rng = torch.Generator().manual_seed(909)
    q = torch.linalg.qr(torch.randn(8, 8, generator=rng, dtype=torch.double)).Q
    source = torch.randn(3, 4, 8, generator=rng, dtype=torch.double); code = source@q.T
    compressed = torch.cat([torch.kron(torch.ones(1, 4, dtype=torch.double)/4, q[:2].contiguous()),
                            torch.kron(torch.eye(4, dtype=torch.double), q[2:].contiguous())])
    return {'zero_gain_controls': gate_rows, 'orthogonal_encode_decode_task_gradient': float(angle.grad),
            'orthogonal_full_slot_inverse_error': float((code@q-source).abs().max()),
            'orthogonal_slot_gram_difference': float((code@code.transpose(1, 2)-source@source.transpose(1, 2)).abs().max()),
            'orthogonal_then_shared_pooling_input_dim': 32, 'orthogonal_then_shared_pooling_rank': int(torch.linalg.matrix_rank(compressed))}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    plan_path = DOC/'native_preservation_plan_20261102.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if not args.run: print(json.dumps(plan, indent=2)); return
    torch.set_num_threads(2)
    assert sha(ROOT/'slotspe_proof/slot_disentanglement.py') == plan['source_adapter_sha256']
    output = DOC/'native_preservation_receipt.json'
    if output.exists(): raise RuntimeError('preserve old receipts')
    files = ['scripts/check_native_preservation.py', 'slotspe_proof/native_preservation_audit.py',
             'slotspe_proof/slot_disentanglement.py', 'slotspe_proof/plugin_host.py',
             'slotspe_proof/prediction_slots.py', 'third_party/SlotSPE/models/SlotSPE.py']
    pins = {p: sha(ROOT/p) for p in files}
    rows = [preservation_case(**case) for case in plan['cases']]
    for row in rows:
        assert row['replacement_max_difference'] < 1e-12
        assert row['bounded_residual_inverse_max_error'] < 1e-8
        assert row['bounded_residual_min_singular_value'] >= 1-row['bounded_residual_contraction_factor']-1e-12
    dependence = summary_dependence_counterexample(); hook = actual_hook_collision()
    assert abs(dependence['source_summary_hsic']) < 1e-12
    assert hook['generated_replaced_predictor_gap'] < 1e-12
    assert all(sha(ROOT/p) == h for p, h in pins.items())
    result = {'schema': 'slotspe_native_preservation_receipt_v1', 'status': 'completed',
              'plan': plan, 'plan_sha256': sha(plan_path), 'source_sha256': pins,
              'adapter_cases': rows, 'summary_dependence_counterexample': dependence,
              'actual_hook_generated_backbone': hook, 'ordinary_residual_cancellation': cancellation_case(),
              'gradient_geometry_controls': gradient_and_geometry_controls(),
              'source_adapter_forwards_excluding_unit_fixtures': 15,
              'neural_training_runs': 0, 'optimizer_updates': 0, 'patient_execution': False, 'server_execution': False,
              'candidate_integrated_into_model_or_queue': False,
              'decision': 'Prioritize source-information-preserving V2 residual research with a norm certificate; ordinary skip alone is not sufficient. Keep invertible full per-slot partition as a backup, not an established semantic solution. No new training authorized by this receipt.'}
    output.write_bytes((json.dumps(result, indent=2)+'\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
