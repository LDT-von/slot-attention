"""Aggregate local API costs and semantic boundary; no optimization or patients."""
import argparse
import hashlib
import itertools
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from slotspe_proof.bounded_native_residual import BoundedNativeResidual
from slotspe_proof.slot_disentanglement import SharedPrivateSlots, separation_terms

DOC=ROOT/'docs/research/slotspe_common_private_20261101'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_control():
    model=BoundedNativeResidual(SharedPrivateSlots(8,2,6,1).double())
    with torch.no_grad():
        model.adapter.shared_encoder.weight.zero_(); model.adapter.shared_encoder.bias.zero_()
        model.adapter.private_encoder.weight.zero_(); model.adapter.private_encoder.bias.zero_()
        model.adapter.shared_encoder.weight[:,1]=1
        model.adapter.private_encoder.weight[:,2]=1
    states=list(itertools.product([-1.,1.],[-1.,1.],[0.,2.]))
    x=torch.zeros(8,4,8,dtype=torch.double)
    for i,(y,n1,n2) in enumerate(states): x[i,:,0:3]=torch.tensor([y,n1,n2])
    z,record=model(x)
    separated=separation_terms(record['shared'],record['private'])
    score=model.inverse(z).mean(1)[:,0]
    grads=torch.autograd.grad(score.square().sum(),list(model.parameters()),allow_unused=True)
    return {'exact_population_states':8,'population_rows_are_iid_patient_samples':False,
            'known_mi_target_shared_nats':0.,'known_mi_target_private_set_nats':0.,
            'source_hsic':float(separated['hsic']),'source_shared_variance_penalty':float(separated['variance']),
            'old_replacement_target_pair_max_difference':float((record['base_latent'][:4]-record['base_latent'][4:]).abs().max()),
            'native_residual_target_pair_distance':float((z[:4]-z[4:]).norm(dim=(1,2)).mean()),
            'privileged_inverse_head_score_max_error':float((score-x[:,0,0]).abs().max()),
            'score_gradient_to_adapter_max':max(float(g.abs().max()) for g in grads if g is not None),
            'actual_fixed_content_reference_gate_exercised':False,
            'interpretation':'Inverse head is an existence control, not a proposed predictor or evidence an ordinary finite head learns it. Low separation plus retained native task information does not identify useful C/U.'}


def cell(plan,seed,dtype_name,batch,arm):
    dtype=getattr(torch,dtype_name)
    adapter=SharedPrivateSlots(seed=seed).to(dtype=dtype)
    model=adapter if arm=='source_v2' else BoundedNativeResidual(adapter,bound=arm)
    rng=torch.Generator().manual_seed(300000+seed+batch)
    x=torch.randn(batch,plan['slots'],plan['dimension'],generator=rng,dtype=torch.double).to(dtype)
    matrix_mean=adapter.shared_projection.weight.double()@adapter.shared_encoder.weight.double()
    matrix_local=adapter.private_projection.weight.double()@adapter.private_encoder.weight.double()
    actual=max(float(torch.linalg.matrix_norm(matrix_local,2)),float(torch.linalg.matrix_norm(matrix_mean+matrix_local,2)))
    diagnostics={'exact_svd_operator_norm':actual}
    with torch.no_grad():
        z,record=model(x)
        if arm!='source_v2':
            coefficient=float(record['residual_coefficient'])
            diagnostics.update(analytic_upper_bound=float(record['norm_upper_bound']),
                               coefficient=coefficient,actual_contraction=coefficient*actual,
                               reported_contraction_upper_bound=float(record['contraction_upper_bound']),
                               inverse_max_error=float((model.inverse(z)-x).abs().max()),
                               output_change_l2_ratio=float((z-x).norm()/x.norm()))
            assert coefficient*actual<plan['contraction']
            assert diagnostics['inverse_max_error']<(2e-6 if dtype==torch.float32 else 1e-10)
    elapsed=[]; finite=True; max_gradient=0.
    warmup=plan['warmup_forward_backward_calls_each']; repeats=plan['timed_forward_backward_calls_each']
    for i in range(warmup+repeats):
        start=time.perf_counter()
        model.zero_grad(set_to_none=True)
        output,_=model(x)
        output.square().mean().backward()
        duration=time.perf_counter()-start
        if i>=warmup: elapsed.append(duration)
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        finite=finite and all(bool(torch.isfinite(g).all()) for g in gradients)
        max_gradient=max(max_gradient,max(float(g.norm()) for g in gradients))
    assert finite
    return {'seed':seed,'dtype':dtype_name,'generated_batch':batch,'arm':arm,
            'parameters':sum(p.numel() for p in model.parameters()),
            'source_adapter_forward_calls':1+warmup+repeats,
            'forward_backward_calls':warmup+repeats,'optimizer_updates':0,
            'raw_timed_seconds':elapsed,'median_forward_backward_seconds':statistics.median(elapsed),
            'finite_gradients':finite,'maximum_single_parameter_gradient_l2':max_gradient,
            'diagnostics':diagnostics}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    plan_path=DOC/'bounded_residual_plan_20261102.json'
    plan=json.loads(plan_path.read_text())
    if not args.run:
        print(json.dumps(plan,indent=2)); return
    receipt_path=DOC/'bounded_residual_receipt.json'
    if receipt_path.exists(): raise RuntimeError('preserve historical receipt')
    torch.set_num_threads(plan['cpu_threads'])
    files=['slotspe_proof/bounded_native_residual.py','scripts/check_bounded_native_residual.py',
           'slotspe_proof/slot_disentanglement.py','slotspe_proof/plugin_host.py',
           'slotspe_proof/prediction_slots.py','third_party/SlotSPE/models/SlotSPE.py']
    pins={p:sha(ROOT/p) for p in files}
    rows=[cell(plan,seed,dtype,batch,arm) for seed,dtype,batch,arm in
          itertools.product(plan['seeds'],plan['dtypes'],plan['batch_sizes'],plan['arms'])]
    assert len(rows)==plan['timing_cells']
    for row in rows:
        baseline=next(r for r in rows if (r['seed'],r['dtype'],r['generated_batch'],r['arm'])==
                      (row['seed'],row['dtype'],row['generated_batch'],'source_v2'))
        row['median_cost_ratio_to_source']=row['median_forward_backward_seconds']/baseline['median_forward_backward_seconds']
    semantic=semantic_control()
    assert abs(semantic['source_hsic'])<1e-12 and semantic['source_shared_variance_penalty']==0
    assert semantic['privileged_inverse_head_score_max_error']<1e-12
    assert semantic['score_gradient_to_adapter_max']<1e-11
    assert all(sha(ROOT/p)==h for p,h in pins.items())
    result={'schema':'slotspe_bounded_native_residual_receipt_v1','status':'completed',
            'plan':plan,'plan_sha256':sha(plan_path),'source_sha256':pins,
            'environment':{'python':platform.python_version(),'torch':torch.__version__,
                           'platform':platform.platform(),'device':'cpu','cpu_threads':plan['cpu_threads']},
            'timing_cells':rows,'semantic_boundary':semantic,
            'source_adapter_forward_calls_excluding_tests':sum(r['source_adapter_forward_calls'] for r in rows)+1,
            'timing_forward_backward_calls_excluding_tests':sum(r['forward_backward_calls'] for r in rows),
            'semantic_score_autograd_calls_excluding_tests':1,
            'synthetic_neural_training_runs':0,'optimizer_updates':0,
            'patient_execution':False,'server_execution':False,'queue_modified':False,
            'new_mi_estimator':False,'new_preset_or_patient_entry_point':False}
    receipt_path.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({'status':'completed','receipt':str(receipt_path),'timing_cells':len(rows),
                      'source_adapter_forward_calls':result['source_adapter_forward_calls_excluding_tests'],
                      'optimizer_updates':0,'semantic_boundary':semantic},indent=2))


if __name__=='__main__': main()
