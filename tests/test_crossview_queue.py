"""Independent generated examples; never open real patients or start training."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from slotspe_proof import crossview_queue as queue
from slotspe_proof import crossview_engine as engine
from slotspe_proof.crossview_reconstruction import CrossViewV2
from slotspe_proof.plugin_engine import _build
from slotspe_proof.redundancy_content import ContentReference

ROOT=Path(__file__).resolve().parents[1]


def test_crossview_scores_exclude_self_from_donors():
    pred=torch.tensor([[0.],[2.],[4.]],dtype=torch.float64)
    target=torch.tensor([[0.],[1.],[3.]],dtype=torch.float64)
    scores=engine._crossview_scores(pred,target,torch.tensor([1.],dtype=torch.float64))
    assert scores['own_mse']==pytest.approx(2/3)
    assert scores['training_constant_mse']==pytest.approx(5/3)
    assert scores['donor_mse']==pytest.approx(40/6)
    one=engine._crossview_scores(pred[:1],target[:1],torch.tensor([1.],dtype=torch.float64))
    assert one['donor_mse'] is None and one['donor_minus_own'] is None


@pytest.fixture(autouse=True)
def fixed_resources():
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(old)


def generated_cases(n=3, length=10):
    rng=torch.Generator().manual_seed(12281)
    return [dict(x_wsi=torch.randn(1,length,8,generator=rng),
                 x_omic1=torch.randn(1,3,generator=rng),
                 x_omic2=torch.randn(1,5,generator=rng),
                 omic_missing=False,y=torch.tensor([i%4]),
                 c=torch.tensor([float(i%2)])) for i in range(n)]


def make_host():
    config={'source_root':str(ROOT/'third_party/SlotSPE'), 'projection_dim':16,
            'slot_iters':2, 'device':'cpu', 'native_alpha':.5,
            'native_reconstruction_weight':.01}
    return _build(config,{'omic_sizes':[3,5],'encoding_dim':8},'V2_separate')


class CaptureOnly(torch.optim.Optimizer):
    """Match zero_grad/step API while making no optimization update."""
    def __init__(self,params):
        super().__init__(params,{})
        self.step_calls=0
    def step(self,closure=None):
        self.step_calls+=1


def test_plan_is_data_free_and_four_arms_keep_final_checkpoint_policy(tmp_path):
    config=json.loads(queue.DEFAULT_CONFIG.read_text(encoding='utf-8'))
    plan=queue.make_plan(config)
    assert [x['name'] for x in plan['arms']]==list(queue.ARMS)
    assert plan['config']['patches']==4096 and plan['config']['eval_repeats']==20
    assert plan['evaluation']['checkpoint_policy']==queue.CHECKPOINT_POLICY
    assert plan['evaluation']['cross_policy_scores_comparable'] is False
    assert not plan['patient_work_executed'] and plan['training_runs']==4
    other=tmp_path/'does_not_exist'
    assert not other.exists()


def test_cli_plan_does_not_import_neural_engine_or_create_output():
    code=('import sys; from slotspe_proof.crossview_queue import main; '
          'rc=main(["--plan"]); assert "torch" not in sys.modules; '
          'assert "slotspe_proof.crossview_engine" not in sys.modules; raise SystemExit(rc)')
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,
                          capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0, result.stderr
    plan=json.loads(result.stdout)
    assert plan['mode']=='plan' and not plan['patient_work_executed']


def test_real_count_uses_sampled_indices_even_when_real_features_are_zero(monkeypatch,tmp_path):
    item={'patient_id':'invented','patches':torch.tensor([[0.,0.],[2.,3.]]),
          'omics':[torch.tensor([1.])]}
    monkeypatch.setattr(engine.torch,'load',lambda *args,**kwargs:item)
    manifest={'patients':{'invented':{'cache_path':'invented.pt','censor':0.}},
              'patches':5,'folds':{'0':{'labels':{'invented':0}}}}
    case,targets,count=engine._case(tmp_path,manifest,'invented','cpu',training=True,seed=10)
    assert count==2 and targets is None and case['x_wsi'].shape==(1,5,2)
    assert torch.equal(case['x_wsi'][0,:2],item['patches'])
    assert torch.count_nonzero(case['x_wsi'][0,2:])==0
    old_case,old_targets=engine.existing._case(tmp_path,manifest,'invented','cpu',training=True,seed=10)
    assert old_targets is None
    for key in case:
        if isinstance(case[key],torch.Tensor): assert torch.equal(case[key],old_case[key])
        else: assert case[key]==old_case[key]


def test_v2_initial_state_is_identical_across_all_v2_arms():
    cases=generated_cases()
    reference=ContentReference([3,5],8,target_dim=32,clusters=4).fit(cases)
    config={'source_root':str(ROOT/'third_party/SlotSPE'), 'projection_dim':16,
            'slot_iters':2, 'device':'cpu', 'native_alpha':.5,
            'native_reconstruction_weight':.01}
    manifest={'omic_sizes':[3,5],'encoding_dim':8}
    initial=engine._cpu_state(engine.existing._build(config,manifest,'B0_native').backbone)
    pairs=[engine._build_arm(config,manifest,arm,initial,reference.state_dict())
           for arm in queue.ARMS[1:]]
    hashes=[engine.state_hash(engine._v2_state(base)) for base,_ in pairs]
    assert len(set(hashes))==1 and hashes[0]
    assert all(engine.state_hash(engine._cpu_state(base.backbone))==engine.state_hash(initial)
               for base,_ in pairs)


def test_compute_control_preserves_v2_full_objective_and_gradient():
    cases=generated_cases()
    reference=ContentReference([3,5],8,target_dim=32,clusters=4).fit(cases)
    targets=[reference.targets(c) for c in cases]
    base=make_host()
    compare=make_host()
    wrapped=CrossViewV2(compare,reference.state_dict(),hidden_dim=12,decoder_seed=77)
    ids=['invented-0','invented-1','invented-2']
    seeds=[17,19,23]
    base_opt=CaptureOnly(base.parameters()); compute_opt=CaptureOnly(wrapped.parameters())
    baseline=engine.logical_update(base,base,queue.ARMS[1],cases,targets,[8]*3,ids,seeds,base_opt)
    state_after_baseline=torch.get_rng_state().clone()
    control=engine.logical_update(compare,wrapped,queue.ARMS[2],cases,targets,[8]*3,ids,seeds,compute_opt)
    assert base_opt.step_calls==compute_opt.step_calls==1
    assert baseline['main_nll']==control['main_nll']
    assert baseline['native_auxiliary']==control['native_auxiliary']
    assert baseline['total_loss']==control['total_loss']
    assert control['view_loss_used']==0 and control['view_loss_observed']>0 and control['view_cases']==3
    assert torch.equal(state_after_baseline,torch.get_rng_state())
    for (name,p),(name2,q) in zip(base.named_parameters(),compare.named_parameters()):
        assert name==name2
        assert (p.grad is None)==(q.grad is None), name
        if p.grad is not None:
            assert torch.equal(p.grad,q.grad), name
    assert wrapped.decoder.readout.weight.grad is not None
    assert torch.count_nonzero(wrapped.decoder.readout.weight.grad)==0


def test_train_arm_adds_nonzero_view_gradient_without_optimizer_update():
    cases=generated_cases()
    reference=ContentReference([3,5],8,target_dim=32,clusters=4).fit(cases)
    targets=[reference.targets(c) for c in cases]
    base=make_host()
    wrapped=CrossViewV2(base,reference.state_dict(),hidden_dim=12,decoder_seed=77)
    before={n:p.detach().clone() for n,p in wrapped.named_parameters()}
    opt=CaptureOnly(wrapped.parameters())
    result=engine.logical_update(base,wrapped,queue.ARMS[3],cases,targets,[8]*3,
                                 ['invented-0','invented-1','invented-2'],[17,19,23],opt)
    assert result['view_loss_used']>0 and result['view_cases']==3 and result['view_skipped']==0
    assert wrapped.decoder.readout.weight.grad is not None
    assert torch.count_nonzero(wrapped.decoder.readout.weight.grad)>0
    assert all(torch.equal(before[n],p.detach()) for n,p in wrapped.named_parameters())
    assert opt.step_calls==1  # CaptureOnly records the call but does not optimize.


def test_checkpoint_keeps_extra_state_and_rejects_other_reference():
    cases=generated_cases()
    ref=ContentReference([3,5],8,target_dim=5,clusters=4).fit(cases).state_dict()
    host=CrossViewV2(make_host(),ref,hidden_dim=12)
    opt=CaptureOnly(host.parameters())
    class NoScheduler:
        def state_dict(self): return {}
    checkpoint=engine._checkpoint(host,opt,NoScheduler(),{'synthetic':True},1,0)
    assert isinstance(checkpoint['state_dict']['_extra_state'],dict)
    other=ContentReference([3,5],8,target_dim=5,clusters=4,seed=55).fit(cases).state_dict()
    with pytest.raises(RuntimeError,match='checkpoint/reference'):
        CrossViewV2(make_host(),other,hidden_dim=12).load_state_dict(checkpoint['state_dict'])


def test_quick_profile_is_separate_and_strict():
    config=json.loads(queue.DEFAULT_CONFIG.read_text(encoding='utf-8'))
    standard=queue.normalize_config(config)
    quick=dict(standard, **queue.QUICK_VALUES, crossview_profile=queue.QUICK_PROFILE)
    normalized=queue.normalize_config(quick)
    assert (standard['epochs'],standard['patches'],standard['eval_repeats']) == (30,4096,20)
    assert (normalized['epochs'],normalized['patches'],normalized['eval_repeats']) == (3,256,3)
    assert queue.normalize_config(normalized)==normalized
    assert queue.QUICK_OUTPUT!=queue.DEFAULT_OUTPUT
    for key in queue.QUICK_VALUES:
        bad=dict(quick); bad[key]=standard[key]
        with pytest.raises(ValueError,match='quick profile'): queue.normalize_config(bad)
    with pytest.raises(ValueError): queue.normalize_config(dict(standard,epochs=3))


def test_cli_quick_plan_is_data_free_and_distinct_output():
    result=subprocess.run([sys.executable,'-B','scripts/run_slotspe_crossview.py','--quick','--plan'],
                          cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0, result.stdout+result.stderr
    plan=json.loads(result.stdout)
    assert plan['config']['crossview_profile']==queue.QUICK_PROFILE
    assert (plan['config']['epochs'],plan['config']['patches'],plan['config']['eval_repeats'])==(3,256,3)
    assert plan['training_runs']==4 and plan['epoch_budget']==12
    assert plan['output_root']==str(queue.QUICK_OUTPUT) and not plan['patient_work_executed']


def test_quick_doctor_retains_effective_profile(monkeypatch,tmp_path):
    metadata=tmp_path/'metadata'; features=tmp_path/'features'; features.mkdir()
    for name in queue.plugin_queue._METADATA_FILES:
        path=metadata/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('fake')
    base=queue.normalize_config(json.loads(queue.DEFAULT_CONFIG.read_text(encoding='utf-8')))
    quick=dict(base, **queue.QUICK_VALUES, crossview_profile=queue.QUICK_PROFILE)
    seen=[]
    def fake_doctor(config):
        seen.append(config)
        return {'config':config}
    monkeypatch.setattr(queue.plugin_queue,'doctor',fake_doctor)
    result=queue.doctor(quick,metadata_root=str(metadata),features_root=str(features),device='cpu')
    assert result['effective_config']['epochs']==3
    assert result['effective_config']['patches']==256
    assert result['effective_config']['eval_repeats']==3
    assert result['effective_config']['crossview_profile']==queue.QUICK_PROFILE
    assert seen[0]['epochs']==30 and seen[0]['patches']==4096 and seen[0]['eval_repeats']==20
    assert 'crossview_profile' not in seen[0]
