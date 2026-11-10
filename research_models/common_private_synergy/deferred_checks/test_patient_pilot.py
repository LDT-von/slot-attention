import copy
import json
import subprocess
import sys

import pytest
import torch

from research_models.common_private_synergy import patient_pilot as cli
from research_models.common_private_synergy.patient_pilot_engine import (
    ContentAdapter, fit_boundaries, label_times, synthetic_smoke)
from research_models.common_private_synergy.components.losses import discrete_hazard_nll


def test_plan_import_has_no_tensor_or_patient_access():
    code="import sys; from research_models.common_private_synergy.patient_pilot import main; main(['--plan']); assert 'torch' not in sys.modules; assert 'pandas' not in sys.modules"
    run=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert json.loads(run.stdout)['execution']['patient_values_read'] is False


def test_train_only_fit_and_validation_shift_does_not_refit():
    rng=torch.Generator().manual_seed(718)
    p=torch.randn(20,5,generator=rng,dtype=torch.float64)
    g=torch.randn(20,7,generator=rng,dtype=torch.float64)
    a=ContentAdapter(p,g,[3,4],3)
    before=copy.deepcopy(a.state())
    transformed=a.target(p+1000,'p')
    assert torch.isfinite(transformed).all()
    for v in ('p','g'):
        for k,t in a.stats[v].items():assert torch.equal(t,before['stats'][v][k])
    _,tokens=a.token_inputs(p[:3],[g[0,:3],g[0,3:]])
    torch.testing.assert_close(tokens[:,-2:],torch.eye(2,dtype=tokens.dtype))
    assert tokens[0,3]==0 # padding does not masquerade as a real gene position


def test_train_only_event_quantiles_and_boundary_labels():
    t=torch.tensor([1.,2.,3.,4.,5.,6.,7.,1000.],dtype=torch.float64)
    e=torch.tensor([True]*7+[False])
    edges=fit_boundaries(t,e,4)
    torch.testing.assert_close(edges,torch.tensor([2.5,4.,5.5],dtype=t.dtype))
    test=torch.tensor([0.,2.5,4.,5.5,10000.],dtype=t.dtype)
    n=label_times(test,torch.ones(5,dtype=torch.bool),edges)
    assert n.tolist()==[0,1,2,3,3]
    # logit zero: event with n previous intervals costs (n+1)*log(2).
    torch.testing.assert_close(discrete_hazard_nll(torch.zeros(5,4,dtype=t.dtype),n,torch.ones(5,dtype=torch.bool),reduction='none'),(n+1)*torch.log(torch.tensor(2.,dtype=t.dtype)))


def test_rank_and_patient_output_guards():
    with pytest.raises(ValueError):ContentAdapter(torch.ones(8,4),torch.ones(8,4),[4],2)
    with pytest.raises(ValueError):cli.private_path(cli.ROOT/'public_patient_output')


def test_full_two_phase_survival_smoke_and_reload(tmp_path):
    torch.set_num_threads(1)
    r=synthetic_smoke(tmp_path/'smoke')
    assert r['status']=='completed' and r['synthetic_only']
    assert r['optimizer_steps']=={'role':12,'task':12}
    assert r['validation']['NLL']>0
    receipt=json.loads((tmp_path/'smoke/run/receipt.json').read_text(encoding='utf-8'))
    assert receipt['roles_frozen_during_task'] and receipt['reference_frozen']
    assert receipt['checkpoint_reload_parameters_equal']
    assert receipt['test_used'] is False
