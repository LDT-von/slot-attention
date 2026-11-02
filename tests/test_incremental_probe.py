import torch
import pytest
from slotspe_proof.incremental_probe import incremental_probe
from scripts.check_incremental_probe import pack


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def partitions(case):
    return [pack(seed,n,case) for seed,n in [(7,100),(8,80),(9,120)]]


def test_test_labels_cannot_change_selected_model():
    a,b,c=partitions('private_signal')
    r=incremental_probe(a,b,c)
    changed=incremental_probe(a,b,(c[0],c[1],c[2]*-100))
    for arm in ['common','joint']:
        for key in ['family','bandwidth','private_weight','regularization','effective_df','validation_mse']:
            assert r[arm][key]==changed[arm][key]


def test_deterministic_copy_not_increment():
    r=incremental_probe(*partitions('quadratic_copy'))
    assert abs(r['gain_relative_to_fit_mean_baseline'])<.01


def test_true_private_signal_detected():
    r=incremental_probe(*partitions('private_signal'))
    assert r['gain_relative_to_fit_mean_baseline']>.5


def test_all_constant_features_are_finite():
    parts=partitions('noise_private')
    r=incremental_probe(*[(torch.zeros_like(g),torch.zeros_like(u),y) for g,u,y in parts])
    assert abs(r['test_mse_gain'])<1e-10


def test_invalid_partition_rejected():
    a,b,c=partitions('private_signal')
    with pytest.raises(ValueError): incremental_probe(a,b,(c[0][:-1],c[1],c[2]))


def test_probe_does_not_backpropagate_into_representations():
    a,b,c=partitions('private_signal')
    a[0].requires_grad_();incremental_probe(a,b,c)
    assert a[0].grad is None
