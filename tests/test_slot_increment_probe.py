import torch
import pytest
from slotspe_proof.slot_increment_probe import slot_increment_probe
from scripts.check_slot_increment_probe import pack


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def parts(case):return [pack(seed,n,case) for seed,n in [(13,80),(14,80),(15,100)]]


def test_duplicate_slots_have_no_gain():
    r=slot_increment_probe(*parts('duplicate'))
    assert abs(r['gain_relative_to_fit_mean_baseline'])<.01


def test_independent_factors_have_gain():
    r=slot_increment_probe(*parts('complementary'))
    assert r['gain_relative_to_fit_mean_baseline']>.4


def test_test_targets_do_not_choose_slot_or_hyperparameters():
    a,b,c=parts('complementary');r=slot_increment_probe(a,b,c)
    changed=slot_increment_probe(a,b,(c[0],c[1],-20*c[2]))
    assert r['best_single_indices']==changed['best_single_indices']
    for arm in ['selected_single','all_slots']:
        for key in ['family','bandwidth','regularization','private_weight','validation_mse']:
            assert r[arm]['joint'][key]==changed[arm]['joint'][key]


def test_modalities_and_width_must_align():
    a,b,c=parts('duplicate')
    with pytest.raises(ValueError):slot_increment_probe(a,b,(c[0],[c[1][0][:,:2]],c[2]))
