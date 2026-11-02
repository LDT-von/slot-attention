import torch
from slotspe_proof.prognostic_effect_probe import effects


def test_xor_effects_do_not_identify_global_private_sign():
    bits = torch.tensor([[1., 1., -1., 1.], [-1., 1., 1., -1.]])
    assert torch.equal(effects(bits, 'paired_xor'), effects(-bits, 'paired_xor'))
    assert not torch.equal(effects(bits, 'factorwise'), effects(-bits, 'factorwise'))
