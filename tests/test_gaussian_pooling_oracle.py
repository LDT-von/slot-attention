import math
import pytest
import torch

from slotspe_proof.gaussian_pooling_oracle import scalar_channel_mi, channel_snrs, raw_mean_bayes_logits
from slotspe_proof.slot_pooling_diagnostics import code_inputs, code_bayes_logits


def test_known_channel_mi_bounds_and_noise_null():
    null = scalar_channel_mi(0.)
    assert abs(null['binary_mi_nats']) < 1e-12 and abs(null['xor_mi_nats']) < 1e-12
    previous = 0.
    for snr in sorted(channel_snrs().values()):
        row = scalar_channel_mi(snr)
        assert 0 <= row['xor_mi_nats'] <= row['binary_mi_nats'] <= math.log(2)+1e-9
        assert row['xor_mi_nats'] >= previous
        previous = row['xor_mi_nats']
    assert channel_snrs()['single_known_role'] > channel_snrs()['repeated_equal_total']
    assert channel_snrs()['repeated_equal_total'] > channel_snrs()['single_untyped_raw_mean']


def test_raw_mean_is_sufficient_only_for_repeated_or_clean_controls():
    torch.manual_seed(7)
    shared = torch.randn(12)
    bits = 2*torch.randint(2, (12, 2)).float()-1
    noise = [torch.randn(12, 4, 4) for _ in range(2)]
    for condition in ['repeated_total1', 'repeated_total4', 'clean_single', 'noise_only']:
        g, v = code_inputs(shared, bits, noise, condition)
        assert torch.allclose(raw_mean_bayes_logits(g, v, condition), code_bayes_logits(g, v, condition), atol=1e-7)
    g, v = code_inputs(shared, bits, noise, 'single_total1')
    assert not torch.allclose(raw_mean_bayes_logits(g, v, 'single_total1'), code_bayes_logits(g, v, 'single_total1'))
    with pytest.raises(ValueError): scalar_channel_mi(-1.)
