"""Known scalar Gaussian channel controls, not neural MI estimators."""
import math
import numpy as np
import torch

from .slot_pooling_diagnostics import gaussian_code
from .xor_readout_diagnostics import mixture_survival_logits


def scalar_channel_mi(snr, order=128):
    """Population binary/XOR MI by Gaussian quadrature; units nats.

    Equal-prior independent signed bits each pass through a known scalar
    Gaussian channel with this SNR. This does not estimate learned-slot MI.
    """
    if not math.isfinite(snr) or snr < 0 or order < 8:
        raise ValueError('nonnegative finite SNR and quadrature order >=8 required')
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    weights = weights/math.sqrt(math.pi)
    # Symmetry permits conditioning the true transmitted bit to +1.
    logits = 2*snr+2*math.sqrt(snr)*math.sqrt(2)*nodes
    bit = math.log(2)-np.dot(weights, np.logaddexp(0., -logits))
    lp = -np.logaddexp(0., -logits)
    ln = -np.logaddexp(0., logits)
    same = np.logaddexp(lp[:, None]+lp[None, :], ln[:, None]+ln[None, :])
    xor = math.log(2)+np.sum(weights[:, None]*weights[None, :]*same)
    return {'binary_mi_nats': float(bit), 'xor_mi_nats': float(xor)}


def channel_snrs(slots=4, total_rate=1.):
    if slots < 1:
        raise ValueError('positive slot count required')
    a, sigma = gaussian_code(total_rate)
    return {'single_known_role': math.expm1(2*total_rate),
            'single_untyped_raw_mean': a*a/(sigma*sigma+slots-1),
            'repeated_equal_total': slots*math.expm1(2*total_rate/slots),
            'repeated_equal_per_slot': slots*math.expm1(2*total_rate)}


def raw_mean_bayes_logits(common, private, condition):
    """Bayes predictor given only the untyped raw mean, not learned phi."""
    if condition == 'noise_only':
        xor = torch.full_like(common[:, 0], .5)
    else:
        ps = []
        for value in private:
            summed = value[:, :, 0].sum(1)
            if condition == 'clean_single':
                p = (summed > 0).to(value.dtype)
            else:
                rate = .25 if condition == 'repeated_total1' else 1.
                amplitude, sigma = gaussian_code(rate)
                variance = sigma*sigma+3 if condition == 'single_total1' else 4*sigma*sigma
                amplitude_sum = amplitude if condition == 'single_total1' else 4*amplitude
                p = torch.sigmoid(2*amplitude_sum*summed/variance)
            ps.append(p)
        p, q = ps
        xor = p*(1-q)+(1-p)*q
    return mixture_survival_logits(common[:, 0], xor)
