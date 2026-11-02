"""Known-answer synthetic code/readout and censored-survival mixture helpers.

These are privileged factor controls, not a new predictive architecture.
The repeated code deliberately carries duplicate information across slots.
"""
import math
import torch

MODES = ('after_phi_clean', 'before_phi_clean_single', 'before_phi_rate1_single',
         'before_phi_rate1_repeated', 'before_phi_noise_only', 'before_phi_mismatched_repeated')


def oracle_inputs(shared_factor, private_bits, private_noise, mode):
    if mode not in MODES or private_bits.shape != (len(shared_factor), 2):
        raise ValueError('oracle mode/factor layout')
    if not bool(((private_bits == -1) | (private_bits == 1)).all()):
        raise ValueError('signed planted bits required')
    if len(private_noise) != 2 or any(e.shape != (len(shared_factor), 4, 4) for e in private_noise):
        raise ValueError('two [N,4,4] private noise arrays required')
    g = torch.zeros(len(shared_factor), 4, dtype=shared_factor.dtype, device=shared_factor.device)
    g[:, 0] = shared_factor
    result = []
    amplitude, sigma = math.sqrt(1-math.exp(-2)), math.exp(-1)
    for j, eps in enumerate(private_noise):
        if mode in ('after_phi_clean', 'before_phi_clean_single'):
            v = torch.zeros_like(eps); v[:, 0, 0] = private_bits[:, j]
        else:
            v = eps.clone()
            if mode == 'before_phi_rate1_single':
                v[:, 0, 0] = amplitude*private_bits[:, j]+sigma*eps[:, 0, 0]
            elif mode in ('before_phi_rate1_repeated', 'before_phi_mismatched_repeated'):
                v[:, :, 0] = amplitude*private_bits[:, j, None]+sigma*eps[:, :, 0]
        result.append(v)
    return g, result


def oracle_forward(net, common, private, mode):
    if mode == 'after_phi_clean':
        width = net.phi[0][-1].out_features
        z = [torch.zeros(len(common), width, dtype=common.dtype, device=common.device) for _ in private]
        for target, v in zip(z, private): target[:, 0] = v[:, 0, 0]
        return net.head(torch.cat([common, *z], -1))
    return net.readout(common, private)


def mixture_survival_logits(shared_factor, xor_probability):
    """Hazard of a survival mixture; posterior component weights change with time."""
    s = shared_factor.double()
    probability = xor_probability.double()
    if probability.shape != s.shape or not bool(((probability >= 0) & (probability <= 1)).all()):
        raise ValueError('one XOR probability per row required')
    risk = .3*s[:, None]+s.new_tensor([-1.5, 1.5])[None]
    hazard = torch.sigmoid(risk[:, :, None]+s.new_tensor([-2., -1., 0., 1.])[None, None])
    previous_survival = torch.cat([torch.ones_like(hazard[:, :, :1]), (1-hazard).cumprod(-1)[:, :, :-1]], -1)
    weights = torch.stack([1-probability, probability], -1)[:, :, None]
    numerator = (weights*previous_survival*hazard).sum(1)
    denominator = (weights*previous_survival).sum(1)
    return torch.logit((numerator/denominator).clamp(1e-12, 1-1e-12))


def oracle_bayes_logits(common, private, mode):
    if mode in ('before_phi_noise_only', 'before_phi_mismatched_repeated'):
        xor = torch.full_like(common[:, 0], .5)
    else:
        ps = []
        for v in private:
            if mode in ('after_phi_clean', 'before_phi_clean_single'):
                p = (v[:, 0, 0] > 0).to(v.dtype)
            else:
                signal = v[:, 0, 0] if mode == 'before_phi_rate1_single' else v[:, :, 0].sum(1)
                amplitude, variance = math.sqrt(1-math.exp(-2)), math.exp(-2)
                p = torch.sigmoid(2*amplitude*signal/variance)
            ps.append(p)
        xor = ps[0]*(1-ps[1])+(1-ps[0])*ps[1]
    return mixture_survival_logits(common[:, 0], xor)


def configure_readout_training(net, head_only=False):
    for name, parameter in net.named_parameters():
        parameter.grad = None
        parameter.requires_grad_(name.startswith('head.') or (not head_only and name.startswith(('phi.', 'role_offsets.'))))
    return [p for p in net.parameters() if p.requires_grad]
