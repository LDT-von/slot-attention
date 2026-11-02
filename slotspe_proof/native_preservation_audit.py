"""Generated boundary counterexamples for V2; not a new host or patient model."""
import itertools

import torch
from torch import nn

from .plugin_host import SlotPluginHost
from .slot_disentanglement import SharedPrivateSlots, content_set_loss, separation_terms, private_summary, summary_hsic


def affine_blocks(adapter):
    mean = adapter.shared_projection.weight@adapter.shared_encoder.weight
    local = adapter.private_projection.weight@adapter.private_encoder.weight
    bias = adapter.shared_projection(adapter.shared_encoder.bias)+adapter.private_projection(adapter.private_encoder.bias)
    return mean, local, bias


def invert_affine_residual(z, adapter, coefficient):
    """Exact affine mean/contrast inverse; no iterative training or fitting."""
    mean, local, bias = affine_blocks(adapter)
    eye = torch.eye(z.shape[-1], dtype=z.dtype, device=z.device)
    zmean = z.mean(1)
    original_mean = torch.linalg.solve(eye+coefficient*(mean+local), (zmean-coefficient*bias).T).T
    contrast = z-zmean[:, None]
    recovered = torch.linalg.solve(eye+coefficient*local, contrast.reshape(-1, z.shape[-1]).T).T.reshape_as(z)
    return original_mean[:, None]+recovered


def collision_inputs(adapter, slots=8, batch=16):
    d = adapter.dimension
    if adapter.private_dim+adapter.target_dim >= d:
        raise ValueError('this independent-target construction needs a nonempty joint nullspace')
    rng = torch.Generator().manual_seed(adapter.seed+7000000)
    reference = torch.randn(adapter.target_dim, d, generator=rng, dtype=torch.double)/d**.5
    constraint = torch.cat([adapter.private_encoder.weight.detach(), reference])
    direction = torch.linalg.svd(constraint, full_matrices=True).Vh[-1]
    base = torch.randn(batch, slots, d, generator=rng, dtype=torch.double)
    base = base-(base@direction)[..., None]*direction
    delta = torch.zeros_like(base); delta[:, 0] = direction; delta[:, 1] = -direction
    return base, base+delta, direction, reference


def preservation_case(dimension=256, shared=64, private=192, target=32, slots=8, seed=731):
    adapter = SharedPrivateSlots(dimension, shared, private, target, seed).double()
    before, after, direction, reference = collision_inputs(adapter, slots)
    with torch.no_grad():
        latent0, record0 = adapter(before); latent1, record1 = adapter(after)
        targets0, targets1 = before@reference.T, after@reference.T
        content0 = content_set_loss(record0['content'], targets0)
        content1 = content_set_loss(record1['content'], targets1)
        separation0 = separation_terms(record0['shared'], record0['private'])
        separation1 = separation_terms(record1['shared'], record1['private'])
        mean, local, bias = affine_blocks(adapter)
        local_rank = int(torch.linalg.matrix_rank(adapter.private_encoder.weight))
        joined_rank = int(torch.linalg.matrix_rank(torch.cat([adapter.shared_encoder.weight, adapter.private_encoder.weight])))
        code_rank = (slots-1)*local_rank+joined_rank
        output_rank = (slots-1)*int(torch.linalg.matrix_rank(local))+int(torch.linalg.matrix_rank(mean+local))
        lipschitz = max(float(torch.linalg.matrix_norm(local, 2)), float(torch.linalg.matrix_norm(mean+local, 2)))
        coefficient = .5/max(1., lipschitz)
        residual = after+coefficient*latent1
        recovered = invert_affine_residual(residual, adapter, coefficient)
        eye = torch.eye(dimension, dtype=torch.double)
        minimum_residual_singular = min(float(torch.linalg.svdvals(eye+coefficient*local).min()),
                                        float(torch.linalg.svdvals(eye+coefficient*(mean+local)).min()))
        native_score0 = (before@direction).square().mean(1)
        native_score1 = (after@direction).square().mean(1)
        constant_set = before[:, :1].expand(-1, slots, -1)
        constant_latent, _ = adapter(constant_set)
        constant_recovered = torch.linalg.solve(mean+local, (constant_latent.mean(1)-bias).T).T
        measured = {
            'dimension': dimension, 'slots': slots, 'shared': shared, 'private': private,
            'encoder_rank_numeric': code_rank, 'encoder_ambient_dimension': slots*dimension,
            'encoder_nullity_numeric': slots*dimension-code_rank,
            'encoder_structural_rank_upper_bound': min(dimension, shared+private)+(slots-1)*min(dimension, private),
            'replacement_rank_numeric': output_rank,
            'private_null_residual': float((adapter.private_encoder.weight@direction).abs().max()),
            'fixed_target_null_residual': float((reference@direction).abs().max()),
            'native_input_difference_norm': float((after-before).norm()),
            'permutation_invariant_native_moment_gap_mean': float((native_score1-native_score0).mean()),
            'replacement_max_difference': float((latent1-latent0).abs().max()),
            'shared_max_difference': float((record1['shared']-record0['shared']).abs().max()),
            'private_max_difference': float((record1['private']-record0['private']).abs().max()),
            'content_target_max_difference': float((targets1-targets0).abs().max()),
            'content_loss_difference': float(content1-content0),
            'hsic_difference': float(separation1['hsic']-separation0['hsic']),
            'variance_penalty_difference': float(separation1['variance']-separation0['variance']),
            'affine_global_operator_norm': lipschitz,
            'bounded_residual_contraction_factor': coefficient*lipschitz,
            'bounded_residual_min_singular_value': minimum_residual_singular,
            'bounded_residual_inverse_max_error': float((recovered-after).abs().max()),
            'bounded_residual_lower_distance_ratio': float((residual-(before+coefficient*latent0)).norm()/(after-before).norm()),
            'identical_slot_manifold_inverse_max_error': float((constant_recovered-before[:, 0]).abs().max()),
            'identical_slot_mean_map_rank': int(torch.linalg.matrix_rank(mean+local)),
            'interpretation': 'source adapter collision on generated states; not observed patient harm or the actual native survival head'}
    return measured


def summary_dependence_counterexample():
    rows = list(itertools.product([-1., 1.], [-1., 2.], [-2., 1.], [.5, 1.5]))
    g = torch.tensor([[b]*64 for b, _, _, _ in rows], dtype=torch.double)
    private = []
    for b, m1, m2, amplitude in rows:
        plus = torch.tensor([m1+amplitude, m2+b*amplitude], dtype=torch.double).repeat(96)
        minus = torch.tensor([m1-amplitude, m2-b*amplitude], dtype=torch.double).repeat(96)
        private.append(torch.stack([plus, minus]*4))
    private = torch.stack(private)
    summary = private_summary(private)
    hsic = summary_hsic(g, summary)
    centered = private-private.mean(1, keepdim=True)
    covariance_sign = (centered[:, :, 0]*centered[:, :, 1]).mean(1).sign()
    assert torch.equal(covariance_sign, g[:, 0])
    return {'enumerated_patient_states': 16, 'slots_per_state': 8, 'private_dim': 192,
            'source_summary_hsic': float(hsic), 'known_mi_shared_full_private_nats': 0.6931471805599453,
            'known_mi_shared_summary_nats': 0., 'cross_covariance_recovers_shared_bit_exactly': True,
            'interpretation': 'exact declared finite population; source HSIC acts on first/second coordinate moments, not the full private slot set'}


class _MomentBackbone(nn.Module):
    """Permutation-invariant generated predictor, NOT protected native SlotSPE."""
    def __init__(self, dimension):
        super().__init__()
        self.slot_attention_wsi = nn.Identity(); self.slot_attention_omic = nn.Identity()
        self.register_buffer('direction', torch.zeros(dimension, dtype=torch.double))

    def forward(self, *, x_wsi, x_omics, omic_missing=False):
        wsi = self.slot_attention_wsi(x_wsi); omic = self.slot_attention_omic(x_omics)
        score = (wsi@self.direction).square().mean(1)+0*omic.sum((1, 2))
        return score[:, None], score.new_zeros(())


def actual_hook_collision():
    backbone = _MomentBackbone(256)
    host = SlotPluginHost(backbone, 'V2_content').double().eval()
    before, after, direction, _ = collision_inputs(host.parts['wsi'])
    backbone.direction.copy_(direction)
    omic = torch.zeros_like(before)
    with torch.no_grad():
        native0 = backbone(x_wsi=before, x_omics=omic)[0]
        native1 = backbone(x_wsi=after, x_omics=omic)[0]
        replaced0 = host(x_wsi=before, x_omics=omic, omic_missing=False)[0]
        replaced1 = host(x_wsi=after, x_omics=omic, omic_missing=False)[0]
    return {'actual_plugin_host_exercised': True, 'backbone_is_protected_native_model': False,
            'generated_native_predictor_gap': float((native1-native0).abs().mean()),
            'generated_replaced_predictor_gap': float((replaced1-replaced0).abs().max()),
            'hook_records_present': set(host.last_records) == {'wsi', 'omic', 'kl'}}


def cancellation_case():
    adapter = SharedPrivateSlots(8, 2, 6, 1).double()
    with torch.no_grad():
        for layer in [adapter.shared_encoder, adapter.private_encoder, adapter.shared_projection, adapter.private_projection]:
            layer.weight.zero_(); layer.bias.zero_()
        adapter.private_encoder.weight[:, :6].copy_(torch.eye(6, dtype=torch.double))
        adapter.private_projection.weight[:6].copy_(-torch.eye(6, dtype=torch.double))
        left = torch.zeros(2, 4, 8, dtype=torch.double); right = left.clone(); right[:, :, 0] = 1
        f0, _ = adapter(left); f1, _ = adapter(right)
        ordinary_gap = float(((left+f0)-(right+f1)).norm())
        bounded_gap = float(((left+.5*f0)-(right+.5*f1)).norm())
    return {'input_gap': float((right-left).norm()), 'ordinary_residual_gap': ordinary_gap,
            'half_gain_residual_gap': bounded_gap, 'interpretation': 'adding native input alone is not a global injectivity guarantee'}
