"""V0 reference only; superseded design is in ARCHITECTURE.md and components/.

Do not use this joint-MLP skeleton as a no-interaction control. The revised
parts have NOT been assembled or tensor-tested. No training entry point.

Seven common-role and one private-role learned initial queries per modality are
roles, not identified biological factors. Interaction is not a PID estimate.
"""
import torch
from torch import nn

from slotspe_proof.common_private_survival import SlotEncoder


def _mlp(dim):
    return nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))


class RoleSlotSurvival(nn.Module):
    """Feature-set scaffold with all eight slots used in each modality."""

    def __init__(self, wsi_dim, omic_dim, width=64, bins=4, common_slots=7):
        super().__init__()
        if not 1 <= common_slots < 8:
            raise ValueError('common_slots must be between 1 and 7')
        self.common_slots = common_slots
        self.encoders = nn.ModuleList([SlotEncoder(d, width, 8) for d in (wsi_dim, omic_dim)])
        self.common_phi = nn.ModuleList([_mlp(width) for _ in range(2)])
        self.private_phi = nn.ModuleList([_mlp(width) for _ in range(2)])
        self.fuse = nn.Sequential(nn.Linear(2 * width, width), nn.GELU())
        self.joint_modality = nn.ModuleList([nn.Linear(2 * width, width) for _ in range(2)])
        self.interaction = _mlp(width)
        self.head = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, bins))

    def forward(self, wsi, omic, wsi_mask=None, omic_mask=None):
        if wsi.ndim != 3 or omic.ndim != 3 or wsi.shape[0] != omic.shape[0]:
            raise ValueError('inputs must be paired batch x tokens x features tensors')
        inputs = (wsi, omic)
        masks = tuple(torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if mask is None else mask
                      for x, mask in zip(inputs, (wsi_mask, omic_mask)))
        slots = [encoder(x, mask) for encoder, x, mask in zip(self.encoders, inputs, masks)]
        common = [phi(s[:, :self.common_slots]).mean(dim=1)
                  for phi, s in zip(self.common_phi, slots)]
        private = [phi(s[:, self.common_slots:]).mean(dim=1)
                   for phi, s in zip(self.private_phi, slots)]
        g = self.fuse(torch.cat(common, dim=-1))
        modality = [layer(torch.cat((c, p), dim=-1))
                    for layer, c, p in zip(self.joint_modality, common, private)]
        interaction = self.interaction(modality[0] * modality[1])
        logits = self.head(torch.cat((g, private[0], private[1], interaction), dim=-1))
        return {'slots': slots, 'common': common, 'private': private,
                'g': g, 'interaction': interaction, 'logits': logits}


def _correlation(x, y, min_variance=1e-12):
    """Cross-patient Pearson matrix; explicit refusal for degenerate coordinates."""
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape or x.shape[0] < 2:
        raise ValueError('paired 2D features with equal dimensions and at least 2 patients required')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('nonfinite features')
    xc, yc = x - x.mean(0), y - y.mean(0)
    vx, vy = xc.square().mean(0), yc.square().mean(0)
    if (vx <= min_variance).any() or (vy <= min_variance).any():
        raise ValueError('zero or near-zero across-patient variance')
    return (xc / vx.sqrt()).T @ (yc / vy.sqrt()) / x.shape[0]


def _identity_loss(correlation, offdiag_weight):
    diagonal = torch.diagonal(correlation)
    eye = torch.eye(correlation.shape[0], dtype=torch.bool, device=correlation.device)
    off = correlation.masked_select(~eye)
    return (diagonal - 1).square().mean() + offdiag_weight * (off.square().mean() if off.numel() else diagonal.new_zeros(()))


def _zero_loss(correlation, offdiag_weight):
    diagonal = torch.diagonal(correlation)
    eye = torch.eye(correlation.shape[0], dtype=torch.bool, device=correlation.device)
    off = correlation.masked_select(~eye)
    return diagonal.square().mean() + offdiag_weight * (off.square().mean() if off.numel() else diagonal.new_zeros(()))


def decur_style_losses(view1, view2, offdiag_weight=1.0):
    """Four loss terms, without weighting or optimizer; views must be paired.

    Intra-modality uses concatenated group summaries; cross-modality terms are
    averaged over both views. Diagonal and off-diagonal losses are separately
    averaged (not summed as in DeCUR); offdiag_weight applies to both cross
    terms and both intra terms. This is a group-level adaptation, not DeCUR.
    Batch size two is mathematically accepted but a statistically poor regime.
    """
    if offdiag_weight < 0:
        raise ValueError('offdiag_weight must be nonnegative')
    for view in (view1, view2):
        if len(view['common']) != 2 or len(view['private']) != 2:
            raise ValueError('two modality representations required')
    full1 = [torch.cat((c, p), -1) for c, p in zip(view1['common'], view1['private'])]
    full2 = [torch.cat((c, p), -1) for c, p in zip(view2['common'], view2['private'])]
    intra = [_identity_loss(_correlation(a, b), offdiag_weight) for a, b in zip(full1, full2)]
    cross_common = torch.stack([_identity_loss(_correlation(*v['common']), offdiag_weight)
                                for v in (view1, view2)]).mean()
    cross_private = torch.stack([_zero_loss(_correlation(*v['private']), offdiag_weight)
                                 for v in (view1, view2)]).mean()
    return {'intra_wsi': intra[0], 'intra_omic': intra[1],
            'cross_common': cross_common, 'cross_private': cross_private}
