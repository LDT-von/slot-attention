"""Explicit hazard-logit additive control and a separately switchable interaction.

    h0 = b + .5*(hc(cw)+hc(co)) + hw(uw) + ho(uo)
    h  = h0 + interaction([cw,uw], [co,uo])

Each summary must depend ONLY on its own modality. No nonlinear fused-G path
is allowed in h0. Sigmoid(h0) is not additive on probability scale. Bilinear
expressivity is not PID; constant encoder coordinates can still mimic main
effects. This is an unexecuted component, not an identified synergy estimator.
"""
import torch
from torch import nn
from .contracts import ArchitectureConfig


def _head(width, bins):
    return nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, bins, bias=False))


class AdditiveInteractionReadout(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        d, r = config.width, config.interaction_rank
        self.common_head = _head(d, config.bins)
        self.private_heads = nn.ModuleList([_head(d, config.bins) for _ in range(2)])
        self.bias = nn.Parameter(torch.zeros(config.bins))
        # Included in both arms for identical state_dict shape; disabled arm
        # never evaluates them. Active capacity still differs and needs controls.
        self.left = nn.Linear(2*d, r, bias=False)
        self.right = nn.Linear(2*d, r, bias=False)
        self.cross_out = nn.Linear(r, config.bins, bias=False)

    def forward(self, common_wsi, private_wsi, common_omic, private_omic):
        values = (common_wsi, private_wsi, common_omic, private_omic)
        shape = common_wsi.shape
        if len(shape) != 2 or shape[0] == 0 or shape[1] != self.config.width:
            raise ValueError('expected nonempty [patients,width] summaries')
        if any(x.shape != shape or x.device != common_wsi.device or x.dtype != common_wsi.dtype for x in values):
            raise ValueError('paired summaries must have same shape, device and dtype')
        if any(x.dtype not in (torch.float32, torch.float64) or not torch.isfinite(x).all() for x in values):
            raise ValueError('finite float32/float64 summaries required')
        zero = common_wsi.new_zeros((shape[0], self.config.bins))
        cw = self.common_head(common_wsi) if self.config.common_slots else zero
        co = self.common_head(common_omic) if self.config.common_slots else zero
        uw = self.private_heads[0](private_wsi) if self.config.private_slots else zero
        uo = self.private_heads[1](private_omic) if self.config.private_slots else zero
        additive = self.bias + .5*(cw+co) + uw + uo
        cross = zero
        if self.config.use_interaction:
            # Ignore supplied values for absent groups, rather than trusting a caller.
            z = torch.zeros_like(common_wsi)
            left = torch.cat((common_wsi if self.config.common_slots else z,
                              private_wsi if self.config.private_slots else z), -1)
            right = torch.cat((common_omic if self.config.common_slots else z,
                               private_omic if self.config.private_slots else z), -1)
            cross = self.cross_out(self.left(left) * self.right(right))
        return {'logits': additive + cross, 'additive_logits': additive,
                'interaction_logits': cross, 'common_wsi_logits': cw + self.bias if self.config.common_slots else None,
                'common_omic_logits': co + self.bias if self.config.common_slots else None,
                'g_diagnostic_only': .5*(common_wsi+common_omic) if self.config.common_slots else torch.zeros_like(common_wsi),
                'common_active': self.config.common_slots > 0,
                'private_active': self.config.private_slots > 0}
