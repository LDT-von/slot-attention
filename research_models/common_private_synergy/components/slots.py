"""Role-specific query banks and full-slot summaries; source-only preparation.

Based on the research SlotEncoder update, not an original SlotSPE replacement.
Role banks are distinct parameters before iteration; updates compete jointly.
Role names alone imply no identified information factors. No training entry.
"""
import torch
from torch import nn
from .contracts import ArchitectureConfig


READ_EPS = 1e-8
READ_NORMALIZATION = 'masked_add_eps_l1_v2'


def masked_slot_read(logits, real_mask):
    """Slot competition followed by real-token L1 normalization.

    Return pre-epsilon mass separately so numerical smoothing cannot conceal
    competition starvation. A nonempty real bag guarantees a positive divisor,
    including when a slot's softmax assignments underflow completely to zero.
    """
    if logits.ndim != 3 or not logits.numel() or logits.dtype not in (torch.float32, torch.float64):
        raise ValueError('float32/float64 [patients,slots,tokens] logits required')
    if real_mask.dtype != torch.bool or real_mask.shape != (logits.shape[0], logits.shape[2]) or real_mask.device != logits.device:
        raise ValueError('aligned boolean real-token mask required')
    if not real_mask.any(1).all() or not torch.isfinite(logits).all():
        raise ValueError('finite logits and at least one real token per bag required')
    mask = real_mask[:, None, :]
    assignment = logits.softmax(1) * mask
    mass = assignment.sum(-1)
    weights = assignment + READ_EPS * mask.to(logits.dtype)
    attention = weights / weights.sum(-1, keepdim=True)
    return attention, mass


class RoleSlotEncoder(nn.Module):
    def __init__(self, input_dim, config: ArchitectureConfig):
        super().__init__()
        d = config.width
        self.config = config
        self.common_queries = nn.Parameter(torch.randn(config.common_slots, d) * .1)
        self.private_queries = nn.Parameter(torch.randn(config.private_slots, d) * .1)
        self.project = nn.Linear(input_dim, d)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.query = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.update = nn.GRUCell(d, d)

    def forward(self, x, real_mask):
        if x.ndim != 3 or not x.is_floating_point() or x.shape[0] == 0:
            raise ValueError('nonempty floating [patients,tokens,features] required')
        if real_mask.dtype != torch.bool or real_mask.shape != x.shape[:2] or real_mask.device != x.device:
            raise ValueError('explicit boolean real_mask on the input device required')
        if not real_mask.any(1).all() or not torch.isfinite(x[real_mask]).all():
            raise ValueError('each bag needs finite real tokens')
        features = self.project(x.masked_fill(~real_mask[..., None], 0))
        k, v = self.key(features), self.value(features)
        initial = torch.cat((self.common_queries, self.private_queries), 0)
        slots = initial[None].expand(x.shape[0], -1, -1)
        read_mass = []
        for _ in range(self.config.iterations):
            logits = self.query(self.norm(slots)) @ k.transpose(1, 2) / k.shape[-1] ** .5
            attention, mass = masked_slot_read(logits, real_mask)
            read_mass.append(mass.detach())
            messages = attention @ v
            slots = self.update(messages.reshape(-1, slots.shape[-1]), slots.reshape(-1, slots.shape[-1])).reshape_as(slots)
        cut = self.config.common_slots
        return {'slots': slots, 'common_slots': slots[:, :cut], 'private_slots': slots[:, cut:],
                'read_attention': attention, 'real_mask': real_mask,
                'raw_read_mass': torch.stack(read_mass, dim=1)}


class AllSlotPool(nn.Module):
    """Same input/output width in ratio arms; mean after nonlinear phi.

    Empty groups return a fixed zero, without biases masquerading as content.
    Symmetric aggregation can lose information; all-slot access is not usage.
    """
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.phi = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, slots):
        if slots.ndim != 3 or slots.shape[-1] != self.width:
            raise ValueError('expected [patients,slots,width]')
        if slots.shape[1] == 0:
            return slots.new_zeros((slots.shape[0], self.width))
        return self.phi(slots).mean(1)


class SummaryProjector(nn.Module):
    """Loss-only low-dimensional projection; instantiate once and reuse for views.

    One instance per modality/role. No batch norm, patient mixing or automatic
    adaptation to validation data. Projector success need not transfer to slots.
    """
    def __init__(self, width, projection_dim=2):
        super().__init__()
        if type(projection_dim) is not int or projection_dim < 1:
            raise ValueError('positive projection dimension required')
        self.net = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, projection_dim))

    def forward(self, summary):
        return self.net(summary)
