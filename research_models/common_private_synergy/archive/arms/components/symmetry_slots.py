"""Per-slot query modulation: an explicit symmetry-breaking variant of the role slot encoder.

Added as a NEW module. `components/slots.py` is byte-for-byte unchanged, because
its SHA-256 is recorded in the archived matched35 receipts and in the Cox-objective
receipts; editing it would invalidate the provenance of those runs.

What this module changes, and nothing else:

  The archived encoder maps every slot through the SAME query projection
  `query = self.query(self.norm(slots))`. Two slots that hold the same content
  therefore receive the same query forever, which is the mechanism that makes
  slot exchangeability an exact symmetry of the update. This variant adds one
  parameter per slot, `slot_query_bias`, which is added to the query of that
  slot only:

      query_i = self.query(self.norm(slots_i)) + slot_query_bias_i

The bias is initialised to EXACTLY ZERO, so at initialisation the forward pass
of this variant is bit-for-bit identical to the archived encoder. Every later
difference is therefore learned, not seeded, and the arm comparison starts from
a matched forward rather than from a matched weight tensor only.

The bias is not an identifier for a biological factor. It is a per-slot degree
of freedom that lets two slots that hold similar content still issue
different queries. Whether that changes the survival score is an empirical
question and is not assumed here.

This is a Masked Multi-Query Slot Attention style device (per-slot query
freedom instead of a shared query), applied to the role slot encoder of the
research architecture. It is not a modification of the original SlotSPE
source and it does not reproduce the original paper's full architecture.
"""
import torch
from torch import nn

from .slots import READ_EPS, RoleSlotEncoder


class SymmetryBrokenRoleSlotEncoder(RoleSlotEncoder):
    """Role slot encoder with one learnable per-slot query bias.

    The inherited weights (project, key, value, query, norm, update and both
    query banks) keep their original names and shapes, so a checkpoint written
    by the archived encoder loads into this class unchanged. The only new
    tensor is `slot_query_bias`, of shape [slots, width], initialised to zero.
    """

    def __init__(self, input_dim, config):
        super().__init__(input_dim, config)
        self.slot_query_bias = nn.Parameter(torch.zeros(config.slots, config.width))

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
        read_mass, diagnostics = [], {}
        bias = self.slot_query_bias
        for _ in range(self.config.iterations):
            query = self.query(self.norm(slots)) + bias
            logits = query @ k.transpose(1, 2) / k.shape[-1] ** .5
            assignment = logits.softmax(1) * real_mask[:, None, :]
            mass = assignment.sum(-1)
            weights = assignment + READ_EPS * real_mask[:, None, :].to(logits.dtype)
            attention = weights / weights.sum(-1, keepdim=True)
            read_mass.append(mass.detach())
            if self.observe_read_diagnostics:
                from .slots import observe_slot_read
                for name, value in observe_slot_read(logits, query, k, real_mask, mass).items():
                    diagnostics.setdefault(name, []).append(value)
            messages = attention @ v
            slots = self.update(messages.reshape(-1, slots.shape[-1]),
                                slots.reshape(-1, slots.shape[-1])).reshape_as(slots)
        cut = self.config.common_slots
        result = {'slots': slots, 'common_slots': slots[:, :cut], 'private_slots': slots[:, cut:],
                  'read_attention': attention, 'real_mask': real_mask,
                  'raw_read_mass': torch.stack(read_mass, dim=1)}
        if self.observe_read_diagnostics:
            result['read_diagnostics'] = {name: torch.stack(values, dim=1)
                                          for name, values in diagnostics.items()}
        return result
