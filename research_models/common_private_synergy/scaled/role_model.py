"""The scaled 16-slot common/private role model.

User specification implemented here
-----------------------------------
  1. slot allocation 4 / 4 / 4 / 4, i.e. 16 slots in total
  2. slot width 256
  3. omics encoding by 329 INDEPENDENT networks
  4. omics input: RAW pathway expression -> 329 MLPs
  5. reconstruction heads: DEFERRED (there is no reconstruction loss here)
  6. batch size 8 (set in the configuration, not in this module)

Slot budget
-----------
`{"p": [4, 4], "g": [4, 4]}` means, per modality, 4 COMMON slots and 4
PRIVATE (residual) slots:

    p_common   4 slots   width 256
    p_residual 4 slots   width 256
    g_common   4 slots   width 256
    g_residual 4 slots   width 256
    ---------------------------
    total     16 slots

What is kept from the archived role model
-----------------------------------------
* the role supervision against the FROZEN train-fitted role reference
  (`FrozenRoleReference`), with the decoder gradients kept connected;
* the ALL-SLOT pool, so that a role group's summary sees every slot of that
  group;
* the BOUNDED readout with a bounded additive part and a centered interaction
  part, so that the survival head cannot amplify without limit;
* the explicit two-phase contract helpers (`freeze_roles`, `freeze_readout`).

What is changed
---------------
* the omics encoder is the 329 independent pathway MLP encoder;
* the slot width is 256 and the slot counts are 4 + 4 per modality;
* the omics ENCODER INPUT is the raw per-pathway expression, standardized per
  pathway on train rows only.

The reconstruction head of the original SlotSPE is DEFERRED by the user's
instruction (point 5).  This model therefore has no reconstruction loss and no
reconstruction head, and it must not be compared against a reconstruction-aware
baseline without saying so.

Slot count and width do not change what the role names MEAN.  `common` and
`residual` remain architectural roles relative to a declared train-fitted
reference, not certified biological factors, not a PID decomposition and not
an independence claim.  More slots of the same kind increase capacity and
increase the risk of slot redundancy; that risk is measured separately, by the
existing slot relation tables, and it is not removed by this module.
"""
from __future__ import annotations

import torch
from torch import nn

from ..components.contracts import ArchitectureConfig
from ..components.role_model import BoundedLinear, RoleAllocation
from ..components.role_reference import FrozenRoleReference
from ..components.slots import AllSlotPool, RoleSlotEncoder
from .omics_encoder import PathwayOmicsEncoder


class ScaledRoleTargetModel(nn.Module):
    """16 slots of width 256 over two modalities, with a 329-network omics encoder.

    The forward pass takes:
      p        [patients, patches, features]   standardized real patch bag
      g        list of [patients, genes_i]     per-pathway standardized expression
      p_mask   [patients, patches] bool        real-token mask
      g_mask   [patients, pathways] bool       pathway mask
    """

    def __init__(self, reference: FrozenRoleReference, *, p_input_dim, omic_sizes,
                 p_allocation: RoleAllocation, g_allocation: RoleAllocation,
                 width, iterations, readout_hidden, readout_bound, interaction,
                 train_role_anchors, omics_dropout=0.25):
        super().__init__()
        self.reference = reference
        self.allocations = {'p': p_allocation, 'g': g_allocation}
        self.width = int(width)

        self.omics_encoder = PathwayOmicsEncoder(omic_sizes, width, dropout=omics_dropout)

        self.encoders, self.pools, self.decoders = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        for modality, input_dim in [('p', p_input_dim), ('g', width)]:
            allocation = self.allocations[modality]
            cfg = ArchitectureConfig(input_dim, input_dim, width=width,
                                     slots=allocation.common + allocation.residual,
                                     common_slots=allocation.common, iterations=iterations)
            self.encoders[modality] = RoleSlotEncoder(input_dim, cfg)
            for role in ('common', 'residual'):
                key = modality + '_' + role
                self.pools[key] = AllSlotPool(width)
                self.decoders[key] = nn.Linear(width, reference.dimension(modality))

        anchors = {m: torch.cat((train_role_anchors[m + '_common'],
                                  train_role_anchors[m + '_residual']), -1)
                   for m in ('p', 'g')}
        # The archived bounded readout is reused unchanged: it expects the role
        # output dimension, which the scaled architecture does not change, and
        # it carries the bounded-readout contract.
        from ..components.role_model import BoundedRoleReadout
        self.readout = BoundedRoleReadout(2 * reference.dimension('p'), 2 * reference.dimension('g'),
                                          hidden=readout_hidden, bound=readout_bound,
                                          interaction=interaction, anchors=anchors)

    # ------------------------------------------------------------------ info
    def slot_budget(self):
        return {m: {'common': self.allocations[m].common, 'residual': self.allocations[m].residual}
                for m in ('p', 'g')}

    def total_slot_count(self):
        return sum(self.allocations[m].common + self.allocations[m].residual for m in ('p', 'g'))

    # --------------------------------------------------------------- forward
    def encode(self, p, g, p_mask, g_mask):
        """Encode both modalities and decode the four role groups."""
        roles, parts = {}, {}
        for modality, x, mask in [('p', p, p_mask), ('g', g, g_mask)]:
            if modality == 'g':
                # The omics side enters through the 329 independent pathway MLPs.
                x = self.omics_encoder(g)
            part = self.encoders[modality](x, mask)
            parts[modality] = part
            for role, slots in [('common', part['common_slots']), ('residual', part['private_slots'])]:
                key = modality + '_' + role
                if slots.shape[1] == 0:
                    roles[key] = slots.new_zeros((len(x), self.reference.dimension(modality)))
                else:
                    decoded = self.decoders[key](self.pools[key](slots))
                    roles[key] = self.reference.project(decoded, modality, role)
        return roles, parts

    def predict_roles(self, roles):
        return self.readout(roles)

    def forward(self, p, g, p_mask, g_mask):
        roles, parts = self.encode(p, g, p_mask, g_mask)
        return {'roles': roles, 'parts': parts, **self.predict_roles(roles)}

    # ------------------------------------------------------------- contracts
    def freeze_roles(self):
        """Explicit two-stage contract: task training cannot move the role encoders."""
        for module in (self.encoders, self.pools, self.decoders, self.omics_encoder):
            module.requires_grad_(False)
            for parameter in module.parameters():
                parameter.grad = None

    def freeze_readout(self):
        """Explicit two-stage contract: task training cannot move the readout."""
        self.readout.requires_grad_(False)
        for parameter in self.readout.parameters():
            parameter.grad = None
