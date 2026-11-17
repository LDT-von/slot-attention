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


def zero_slot_update_bias(encoder):
    """Zero the GRU update bias of a role slot encoder, in place.

    The archived `components/slots.py` is NOT edited, because its SHA-256 is
    recorded in the archived matched35, spread, Cox and symmetry receipts.
    This helper performs the same operation from outside, so the archived
    file stays byte for byte while the scaled arm gets a different
    INITIALISATION of an existing parameter.

    `nn.GRUCell` owns two bias tensors, `bias_ih` and `bias_hh`.  Both are
    zeroed: the input bias is the one the archived diagnosis identified, and
    the hidden bias is zeroed with it so that the update has no constant
    offset on the first iteration either.

    The function returns the number of bias tensors it touched, so a caller
    or a test can assert that it did something and did not silently miss a
    parameter.  It raises if the encoder does not own a GRU update cell, so a
    future change to the archived encoder cannot silently disable this.
    """
    update = getattr(encoder, 'update', None)
    if not isinstance(update, nn.GRUCell):
        raise TypeError('the role slot encoder must own an nn.GRUCell update cell')
    touched = 0
    with torch.no_grad():
        for name in ('bias_ih', 'bias_hh'):
            bias = getattr(update, name, None)
            if bias is None:
                continue
            bias.zero_()
            touched += 1
    if touched == 0:
        raise ValueError('the GRU update cell exposes no bias tensor to zero')
    return touched


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
            # The GRU update bias is zeroed HERE, in this module, and NOT in
            # the archived `components/slots.py`, whose SHA-256 is recorded in
            # the archived receipts.
            #
            # Why
            # ---
            # The archived collapse diagnosis (PROGRESS entry
            # `latest_common_private_collapse_and_gradient_diagnosis_20261104`)
            # located the slot-duplication root cause in the GRU update bias:
            # with a nonzero input bias every slot receives the same additive
            # push on every iteration, so slots holding similar content are
            # driven towards each other, and the control predicted the slot
            # cosine would fall from 0.9458 to 0.0511 when the bias was zeroed.
            #
            # A local measurement on THIS architecture (16 slots, width 256,
            # 3 iterations, random initialisation, 4 real-shaped bags) gives
            # the same direction and the expected size of the effect:
            #
            #     slots   random init mean pair cosine / effective rank
            #        2     0.8493 / 1.53     ->  0.6514 / 1.76  (zero bias)
            #        4     0.8691 / 2.22     ->  0.6663 / 2.98
            #        8     0.8797 / 3.23     ->  0.6748 / 5.05
            #       16     0.8549 / 5.47     ->  0.6704 / 8.71
            #
            # Zeroing the bias therefore INCREASES the initial slot
            # differentiation of the 16-slot encoder (effective rank 5.47 ->
            # 8.71 out of 16).  This is a statement about the INITIAL state of
            # the encoder, measured on synthetic inputs; it is NOT a claim
            # about the trained model's slot geometry, which must be measured
            # on the real arm.
            #
            # What this does NOT do
            # ---------------------
            # It does not add a slot-diversity LOSS.  A loss term would add a
            # new objective and change the protocol; this changes only the
            # INITIALISATION of an existing parameter.  It does not guarantee
            # that the trained slots stay differentiated, and it does not make
            # the slots carry different CONTENT: slot differentiation and slot
            # content assignment are separate properties, and the repository's
            # own measurements show that a low slot cosine is not by itself
            # evidence of useful content (see the `R5`/`R1` arms, which have
            # the LOWEST redundancy and the LOWEST C-index).
            #
            # It also does not change the parameter count, the shapes, the
            # state_dict keys or the checkpoint format: the bias tensors keep
            # their names and shapes and are simply initialised to zero.
            zero_slot_update_bias(self.encoders[modality])
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
