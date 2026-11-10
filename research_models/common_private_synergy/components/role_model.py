"""Research role-output model; reuses recurrence, never changes original SlotSPE.

Explicit per-modality slot allocations are required. Predictions see only decoded
role outputs. A bounded readout limits amplification; this is not independence.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import ArchitectureConfig
from .slots import RoleSlotEncoder, AllSlotPool
from .role_reference import FrozenRoleReference


@dataclass(frozen=True)
class RoleAllocation:
    common: int
    residual: int

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in (self.common, self.residual)) or self.common + self.residual < 1:
            raise ValueError('explicit nonnegative role counts with positive total required')


class BoundedLinear(nn.Module):
    """Frobenius norm bounds spectral norm; smooth normalization, including at 0."""
    def __init__(self, inputs, outputs, bound):
        super().__init__()
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError('positive finite bound required')
        self.bound = float(bound)
        self.weight = nn.Parameter(torch.randn(outputs, inputs) / math.sqrt(inputs))
        self.bias = nn.Parameter(torch.zeros(outputs))

    def forward(self, x):
        w = self.weight * (self.bound / torch.sqrt(self.weight.square().sum() + self.bound ** 2))
        return F.linear(x, w, self.bias)


class BoundedRoleReadout(nn.Module):
    def __init__(self, p_dim, g_dim, *, hidden, bound, interaction, anchors):
        super().__init__()
        self.bound, self.interaction = float(bound), bool(interaction)
        self.p_head = nn.Sequential(BoundedLinear(p_dim, hidden, bound), nn.Tanh(), BoundedLinear(hidden, 1, bound))
        self.g_head = nn.Sequential(BoundedLinear(g_dim, hidden, bound), nn.Tanh(), BoundedLinear(hidden, 1, bound))
        self.left, self.right = BoundedLinear(p_dim, hidden, bound), BoundedLinear(g_dim, hidden, bound)
        self.cross = BoundedLinear(hidden, 1, bound)
        self.register_buffer('p_anchors', anchors['p'].detach().clone())
        self.register_buffer('g_anchors', anchors['g'].detach().clone())

    @property
    def lipschitz_upper_bound(self):
        # Each centered factor is bounded by 2. Reference means are fixed inputs;
        # gradients to weights through the exact means remain connected.
        return math.sqrt(2) * self.bound ** 2 * (3 if self.interaction else 1)

    def forward(self, roles):
        p = torch.cat((roles['p_common'], roles['p_residual']), -1)
        g = torch.cat((roles['g_common'], roles['g_residual']), -1)
        additive = self.p_head(p) + self.g_head(g)
        joint = torch.zeros_like(additive)
        if self.interaction:
            a = self.left(p).tanh() - self.left(self.p_anchors).tanh().mean(0)
            b = self.right(g).tanh() - self.right(self.g_anchors).tanh().mean(0)
            # No cross bias: it would add an uncentered constant contribution.
            w = self.cross.weight * (self.bound / torch.sqrt(self.cross.weight.square().sum() + self.bound ** 2))
            joint = F.linear(a * b, w)
        return {'prediction': additive + joint, 'additive': additive, 'interaction': joint}


class RoleTargetModel(nn.Module):
    def __init__(self, reference: FrozenRoleReference, *, p_input_dim, g_input_dim,
                 p_allocation: RoleAllocation, g_allocation: RoleAllocation,
                 width, iterations, readout_hidden, readout_bound, interaction, train_role_anchors):
        super().__init__()
        self.reference = reference
        self.allocations = {'p': p_allocation, 'g': g_allocation}
        self.encoders, self.pools, self.decoders = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        for modality, input_dim in [('p', p_input_dim), ('g', g_input_dim)]:
            allocation = self.allocations[modality]
            cfg = ArchitectureConfig(input_dim, input_dim, width=width,
                                     slots=allocation.common + allocation.residual,
                                     common_slots=allocation.common, iterations=iterations)
            self.encoders[modality] = RoleSlotEncoder(input_dim, cfg)
            for role in ('common', 'residual'):
                key = modality + '_' + role
                self.pools[key] = AllSlotPool(width)
                self.decoders[key] = nn.Linear(width, reference.dimension(modality))
        anchors = {m: torch.cat((train_role_anchors[m + '_common'], train_role_anchors[m + '_residual']), -1)
                   for m in ('p', 'g')}
        self.readout = BoundedRoleReadout(2 * reference.dimension('p'), 2 * reference.dimension('g'),
                                          hidden=readout_hidden, bound=readout_bound,
                                          interaction=interaction, anchors=anchors)

    def encode(self, p, g, p_mask, g_mask):
        roles, parts = {}, {}
        for modality, x, mask in [('p', p, p_mask), ('g', g, g_mask)]:
            part = self.encoders[modality](x, mask)
            parts[modality] = part
            for role, slots in [('common', part['common_slots']), ('residual', part['private_slots'])]:
                key = modality + '_' + role
                if slots.shape[1] == 0:
                    # Fixed zero; no decoder bias masquerades as missing content.
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

    def freeze_roles(self):
        """Explicit two-stage contract: task training cannot move role encoders."""
        for module in (self.encoders, self.pools, self.decoders):
            module.requires_grad_(False)
            for parameter in module.parameters():
                parameter.grad = None
