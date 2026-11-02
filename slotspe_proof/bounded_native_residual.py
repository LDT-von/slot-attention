"""Independent affine V2 residual candidate; no preset or patient entry point.

The norm bound is an exact-real sufficient condition, not an interval-arithmetic
certificate for floating point. Only float32/float64 without autocast are scoped.
Neither invertibility nor this wrapper identifies common/private semantics.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .slot_disentanglement import SharedPrivateSlots, _tensor


class BoundedNativeResidual(nn.Module):
    """Z=S+aF(S), a=q/((1+margin)*max(1,B)); differentiate through B.

    composed_frobenius: B=max(||P_u A_u||F,||P_s A_s+P_u A_u||F).
    factor_frobenius: B=||P_u||F||A_u||F+||P_s||F||A_s||F.
    Both bound the global slot-set operator norm in exact arithmetic. The
    composed version is the specified research default; the factor version is
    a cost/conservatism control, not a tuned prediction arm. Bias is unbounded.
    """
    def __init__(self, adapter: SharedPrivateSlots, *, contraction=0.5,
                 bound="composed_frobenius", margin=1e-4):
        super().__init__()
        if type(adapter) is not SharedPrivateSlots:
            raise TypeError("only the unchanged affine SharedPrivateSlots is scoped")
        if isinstance(contraction, bool) or not isinstance(contraction, (int, float)) or not 0 < contraction < 1:
            raise ValueError("contraction must be strictly between0 and1")
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin <= 0:
            raise ValueError("positive finite numerical margin required")
        if bound not in {"composed_frobenius", "factor_frobenius"}:
            raise ValueError("unknown analytic norm bound")
        self.adapter = adapter
        self.contraction, self.bound, self.margin = float(contraction), bound, float(margin)

    def get_extra_state(self):
        return {"schema": "bounded_native_residual_v1", "contraction": self.contraction,
                "bound": self.bound, "margin": self.margin, "differentiate_coefficient": True}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("residual configuration mismatch; recreate matching constructor")

    def _check(self, value):
        _tensor("slots", value, 3)
        parameter = self.adapter.shared_encoder.weight
        if value.dtype not in (torch.float32, torch.float64) or value.dtype != parameter.dtype:
            raise ValueError("matched float32 or float64 required")
        if value.device != parameter.device or value.shape[-1] != self.adapter.dimension:
            raise ValueError("slots must match adapter dimension and device")
        if torch.is_autocast_enabled(value.device.type):
            raise ValueError("autocast is outside this candidate's acceptance scope")
        for layer in (self.adapter.shared_encoder, self.adapter.private_encoder,
                      self.adapter.shared_projection, self.adapter.private_projection):
            if type(layer) is not nn.Linear:
                raise TypeError("affine norm derivation requires unchanged Linear layers")

    def _blocks(self):
        a = self.adapter
        mean = a.shared_projection.weight.double() @ a.shared_encoder.weight.double()
        local = a.private_projection.weight.double() @ a.private_encoder.weight.double()
        bias = (a.shared_projection.weight.double() @ a.shared_encoder.bias.double()
                + a.shared_projection.bias.double()
                + a.private_projection.weight.double() @ a.private_encoder.bias.double()
                + a.private_projection.bias.double())
        return mean, local, bias

    def coefficient(self):
        """Input-independent coefficient, with parameter-dependent gradient."""
        a = self.adapter
        if self.bound == "composed_frobenius":
            mean, local, _ = self._blocks()
            upper = torch.maximum(torch.linalg.vector_norm(local), torch.linalg.vector_norm(mean + local))
        else:
            upper = (torch.linalg.vector_norm(a.private_projection.weight.double())
                     * torch.linalg.vector_norm(a.private_encoder.weight.double())
                     + torch.linalg.vector_norm(a.shared_projection.weight.double())
                     * torch.linalg.vector_norm(a.shared_encoder.weight.double()))
        if not bool(torch.isfinite(upper)):
            raise ValueError("nonfinite analytic norm upper bound")
        coefficient = self.contraction / ((1 + self.margin) * upper.clamp_min(1.))
        return coefficient, upper

    def forward(self, slots):
        self._check(slots)
        coefficient, upper = self.coefficient()
        latent, record = self.adapter(slots)
        z = slots + coefficient.to(slots.dtype) * latent
        _tensor("bounded residual", z, 3)
        record.update(native=slots, base_latent=latent, latent=z,
                      residual_coefficient=coefficient, norm_upper_bound=upper,
                      contraction_upper_bound=coefficient * upper)
        return z, record

    def inverse(self, z):
        """Acceptance diagnostic; requires exactly the same model parameters.

        Uses double-precision linear solves and returns the input dtype. This
        is not a prediction head or a recovery claim after training changes.
        """
        self._check(z)
        coefficient, _ = self.coefficient()
        mean, local, bias = self._blocks()
        eye = torch.eye(z.shape[-1], device=z.device, dtype=torch.float64)
        values = z.double()
        zmean = values.mean(1)
        original_mean = torch.linalg.solve(eye + coefficient * (mean + local),
                                           (zmean - coefficient * bias).T).T
        contrast = values - zmean[:, None]
        recovered = torch.linalg.solve(eye + coefficient * local,
                                       contrast.reshape(-1, z.shape[-1]).T).T.reshape_as(values)
        result = (original_mean[:, None] + recovered).to(z.dtype)
        _tensor("recovered slots", result, 3)
        return result
