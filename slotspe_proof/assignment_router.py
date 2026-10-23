"""Research-only alternatives to the normalization INSIDE Slot Attention.

The MESH-style arm follows Zhang et al. (ICML 2023), equations 9--12:
https://proceedings.mlr.press/v202/zhang23ba.html
This is an independently written, uniform-marginal adaptation, not a complete
reproduction of the authors' model (which learns its marginals). No downloaded
source is vendored here. No patient runner or default model imports this file.

All arms consume the same scaled QK logits [B, H, S, N]. Returned weights sum
to one over N, as needed for a slot's weighted-mean value update. MESH changes
assignment during aggregation; it is neither a final Top-K gate nor an extra
cross-modal transport stage. It does not guarantee biological specialization.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AssignmentConfig:
    mode: str = "original"
    eps: float = 1e-8
    temperature: float = 1.0
    sinkhorn_steps: int = 20
    mesh_steps: int = 4
    mesh_lr: float = 1.0
    noise_std: float = 0.001

    def __post_init__(self):
        if self.mode not in {"original", "sinkhorn", "mesh_uniform"}:
            raise ValueError("unknown assignment mode")
        for name in ("eps", "temperature", "mesh_lr", "noise_std"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("sinkhorn_steps", "mesh_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def _uniform_log_plan(cost, token_mask, config):
    """Log-domain rectangular Sinkhorn; total mass 1, slot mass 1/S.

    Finite iteration counts approximate the row marginal. The last column
    normalization enforces uniform mass over VALID tokens. Invalid columns
    always remain -inf. No uniform square-matrix assumption is made.
    """
    valid = token_mask[:, None, None, :]
    count = token_mask.sum(-1).to(cost.dtype)[:, None, None, None]
    log_p = (-cost / config.temperature).masked_fill(~valid, -torch.inf)
    for _ in range(config.sinkhorn_steps):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - math.log(cost.shape[-2])
        column = torch.logsumexp(log_p.masked_fill(~valid, 0), dim=-2, keepdim=True)
        # Avoid both -inf - -inf and backward through an all--inf reduction.
        log_p = log_p - column.masked_fill(~valid, 0) - count.log()
    return log_p


class SlotAssignmentRouter(nn.Module):
    """Parameter-free assignment module with explicit tie-breaking noise.

    mesh_uniform requires caller-supplied noise, so a forward consumes no RNG.
    Reuse that exact tensor for paired probes and jointly permute it with slots
    or tokens for equivariance checks. Different noise can give different tie
    solutions. torch.no_grad() evaluation works; inference_mode is rejected
    because entropy descent requires local automatic differentiation.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or AssignmentConfig()

    def forward(self, logits, *, token_mask=None, noise=None):
        c = self.config
        if logits.ndim != 4 or min(logits.shape) < 1:
            raise ValueError("logits must be nonempty [B, H, S, N]")
        if not logits.is_floating_point() or not torch.isfinite(logits).all():
            raise ValueError("logits must be finite floating-point values")
        if token_mask is None:
            token_mask = torch.ones(
                logits.shape[0], logits.shape[-1], dtype=torch.bool,
                device=logits.device,
            )
        if (token_mask.dtype != torch.bool or
                token_mask.shape != (logits.shape[0], logits.shape[-1]) or
                token_mask.device != logits.device or
                not token_mask.any(-1).all()):
            raise ValueError("token_mask must be boolean [B, N] with valid tokens")
        valid = token_mask[:, None, None, :]
        if c.mode == "original":
            # Exact native expression when all tokens are valid. A partial mask
            # is an optional extension and is not advertised as native SlotSPE.
            attention = (logits.softmax(dim=-2) + c.eps).masked_fill(~valid, 0)
            return F.normalize(attention, p=1, dim=-1)

        # fp16/bf16 entropy descent is deliberately performed in float32.
        work = logits if logits.dtype == torch.float64 else logits.float()
        if c.mode == "sinkhorn":
            log_plan = _uniform_log_plan(-work, token_mask, c)
        else:
            if torch.is_inference_mode_enabled():
                raise RuntimeError("mesh_uniform needs no_grad(), not inference_mode()")
            if (noise is None or noise.shape != logits.shape or
                    noise.device != logits.device or
                    not noise.is_floating_point() or not torch.isfinite(noise).all()):
                raise ValueError("mesh_uniform requires finite noise matching logits")
            outer_grad = torch.is_grad_enabled()
            with torch.enable_grad():
                cost = -work + c.noise_std * noise.to(work.dtype).detach()
                if not cost.requires_grad:
                    cost = cost.detach().requires_grad_(True)
                for _ in range(c.mesh_steps):
                    log_plan = _uniform_log_plan(cost, token_mask, c)
                    safe_log = log_plan.masked_fill(~valid, 0)
                    entropy = -(log_plan.exp() * safe_log).sum()
                    # As in the supplied authors' implementation, do not form
                    # Hessians of the inner cost-gradient updates. The final
                    # plan still differentiates through cost to Q/K logits.
                    gradient, = torch.autograd.grad(entropy, cost)
                    norm = gradient.flatten(2).norm(dim=-1)[:, :, None, None]
                    cost = cost - c.mesh_lr * gradient / norm.clamp_min(1e-20)
                log_plan = _uniform_log_plan(cost, token_mask, c)
                if not outer_grad:
                    log_plan = log_plan.detach()

        attention = (log_plan - torch.logsumexp(
            log_plan, dim=-1, keepdim=True,
        )).exp()
        return attention.to(logits.dtype)


def replay_with_assignment(native, inputs, initial, router, *, noises=None,
                           token_mask=None, iterations=None):
    """Reuse native projections/GRU/MLP, replacing only assignment.

    inputs/initial must be supplied explicitly; this function never samples
    slots or loads checkpoints. mesh_uniform needs one noise tensor per update.
    It does not install hooks or mutate native methods/parameters. State-dict
    compatibility follows because the router has no parameters or buffers.
    """
    count = native.iters if iterations is None else iterations
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("iterations must be a positive integer")
    if (inputs.ndim != 3 or initial.ndim != 3 or
            inputs.shape[0] != initial.shape[0] or
            inputs.shape[-1] != native.dim or initial.shape[-1] != native.dim):
        raise ValueError("inputs and initial must match native batch and dimension")
    if router.config.mode == "original" and router.config.eps != native.eps:
        raise ValueError("original control must use native eps")
    if noises is not None and len(noises) != count:
        raise ValueError("one noise tensor is required per iteration")
    if router.config.mode == "mesh_uniform" and noises is None:
        raise ValueError("mesh_uniform requires explicit per-iteration noises")
    normalized = native.norm_input(inputs)
    keys = native.split_heads(native.to_k(normalized))
    values = native.split_heads(native.to_v(normalized))
    batch, slots_count, dimension = initial.shape
    slots = initial
    states, attentions = [initial], []
    for step in range(count):
        previous = slots
        queries = native.split_heads(native.to_q(native.norm_slots(slots)))
        logits = torch.einsum("... i d, ... j d -> ... i j", queries, keys)
        attention = router(
            logits * native.scale, token_mask=token_mask,
            noise=None if noises is None else noises[step],
        )
        updates = torch.einsum("... j d, ... i j -> ... i d", values, attention)
        updates = native.combine_heads(native.merge_heads(updates))
        slots = native.gru(
            updates.reshape(-1, dimension), previous.reshape(-1, dimension),
        ).reshape(batch, slots_count, dimension)
        slots = slots + native.mlp(native.norm_pre_ff(slots))
        states.append(slots)
        attentions.append(attention)
    return {"final": slots, "states": states, "attentions": attentions}
