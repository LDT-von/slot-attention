"""Controlled post-slot relation residuals for synthetic and paired audits.

The relation and capacity modes share an identity initialization, but only the
former exchanges private features between slots. Neither mode identifies a
biological relation or guarantees improved patient predictions.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["SlotRelationResidual"]


class SlotRelationResidual(nn.Module):
    """Add a permutation-equivariant relation or per-slot capacity residual.

    At the native dimensions, relation mode has 115072 parameters and capacity
    mode has 115168. Their parameter counts are close, not identical. Parameters
    are initialized on CPU inside an isolated RNG context; constructing this
    module does not reseed CUDA or consume the caller's CPU random stream.
    """

    def __init__(
        self,
        private_dim: int = 192,
        shared_dim: int = 64,
        output_dim: int = 256,
        mode: str = "relation",
        seed: int = 901,
    ) -> None:
        super().__init__()
        for name, value in (
            ("private_dim", private_dim),
            ("shared_dim", shared_dim),
            ("output_dim", output_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(mode, str) or mode not in {"relation", "capacity"}:
            raise ValueError("mode must be 'relation' or 'capacity'")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        self.private_dim = private_dim
        self.shared_dim = shared_dim
        self.output_dim = output_dim
        self.mode = mode
        self.seed = seed
        input_dim = 3 * private_dim + shared_dim if mode == "relation" else private_dim + shared_dim
        hidden_dim = 128 if mode == "relation" else 224
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.g = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=True, device="cpu"),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim, bias=True, device="cpu"),
            )
            # Zeroing only the last layer leaves an immediate learning path.
            with torch.no_grad():
                self.g[-1].weight.zero_()
                self.g[-1].bias.zero_()

    def _validate(self, latent: Tensor, shared: Tensor, private: Tensor) -> None:
        for name, value in (("latent", latent), ("shared", shared), ("private", private)):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} must have a real floating dtype")
        if latent.ndim != 3 or latent.shape[-1] != self.output_dim:
            raise ValueError("latent must have shape [B, K, output_dim]")
        batch, slots, _ = latent.shape
        if batch < 1 or slots < 1:
            raise ValueError("latent requires a nonempty batch and slot axis")
        if self.mode == "relation" and slots < 2:
            raise ValueError("relation mode requires at least two slots")
        if shared.shape != (batch, self.shared_dim):
            raise ValueError("shared must have shape [B, shared_dim]")
        if private.shape != (batch, slots, self.private_dim):
            raise ValueError("private must have shape [B, K, private_dim]")
        reference = self.g[0].weight
        for name, value in (("latent", latent), ("shared", shared), ("private", private)):
            if value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(f"{name} must match the module parameter dtype and device")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values")

    def forward(
        self,
        latent: Tensor,
        shared: Tensor,
        private: Tensor,
        *,
        enabled: bool = True,
    ) -> Tensor:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        if not enabled:
            return latent
        self._validate(latent, shared, private)
        batch, slots, _ = private.shape
        if self.mode == "capacity":
            common = shared[:, None, :].expand(batch, slots, self.shared_dim)
            residual = self.g(torch.cat((private, common), dim=-1))
        else:
            source = private[:, :, None, :].expand(batch, slots, slots, self.private_dim)
            neighbor = private[:, None, :, :].expand(batch, slots, slots, self.private_dim)
            common = shared[:, None, None, :].expand(batch, slots, slots, self.shared_dim)
            messages = self.g(torch.cat((source, neighbor, source * neighbor, common), dim=-1))
            diagonal = torch.eye(slots, device=private.device, dtype=torch.bool)[None, :, :, None]
            residual = messages.masked_fill(diagonal, 0).sum(dim=2) / (slots - 1)
        result = latent + residual
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("relation residual produced a nonfinite output")
        return result
