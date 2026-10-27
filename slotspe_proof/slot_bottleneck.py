"""Experimental slot VIB, separate from the protected eleven-arm queue.

The analytic KL bounds input information for the stochastic posterior. It
does not establish independent slots, patch deletion or patient efficacy.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .prediction_slots import prediction_slot_outputs


def diagonal_gaussian_kl(mean, log_variance):
    """KL(q || N(0,I)), summed over K,D and returned per batch example.

    ``log_variance`` must be the SAME effective value used for sampling.
    expm1 avoids cancellation around zero without altering the formula.
    """
    if mean.ndim != 3 or mean.shape != log_variance.shape or min(mean.shape) < 1:
        raise ValueError("expected matching nonempty [B,K,D] posterior parameters")
    if not mean.is_floating_point() or mean.dtype != log_variance.dtype or mean.device != log_variance.device:
        raise ValueError("posterior parameters must share floating dtype and device")
    if not bool(torch.isfinite(mean).all() and torch.isfinite(log_variance).all()):
        raise ValueError("posterior parameters must be finite")
    result = .5 * (mean.square() + torch.expm1(log_variance) - log_variance).sum(dim=(-2, -1))
    if not bool(torch.isfinite(result).all()):
        raise ValueError("posterior KL overflow")
    return result


class GaussianSlotBottleneck(nn.Module):
    """Shared slotwise encoder; independent K,D Gaussian posterior noise.

    The initial mean is exactly the input. Initialization draws no random
    values, so constructing this module cannot perturb the native RNG.
    """

    def __init__(self, dimension, initial_log_variance=-6., log_variance_bounds=(-12., 8.)):
        super().__init__()
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        low, high = log_variance_bounds
        if not all(math.isfinite(value) for value in (initial_log_variance, low, high)) or not low <= initial_log_variance <= high:
            raise ValueError("invalid log variance or bounds")
        self.dimension = dimension
        self.bounds = (float(low), float(high))
        # torch.empty and nn.Parameter do not sample a global random stream.
        self.mean_weight = nn.Parameter(torch.eye(dimension))
        self.mean_bias = nn.Parameter(torch.zeros(dimension))
        self.logvar_weight = nn.Parameter(torch.zeros(dimension, dimension))
        self.logvar_bias = nn.Parameter(torch.full((dimension,), float(initial_log_variance)))

    def forward(self, slots, *, generator=None, noise=None, stochastic=True):
        if slots.ndim != 3 or min(slots.shape) < 1 or slots.shape[-1] != self.dimension:
            raise ValueError("expected nonempty [B,K,dimension] slots")
        if not slots.is_floating_point() or not bool(torch.isfinite(slots).all()):
            raise ValueError("slots must be finite floating tensors")
        mean = torch.nn.functional.linear(slots, self.mean_weight, self.mean_bias)
        logvar = torch.nn.functional.linear(slots, self.logvar_weight, self.logvar_bias).clamp(*self.bounds)
        kl = diagonal_gaussian_kl(mean, logvar)
        if stochastic:
            if noise is None:
                if generator is None:
                    raise ValueError("stochastic mode needs an isolated generator or explicit noise")
                noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype, device=mean.device)
            elif noise.shape != mean.shape or noise.dtype != mean.dtype or noise.device != mean.device or not bool(torch.isfinite(noise).all()):
                raise ValueError("noise must match posterior shape, dtype and device and be finite")
            latent = mean + torch.exp(.5 * logvar) * noise
        else:
            if noise is not None or generator is not None:
                raise ValueError("mean mode must not consume noise")
            latent = mean
        return latent, kl


class SlotVIBHost(nn.Module):
    """Scoped post-slot adapter for the unmodified native SlotSPE host.

    Only the first WSI/OMIC call in a complete-modality forward is replaced.
    The later OMIC-from-WSI reconstruction call stays native. The existing
    decoder, self/cross attention, auxiliary losses and padding stay native.
    The caller adds beta * kl_loss() to native main + native auxiliary loss.
    Both training and evaluation sample in stochastic mode; eval() does not
    silently replace the posterior by its mean. Each forward needs vib_seed.
    """

    MODES = {"off", "mean", "stochastic"}

    def __init__(self, backbone, dimension, mode="stochastic"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError("unknown bottleneck mode")
        self.backbone = backbone
        self.mode = mode
        self.bottlenecks = nn.ModuleDict({name: GaussianSlotBottleneck(dimension)
                                         for name in ("wsi", "omic")} if mode != "off" else {})
        self._kl = None
        self._latent_dimensions = None
        self._forward_active = False
        self.last_slots = {}

    def forward(self, *, vib_seed=None, **kwargs):
        if self._forward_active:
            raise RuntimeError("reentrant adapter forward is unsupported")
        self._kl, self._latent_dimensions = None, None
        self.last_slots = {}
        if self.mode == "off":
            return self.backbone(**kwargs)
        if kwargs.get("omic_missing") is not False:
            raise ValueError("this experiment requires explicit complete modalities")
        if self.mode == "stochastic" and (isinstance(vib_seed, bool) or not isinstance(vib_seed, int) or not 0 <= vib_seed < 2**63):
            raise ValueError("stochastic mode requires an integer vib_seed in [0,2**63)")
        terms, dimensions = {}, {}

        def replace(name, output):
            generator = None
            if self.mode == "stochastic":
                generator = torch.Generator(device=output.device)
                generator.manual_seed((vib_seed + (0 if name == "wsi" else 104729)) % (2**63))
            latent, terms[name] = self.bottlenecks[name](output, generator=generator,
                                                        stochastic=self.mode == "stochastic")
            dimensions[name] = output.shape[1] * output.shape[2]
            # The mean is a detached diagnostic, never a new prediction path.
            mean = torch.nn.functional.linear(output.detach(), self.bottlenecks[name].mean_weight.detach(),
                                               self.bottlenecks[name].mean_bias.detach())
            return latent, {"native": output, "latent": latent, "mean": mean}

        self._forward_active = True
        try:
            with prediction_slot_outputs(self.backbone, replace) as records:
                result = self.backbone(**kwargs)
            if set(terms) != {"wsi", "omic"} or terms["wsi"].shape != terms["omic"].shape:
                raise RuntimeError("prediction slot calls did not match the two-modality contract")
            self._kl = terms["wsi"] + terms["omic"]
            self._latent_dimensions = sum(dimensions.values())
            self.last_slots = records
            return result
        finally:
            self._forward_active = False

    def kl_loss(self, *, reduction="mean_per_dimension"):
        """Default: mean over patients of joint KL / joint latent dimensions.

        At native K=8,D=256 in two modalities, total dimensions = 4096.
        This normalized quantity must be multiplied by 4096 for the MI bound.
        Mean mode reports its diagnostic Gaussian KL, but is not a stochastic
        bottleneck; its deterministic prediction has no such MI guarantee.
        """
        if reduction not in {"none", "mean", "mean_per_dimension"}:
            raise ValueError("unknown KL reduction")
        if self.mode == "off":
            return next(self.backbone.parameters()).new_zeros(())
        if self._kl is None:
            raise RuntimeError("KL unavailable before a successful forward")
        if reduction == "none":
            return self._kl
        return self._kl.mean() / (self._latent_dimensions if reduction == "mean_per_dimension" else 1)
