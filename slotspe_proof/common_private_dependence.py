"""Finite, differentiable public/private dependence proxy.

This module is deliberately named a *proxy*: the batch statistic below is not
an MI estimator or a conditional-independence test.  It uses a fixed small
polynomial lift before measuring centred cross-covariance, so a nonlinear copy
such as ``U = G**2`` is visible to the mechanism while independent factors
remain close to zero in expectation.
"""
from __future__ import annotations

import torch


def _matrix(x: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"{name} must be a [batch, features] tensor with batch >= 2")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError(f"{name} must be finite floating point")
    return x


def _standardize(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(0, keepdim=True)
    scale = x.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (x - mean) / scale


def polynomial_lift(x: torch.Tensor) -> torch.Tensor:
    """A fixed low-order lift that includes linear and quadratic effects."""
    x = _matrix(x, "x")
    z = _standardize(x)
    return torch.cat([z, z.square()], dim=1)


def cross_covariance_dependence(common: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
    """Return mean squared lifted cross-covariance for aligned patients.

    The value is non-negative and differentiable with respect to both inputs.
    It is a finite-batch regularizer only; no MI interpretation is attached.
    """
    common = _matrix(common, "common")
    private = _matrix(private, "private")
    if common.shape[0] != private.shape[0]:
        raise ValueError("common and private must have the same batch size")
    a = polynomial_lift(common)
    b = polynomial_lift(private)
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    cov = a.transpose(0, 1) @ b / (a.shape[0] - 1)
    return cov.square().mean()


def public_private_separation(common: torch.Tensor, private_features: list[torch.Tensor]) -> torch.Tensor:
    """Average the proxy over private modality branches."""
    if not private_features:
        raise ValueError("at least one private branch is required")
    penalties = [cross_covariance_dependence(common, branch) for branch in private_features]
    return torch.stack(penalties).mean()

