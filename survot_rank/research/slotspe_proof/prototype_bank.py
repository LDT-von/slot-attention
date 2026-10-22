"""Explicitly fitted prototype memory for the local MetaSlot-style adaptation.

This is an independent, small-scale implementation of the paper's nearest-L2
and centroid-EMA mechanism, not vendored author code.  Farthest-point fitting,
the caller's codebook size/EMA rate/timeout, and the Lloyd iteration budget are
local adaptation choices.  No data is loaded, fitted, or queued automatically;
the caller is responsible for supplying training-only vectors.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


class PrototypeBank:
    """A non-gradient codebook with one explicit update per logical batch.

    ``observations`` counts fitting vectors and committed training vectors.
    ``updates`` counts nonempty commits; queueing alone changes neither count.
    A second fit resets the bank, but cannot discard an uncommitted queue.
    """

    _SCHEMA = "slotspe_local_prototype_bank_v1"
    _CONFIG_FIELDS = ("prototype_count", "ema_rate", "dead_timeout", "kmeans_steps")

    def __init__(self, config: Any):
        self.config = config
        self._metadata = {
            name: config[name] if isinstance(config, Mapping) else getattr(config, name)
            for name in self._CONFIG_FIELDS
        }
        for name in ("prototype_count", "dead_timeout", "kmeans_steps"):
            value = self._metadata[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self._metadata["prototype_count"] < 1:
            raise ValueError("prototype_count must be positive")
        if self._metadata["dead_timeout"] < 0 or self._metadata["kmeans_steps"] < 0:
            raise ValueError("dead_timeout and kmeans_steps must be nonnegative")
        rate = self._metadata["ema_rate"]
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise ValueError("ema_rate must be a finite number in (0, 1]")
        if not math.isfinite(rate) or not 0 < rate <= 1:
            raise ValueError("ema_rate must be a finite number in (0, 1]")
        self._metadata["ema_rate"] = float(rate)
        self.prototypes: Tensor | None = None
        self.ages: Tensor | None = None
        self.pending: list[Tensor] = []
        self.updates = 0
        self.observations = 0
        self.fit_observations = 0

    @staticmethod
    def _vectors(vectors: Tensor, *, allow_empty: bool = False) -> Tensor:
        if not isinstance(vectors, Tensor) or vectors.ndim != 2:
            raise ValueError("vectors must have shape [N, D]")
        if not vectors.is_floating_point() or vectors.shape[1] < 1:
            raise ValueError("vectors must have a positive feature dimension and floating dtype")
        if not allow_empty and vectors.shape[0] < 1:
            raise ValueError("vectors must contain at least one row")
        if not torch.isfinite(vectors).all().item():
            raise ValueError("vectors must be finite")
        dtype = torch.float32 if vectors.dtype in (torch.float16, torch.bfloat16) else vectors.dtype
        return vectors.detach().to(dtype=dtype).clone()

    @staticmethod
    def _distance(vectors: Tensor, centers: Tensor) -> Tensor:
        return (
            vectors.square().sum(-1, keepdim=True)
            - 2 * vectors @ centers.transpose(-1, -2)
            + centers.square().sum(-1).unsqueeze(0)
        ).clamp_min(0)

    def _require_fit(self) -> tuple[Tensor, Tensor]:
        if self.prototypes is None or self.ages is None:
            raise RuntimeError("fit(training_vectors) must be called explicitly before use")
        return self.prototypes, self.ages

    @torch.no_grad()
    def fit(self, vectors: Tensor) -> "PrototypeBank":
        if self.pending:
            raise RuntimeError("commit pending training vectors before refitting")
        data = self._vectors(vectors)
        count = self._metadata["prototype_count"]
        if data.shape[0] < count:
            raise ValueError(f"fit requires N >= prototype_count, got {data.shape[0]} < {count}")
        selected = torch.zeros(data.shape[0], dtype=torch.bool, device=data.device)
        indices = [0]
        selected[0] = True
        minimum = self._distance(data, data[:1]).squeeze(1)
        for _ in range(1, count):
            candidate = minimum.masked_fill(selected, -torch.inf).argmax().item()
            indices.append(candidate)
            selected[candidate] = True
            distance = self._distance(data, data[candidate : candidate + 1]).squeeze(1)
            minimum = torch.minimum(minimum, distance)
        centers = data[indices].clone()
        for _ in range(self._metadata["kmeans_steps"]):
            assignments = self._distance(data, centers).argmin(-1)
            revised = centers.clone()
            for index in range(count):
                members = data[assignments == index]
                if members.shape[0]:
                    revised[index] = members.mean(0)
            centers = revised
        self.prototypes = centers.detach().clone()
        self.ages = torch.zeros(count, dtype=torch.long, device=data.device)
        self.updates = 0
        self.fit_observations = data.shape[0]
        self.observations = self.fit_observations
        return self

    @torch.no_grad()
    def lookup(self, slots: Tensor) -> tuple[Tensor, Tensor]:
        prototypes, _ = self._require_fit()
        if not isinstance(slots, Tensor) or slots.ndim < 1 or not slots.is_floating_point():
            raise ValueError("slots must be a floating tensor with a final feature dimension")
        if slots.shape[-1] != prototypes.shape[-1]:
            raise ValueError("slot and prototype feature dimensions must match")
        if not torch.isfinite(slots).all().item():
            raise ValueError("slots must be finite")
        flat = slots.detach().reshape(-1, slots.shape[-1]).to(prototypes)
        indices = self._distance(flat, prototypes).argmin(-1)
        quantized = prototypes[indices].reshape(slots.shape).to(slots).detach().clone()
        return indices.reshape(slots.shape[:-1]).to(slots.device), quantized

    @torch.no_grad()
    def queue(self, final_active_vectors: Tensor) -> None:
        """Queue detached final *active* training slots, without updating codes."""
        prototypes, _ = self._require_fit()
        data = self._vectors(final_active_vectors, allow_empty=True)
        if data.shape[1] != prototypes.shape[1]:
            raise ValueError("queued vectors and prototypes must have the same feature dimension")
        if data.shape[0]:
            self.pending.append(data.to(prototypes).detach().clone())

    @torch.no_grad()
    def commit(self) -> None:
        """Commit all queued calls together as a single logical training batch."""
        if not self.pending:
            return
        prototypes, ages = self._require_fit()
        data = self._vectors(torch.cat(self.pending, dim=0))
        if data.shape[1] != prototypes.shape[1]:
            raise ValueError("pending vectors and prototypes must have the same feature dimension")
        data = data.to(prototypes)
        assignments = self._distance(data, prototypes).argmin(-1)
        counts = torch.bincount(assignments, minlength=prototypes.shape[0])
        totals = torch.zeros_like(prototypes)
        totals.index_add_(0, assignments, data)
        hit = counts > 0
        revised = prototypes.clone()
        means = totals[hit] / counts[hit, None]
        rate = self._metadata["ema_rate"]
        revised[hit] = revised[hit] * (1 - rate) + means * rate
        revised_ages = ages.clone() + 1
        revised_ages[hit] = 0

        dead = revised_ages > self._metadata["dead_timeout"]
        active = ~dead
        normalized_data = F.normalize(data, dim=-1)
        for index in dead.nonzero(as_tuple=False).flatten().tolist():
            if active.any().item():
                normalized_active = F.normalize(revised[active], dim=-1)
                maximum_similarity = (normalized_data @ normalized_active.T).max(-1).values
                candidate = maximum_similarity.argmin().item()
            else:
                candidate = 0
            revised[index] = data[candidate]
            revised_ages[index] = 0
            active[index] = True

        self.prototypes = revised.detach().clone()
        self.ages = revised_ages
        self.updates += 1
        self.observations += data.shape[0]
        self.pending.clear()

    def state_dict(self) -> dict[str, Any]:
        if self.pending:
            raise RuntimeError("cannot serialize a bank with uncommitted pending vectors")
        return {
            "schema": self._SCHEMA,
            "config": dict(self._metadata),
            "prototypes": None if self.prototypes is None else self.prototypes.detach().clone(),
            "ages": None if self.ages is None else self.ages.detach().clone(),
            "updates": self.updates,
            "observations": self.observations,
            "fit_observations": self.fit_observations,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> "PrototypeBank":
        if self.pending:
            raise RuntimeError("commit pending training vectors before loading bank state")
        if state.get("schema") != self._SCHEMA or state.get("config") != self._metadata:
            raise ValueError("incompatible prototype bank schema or configuration")
        prototypes, ages = state.get("prototypes"), state.get("ages")
        if (prototypes is None) != (ages is None):
            raise ValueError("state must contain both prototypes and ages, or neither")
        if prototypes is not None:
            prototypes = self._vectors(prototypes)
            if prototypes.shape[0] != self._metadata["prototype_count"]:
                raise ValueError("state prototype count does not match configuration")
            if not isinstance(ages, Tensor) or ages.shape != (prototypes.shape[0],):
                raise ValueError("state ages must have shape [prototype_count]")
            if ages.dtype != torch.long or (ages < 0).any().item():
                raise ValueError("state ages must be nonnegative int64")
            ages = ages.detach().to(device=prototypes.device).clone()
        counters = {}
        for name in ("updates", "observations", "fit_observations"):
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"state {name} must be a nonnegative integer")
            counters[name] = value
        if counters["fit_observations"] > counters["observations"]:
            raise ValueError("state fit_observations cannot exceed observations")
        self.prototypes = prototypes
        self.ages = ages
        self.updates = counters["updates"]
        self.observations = counters["observations"]
        self.fit_observations = counters["fit_observations"]
        self.pending = []
        return self
