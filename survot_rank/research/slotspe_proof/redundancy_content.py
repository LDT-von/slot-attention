"""Train-fold references and query-free targets for the redundancy queue.

This is an input-feature reconstruction contract, not biological supervision.
Each raw feature type has its own fixed random projection after standardization
by training-fold moments. Neither outcomes nor held-out inputs fit the state.
Raw features are not retained in the fitted object or its checkpoint. The static
dictionary uses a bounded, training-only reservoir and is a diagnostic control.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_tensor(value, name):
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _centres(points, clusters, steps):
    """Deterministic farthest initialization followed by fixed-count Lloyd steps."""
    if points.ndim != 2 or min(points.shape) < 1:
        raise ValueError("k-means requires nonempty [tokens,features]")
    points = points.detach().cpu().double()
    count = min(clusters, len(points))
    first = int((points - points.mean(0)).square().sum(-1).argmax())
    selected = [first]
    distance = (points - points[first]).square().sum(-1)
    for _ in range(1, count):
        candidate = distance.clone()
        candidate[selected] = -1
        index = int(candidate.argmax())
        selected.append(index)
        distance = torch.minimum(distance, (points - points[index]).square().sum(-1))
    centres = points[selected].clone()
    for _ in range(steps):
        labels = (points[:, None] - centres[None]).square().sum(-1).argmin(-1)
        centres = torch.stack([
            points[labels == i].mean(0) if bool((labels == i).any()) else centres[i]
            for i in range(count)
        ])
    return centres


class _Moments:
    def __init__(self, dimension):
        self.count = 0
        self.mean = torch.zeros(dimension, dtype=torch.float64)
        self.m2 = torch.zeros_like(self.mean)

    def update(self, values):
        values = values.detach().cpu().double()
        count = len(values)
        mean = values.mean(0)
        m2 = (values - mean).square().sum(0)
        total = self.count + count
        difference = mean - self.mean
        self.m2 = self.m2 + m2 + difference.square() * (self.count * count / total)
        self.mean = self.mean + difference * (count / total)
        self.count = total

    def std(self, floor):
        return (self.m2 / self.count).clamp_min(0).sqrt().clamp_min(floor)


class _Reservoir:
    """A bounded local fit buffer; never retained on ContentReference."""
    def __init__(self, capacity, generator):
        self.capacity = capacity
        self.generator = generator
        self.seen = 0
        self.values = []

    def update(self, values):
        for value in values.detach().cpu().double():
            self.seen += 1
            if len(self.values) < self.capacity:
                self.values.append(value.clone())
            else:
                index = int(torch.randint(self.seen, (), generator=self.generator))
                if index < self.capacity:
                    self.values[index] = value.clone()


class ContentReference:
    """Frozen train-fold normalization and per-input-type random projections.

    ``fit`` accepts a one-pass iterable of cases or batches. WSI moments pool all
    provided training patches; each heterogeneous pathway has independent raw
    moments and a separate projection. Callers must supply training indices only.
    ``targets`` accepts one patient at a time and never changes fitted state.
    Projection and k-means describe feature content, not identified tissue types.
    """
    SCHEMA = "slotspe_redundancy_content_reference_v1"
    STD_FLOOR = 1e-4

    def __init__(self, omic_sizes, encoding_dim, target_dim=32, clusters=8,
                 seed=731, kmeans_steps=5):
        self.omic_sizes = tuple(_positive_int(x, "omic size") for x in omic_sizes)
        if not self.omic_sizes:
            raise ValueError("at least one omic pathway is required")
        self.encoding_dim = _positive_int(encoding_dim, "encoding_dim")
        self.target_dim = _positive_int(target_dim, "target_dim")
        self.clusters = _positive_int(clusters, "clusters")
        self.kmeans_steps = _positive_int(kmeans_steps, "kmeans_steps")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0,2**63)")
        self.seed = seed
        self._fitted = False

    @property
    def fitted(self):
        return self._fitted

    def _inputs(self, batch):
        if not isinstance(batch, Mapping):
            raise ValueError("a case must be a mapping of feature tensors")
        wsi = batch["x_wsi"]
        _finite_tensor(wsi, "x_wsi")
        if wsi.ndim == 2:
            wsi = wsi.unsqueeze(0)
        if wsi.ndim != 3 or min(wsi.shape) < 1 or wsi.shape[-1] != self.encoding_dim:
            raise ValueError("x_wsi must have shape [B,N,encoding_dim] or [N,encoding_dim]")
        omics = []
        for index, size in enumerate(self.omic_sizes, 1):
            value = batch[f"x_omic{index}"]
            _finite_tensor(value, f"x_omic{index}")
            if value.ndim == 1:
                value = value.unsqueeze(0)
            if value.ndim != 2 or value.shape != (len(wsi), size):
                raise ValueError(f"x_omic{index} must have shape [B,{size}] or [{size}]")
            omics.append(value)
        return wsi, omics

    def _projection(self, dimension, index):
        generator = torch.Generator(device="cpu")
        generator.manual_seed((self.seed + 104729 * index) % (2**63))
        return torch.randn(dimension, self.target_dim, generator=generator,
                           dtype=torch.float64) / math.sqrt(dimension)

    def fit(self, cases):
        if self._fitted:
            raise RuntimeError("reference is frozen; construct a new reference to refit")
        moments = [_Moments(x) for x in (self.encoding_dim, *self.omic_sizes)]
        reservoirs = []
        for index in range(len(moments)):
            generator = torch.Generator(device="cpu")
            generator.manual_seed((self.seed + 130363 * (index + 1)) % (2**63))
            reservoirs.append(_Reservoir(512 if index == 0 else 64, generator))
        case_count = 0
        for batch in cases:
            wsi, omics = self._inputs(batch)
            values = [wsi.reshape(-1, self.encoding_dim), *omics]
            for moment, reservoir, value in zip(moments, reservoirs, values):
                moment.update(value)
                reservoir.update(value)
            case_count += len(wsi)
        if not case_count:
            raise ValueError("cannot fit a content reference on an empty training fold")
        means = [value.mean.clone() for value in moments]
        stds = [value.std(self.STD_FLOOR) for value in moments]
        projections = [self._projection(dimension, index) for index, dimension in
                       enumerate((self.encoding_dim, *self.omic_sizes))]
        dictionaries = [
            ((torch.stack(reservoir.values) - mean) / std) @ projection
            for reservoir, mean, std, projection in zip(reservoirs, means, stds, projections)
        ]
        static_targets = {
            "wsi": _centres(dictionaries[0], self.clusters, self.kmeans_steps),
            "omic": _centres(torch.cat(dictionaries[1:]), self.clusters, self.kmeans_steps),
        }
        state = self._metadata()
        state.update({
            "train_case_count": case_count,
            "feature_counts": [value.count for value in moments],
            "means": means, "stds": stds, "projections": projections,
            "static_targets": static_targets,
        })
        self.load_state_dict(state)
        return self

    def targets(self, batch):
        if not self._fitted:
            raise RuntimeError("fit the content reference on the training fold first")
        wsi, omics = self._inputs(batch)
        if len(wsi) != 1:
            raise ValueError("content targets require batch size one (ragged patient sets)")
        raw = [wsi[0], *[value[0].unsqueeze(0) for value in omics]]
        projected = [
            ((value.detach().cpu().double() - mean) / std) @ projection
            for value, mean, std, projection in
            zip(raw, self.means, self.stds, self.projections)
        ]
        result = {
            "wsi": _centres(projected[0], self.clusters, self.kmeans_steps).to(wsi),
            "omic": _centres(torch.cat(projected[1:]), self.clusters,
                             self.kmeans_steps).to(wsi),
        }
        for value in result.values():
            _finite_tensor(value, "projected content target")
        return result

    def _metadata(self):
        return {
            "schema": self.SCHEMA, "omic_sizes": list(self.omic_sizes),
            "encoding_dim": self.encoding_dim, "target_dim": self.target_dim,
            "clusters": self.clusters, "seed": self.seed,
            "kmeans_steps": self.kmeans_steps, "std_floor": self.STD_FLOOR,
        }

    def state_dict(self):
        if not self._fitted:
            raise RuntimeError("an unfitted reference has no checkpoint state")
        state = self._metadata()
        state.update({
            "train_case_count": self.train_case_count,
            "feature_counts": list(self.feature_counts),
            "means": [x.clone() for x in self.means],
            "stds": [x.clone() for x in self.stds],
            "projections": [x.clone() for x in self.projections],
            "static_targets": {key: value.clone() for key, value in self.static_targets.items()},
        })
        return state

    def load_state_dict(self, state):
        expected = set(self._metadata()) | {
            "train_case_count", "feature_counts", "means", "stds", "projections", "static_targets",
        }
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("content reference state keys do not match its schema")
        for key, value in self._metadata().items():
            if state[key] != value:
                raise ValueError(f"content reference metadata mismatch: {key}")
        case_count = _positive_int(state["train_case_count"], "train_case_count")
        dimensions = (self.encoding_dim, *self.omic_sizes)
        lists = {}
        for name in ("feature_counts", "means", "stds", "projections"):
            values = state[name]
            if not isinstance(values, (tuple, list)) or len(values) != len(dimensions):
                raise ValueError(f"reference {name} must have one entry per raw feature type")
            lists[name] = values
        counts = [_positive_int(x, "feature count") for x in lists["feature_counts"]]
        if counts[0] < case_count or any(x != case_count for x in counts[1:]):
            raise ValueError("feature counts disagree with the number of training cases")
        tensors = {}
        for name in ("means", "stds", "projections"):
            tensors[name] = []
            for dimension, value in zip(dimensions, lists[name]):
                _finite_tensor(value, name)
                shape = (dimension, self.target_dim) if name == "projections" else (dimension,)
                if value.shape != shape or (name == "stds" and bool((value < self.STD_FLOOR).any())):
                    raise ValueError(f"invalid reference {name} shape or scale")
                tensors[name].append(value.detach().cpu().double().clone())
        dictionaries = state["static_targets"]
        if not isinstance(dictionaries, dict) or set(dictionaries) != {"wsi", "omic"}:
            raise ValueError("static targets require both modalities")
        static_targets = {}
        for name, value in dictionaries.items():
            _finite_tensor(value, "static targets")
            if value.ndim != 2 or not 1 <= len(value) <= self.clusters or value.shape[-1] != self.target_dim:
                raise ValueError("invalid static target shape")
            static_targets[name] = value.detach().cpu().double().clone()
        self.train_case_count = case_count
        self.feature_counts = tuple(counts)
        self.means, self.stds, self.projections = (
            tuple(tensors[name]) for name in ("means", "stds", "projections")
        )
        self.static_targets = static_targets
        self._fitted = True
        return self


def coverage_set_mse(pred, target, return_details=False):
    """Rectangular Hungarian MSE with costs for every unmatched target/slot.

    Only the discrete matching is detached. Every target is represented in the
    objective even after DIAS reduces slot count. Equal counts are exactly the
    original matched-set MSE; no padding, duplicate slots, or query inputs exist.
    """
    from scipy.optimize import linear_sum_assignment

    _finite_tensor(pred, "pred")
    _finite_tensor(target, "target")
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if target.ndim == 3 and target.shape[0] == 1:
        target = target[0]
    if pred.ndim != 2 or target.ndim != 2 or min(*pred.shape, *target.shape) < 1:
        raise ValueError("sets require nonempty [K,D] or [1,K,D] tensors")
    if pred.shape[-1] != target.shape[-1]:
        raise ValueError("prediction and target feature dimensions must match")
    target = target.detach().to(pred)
    costs = (pred[:, None] - target[None]).square().mean(-1)
    if not bool(torch.isfinite(costs).all()):
        raise ValueError("set MSE overflowed; prediction and target scales must remain finite")
    rows, columns = linear_sum_assignment(costs.detach().cpu().double().numpy())
    row_index = torch.as_tensor(rows, device=pred.device)
    column_index = torch.as_tensor(columns, device=pred.device)
    total = costs[row_index, column_index].sum()
    unused_rows = torch.ones(len(pred), dtype=torch.bool, device=pred.device)
    unused_columns = torch.ones(len(target), dtype=torch.bool, device=pred.device)
    unused_rows[row_index] = False
    unused_columns[column_index] = False
    if bool(unused_rows.any()):
        total = total + costs[unused_rows].min(-1).values.sum()
    if bool(unused_columns.any()):
        total = total + costs[:, unused_columns].min(0).values.sum()
    loss = total / max(len(pred), len(target))
    if return_details:
        return loss, {
            "prediction_count": len(pred), "target_count": len(target),
            "matched_count": len(rows),
            "unmatched_prediction_count": int(unused_rows.sum()),
            "unmatched_target_count": int(unused_columns.sum()),
        }
    return loss
