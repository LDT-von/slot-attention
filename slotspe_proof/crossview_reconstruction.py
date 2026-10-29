"""Explicit V2 cross-view experiment; the native full-bag forward is unchanged.

Reference fitting and patient-level train/validation separation are the caller's
responsibility.  This module receives a *fitted* ContentReference state; it
does not create, fit, or discover any reference from evaluation patients.

This is an experimental reconstruction objective, not a proof of slot-wise
uniqueness, biological meaning, conditional mutual information, or survival
benefit.  A case contains exactly one patient.  The caller supplies the actual
number of real rows, with any loader padding following those rows.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .plugin_host import SlotPluginHost
from .redundancy_content import ContentReference


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _seed(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 2**63:
        raise ValueError(f"{name} must be an integer in [0, 2**63)")
    return int(value)


def _checked_reference(state: Mapping):
    """Use the existing reference validator rather than accept ad-hoc arrays."""
    if not isinstance(state, dict):
        raise ValueError("reference_state must be a fitted ContentReference state dict")
    try:
        reference = ContentReference(
            state["omic_sizes"], state["encoding_dim"], target_dim=state["target_dim"],
            clusters=state["clusters"], seed=state["seed"], kmeans_steps=state["kmeans_steps"],
        ).load_state_dict(state)
    except (KeyError, TypeError, RuntimeError) as error:
        raise ValueError("invalid fitted ContentReference state") from error
    # The validator clones all data; clone the four required WSI arrays again to
    # make this ownership boundary explicit and leave no trainable target path.
    tensors = tuple(x.detach().cpu().double().contiguous().clone() for x in (
        reference.means[0], reference.stds[0], reference.projections[0],
        reference.static_targets["wsi"],
    ))
    meta = {
        "schema": state["schema"], "encoding_dim": reference.encoding_dim,
        "target_dim": reference.target_dim, "clusters": reference.clusters,
        "seed": reference.seed, "kmeans_steps": reference.kmeans_steps,
        "std_floor": reference.STD_FLOOR, "train_case_count": reference.train_case_count,
        "wsi_feature_count": reference.feature_counts[0],
    }
    digest = hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for value in tensors:
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        # Frozen references are canonical float64, independent of model dtype.
        digest.update(value.numpy().astype("<f8", copy=False).tobytes(order="C"))
    return tensors, meta, digest.hexdigest()


class _SetHistogramDecoder(nn.Module):
    """Shared slot map predicting anchor mass and weighted feature moments."""

    def __init__(self, private_dim: int, hidden_dim: int, outputs: int, target_dim: int):
        super().__init__()
        self.private_dim = private_dim
        self.outputs = outputs
        self.target_dim = target_dim
        self.slot_map = nn.Sequential(
            nn.Linear(private_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.readout = nn.Linear(hidden_dim, outputs)
        self.moment_readout = nn.Linear(hidden_dim, outputs * target_dim)

    def forward(self, private_slots: Tensor, *, return_moments: bool = False):
        if not isinstance(private_slots, Tensor) or private_slots.ndim != 3:
            raise ValueError("private_slots must have shape [B,K,private_dim]")
        if min(private_slots.shape) < 1 or private_slots.shape[-1] != self.private_dim:
            raise ValueError("private_slots has invalid dimensions")
        if not bool(torch.isfinite(private_slots).all()):
            raise ValueError("private_slots must be finite")
        pooled = self.slot_map(private_slots).mean(dim=1)
        logits = self.readout(pooled)
        # MSE in at least fp32 avoids unsupported/unstable half-precision loss.
        work_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        mass = F.softmax(logits, dim=-1, dtype=work_dtype)
        if not return_moments:
            return mass
        moments = self.moment_readout(pooled).to(dtype=work_dtype).reshape(
            len(pooled), self.outputs, self.target_dim)
        return mass, moments


class CrossViewV2(nn.Module):
    """V2 host plus optional A-view slots / B-view fixed-histogram supervision.

    Call ``forward(..., mode='off', run_aux=False)`` for the unchanged host.
    ``mode='compute'`` performs the same auxiliary graph as ``mode='train'``
    but exposes zero auxiliary loss; it is a matched extra-computation control.
    When run_aux is true both a real row count and a separate seed are required.

    ``crossview_loss`` is reset on every call.  ``last_crossview`` contains only
    scalar metadata, not a patient's feature, target, prediction, or index
    arrays.  The returned logits and native auxiliary loss come directly from
    the host and are computed *before* the optional branch.

    Reference buffers are deliberately nonpersistent: checkpoints must be
    re-created with the separately saved, matching reference state.  Extra
    checkpoint state rejects a different reference or temperature on loading.
    """

    def __init__(self, host: SlotPluginHost, reference_state: dict, *,
                 temperature: float = 1.0, hidden_dim: int | None = None,
                 decoder_seed: int = 731):
        super().__init__()
        if not isinstance(host, SlotPluginHost) or host.preset not in {"V2_content", "V2_separate"}:
            raise ValueError("host must be an existing V2_content or V2_separate SlotPluginHost")
        if isinstance(temperature, bool) or not isinstance(temperature, Real):
            raise ValueError("temperature must be a positive finite real number")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a positive finite real number")
        decoder_seed = _seed("decoder_seed", decoder_seed)
        arrays, metadata, signature = _checked_reference(reference_state)
        private_dim = _positive_int("private_dim", host.parts["wsi"].private_dim)
        hidden_dim = private_dim if hidden_dim is None else _positive_int("hidden_dim", hidden_dim)
        self.host = host
        self.temperature = temperature
        self.reference_signature = signature
        self.reference_metadata = metadata
        self.decoder_seed = decoder_seed
        self.hidden_dim = hidden_dim
        self.register_buffer("reference_mean", arrays[0], persistent=False)
        self.register_buffer("reference_std", arrays[1], persistent=False)
        self.register_buffer("reference_projection", arrays[2], persistent=False)
        self.register_buffer("reference_anchors", arrays[3], persistent=False)
        # Decoder construction never advances the caller's CPU random stream.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(decoder_seed)
            decoder = _SetHistogramDecoder(private_dim, hidden_dim, len(arrays[3]), metadata["target_dim"])
        parameter = host.parts["wsi"].private_encoder.weight
        self.decoder = decoder.to(device=parameter.device, dtype=parameter.dtype)
        self.crossview_loss = None
        self.crossview_raw_loss = None
        self.last_crossview = None
        self.last_crossview_tensors = None

    @property
    def backbone(self):
        return self.host.backbone

    def get_extra_state(self):
        return {"schema": "slotspe_crossview_v1", "reference_signature": self.reference_signature,
                "temperature": self.temperature, "hidden_dim": self.hidden_dim,
                "decoder_seed": self.decoder_seed, "split_policy": "cpu_permutation_half_v1",
                "target_policy": "fixed_anchor_mass_and_first_moment_v1", "moment_weight": 1.0}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("cross-view checkpoint/reference configuration mismatch")

    def _statistics(self, view_b: Tensor):
        """Fixed B mass and q-weighted first moments; no target gradients."""
        if not isinstance(view_b, Tensor) or view_b.ndim != 3 or view_b.shape[0] != 1:
            raise ValueError("view_b must have shape [1,N,D]")
        if view_b.shape[1] < 1 or view_b.shape[2] != self.reference_mean.numel():
            raise ValueError("view_b must contain real rows in the reference dimension")
        if not view_b.is_floating_point() or not bool(torch.isfinite(view_b).all()):
            raise ValueError("view_b must contain finite real floating-point values")
        dtype = torch.float64 if view_b.dtype == torch.float64 else torch.float32
        device = view_b.device
        with torch.no_grad():
            value = view_b.detach().to(dtype=dtype)
            mean = self.reference_mean.to(device=device, dtype=dtype)
            std = self.reference_std.to(device=device, dtype=dtype)
            projection = self.reference_projection.to(device=device, dtype=dtype)
            anchors = self.reference_anchors.to(device=device, dtype=dtype)
            projected = ((value - mean) / std) @ projection
            distance = (projected.unsqueeze(2) - anchors.view(1, 1, *anchors.shape)).square().sum(-1)
            if not bool(torch.isfinite(distance).all()):
                raise ValueError("nonfinite reference distance; check scale and feature range")
            assignments = F.softmax(-distance / self.temperature, dim=-1)
            target = assignments.mean(dim=1)
            moments = (assignments.unsqueeze(-1) * projected.unsqueeze(-2)).mean(dim=1)
            if not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(moments).all()):
                raise ValueError("nonfinite fixed content targets")
            return target.detach(), moments.detach()

    def _histogram(self, view_b: Tensor) -> Tensor:
        """Return mass only, for inspection or aggregate training baselines."""
        return self._statistics(view_b)[0]

    def forward(self, *, plugin_seed=None, real_count=None, aux_seed=None,
                mode="off", run_aux=False, capture_aux=False, **case):
        if type(capture_aux) is not bool or (capture_aux and (not run_aux or torch.is_grad_enabled())):
            raise ValueError("auxiliary tensor capture requires an auxiliary no-grad evaluation")
        if mode not in {"off", "compute", "train"}:
            raise ValueError("mode must be off, compute, or train")
        if type(run_aux) is not bool or run_aux != (mode != "off"):
            raise ValueError("run_aux must be true exactly for compute/train modes")
        if run_aux:
            aux_seed = _seed("aux_seed", aux_seed)
            if isinstance(real_count, bool) or not isinstance(real_count, Integral) or real_count < 0:
                raise ValueError("real_count must be a nonnegative integer")
            real_count = int(real_count)
            x = case.get("x_wsi")
            if not isinstance(x, Tensor) or x.ndim != 3 or x.shape[0] != 1:
                raise ValueError("cross-view requires one patient's x_wsi [1,N,D]")
            if x.shape[1] < 1 or x.shape[2] != self.reference_mean.numel() or real_count > x.shape[1]:
                raise ValueError("real_count or WSI feature dimensions do not match the input")
            if x.device.type not in {"cpu", "cuda"}:
                raise ValueError("cross-view RNG isolation supports CPU or CUDA only")
        self.crossview_loss = None
        self.crossview_raw_loss = None
        self.last_crossview = None
        self.last_crossview_tensors = None
        result = self.host(plugin_seed=plugin_seed, **case)
        if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], Tensor):
            raise RuntimeError("V2 host must return (logits, native_auxiliary)")
        self.crossview_loss = result[0].new_zeros(())
        self.crossview_raw_loss = result[0].new_zeros(())
        self.last_crossview = {
            "mode": mode, "skipped": not run_aux or real_count < 2,
            "real_count": real_count if run_aux else None, "a_count": 0, "b_count": 0,
            "reference_signature": self.reference_signature,
        }
        if not run_aux or real_count < 2:
            return result

        # Dedicated CPU generator decides only the split, never global RNG.
        split_gen = torch.Generator(device="cpu")
        split_gen.manual_seed(aux_seed)
        indices = torch.randperm(real_count, generator=split_gen, device="cpu")
        a_count = real_count // 2
        index_a = indices[:a_count].to(device=x.device)
        index_b = indices[a_count:].to(device=x.device)
        view_a = x.index_select(1, index_a)
        view_b = x.index_select(1, index_b)
        target, target_moments = self._statistics(view_b)

        if self.host._active:
            raise RuntimeError("native host hooks have not been released")
        device_indices = [] if x.device.type == "cpu" else [x.device.index]
        if device_indices and device_indices[0] is None:
            raise RuntimeError("CUDA tensor must have an explicit device index")
        auxiliary_seed = (aux_seed + 1) % (2**63)
        # Save/restore only CPU and this particular CUDA stream.  In particular,
        # torch.manual_seed() is avoided because it also modifies all GPUs.
        with torch.random.fork_rng(devices=device_indices, enabled=True):
            torch.default_generator.manual_seed(auxiliary_seed)
            if device_indices:
                torch.cuda.default_generators[device_indices[0]].manual_seed(auxiliary_seed)
            projected_a = self.host.backbone.wsi_mlp(view_a)
            native_slots_a = self.host.backbone.slot_attention_wsi(projected_a)
            _, slot_record = self.host.parts["wsi"](native_slots_a)
            prediction, predicted_moments = self.decoder(slot_record["private"], return_moments=True)
            if prediction.shape != target.shape:
                raise RuntimeError("histogram prediction and fixed target shapes differ")
            if predicted_moments.shape != target_moments.shape:
                raise RuntimeError("moment prediction and fixed target shapes differ")
            mass_loss = F.mse_loss(prediction, target.to(device=prediction.device, dtype=prediction.dtype))
            moment_loss = F.mse_loss(predicted_moments, target_moments.to(
                device=predicted_moments.device, dtype=predicted_moments.dtype))
            loss = mass_loss + moment_loss
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite cross-view loss")
        # Keep the same auxiliary backward graph in the compute control while
        # imposing exactly zero auxiliary gradient. Detaching would omit that
        # computation and make the intended compute comparison misleading.
        self.crossview_raw_loss = loss.detach()
        self.crossview_loss = loss if mode == "train" else loss * 0.0
        self.last_crossview.update(skipped=False, a_count=a_count, b_count=real_count-a_count,
                                   mass_loss=float(mass_loss.detach()), moment_loss=float(moment_loss.detach()))
        if capture_aux:
            self.last_crossview_tensors = tuple(value.detach().cpu() for value in
                (prediction, predicted_moments, target, target_moments))
        return result
