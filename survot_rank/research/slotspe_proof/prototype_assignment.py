"""Research bridge for DCT's post-SA prototype coordinate pooling.

DCT v2.3 inherits DCTFeatureCore._slots: local Slot Attention -> soft prototype
coordinates -> downstream cross-modal transport. Prototype coordinates are not
a VQ codebook, duplicate-removal mask, or prototype-initialized Slot Attention.
This bridge changes only the requested assignment stage and reuses supplied
prototype parameters. It does not import DCT, change its files, or add transport.
"""
import math

import torch
from torch.nn import functional as F

from .assignment_router import (
    AssignmentConfig, SlotAssignmentRouter, replay_with_assignment,
)


def prototype_coordinates(local_slots, prototypes, *, temperature=0.3,
                          mode="dct_soft", noise=None,
                          assignment_config=None):
    """Return pooled coordinates [B,K,D] and weights [B,K,L].

    dct_soft intentionally follows the existing DCT semantic_slots expression
    exactly, including its clamp instead of the native SA's added epsilon.
    Alternatives operate on these same cosine scores and preserve value scale
    by returning row-normalized weights. They are MESH/Sinkhorn adaptations,
    not MetaSlot quantization. Noise for mesh_uniform is explicit [B,1,K,L].
    """
    if (local_slots.ndim != 3 or prototypes.ndim != 2 or
            min(local_slots.shape) < 1 or min(prototypes.shape) < 1 or
            local_slots.shape[-1] != prototypes.shape[-1]):
        raise ValueError("expected local_slots[B,L,D], prototypes[K,D]")
    if (not local_slots.is_floating_point() or not prototypes.is_floating_point() or
            local_slots.dtype != prototypes.dtype or
            local_slots.device != prototypes.device or
            not torch.isfinite(local_slots).all() or
            not torch.isfinite(prototypes).all()):
        raise ValueError("slots and prototypes need matching finite dtype/device")
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    scores = torch.einsum(
        "bnd,kd->bkn", F.normalize(local_slots, dim=-1),
        F.normalize(prototypes, dim=-1),
    ) / temperature
    if mode == "dct_soft":
        if assignment_config is not None:
            raise ValueError("dct_soft does not consume router configuration")
        assignments = torch.softmax(scores, dim=1)
        weights = assignments / assignments.sum(-1, keepdim=True).clamp_min(1e-8)
    elif mode in {"sinkhorn_uniform", "mesh_uniform"}:
        router_mode = "sinkhorn" if mode == "sinkhorn_uniform" else mode
        config = assignment_config or AssignmentConfig(mode=router_mode)
        if config.mode != router_mode:
            raise ValueError("coordinate mode and router mode disagree")
        weights = SlotAssignmentRouter(config)(scores[:, None], noise=noise)[:, 0]
    else:
        raise ValueError("unknown prototype coordinate mode")
    return torch.einsum("bkn,bnd->bkd", weights, local_slots), weights


def replay_prototype_pipeline(native, inputs, initial, prototypes, *,
                              local_config=None, local_noises=None,
                              coordinate_mode="dct_soft", coordinate_noise=None,
                              coordinate_temperature=0.3,
                              coordinate_config=None, token_mask=None):
    """Independent two-stage research probe; does not install a model adapter.

    Caller controls initial states and existing learned prototypes. local_config
    changes the assignment inside SA; coordinate_mode changes post-SA pooling.
    Keep downstream model/loss fixed when integrating an explicit experiment.
    """
    config = local_config or AssignmentConfig(eps=native.eps)
    # DCT masks padding before LayerNorm, including nonfinite padding values.
    # Its core dispatches only patients with a nonempty modality to this stage.
    if token_mask is not None:
        if (token_mask.dtype != torch.bool or token_mask.device != inputs.device or
                token_mask.shape != inputs.shape[:2]):
            raise ValueError("token_mask must be boolean [B,N] on the inputs device")
        inputs = inputs.masked_fill(~token_mask.unsqueeze(-1), 0)
    local = replay_with_assignment(
        native, inputs, initial, SlotAssignmentRouter(config),
        noises=local_noises, token_mask=token_mask,
    )
    coordinates, weights = prototype_coordinates(
        local["final"], prototypes, temperature=coordinate_temperature,
        mode=coordinate_mode, noise=coordinate_noise,
        assignment_config=coordinate_config,
    )
    return {"local": local, "coordinates": coordinates, "coordinate_weights": weights}


def replay_assignment_variant(spec, native, inputs, initial, prototypes, *,
                              local_noises=None, coordinate_noise=None,
                              token_mask=None):
    """Run an explicit stage-spec dict; never a training configuration.

    A host adapter and matched training driver are separate work. No arbitrary
    method import, patient file loading, source patching, or queue dispatch is
    selected by this registry.
    """
    if spec.get("schema") != "dct_prototype_assignment_stage_spec_v1":
        raise ValueError("expected a versioned assignment-stage specification")
    return replay_prototype_pipeline(
        native, inputs, initial, prototypes,
        local_config=AssignmentConfig(**spec["local_assignment"]),
        local_noises=local_noises,
        coordinate_mode=spec["prototype_assignment"],
        coordinate_noise=coordinate_noise,
        coordinate_temperature=spec["coordinate_temperature"],
        token_mask=token_mask,
    )
