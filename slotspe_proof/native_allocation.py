"""Temporary allocation-only interventions in the unchanged native SlotSPE.

This is a generated-data research adapter, not a patient-training entry point.
Both native omic attention calls are replaced, including reconstruction. Native
Gaussian initial slots, Gumbel selection, readout, and losses are retained.
"""
from types import MethodType

import torch

from .assignment_router import (
    AssignmentConfig, SlotAssignmentRouter, replay_with_assignment,
)


VARIANTS = {
    "S0": {"wsi": "original", "omic": "original"},
    "S1": {"wsi": "sinkhorn", "omic": "sinkhorn"},
    "S2": {"wsi": "mesh_uniform", "omic": "original"},
    "S3": {"wsi": "original", "omic": "mesh_uniform"},
    "S4": {"wsi": "mesh_uniform", "omic": "mesh_uniform"},
}


class NativeAllocation:
    """Replace only the two native slot modules' forward inside a context.

    MESH noise is derived from a separate CPU generator, indexed by modality
    and invocation. Resetting noise_seed replays noise, without changing the
    global native RNG stream. This context is deliberately not thread safe.
    Detached traces may contain input-derived representations: callers using
    real data must keep them private. No patient runner imports this adapter.
    """

    def __init__(self, model, variant="S0", noise_seed=123, capture=True):
        if variant not in VARIANTS:
            raise ValueError(f"unknown native allocation variant: {variant}")
        self.model = model
        self.variant = variant
        self.capture = capture
        self.set_noise_seed(noise_seed)
        self.records = []
        self._calls = {"wsi": 0, "omic": 0}
        self._saved = []
        self._hook = None
        self._active = False

    def set_noise_seed(self, seed):
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("noise_seed must be an integer")
        self.noise_seed = seed

    def _reset(self, _model, _args):
        self.records.clear()
        self._calls = {"wsi": 0, "omic": 0}

    def _forward(self, attention, inputs, modality, num_slots=None):
        index = self._calls[modality]
        self._calls[modality] += 1
        mode = VARIANTS[self.variant][modality]
        slots_count = attention.num_slots if num_slots is None else num_slots
        batch = inputs.shape[0]
        # Same expansion, order, dtype and global Gaussian RNG as native code.
        mu = attention.slots_mu.expand(batch, slots_count, -1)
        sigma = attention.slots_logsigma.exp().expand(batch, slots_count, -1)
        initial = mu + sigma * torch.randn(
            mu.shape, device=inputs.device, dtype=inputs.dtype,
        )
        router = SlotAssignmentRouter(AssignmentConfig(mode=mode, eps=attention.eps))
        noises = None
        if mode == "mesh_uniform":
            head_probe = torch.zeros(
                1, 1, attention.to_q.out_features,
                device=inputs.device, dtype=inputs.dtype,
            )
            heads = attention.split_heads(head_probe).shape[1]
            generator = torch.Generator(device="cpu")
            stream = self.noise_seed + (0 if modality == "wsi" else 1000003) + index * 10007
            generator.manual_seed(stream % (2**63 - 1))
            noises = [torch.randn(
                (batch, heads, slots_count, inputs.shape[1]),
                generator=generator, dtype=inputs.dtype, device="cpu",
            ).to(inputs.device) for _ in range(attention.iters)]
        result = replay_with_assignment(
            attention, inputs, initial, router, noises=noises,
        )
        if self.capture:
            self.records.append({
                "modality": modality, "call_index": index, "mode": mode,
                "source_input_count": inputs.shape[1],
                "initial": initial.detach().cpu(),
                "final": result["final"].detach().cpu(),
                "attentions": [value.detach().cpu() for value in result["attentions"]],
            })
        return result["final"]

    def __enter__(self):
        if self._active or getattr(self.model, "_native_allocation_active", False):
            raise RuntimeError("native allocation contexts cannot be nested")
        self._active = True
        self.model._native_allocation_active = True
        try:
            self._hook = self.model.register_forward_pre_hook(self._reset)
            for modality in ("wsi", "omic"):
                module = getattr(self.model, f"slot_attention_{modality}")
                # Remember whether a method existed on the instance, so exit
                # restores class dispatch rather than leaving a bound override.
                self._saved.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))

                def replacement(attention, inputs, num_slots=None, _modality=modality):
                    return self._forward(attention, inputs, _modality, num_slots)

                module.forward = MethodType(replacement, module)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        for module, had_override, original in reversed(self._saved):
            if had_override:
                module.forward = original
            else:
                del module.forward
        self._saved.clear()
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        if self._active:
            delattr(self.model, "_native_allocation_active")
        self._active = False
        return False
