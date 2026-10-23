"""DIAS/MetaSlot redundancy-core adaptations for the unchanged SlotSPE.

S5 implements DIAS section 3.2, not its extra distillation/image decoder.
S6 follows the MetaSlot paper's deterministic quantization, final-step gradient
and EMA codebook scheme, not every detail of its released image framework.
Removed slots are physically compacted, so they never enter native attention,
reconstruction or means as misleading zero vectors. Native model forwards are
therefore single-example; forward_compact accumulates a logical minibatch.
"""
from dataclasses import dataclass
import math
from types import MethodType

import torch
from torch.nn import functional as F

from .assignment_router import AssignmentConfig, SlotAssignmentRouter, replay_with_assignment
from .prototype_bank import PrototypeBank


@dataclass(frozen=True)
class PruningConfig:
    # Local prespecified adaptation settings; DIAS has no published threshold.
    dias_distance: float = 0.05
    prototype_count: int = 8
    ema_rate: float = 0.05
    noise_std: float = 0.5
    dead_timeout: int = 100
    kmeans_steps: int = 10

    def __post_init__(self):
        if not 0 <= self.dias_distance <= 2 or not math.isfinite(self.dias_distance):
            raise ValueError("dias_distance must be finite in [0,2]")
        if not 0 < self.ema_rate <= 1 or not math.isfinite(self.ema_rate):
            raise ValueError("ema_rate must be in (0,1]")
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be nonnegative and finite")
        for name in ("prototype_count", "dead_timeout", "kmeans_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def cosine_clusters(slots, distance_threshold):
    """Deterministic average-linkage clustering of detached slot directions.

    Return original member indices. Cluster means are computed by the caller
    without detaching, so DIAS reaggregation still trains retained content.
    This threshold/linkage/representative choice is our declared adaptation.
    """
    if slots.ndim != 2 or min(slots.shape) < 1 or not torch.isfinite(slots).all():
        raise ValueError("slots must be finite nonempty [S,D]")
    if not math.isfinite(distance_threshold) or not 0 <= distance_threshold <= 2:
        raise ValueError("distance threshold must be in [0,2]")
    unit = F.normalize(slots.detach().double(), dim=-1)
    distance = (1 - unit @ unit.T).clamp(0, 2).cpu()
    groups = [[index] for index in range(len(slots))]
    while len(groups) > 1:
        candidates = [(float(distance[a][:, b].mean()), left, right)
                      for left, a in enumerate(groups)
                      for right, b in enumerate(groups) if right > left]
        best, left, right = min(candidates)
        if best > distance_threshold:
            break
        groups[left] = sorted(groups[left] + groups[right])
        del groups[right]
    return groups


def forward_compact(model, adapter, batch, *, native_seed, noise_seed):
    """Same original model/loss per example; average auxiliary loss per batch.

    Explicit per-example native seeds pair all arms despite ragged slot counts.
    Different counts still change native Gumbel/dropout tensor shapes, so they
    do not imply elementwise-identical random draws at every downstream layer.
    Use this same protocol for the S0 reference, not older vectorized results.
    """
    count = batch["x_wsi"].shape[0]
    if count < 1:
        raise ValueError("empty logical batch")
    outputs, auxiliary, records = [], [], []
    try:
        for index in range(count):
            item = {key: value[index:index + 1] if torch.is_tensor(value) else value
                    for key, value in batch.items()}
            torch.manual_seed(native_seed + index * 10007)
            adapter.set_noise_seed(noise_seed + index * 10007)
            logits, loss = model(**item)
            outputs.append(logits)
            auxiliary.append(torch.as_tensor(loss, device=logits.device))
            records.extend(dict(record, sample_index=index) for record in adapter.records)
    except BaseException:
        # A caught failed microbatch must not silently contaminate the next
        # successful EMA commit, even while the outer context stays active.
        if hasattr(adapter, "abort_codebooks"):
            adapter.abort_codebooks()
        raise
    adapter.batch_records = records
    return torch.cat(outputs, dim=0), torch.stack(auxiliary).mean()


class NativePruning:
    def __init__(self, model, variant="S5", config=None, banks=None,
                 noise_seed=123, capture=True):
        if variant not in {"S5", "S6"}:
            raise ValueError("NativePruning supports S5 or S6")
        self.model = model
        self.variant = variant
        self.config = config or PruningConfig()
        self.banks = banks or {}
        if variant == "S6":
            for modality in ("wsi", "omic"):
                if modality not in self.banks or self.banks[modality].prototypes is None:
                    raise ValueError("S6 requires separately initialized train-only wsi/omic codebooks")
                bank = self.banks[modality]
                if bank.pending:
                    raise ValueError("S6 requires committed codebooks without pending vectors")
                expected = {name: getattr(self.config, name) for name in
                            ("prototype_count", "ema_rate", "dead_timeout", "kmeans_steps")}
                if bank.state_dict()["config"] != expected:
                    raise ValueError("S6 codebook configuration differs from the active pruning configuration")
        self.capture = capture
        self.set_noise_seed(noise_seed)
        self.records, self.batch_records = [], []
        self._calls = {"wsi": 0, "omic": 0}
        self._saved, self._handles, self._decoders = [], [], []
        self._active = False

    def set_noise_seed(self, seed):
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("noise_seed must be an integer")
        self.noise_seed = seed

    def _root_start(self, _model, _args, kwargs):
        if kwargs["x_wsi"].shape[0] != 1:
            raise ValueError("ragged pruning requires B=1; use forward_compact for a logical batch")
        if kwargs.get("omic_missing", False):
            raise ValueError("current pruning adapter requires complete paired inputs")
        self.records.clear()
        self._calls = {"wsi": 0, "omic": 0}

    def _forward(self, attention, inputs, modality, num_slots=None):
        if inputs.shape[0] != 1 or attention.iters < 2:
            raise ValueError("pruning requires single-example inputs and at least two SA iterations")
        count = attention.num_slots if num_slots is None else num_slots
        if count < 1:
            raise ValueError("at least one candidate slot required")
        index = self._calls[modality]
        self._calls[modality] += 1
        mu = attention.slots_mu.expand(1, count, -1)
        sigma = attention.slots_logsigma.exp().expand(1, count, -1)
        initial = mu + sigma * torch.randn(mu.shape, device=inputs.device, dtype=inputs.dtype)
        router = SlotAssignmentRouter(AssignmentConfig(mode="original", eps=attention.eps))
        groups, prototype_indices = None, None
        if self.variant == "S5":
            prefix = replay_with_assignment(attention, inputs, initial, router, iterations=attention.iters - 1)
            groups = cosine_clusters(prefix["final"][0], self.config.dias_distance)
            retained = torch.stack([prefix["final"][0, group].mean(0) for group in groups])[None]
            final = replay_with_assignment(attention, inputs, retained, router, iterations=1)
            first_iters, refine_iters = attention.iters - 1, 1
        else:
            with torch.no_grad():
                prefix = replay_with_assignment(attention, inputs, initial, router)
                prototype_indices, quantized = self.banks[modality].lookup(prefix["final"])
            # First occurrence per prototype; keep exact prototype values as in
            # the paper, rather than pretending selection alone is MetaSlot.
            seen, positions = set(), []
            for position, code in enumerate(prototype_indices[0].tolist()):
                if code not in seen:
                    seen.add(code)
                    positions.append(position)
            retained = quantized[:, positions].to(inputs.device, inputs.dtype)
            slots = retained
            attentions, states = [], [retained]
            generator = torch.Generator(device="cpu")
            stream = self.noise_seed + (0 if modality == "wsi" else 1000003) + index * 10007
            generator.manual_seed(stream % (2**63 - 1))
            for step in range(attention.iters):
                noisy_inputs = inputs
                amplitude = self.config.noise_std * (1 - step / (attention.iters - 1))
                if self.model.training and amplitude > 0:
                    noise = torch.randn(inputs.shape, generator=generator, dtype=inputs.dtype)
                    noisy_inputs = inputs + amplitude * noise.to(inputs.device)
                if step < attention.iters - 1:
                    with torch.no_grad():
                        update = replay_with_assignment(attention, noisy_inputs, slots, router, iterations=1)
                else:
                    # Paper Algorithm 1 reconnects the original learnable
                    # queries, not the non-gradient EMA prototype values.
                    # The forward state stays refined prototypes; gradients
                    # reach the retained native Gaussian queries only here.
                    query = initial[:, positions]
                    slots = slots.detach() + query - query.detach()
                    update = replay_with_assignment(attention, noisy_inputs, slots, router, iterations=1)
                slots = update["final"]
                states.append(slots)
                attentions.extend(update["attentions"])
            final = {"final": slots, "attentions": attentions, "states": states}
            if self.model.training:
                self.banks[modality].queue(slots[0])
            first_iters = refine_iters = attention.iters
        if self.capture:
            self.records.append({
                "modality": modality, "call_index": index,
                "mode": "dias" if self.variant == "S5" else "metaslot",
                "source_input_count": inputs.shape[1],
                "initial": initial.detach().cpu(),
                "pre_prune": prefix["final"].detach().cpu(),
                "retained_before_refine": retained.detach().cpu(),
                "final": final["final"].detach().cpu(),
                "active_count": final["final"].shape[1],
                "first_stage_iterations": first_iters, "refine_iterations": refine_iters,
                "attentions": [value.detach().cpu() for value in final["attentions"]],
                "groups": groups,
                "prototype_indices": None if prototype_indices is None else prototype_indices.detach().cpu(),
            })
        return final["final"]

    def commit_codebooks(self):
        if self.variant == "S6":
            for bank in self.banks.values():
                bank.commit()

    def abort_codebooks(self):
        """Discard the uncommitted batch; keep all successful prior commits.

        forward_compact and exceptional context exit call this automatically.
        A caller that catches a backward/optimizer error *inside* the context
        must explicitly abort before retrying that batch.
        """
        if self.variant == "S6":
            for bank in self.banks.values():
                bank.pending.clear()

    def __enter__(self):
        if self._active or getattr(self.model, "_native_allocation_active", False):
            raise RuntimeError("allocation/pruning contexts cannot be nested")
        self.model._native_allocation_active = True
        self._active = True
        try:
            self._handles.append(self.model.register_forward_pre_hook(self._root_start, with_kwargs=True))
            for modality in ("wsi", "omic"):
                module = getattr(self.model, f"slot_attention_{modality}")
                self._saved.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))

                def replacement(attention, inputs, num_slots=None, _modality=modality):
                    return self._forward(attention, inputs, _modality, num_slots)

                module.forward = MethodType(replacement, module)
                decoder = getattr(self.model, f"slot_decoder_{modality}")
                original_k = decoder.k
                self._decoders.append((decoder, original_k))

                def cap_k(decoder, inputs, _original=original_k):
                    decoder.k = min(_original, inputs[0].shape[1])

                self._handles.append(decoder.register_forward_pre_hook(cap_k))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        if _exc_type is not None:
            self.abort_codebooks()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for decoder, original in self._decoders:
            decoder.k = original
        self._decoders.clear()
        for module, had_override, original in reversed(self._saved):
            if had_override:
                module.forward = original
            else:
                del module.forward
        self._saved.clear()
        if self._active:
            delattr(self.model, "_native_allocation_active")
        self._active = False
        return False
