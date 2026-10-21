"""Read-only native Slot Attention traces and deterministic slot replay.

Importing this module does not import torch, instantiate a model, or process data.
The hooks observe native execution; replay starts from a supplied initial tensor
and never samples initial slots.  The replay expressions intentionally follow the
vendored SlotSPE implementation, including its attention normalization and scale.
"""

from contextlib import contextmanager
from importlib import import_module


def _torch():
    return import_module("torch")


def rng_state():
    """Copy CPU and already-initialized CUDA generator states without draws."""
    torch = _torch()
    cuda = {}
    if torch.cuda.is_initialized():
        cuda = {
            index: state.clone()
            for index, state in enumerate(torch.cuda.get_rng_state_all())
        }
    return {"cpu": torch.get_rng_state().clone(), "cuda": cuda}


@contextmanager
def replay_rng(state):
    """Replay recorded generators, restoring every outer generator on exit.

    The state is produced by :func:`rng_state`.  CUDA generators are touched only
    when CUDA was initialized at capture.  No random seed or random draw is used.
    """
    torch = _torch()
    cuda = state.get("cuda", {})
    with torch.random.fork_rng(devices=list(cuda)):
        torch.set_rng_state(state["cpu"])
        for device, device_state in cuda.items():
            torch.cuda.set_rng_state(device_state, device=device)
        yield


def _attention(module, queries, keys):
    torch = _torch()
    dots = torch.einsum("... i d, ... j d -> ... i j", queries, keys)
    attention = (dots * module.scale).softmax(dim=-2)
    return torch.nn.functional.normalize(
        attention + module.eps, p=1, dim=-1
    )


def replay_slots(module, inputs, initial, iterations=None):
    """Replay native iterations without initial-slot sampling or detaching.

    ``inputs`` is the encoded input at the Slot Attention module boundary, before
    ``norm_input``.  ``states`` contains the supplied initial state followed by
    one state per complete GRU/MLP iteration.  Autograd remains available to the
    caller for inputs, initial slots, and module parameters.
    """
    torch = _torch()
    count = module.iters if iterations is None else iterations
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("iterations must be a non-negative integer")
    if initial.ndim != 3 or initial.shape[-1] != module.dim:
        raise ValueError("initial must have shape [batch, slots, module.dim]")
    if inputs.shape[0] != initial.shape[0]:
        raise ValueError("inputs and initial must have the same batch size")

    normalized_inputs = module.norm_input(inputs)
    keys = module.split_heads(module.to_k(normalized_inputs))
    values = module.split_heads(module.to_v(normalized_inputs))
    batch, slots_count, dimension = initial.shape
    slots = initial
    states = [slots]
    attentions = []
    for _ in range(count):
        slots_previous = slots
        slots = module.norm_slots(slots)
        queries = module.split_heads(module.to_q(slots))
        attention = _attention(module, queries, keys)
        attentions.append(attention)
        updates = torch.einsum(
            "... j d, ... i j -> ... i d", values, attention
        )
        updates = module.combine_heads(module.merge_heads(updates))
        slots = module.gru(
            updates.reshape(-1, dimension),
            slots_previous.reshape(-1, dimension),
        ).reshape(batch, slots_count, dimension)
        slots = slots + module.mlp(module.norm_pre_ff(slots))
        states.append(slots)
    return {"final": slots, "states": states, "attentions": attentions}


class NativeSlotTrace:
    """Install detached, read-only hooks on both native slot modules.

    ``records[modality]`` always refers to the first call since ``reset()``.
    ``calls[modality]`` retains every call in invocation order, so repeated model
    calls cannot silently replace the native observation used for a comparison.
    Each record contains inputs, entry RNG state, initial slots, complete states,
    per-iteration [batch, heads, slots, tokens] attention, and final slots.
    """

    def __init__(self, model):
        self.model = model
        self.records = {}
        self.calls = {"wsi": [], "omic": []}
        self._active = {"wsi": [], "omic": []}
        self._handles = []
        self._install()

    def _install(self):
        if self._handles:
            return
        try:
            for modality in self.calls:
                module = getattr(self.model, "slot_attention_" + modality)
                self._handles.append(
                    module.register_forward_pre_hook(self._before(modality))
                )
                self._handles.append(
                    module.norm_slots.register_forward_pre_hook(
                        self._before_norm(modality)
                    )
                )
                self._handles.append(
                    module.to_k.register_forward_hook(self._keys(modality, module))
                )
                self._handles.append(
                    module.to_q.register_forward_hook(
                        self._queries(modality, module)
                    )
                )
                self._handles.append(
                    module.register_forward_hook(self._after(modality))
                )
        except Exception:
            self.close()
            raise

    @staticmethod
    def _copy(tensor):
        return tensor.detach().clone()

    def _current(self, modality):
        if not self._active[modality]:
            # replay_slots may invoke these submodules directly while the trace
            # remains installed. Only native parent forwards are observations.
            return None
        return self._active[modality][-1]

    def _before(self, modality):
        def hook(module, args):
            if not args:
                raise ValueError("native Slot Attention inputs must be positional")
            record = {
                "inputs": self._copy(args[0]),
                "rng_entry": rng_state(),
                "initial": None,
                "states": [],
                "attentions": [],
            }
            self.calls[modality].append(record)
            self.records.setdefault(modality, record)
            self._active[modality].append(record)
            return None

        return hook

    def _before_norm(self, modality):
        def hook(module, args):
            record = self._current(modality)
            if record is None:
                return None
            state = self._copy(args[0])
            if record["initial"] is None:
                record["initial"] = state
            record["states"].append(state)
            return None

        return hook

    def _keys(self, modality, slot_module):
        def hook(module, args, output):
            record = self._current(modality)
            if record is None:
                return None
            torch = _torch()
            with torch.no_grad():
                record["_keys"] = self._copy(
                    slot_module.split_heads(output.detach())
                )
            return None

        return hook

    def _queries(self, modality, slot_module):
        def hook(module, args, output):
            torch = _torch()
            record = self._current(modality)
            if record is None:
                return None
            with torch.no_grad():
                queries = slot_module.split_heads(output.detach())
                attention = _attention(slot_module, queries, record["_keys"])
                record["attentions"].append(self._copy(attention))
            return None

        return hook

    def _after(self, modality):
        def hook(module, args, output):
            record = self._current(modality)
            if record is None:
                raise RuntimeError("native slot call has no active trace record")
            record["final"] = self._copy(output)
            # Zero-iteration native calls have no norm_slots invocation.
            if record["initial"] is None:
                record["initial"] = self._copy(output)
            record["states"].append(record["final"])
            record.pop("_keys", None)
            self._active[modality].pop()
            return None

        return hook

    def reset(self):
        """Clear observations between forwards without reinstalling hooks."""
        if any(self._active.values()):
            raise RuntimeError("cannot reset while a slot forward is active")
        self.records.clear()
        for calls in self.calls.values():
            calls.clear()

    def close(self):
        """Remove all hooks; completed records remain available."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for active in self._active.values():
            active.clear()

    def __enter__(self):
        self._install()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
