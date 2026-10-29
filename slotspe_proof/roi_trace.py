"""Verified, in-memory observation of a frozen SlotSPE evaluation forward.

No model, input, checkpoint or patient file is loaded by this module.  Calling
``capture_verified`` does run two model forwards, so the caller must separately
obtain authorization for any real-patient use.  Nothing is saved or printed.

``row_manifest`` is caller-supplied provenance, *not* proof that coordinates or
feature-row order are correct.  It is checked for internal consistency and bound
to the returned token order; independent coordinate/source verification remains
required before interpreting cancer-region labels or drawing slide overlays.
"""

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
import re

from .problem_trace import NativeSlotTrace, replay_rng, replay_slots, rng_state


def _torch():
    import torch
    return torch


def _clone(value):
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported forward input/output type: {type(value).__name__}")


def _to_cpu(value):
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def row_manifest_sha256(row_manifest):
    """Hash caller-provided rows in order using UTF-8 canonical JSON."""
    encoded = json.dumps(list(row_manifest), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _equal(left, right, location):
    """Exact recursive comparison; reject nonfinite floating-point outputs."""
    torch = _torch()
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype:
            raise RuntimeError(f"{location} tensor type/dtype mismatch")
        if left.is_floating_point() or left.is_complex():
            if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
                raise RuntimeError(f"{location} contains nonfinite values")
        if left.shape != right.shape or left.device != right.device or not torch.equal(left, right):
            raise RuntimeError(f"{location} differs between native and traced forwards")
        return
    if type(left) is not type(right):
        raise RuntimeError(f"{location} output type mismatch")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise RuntimeError(f"{location} dictionary keys differ")
        for key in left:
            _equal(left[key], right[key], f"{location}.{key}")
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise RuntimeError(f"{location} sequence lengths differ")
        for index, (a, b) in enumerate(zip(left, right)):
            _equal(a, b, f"{location}[{index}]")
    elif left is None or isinstance(left, (str, bool, int, float)):
        if isinstance(left, float) and (not math.isfinite(left) or not math.isfinite(right)):
            raise RuntimeError(f"{location} contains nonfinite values")
        if left != right:
            raise RuntimeError(f"{location} differs between native and traced forwards")
    else:
        raise TypeError(f"unsupported forward output type: {type(left).__name__}")


def _hash_tensor(hasher, tensor):
    torch = _torch()
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise TypeError("only dense tensor states are supported")
    value = tensor.detach().cpu().contiguous()
    hasher.update(str(value.dtype).encode())
    hasher.update(str(tuple(value.shape)).encode())
    hasher.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())


def _state_digest(model):
    hasher = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        _hash_tensor(hasher, value)
    return hasher.hexdigest()


def _rng_digest(state):
    hasher = hashlib.sha256()
    _hash_tensor(hasher, state["cpu"])
    for key, value in sorted(state["cuda"].items()):
        hasher.update(str(key).encode())
        _hash_tensor(hasher, value)
    return hasher.hexdigest()


def _input_digests(batch):
    """Hash actual WSI and named omic tensors, including shape and dtype."""
    torch = _torch()
    wsi_hash = hashlib.sha256()
    _hash_tensor(wsi_hash, batch["x_wsi"])
    keys = sorted(key for key in batch if isinstance(key, str) and
                  (key in ("x_omic", "x_omics") or re.fullmatch(r"x_omic[1-9][0-9]*", key)))
    if not keys:
        raise ValueError("At least one named omic input tensor is required")
    omic_hash = hashlib.sha256()
    for key in keys:
        value = batch[key]
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError("Omic input tensors must be finite")
        omic_hash.update(key.encode("utf-8"))
        _hash_tensor(omic_hash, value)
    return wsi_hash.hexdigest(), omic_hash.hexdigest()


def _assert_rng_equal(left, right, location):
    torch = _torch()
    if left["cuda"].keys() != right["cuda"].keys() or not torch.equal(left["cpu"], right["cpu"]):
        raise RuntimeError(f"{location}: CPU/CUDA RNG state mismatch")
    if any(not torch.equal(left["cuda"][key], right["cuda"][key]) for key in left["cuda"]):
        raise RuntimeError(f"{location}: CUDA RNG state mismatch")


def _validate_rows(batch, row_manifest, pathway_names):
    torch = _torch()
    if not isinstance(batch, dict) or "x_wsi" not in batch:
        raise ValueError("batch must be a mapping containing x_wsi")
    wsi = batch["x_wsi"]
    if not isinstance(wsi, torch.Tensor) or wsi.ndim != 3 or wsi.shape[0] != 1 or wsi.shape[1] < 1:
        raise ValueError("x_wsi must have shape [1, positive tokens, features]")
    if not bool(torch.isfinite(wsi).all()):
        raise ValueError("x_wsi must be finite")
    if batch.get("omic_missing", None) is not False:
        raise ValueError("omic_missing must explicitly be False")
    if not isinstance(row_manifest, (list, tuple)) or len(row_manifest) != wsi.shape[1]:
        raise ValueError("row_manifest length must match the WSI token count")
    if not isinstance(pathway_names, (list, tuple)) or not pathway_names:
        raise ValueError("ordered pathway names are required")
    if any(not isinstance(name, str) or not name.strip() for name in pathway_names):
        raise ValueError("pathway names must be nonempty strings")
    if len(set(pathway_names)) != len(pathway_names):
        raise ValueError("pathway names must be unique")

    seen = set()
    padding_indices = []
    for index, row in enumerate(row_manifest):
        if (not isinstance(row, dict) or not {"token_index", "is_real", "slide_id", "feature_row"} <= row.keys()
                or type(row.get("token_index")) is not int or row["token_index"] != index):
            raise ValueError("row_manifest token_index must be contiguous and ordered")
        if type(row.get("is_real")) is not bool:
            raise ValueError("each row must explicitly declare boolean is_real")
        slide = row.get("slide_id")
        feature_row = row.get("feature_row")
        if row["is_real"]:
            if not isinstance(slide, str) or not slide.strip() or type(feature_row) is not int or feature_row < 0:
                raise ValueError("real rows require a slide_id and nonnegative feature_row")
            identity = (slide, feature_row)
            if identity in seen:
                raise ValueError("a real slide/feature row cannot occur twice")
            seen.add(identity)
        elif slide is not None or feature_row is not None:
            raise ValueError("padding rows cannot have slide or feature provenance")
        else:
            padding_indices.append(index)
        try:
            json.dumps(row, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("row manifest must be JSON-compatible and finite") from exc
    if not seen:
        raise ValueError("at least one real WSI row is required")
    if padding_indices and bool(torch.count_nonzero(wsi[0, padding_indices])):
        raise ValueError("padding rows must be actual zero-valued WSI input rows")


class _OutputObserver:
    """Observe slot outputs and decoder outputs without modifying them."""

    def __init__(self, model):
        self.handles = []
        self.slots = {"wsi": [], "omic": []}
        self.decoders = {"wsi": [], "omic": []}
        try:
            for modality in ("wsi", "omic"):
                slot = getattr(model, "slot_attention_" + modality)
                decoder = getattr(model, "slot_decoder_" + modality)

                def slot_hook(_module, _args, output, name=modality):
                    self.slots[name].append(_clone(output))

                def decoder_hook(_module, _args, output, name=modality):
                    self.decoders[name].append(_clone(output))

                self.handles.append(slot.register_forward_hook(slot_hook))
                self.handles.append(decoder.register_forward_hook(decoder_hook))
        except Exception:
            self.close()
            raise

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()
        return False


def _run(model, batch, entry, traced):
    torch = _torch()
    with replay_rng(entry), torch.no_grad(), ExitStack() as stack:
        observer = stack.enter_context(_OutputObserver(model))
        trace = stack.enter_context(NativeSlotTrace(model)) if traced else None
        inputs = _clone(batch)
        unchanged = _clone(inputs)
        output = _clone(model(**inputs))
        _equal(unchanged, inputs, "forward input")
        after = rng_state()
        return output, observer.slots, observer.decoders, (trace.records if trace else None), after


def _logits(output):
    torch = _torch()
    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, (tuple, list)) and output:
        logits = output[0]
    elif isinstance(output, dict) and "logits" in output:
        logits = output["logits"]
    else:
        raise ValueError("forward output must expose logits as tensor, first item, or logits key")
    if not isinstance(logits, torch.Tensor) or logits.ndim < 1 or logits.shape[0] != 1:
        raise ValueError("logits must be a batch-one tensor")
    return logits.detach().cpu().clone()


def capture_verified(model, batch, *, row_manifest, pathway_names, source_root=None):
    """Return exactly verified dual-modality observations without saving files.

    Inputs: eval-mode model, batch-one native keyword arguments, ordered pathway
    names, and one row manifest entry per WSI token. Required row fields are
    ``token_index``, ``is_real``, ``slide_id`` and ``feature_row``; padding has
    null slide/feature provenance. The function does not validate real-world
    identity, hashes, coordinate scales or feature-extraction provenance.

    Tensors in the result are detached CPU tensors, including raw per-head
    attention and RNG states. They must be treated as private data when input is
    real. ``checks`` and the caller's row/pathway metadata are JSON-friendly.
    """
    torch = _torch()
    purity = None
    if source_root is not None:
        from .native_audit import model_purity
        purity = model_purity(model, source_root)
    _validate_rows(batch, row_manifest, pathway_names)
    wsi_input_hash, omic_input_hash = _input_digests(batch)
    if any(module.training for module in model.modules()):
        raise ValueError("the whole model must already be in eval mode")
    if any(value.device.type not in ("cpu", "cuda") for value in model.state_dict().values()):
        raise ValueError("only CPU and CUDA model states are supported")
    before_state = _state_digest(model)
    entry = rng_state()
    plain, plain_slots, plain_decoders, _, plain_after = _run(model, batch, entry, False)
    if _state_digest(model) != before_state:
        raise RuntimeError("model state changed during native forward")
    traced, traced_slots, traced_decoders, records, traced_after = _run(model, batch, entry, True)
    if _state_digest(model) != before_state:
        raise RuntimeError("model state changed during traced forward")
    _equal(plain, traced, "complete forward output")
    _assert_rng_equal(plain_after, traced_after, "post-forward RNG")
    _assert_rng_equal(entry, rng_state(), "outer RNG restoration")

    output = {}
    for modality in ("wsi", "omic"):
        if any(len(group[modality]) != 1 for group in (plain_slots, traced_slots, plain_decoders, traced_decoders)):
            raise RuntimeError(f"expected exactly one {modality} slot and decoder call")
        _equal(plain_slots[modality], traced_slots[modality], modality + " slots")
        _equal(plain_decoders[modality], traced_decoders[modality], modality + " decoder")
        record = records.get(modality)
        if record is None or not record["attentions"]:
            raise RuntimeError(f"{modality} slot attention trace is missing")
        _equal(plain_slots[modality][0], record["final"], modality + " captured final slots")
        # A hook can be non-invasive while still computing the wrong attention.
        # Replay the actual captured input/initial slots with the native equations
        # and require exact agreement with every observed iteration and output.
        with torch.no_grad():
            replay = replay_slots(getattr(model, "slot_attention_" + modality),
                                  record["inputs"], record["initial"])
        _equal(record["states"], replay["states"], modality + " replay states")
        _equal(record["attentions"], replay["attentions"], modality + " replay attentions")
        _equal(record["final"], replay["final"], modality + " replay final")
        decoder = traced_decoders[modality][0]
        if not isinstance(decoder, (tuple, list)) or len(decoder) != 3:
            raise RuntimeError(f"{modality} decoder must return logits, gates, hard mask")
        gates, keep = decoder[1], decoder[2]
        slots = record["final"]
        attention = torch.stack(record["attentions"], dim=0)
        expected_tokens = batch["x_wsi"].shape[1] if modality == "wsi" else len(pathway_names)
        if (not isinstance(gates, torch.Tensor) or not isinstance(keep, torch.Tensor)
                or gates.shape != keep.shape or keep.shape != slots.shape[:2]
                or slots.shape[0] != 1 or attention.ndim != 5
                or attention.shape[1] != 1 or attention.shape[3] != slots.shape[1]
                or attention.shape[4] != expected_tokens):
            raise RuntimeError(f"{modality} attention/slot/decoder dimensions do not match")
        if not bool(torch.isfinite(attention).all()) or not bool(torch.isfinite(gates).all()):
            raise RuntimeError(f"{modality} attention or gates are nonfinite")
        if bool((attention < 0).any()) or bool((gates < 0).any()) or not bool(((keep == 0) | (keep == 1)).all()):
            raise RuntimeError(f"{modality} has invalid attention, gate or hard mask values")
        totals = attention.sum(dim=-1)
        if not torch.allclose(totals, torch.ones_like(totals), atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"{modality} attention is not normalized across tokens")
        output[modality] = {
            "attention_by_iteration": attention[:, 0].detach().cpu().clone(),
            "last_attention_by_head": attention[-1, 0].detach().cpu().clone(),
            "slot_states": torch.stack(record["states"], dim=0)[:, 0].detach().cpu().clone(),
            "slots": slots[0].detach().cpu().clone(),
            "slot_gate": gates[0].detach().cpu().clone(),
            "selected_mask": keep[0].detach().cpu().clone(),
            "selected_indices": (keep[0] > 0).nonzero().flatten().cpu().tolist(),
        }
    _assert_rng_equal(entry, rng_state(), "outer RNG after replay")
    if _state_digest(model) != before_state:
        raise RuntimeError("model state changed during slot replay")
    if purity is not None:
        after_purity = model_purity(model, source_root)
        if after_purity["source_sha256"] != purity["source_sha256"]:
            raise RuntimeError("native source changed during observation")
    return {
        "schema": "slotspe_roi_verified_trace_v1",
        "logits": _logits(traced),
        "native_output": _to_cpu(traced),
        "modalities": output,
        "row_manifest": deepcopy(list(row_manifest)),
        "row_manifest_sha256": row_manifest_sha256(row_manifest),
        "pathway_names": list(pathway_names),
        "rng_entry": {"cpu": entry["cpu"].cpu().clone(),
                      "cuda": {key: value.cpu().clone() for key, value in entry["cuda"].items()}},
        "checks": {"entire_forward_output_exact": True, "slot_and_decoder_output_exact": True,
                   "slot_replay_states_and_attention_exact": True,
                   "post_forward_rng_exact": True, "outer_rng_restored": True,
                   "model_state_unchanged": True, "model_state_sha256": before_state,
                   "wsi_input_sha256": wsi_input_hash, "omic_input_sha256": omic_input_hash,
                   "rng_entry_sha256": _rng_digest(entry),
                   "row_manifest_provenance": "caller_supplied_internal_consistency_only",
                   "rng_scope": "torch_cpu_and_already_initialized_cuda"},
        "native_purity": None if purity is None else {"checks": purity["checks"],
                            "source_sha256": purity["source_sha256"]},
    }
