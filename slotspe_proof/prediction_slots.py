"""One scoped hook implementation for native prediction-slot adapters."""
from contextlib import contextmanager

import torch


@contextmanager
def prediction_slot_outputs(backbone, transform):
    """Transform call zero per modality; leave joint reconstruction untouched.

    ``transform(name, output)`` returns (replacement, record). A None
    replacement observes only. Shape/device/dtype changes are prohibited.
    Call-order drift and failures clear the records and remove every hook.
    """
    records, order, handles = {}, [], []

    def hook_for(name):
        def hook(module, inputs, output):
            first = name not in order
            order.append(name)
            if not first:
                return None
            if not torch.is_tensor(output) or output.ndim != 3:
                raise ValueError("native prediction slots require [B,K,D]")
            replacement, record = transform(name, output)
            if replacement is not None and (not torch.is_tensor(replacement) or
                    replacement.shape != output.shape or replacement.device != output.device or
                    replacement.dtype != output.dtype):
                raise ValueError("slot adapter changed native shape/device/dtype")
            records[name] = record
            return replacement
        return hook

    try:
        for name in ("wsi", "omic"):
            handles.append(getattr(backbone, "slot_attention_" + name).register_forward_hook(hook_for(name)))
        yield records
        expected = ["wsi", "omic", "omic"] if backbone.training else ["wsi", "omic"]
        if order != expected:
            raise RuntimeError("prediction slot calls did not match the two-modality contract")
    except BaseException:
        records.clear()
        raise
    finally:
        for handle in handles:
            handle.remove()
