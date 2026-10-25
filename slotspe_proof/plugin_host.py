"""One native host wrapper for the nine declared plugin presets."""
from __future__ import annotations

import torch
from torch import nn

from .plugin_queue import PRESETS
from .prediction_slots import prediction_slot_outputs
from .slot_bottleneck import SlotVIBHost
from .slot_disentanglement import SharedPrivateSlots, content_set_loss, separation_terms
from .slot_regularization import slot_gram_loss
from .slot_relations import SlotRelationResidual


class SlotPluginHost(nn.Module):
    def __init__(self, backbone, preset, dimension=256):
        super().__init__()
        if preset not in PRESETS:
            raise ValueError("unknown plugin preset")
        self.preset = preset
        mode = "mean" if preset == "V1_mean" else "stochastic" if preset in {"V1_noise", "V1_vib"} else "off"
        self.core = SlotVIBHost(backbone, dimension, mode)
        self.parts = nn.ModuleDict()
        self.relations = nn.ModuleDict()
        if preset.startswith(("V2_", "V3_")):
            for index, name in enumerate(("wsi", "omic")):
                self.parts[name] = SharedPrivateSlots(dimension=dimension, seed=731 + index * 104729)
                if preset.startswith("V3_"):
                    self.relations[name] = SlotRelationResidual(output_dim=dimension,
                        mode="relation" if preset == "V3_relation" else "capacity", seed=901 + index * 104729)
        self.last_records = None
        self._active = False

    @property
    def backbone(self):
        return self.core.backbone

    def forward(self, *, plugin_seed=None, relations_enabled=True, **kwargs):
        if self._active:
            raise RuntimeError("reentrant plugin forward is unsupported")
        self.last_records = None
        if kwargs.get("omic_missing") is not False:
            raise ValueError("plugin study requires explicit complete modalities")
        self._active = True
        try:
            if self.preset.startswith("V1_"):
                result = self.core(vib_seed=plugin_seed, **kwargs)
                records = dict(self.core.last_slots)
                records["kl"] = self.core.kl_loss()
            else:
                def transform(name, output):
                    if not self.parts:
                        return None, {"native": output, "latent": output, "mean": output}
                    latent, record = self.parts[name](output)
                    record.update(native=output, base_latent=latent)
                    if self.relations:
                        latent = self.relations[name](latent, record["shared"], record["private"], enabled=relations_enabled)
                    record.update(latent=latent, mean=latent)
                    return latent, record
                with prediction_slot_outputs(self.backbone, transform) as records:
                    result = self.backbone(**kwargs)
                records["kl"] = next(self.parameters()).new_zeros(())
            self.last_records = records
            return result
        finally:
            self._active = False

    def added_losses(self, records, targets=None):
        """One logical batch, explicitly joining patient graphs for HSIC.

        Passing eight slots from one patient does not satisfy the batch axis.
        The caller controls distinct patient identities in each logical batch.
        """
        if not records:
            raise ValueError("nonempty logical patient batch required")
        zero = records[0]["wsi"]["latent"].new_zeros(())
        gram, content, hsic, variance = zero, zero, zero, zero
        kl = torch.stack([item["kl"] for item in records]).mean()
        if self.preset == "B1_gram":
            gram = torch.stack([slot_gram_loss(item[name]["native"]) for item in records
                                for name in ("wsi", "omic")]).mean()
        if self.parts:
            if targets is None or len(targets) != len(records):
                raise ValueError("one fixed content target map per patient required")
            content = torch.stack([content_set_loss(item[name]["content"], targets[index][name].unsqueeze(0)
                          if targets[index][name].ndim == 2 else targets[index][name])
                          for index, item in enumerate(records) for name in ("wsi", "omic")]).mean()
            if self.preset != "V2_content":
                separated = [separation_terms(torch.cat([item[name]["shared"] for item in records]),
                             torch.cat([item[name]["private"] for item in records])) for name in ("wsi", "omic")]
                hsic = torch.stack([item["hsic"] for item in separated]).mean()
                variance = torch.stack([item["variance"] for item in separated]).mean()
        total = 10 * gram + (.001 * kl if self.preset == "V1_vib" else zero) + content + .1 * hsic + .1 * variance
        return {"added": total, "gram": gram, "kl_mean_per_dimension": kl,
                "content": content, "hsic": hsic, "variance": variance}
