"""Portable, parameter-matched SlotSPE mechanism controls.

The ``original`` arm executes the explicitly selected source without repairs.
The two repair arms are engineering controls, not a transport method.  Only
state dictionaries are intended for serialization; source files are read only.
Importing this module does not import torch or change the import search path.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace


MODEL_SOURCE_FILES = (
    "models/SlotSPE.py",
    "models/slot_attention.py",
    "models/transformer.py",
    "models/omics_encoder.py",
    "utils/loss_func.py",
)
_SOURCE = None
_REPAIR_CLASSES = None
_LOAD_LOCK = threading.RLock()
ATTRIBUTION_ARMS = ("A", "B", "C")
PROOF_ARMS = ("original", "eval_fix", "readout_fix", *ATTRIBUTION_ARMS)


def make_args(omic_sizes, encoding_dim=1024):
    """Keep the original diagnostic architecture and loss settings."""
    return SimpleNamespace(
        omic_sizes=list(omic_sizes), n_classes=4, encoding_dim=encoding_dim,
        wsi_projection_dim=256, rna_format="Pathways", slot_num_wsi=8,
        slot_num_omics=8, slot_iters=10, temperature=0.01, topk_ratio=0.25,
        top_k_method="parallel_topk_st", bag_loss="nll_surv", alpha_surv=0.5,
        lambda_recon_loss=0.01,
    )


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _package(name, path):
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    return module


def _inside(path, root):
    try:
        Path(path).resolve().relative_to(root)
    except (ValueError, TypeError, OSError):
        return False
    return True


def _check_cached_source(root):
    """Refuse ambiguous absolute imports before exposing temporary aliases."""
    for name, module in tuple(sys.modules.items()):
        if not (name in ("models", "utils") or
                name.startswith(("models.", "utils."))):
            continue
        if module is None:
            raise RuntimeError(f"Conflicting import cache entry {name!r} is None")
        file = getattr(module, "__file__", None)
        paths = list(getattr(module, "__path__", ()))
        locations = [file] if file else paths
        if not locations or not all(_inside(p, root) for p in locations):
            raise RuntimeError(
                f"Conflicting cached module {name!r} from {file or paths!r}; "
                f"SlotSPE source is {str(root)!r}. Use a fresh process."
            )


def load_source(source_root):
    """Load one explicit source tree without lasting models/utils aliases.

    A process may use only one resolved source root.  Private module names keep
    source class identities available after temporary absolute-import aliases
    have been removed.  Existing foreign ``models`` or ``utils`` imports cause
    a failure rather than silently changing the baseline.
    """
    global _SOURCE
    root = Path(source_root).expanduser().resolve()
    with _LOAD_LOCK:
        if _SOURCE is not None:
            if root != _SOURCE.source_root:
                raise RuntimeError(
                    f"SlotSPE source is already locked to {_SOURCE.source_root}; "
                    f"cannot switch to {root} in the same process."
                )
            current = {p: _digest(root / p) for p in MODEL_SOURCE_FILES}
            if current != _SOURCE.source_hashes:
                raise RuntimeError("SlotSPE model source changed after loading")
            _check_cached_source(root)
            return _SOURCE
        missing = [p for p in MODEL_SOURCE_FILES if not (root / p).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete SlotSPE source {root}: {missing}")
        _check_cached_source(root)
        before_hashes = {p: _digest(root / p) for p in MODEL_SOURCE_FILES}

        # Import dependencies before temporarily exposing generic package names.
        # Nothing here imports, installs, or invokes the transport plugin.
        import torch  # noqa: F401
        import einops  # noqa: F401

        prefix = "_slotspe_proof_source_" + hashlib.sha256(
            str(root).encode("utf-8")
        ).hexdigest()[:16]
        private_names = []
        saved_aliases = {
            name: module for name, module in tuple(sys.modules.items())
            if name in ("models", "utils") or
            name.startswith(("models.", "utils."))
        }
        previous_bytecode = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            private_root = _package(prefix, root)
            sys.modules[prefix] = private_root
            private_names.append(prefix)
            for name in ("models", "utils"):
                package = _package(f"{prefix}.{name}", root / name)
                sys.modules[package.__name__] = package
                private_names.append(package.__name__)
                setattr(private_root, name, package)
                sys.modules[name] = package

            modules = {}
            # Source SlotSPE.py imports these siblings by absolute names.
            for relative in (
                "utils/loss_func.py", "models/slot_attention.py",
                "models/transformer.py", "models/omics_encoder.py",
                "models/SlotSPE.py",
            ):
                alias = relative[:-3].replace("/", ".")
                name = f"{prefix}.{alias}"
                spec = importlib.util.spec_from_file_location(name, root / relative)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load SlotSPE module {root / relative}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                private_names.append(name)
                sys.modules[alias] = module
                package_name, child = alias.rsplit(".", 1)
                setattr(sys.modules[package_name], child, module)
                spec.loader.exec_module(module)
                if Path(module.__file__).resolve() != (root / relative).resolve():
                    raise RuntimeError(f"Unexpected loaded source for {alias}")
                modules[alias] = module
            after_hashes = {p: _digest(root / p) for p in MODEL_SOURCE_FILES}
            if after_hashes != before_hashes:
                raise RuntimeError("SlotSPE source changed while loading")
            slots = modules["models.slot_attention"]
            model = modules["models.SlotSPE"]
            _SOURCE = SimpleNamespace(
                source_root=root, source_hashes=before_hashes,
                module_paths={key: str(Path(value.__file__).resolve())
                              for key, value in modules.items()},
                SlotSPE=model.SlotSPE, MoESlotDecoder=model.MoESlotDecoder,
                MultiHeadSlotAttention=slots.MultiHeadSlotAttention,
                gumbel_topk_st=slots.gumbel_topk_st,
                parallel_topk_st=slots.parallel_topk_st,
                modules=modules,
            )
        except BaseException:
            for name in private_names:
                sys.modules.pop(name, None)
            raise
        finally:
            for name in tuple(sys.modules):
                if (name in ("models", "utils") or
                        name.startswith(("models.", "utils."))):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_aliases)
            sys.dont_write_bytecode = previous_bytecode
        return _SOURCE


def selected_pool(x, keep):
    """Read only selected self outputs while preserving the ST gradient."""
    return (x * keep.unsqueeze(-1)).sum(1) / keep.sum(1, keepdim=True).clamp_min(1.)


def _repair_classes(source):
    global _REPAIR_CLASSES
    if _REPAIR_CLASSES is not None:
        return _REPAIR_CLASSES
    import torch
    from torch.nn import functional as F

    class ReproducibleSlotAttention(source.MultiHeadSlotAttention):
        def forward(self, inputs, num_slots=None):
            if self.training:
                return super().forward(inputs, num_slots=num_slots)
            b, n, d = inputs.shape
            n_s = self.num_slots if num_slots is None else num_slots
            if n_s != self._proof_eval_noise.shape[1]:
                raise ValueError("Diagnostic eval requires the configured slot count.")
            # Distinct fixed noise is independent of patient ID and batch order.
            slots = (self.slots_mu + self.slots_logsigma.exp() *
                     self._proof_eval_noise.to(inputs))
            slots = slots.expand(b, -1, -1)
            normalized = self.norm_input(inputs)
            k, v = map(self.split_heads,
                       (self.to_k(normalized), self.to_v(normalized)))
            for _ in range(self.iters):
                previous = slots
                q = self.split_heads(self.to_q(self.norm_slots(slots)))
                dots = torch.einsum("... i d, ... j d -> ... i j", q, k) * self.scale
                attn = F.normalize(dots.softmax(dim=-2) + self.eps, p=1, dim=-1)
                updates = torch.einsum("... j d, ... i j -> ... i d", v, attn)
                updates = self.combine_heads(self.merge_heads(updates))
                slots = self.gru(updates.reshape(-1, d),
                                 previous.reshape(-1, d)).reshape(b, n_s, d)
                slots = slots + self.mlp(self.norm_pre_ff(slots))
            return slots

    class StableSlotDecoder(source.MoESlotDecoder):
        def forward(self, slots):
            mapped = self.map(slots)
            logits = self.decoder(mapped)
            scores = self.pred_keep_slot(mapped).squeeze(-1)
            if self.training:
                if self.top_k_method == "gumbel_topk_st":
                    choose = source.gumbel_topk_st
                elif self.top_k_method == "parallel_topk_st":
                    choose = source.parallel_topk_st
                else:
                    raise ValueError(f"Invalid top_k_method: {self.top_k_method}")
                sampled, _ = choose(scores, k=self.k, temperature=self.temperature)
                # Original hard forward; temperature-1 surrogate avoids the
                # previously rejected numerically saturated gradient control.
                surrogate = self.k * (scores / 1.0).softmax(-1)
                keep = sampled.detach() + (surrogate - surrogate.detach())
            else:
                indices = scores.topk(self.k, dim=-1).indices
                keep = torch.zeros_like(scores).scatter_(1, indices, 1.)
            hard = keep.detach() > 0.5
            gate = (scores / self.temperature).masked_fill(
                ~hard, float("-inf")
            ).softmax(-1)
            gate = gate * keep
            gate = gate / gate.sum(-1, keepdim=True).clamp_min(
                torch.finfo(gate.dtype).tiny
            )
            return torch.einsum("bs,bsc->bc", gate, logits), gate, keep

    class ReadoutFixedSlotSPE(source.SlotSPE):
        def fuse_slots(self, slots_wsi, slots_omic, keep_wsi, keep_omic):
            # Cross remains the baseline dense branch; exclusion is only a
            # claim about the selected self-branch readout.
            cross = self.cross_attention(slots_wsi, slots_omic)
            intra_wsi = self.self_attention_wsi(
                slots_wsi, mask=keep_wsi.detach() > 0.5
            )
            intra_omic = self.self_attention_omic(
                slots_omic, mask=keep_omic.detach() > 0.5
            )
            arm = getattr(self, "_proof_arm", "readout_fix")
            if arm == "A":
                # Original mean over ALL self outputs, with the same decoder,
                # auxiliary graph and evaluation policy as B/C.
                pooled_wsi, pooled_omic = intra_wsi.mean(1), intra_omic.mean(1)
            else:
                # Detach ONLY the selection weights on the final fused-loss
                # path. The original keep tensors still train the auxiliary
                # decoder; B and C execute identical arithmetic/RNG calls.
                pool_wsi = keep_wsi.detach() if arm == "B" else keep_wsi
                pool_omic = keep_omic.detach() if arm == "B" else keep_omic
                pooled_wsi = selected_pool(intra_wsi, pool_wsi)
                pooled_omic = selected_pool(intra_omic, pool_omic)
            return torch.cat([cross.mean(1), pooled_wsi, pooled_omic], dim=1)

        def forward(self, **kwargs):
            x_wsi_proj = self.wsi_mlp(kwargs["x_wsi"])
            if self.args.rna_format != "Pathways":
                raise ValueError("This scoped proof control supports Pathways only.")
            x_omics = torch.stack([
                self.sig_networks[i](kwargs[f"x_omic{i+1}"])
                for i in range(self.num_pathways)
            ], dim=1)
            if not self.training and kwargs.get("omic_missing", False):
                x_omics = self.reconstruction_head_joint(
                    x_omics, self.slot_attention_omic(x_wsi_proj)
                )
            slots_wsi = self.slot_attention_wsi(x_wsi_proj)
            slots_omic = self.slot_attention_omic(x_omics)
            logits_wsi, _, keep_wsi = self.slot_decoder_wsi(slots_wsi)
            logits_omic, _, keep_omic = self.slot_decoder_omic(slots_omic)
            fused = self.fuse_slots(slots_wsi, slots_omic, keep_wsi, keep_omic)
            prediction = self.to_logits(fused)
            if not self.training:
                return prediction, 0.0
            target = x_omics.detach()
            recon_omic = self.reconstruction_head_omic(
                x_omics, slots_omic, mask=keep_omic.detach() > 0.5
            )
            recon_mse = F.mse_loss(recon_omic, target)
            from_wsi = self.slot_attention_omic(x_wsi_proj)
            recon_joint = self.reconstruction_head_joint(x_omics, from_wsi)
            recon_mse = recon_mse + F.mse_loss(recon_joint, target)
            recon_wsi = self.reconstruction_head_wsi(
                x_wsi_proj, slots_wsi, mask=keep_wsi.detach() > 0.5
            )
            recon_wsi = F.normalize(recon_wsi, dim=-1)
            target_wsi = F.normalize(x_wsi_proj, dim=-1)
            recon_loss = (recon_mse + 1 - F.cosine_similarity(
                recon_wsi, target_wsi, dim=-1
            ).mean())
            decoder_loss = (
                self.loss_fn(logits_wsi, y=kwargs["y"], c=kwargs["c"], t=None) +
                self.loss_fn(logits_omic, y=kwargs["y"], c=kwargs["c"], t=None)
            ) / len(kwargs["y"])
            return prediction, decoder_loss + self.args.lambda_recon_loss * recon_loss

    _REPAIR_CLASSES = SimpleNamespace(
        ReproducibleSlotAttention=ReproducibleSlotAttention,
        StableSlotDecoder=StableSlotDecoder, ReadoutFixedSlotSPE=ReadoutFixedSlotSPE,
    )
    return _REPAIR_CLASSES


def apply_repairs(net, arm, source_root=None):
    """Apply the historical controls without adding trainable parameters."""
    if arm not in PROOF_ARMS:
        raise ValueError(f"Unknown proof arm: {arm}")
    if _SOURCE is None and source_root is None:
        raise RuntimeError("Call load_source or provide an explicit source_root first")
    source = load_source(source_root if source_root is not None else _SOURCE.source_root)
    if not isinstance(net, source.SlotSPE):
        raise TypeError("Model is not from the explicitly loaded SlotSPE source")
    if arm == "original":
        if type(net) is not source.SlotSPE:
            raise ValueError("The original arm must be an unrepaired source model")
        return net
    import torch

    classes = _repair_classes(source)
    if arm == "readout_fix" or arm in ATTRIBUTION_ARMS:
        net.__class__ = classes.ReadoutFixedSlotSPE
        net._proof_arm = arm
    elif type(net) is not source.SlotSPE:
        raise ValueError("eval_fix must retain the original SlotSPE readout")
    for index, name in enumerate(("slot_attention_wsi", "slot_attention_omic")):
        module = getattr(net, name)
        module.__class__ = classes.ReproducibleSlotAttention
        generator = torch.Generator(device="cpu").manual_seed(160018 + index)
        noise = torch.randn(1, module.num_slots, module.dim, generator=generator)
        module.register_buffer("_proof_eval_noise", noise, persistent=False)
    for name in ("slot_decoder_wsi", "slot_decoder_omic"):
        getattr(net, name).__class__ = classes.StableSlotDecoder
    return net


def build_model(source_root, args, arm):
    """Create original/eval_fix/readout_fix from one explicitly bound source."""
    if arm not in PROOF_ARMS:
        raise ValueError(f"Unknown proof arm: {arm}")
    source = load_source(source_root)
    return apply_repairs(source.SlotSPE(args), arm, source.source_root)
