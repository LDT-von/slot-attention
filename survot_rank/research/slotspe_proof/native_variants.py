"""Seven native SlotSPE research arms, separate from historical DCT configs.

The original allocation registry remains unchanged for reproducible S0--S4
receipts. This registry adds the two explicitly requested redundancy methods.
It does not schedule or execute patient work.
"""
from .native_allocation import NativeAllocation, VARIANTS as ASSIGNMENT_VARIANTS
from .native_pruning import NativePruning, PruningConfig


ALL_VARIANTS = {
    **{name: {"family": "allocation", **modes} for name, modes in ASSIGNMENT_VARIANTS.items()},
    "S5": {"family": "pruning", "wsi": "dias", "omic": "dias"},
    "S6": {"family": "pruning", "wsi": "metaslot", "omic": "metaslot"},
}


def make_native_adapter(model, variant, *, config=None, banks=None,
                        noise_seed=123, capture=True):
    """Construct a reversible adapter; caller still explicitly enters it."""
    if variant not in ALL_VARIANTS:
        raise ValueError(f"unknown native SlotSPE variant: {variant}")
    if ALL_VARIANTS[variant]["family"] == "allocation":
        if banks is not None or config is not None:
            raise ValueError("S0--S4 use the unchanged allocation configuration")
        return NativeAllocation(model, variant, noise_seed=noise_seed, capture=capture)
    return NativePruning(model, variant, config=config or PruningConfig(),
                         banks=banks, noise_seed=noise_seed, capture=capture)
