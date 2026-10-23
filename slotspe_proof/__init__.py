"""SlotSPE research harness (top-level package).

This package holds the independent SlotSPE diagnostic and A/B/C experiment code
that was historically located under ``survot_rank/research/slotspe_proof/``.
The name ``survot_rank`` is a legacy project name and was retained only to keep
historical import paths working; the canonical import path for new code is
``slotspe_proof``.

Modules moved here keep their original relative imports (``from .model import ...``)
unchanged. ``survot_rank.research.slotspe_proof`` is now a thin compatibility
shim that re-exports the very same module objects, so the old path and the new
path never load two divergent copies of the code.
"""
