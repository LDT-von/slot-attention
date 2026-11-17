"""Scaled 16-slot common/private role architecture.

A NEW research module.  The archived `common_private_synergy` components and
the archived matched35 / spread arms are NOT modified by this package.

The user's specification for this architecture:

  1. slot allocation 4 / 4 / 4 / 4 -> 16 slots in total
  2. slot width 256
  3. omics encoding by 329 INDEPENDENT networks
  4. omics input: RAW pathway expression -> 329 MLPs
  5. reconstruction heads: DEFERRED (no reconstruction loss here)
  6. batch size 8 (in the configuration)

Importing this package does not build a model, read patient data or start an
experiment.
"""
from .omics_encoder import PathwayOmicsEncoder, pathway_block
from .adapter import ScaledContentAdapter, fit_pathway_moments
from .role_model import ScaledRoleTargetModel

__all__ = [
    'PathwayOmicsEncoder',
    'pathway_block',
    'ScaledContentAdapter',
    'fit_pathway_moments',
    'ScaledRoleTargetModel',
]
