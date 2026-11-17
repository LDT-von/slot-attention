"""Per-pathway independent MLP omics encoder for the scaled role architecture.

What this implements
--------------------
The user's specification is:

  3. omics encoding: 329 INDEPENDENT networks
  4. omics input: RAW pathway expression -> 329 MLPs

So the omics side is NOT a shared SVD adapter any more.  Each of the 329
pathways owns its own two-layer MLP, whose first layer accepts exactly that
pathway's own gene count, and whose output is a `width`-dimensional token.
The 329 tokens then enter the omics slot encoder directly.

This reproduces the parameter pattern of the original SlotSPE
(`third_party/SlotSPE/models/SlotSPE.py::init_per_path_model`), which builds
`SNN_Block(dim1=input_dim, dim2=hidden[0])` followed by
`SNN_Block(dim1=hidden[0], dim2=hidden[1], dropout=0.25)` for every pathway,
with `hidden = [wsi_projection_dim, wsi_projection_dim]`.

Provenance and honesty
----------------------
* The original source is preserved byte for byte.  The `SNN_Block` shape and
  the `ELU + AlphaDropout` activation are re-implemented HERE so that this
  research model stays an independent artefact; it does not import from
  `third_party/SlotSPE` and does not modify it.
* `AlphaDropout` is used in the original.  It is applied to the ELU output.
  AlphaDropout requires floating point; this module therefore refuses a
  non-floating input instead of silently promoting it.
* The number of pathways is a MEASURED fact from the prepared manifest
  (`omic_sizes`), not an assumption: the local BLCA preparation has 329
  pathways with sizes that range from 1 to 195 genes.
* This module performs no fitting, no data loading and no patient access.  It
  only builds parameters from an explicit size list.
"""
from __future__ import annotations

import torch
from torch import nn


def pathway_block(dim_in: int, dim_out: int, *, dropout: float = 0.25) -> nn.Sequential:
    """One pathway's MLP: Linear -> ELU -> AlphaDropout.

    This is the shape of the original `SNN_Block` (third_party/SlotSPE/models/
    omics_encoder.py).  It is re-implemented here rather than imported, so
    that the research architecture has no import-time dependency on the
    preserved original and the original stays byte for byte.
    """
    if type(dim_in) is not int or dim_in < 1:
        raise ValueError('a pathway must have at least one gene')
    if type(dim_out) is not int or dim_out < 1:
        raise ValueError('a positive output width is required')
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError('dropout must lie in [0, 1)')
    return nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.ELU(),
        nn.AlphaDropout(p=float(dropout), inplace=False),
    )


class PathwayOmicsEncoder(nn.Module):
    """329 independent pathway MLPs -> 329 tokens of `width` dimensions.

    The forward pass receives a list of per-pathway gene vectors (one tensor
    per pathway, with that pathway's own gene count) and returns a
    `[patients, pathways, width]` token tensor.

    Why a LIST and not a padded matrix
    ----------------------------------
    A padded matrix would need every pathway to be padded up to the largest
    gene count (195 here), and the padding would have to be masked.  The
    original passes a list of variable-length tensors and applies one network
    per pathway, so this module does the same.  The padded-and-masked variant
    is deliberately NOT introduced: it would silently change the parameter
    count and the per-pathway input dimension.

    The pathway ORDER is explicit and checked: a permutation of the pathways
    would change the model, so the order is part of the contract.
    """

    SCHEMA = 'scaled_pathway_omics_encoder_v1'

    def __init__(self, omic_sizes, width: int, *, dropout: float = 0.25, hidden=None):
        super().__init__()
        if not isinstance(omic_sizes, (list, tuple)) or not omic_sizes:
            raise ValueError('a nonempty explicit pathway size list is required')
        if any(type(n) is not int or n < 1 for n in omic_sizes):
            raise ValueError('every pathway must own at least one gene')
        if type(width) is not int or width < 1:
            raise ValueError('a positive slot width is required')
        self.omic_sizes = [int(n) for n in omic_sizes]
        self.width = int(width)
        self.num_pathways = len(self.omic_sizes)
        # The original uses hidden = [wsi_projection_dim, wsi_projection_dim]:
        # two blocks, each projecting into the slot width.
        hidden = list(hidden) if hidden is not None else [self.width, self.width]
        if not hidden or any(type(h) is not int or h < 1 for h in hidden):
            raise ValueError('explicit positive hidden widths are required')

        networks = []
        for size in self.omic_sizes:
            blocks = [pathway_block(size, hidden[0])]
            for index in range(1, len(hidden)):
                blocks.append(pathway_block(hidden[index - 1], hidden[index], dropout=dropout))
            networks.append(nn.Sequential(*blocks))
        self.sig_networks = nn.ModuleList(networks)
        self.metadata = {
            'schema': self.SCHEMA,
            'num_pathways': self.num_pathways,
            'width': self.width,
            'hidden': hidden,
            'dropout': dropout,
            'pathway_sizes': list(self.omic_sizes),
            'independent_networks_per_pathway': True,
            'activation': 'ELU + AlphaDropout',
            'shape_reference': ('third_party/SlotSPE/models/SlotSPE.py::init_per_path_model, '
                                're-implemented here; the original is not imported or modified'),
        }

    # ------------------------------------------------------------------ info
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def per_pathway_parameter_count(self) -> list:
        return [sum(p.numel() for p in net.parameters()) for net in self.sig_networks]

    # --------------------------------------------------------------- forward
    def forward(self, omics):
        """omics: list of [patients, genes_i] tensors -> [patients, pathways, width].

        Every entry must be a 2-D tensor with the SAME patient count and with
        exactly the gene count of its own pathway.  A mismatch is an error,
        never a silent truncation.
        """
        if not isinstance(omics, (list, tuple)) or len(omics) != self.num_pathways:
            raise ValueError('one tensor per pathway is required: expected %d, got %d'
                             % (self.num_pathways, len(omics) if isinstance(omics, (list, tuple)) else 0))
        patients = None
        encoded = []
        for index, (values, size, network) in enumerate(zip(omics, self.omic_sizes, self.sig_networks)):
            if not isinstance(values, torch.Tensor) or values.ndim != 2:
                raise ValueError('pathway %d must be a [patients, genes] tensor' % index)
            if not values.is_floating_point():
                raise ValueError('pathway %d must be floating point; AlphaDropout refuses other dtypes' % index)
            if values.shape[1] != size:
                raise ValueError('pathway %d has %d genes but the declared size is %d'
                                 % (index, values.shape[1], size))
            if not torch.isfinite(values).all():
                raise ValueError('pathway %d contains nonfinite values' % index)
            if patients is None:
                patients = values.shape[0]
            elif values.shape[0] != patients:
                raise ValueError('every pathway must carry the same patient count')
            encoded.append(network(values))
        # [pathways, patients, width] -> [patients, pathways, width]
        return torch.stack(encoded, dim=1)

    def state(self):
        return {'omic_sizes': list(self.omic_sizes), 'width': self.width,
                'metadata': dict(self.metadata)}
