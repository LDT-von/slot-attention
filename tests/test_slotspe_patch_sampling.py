"""The patch sampler must stay equivalent to the original published loader.

third_party/SlotSPE/dataset/dataset_survival.py lines 285-292 draw
``min(num_patches, available)`` patches without replacement, sort the indices
and then zero-pad the remainder up to ``num_patches``. The original never
removes a patient for owning fewer patches than the budget, and it draws
inside __getitem__, so every epoch sees a different subset.

These checks compare the harness sampler against a direct transcription of
that code. They use invented tensors and never read patient data.
"""
from __future__ import annotations

import numpy as np
import torch

from slotspe_proof.patient import _sample_bag


def _original_sampler(wsi, num_patches, rng):
    """A direct transcription of the published sampling block."""
    n_samples = min(num_patches, wsi.size(0))
    idx = np.sort(rng.choice(wsi.size(0), n_samples, replace=False))
    out = wsi[idx, :]
    if n_samples < num_patches:
        out = torch.cat([out, torch.zeros(num_patches - n_samples, wsi.size(1))], dim=0)
    return out


def test_sampler_matches_published_loader_bitwise():
    generator = torch.Generator().manual_seed(51091)
    for total, budget in ((123, 4096), (2000, 4096), (4096, 4096), (9000, 4096), (100, 100)):
        wsi = torch.randn(total, 8, generator=generator)
        ours, _ = _sample_bag(wsi, budget, np, torch, 7)
        theirs = _original_sampler(wsi, budget, np.random.default_rng(7))
        assert ours.shape == theirs.shape == (budget, 8)
        assert torch.equal(ours, theirs)


def test_short_bag_is_zero_padded_and_keeps_every_real_patch():
    wsi = torch.randn(123, 8)
    bag, indices = _sample_bag(wsi, 4096, np, torch, 3)
    assert bag.shape == (4096, 8)
    assert len(indices) == 123
    # Every real patch is retained, and only the padding is zero.
    assert int(torch.count_nonzero(bag[:123])) == int(torch.count_nonzero(wsi))
    assert int(torch.count_nonzero(bag[123:])) == 0


def test_resampling_draws_different_subsets_but_a_fixed_seed_repeats():
    wsi = torch.randn(9000, 8)
    first, _ = _sample_bag(wsi, 4096, np, torch, None)
    np.random.seed(1)
    second, _ = _sample_bag(wsi, 4096, np, torch, None)
    np.random.seed(2)
    third, _ = _sample_bag(wsi, 4096, np, torch, None)
    assert not torch.equal(first, second)
    assert not torch.equal(second, third)
    fixed_a, _ = _sample_bag(wsi, 4096, np, torch, 0)
    fixed_b, _ = _sample_bag(wsi, 4096, np, torch, 0)
    assert torch.equal(fixed_a, fixed_b)


def test_no_patient_is_excluded_for_owning_fewer_patches_than_the_budget():
    # The harness must not treat a short bag as a missing patient. The reason
    # string that used to be raised is gone from the preparation code.
    from slotspe_proof import patient

    source = patient.__file__
    text = open(source, encoding="utf-8").read()
    assert "fewer_real_patches_than_configured_no_padding" not in text
