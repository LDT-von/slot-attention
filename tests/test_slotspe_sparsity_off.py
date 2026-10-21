"""Synthetic checks of selection controls, separate from real-data execution."""
import pytest
import torch
from survot_rank.research.slotspe_proof.sparsity_ablation import native_dense_decoder, original_decoder_from_mask, mean_pair_cosine


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k = 2
        self.temperature = 1
        self.map = torch.nn.Identity()
        self.pred_keep_slot = torch.nn.Linear(3, 1)
        self.decoder = torch.nn.Linear(3, 2)

    def forward(self, slots):
        # Draw despite K=S to check that the outer RNG is restored.
        score = self.pred_keep_slot(slots).squeeze(-1)
        noise = torch.rand_like(score)
        keep = torch.zeros_like(score).scatter_(1, (score + noise).topk(self.k).indices, 1)
        logits, gate, _, _ = original_decoder_from_mask(self, slots, keep)
        return logits, gate, keep


def test_dense_uses_all_slots_and_restores_k_and_rng():
    decoder = TinyDecoder().eval()
    slots = torch.randn(2, 8, 3)
    before = torch.get_rng_state().clone()
    logits, gate, keep = native_dense_decoder(decoder, slots, 17)
    assert torch.equal(before, torch.get_rng_state())
    assert decoder.k == 2
    assert torch.equal(keep, torch.ones_like(keep))
    assert torch.allclose(gate.sum(-1), torch.ones(2))
    other, _, _ = native_dense_decoder(decoder, slots, 29)
    assert torch.equal(logits, other)


def test_sparse_and_dense_read_different_expert_predictions():
    decoder = TinyDecoder().eval()
    with torch.no_grad():
        decoder.pred_keep_slot.weight.zero_(); decoder.pred_keep_slot.bias.zero_()
        decoder.decoder.weight.copy_(torch.tensor([[1., 0, 0], [0, 1., 0]])); decoder.decoder.bias.zero_()
    slots = torch.zeros(1, 8, 3); slots[0, 2:, 0] = 8
    keep = torch.tensor([[1., 1., 0, 0, 0, 0, 0, 0]])
    sparse = original_decoder_from_mask(decoder, slots, keep)[0]
    dense = native_dense_decoder(decoder, slots, 3)[0]
    assert sparse[0, 0] == 0
    assert dense[0, 0].item() == pytest.approx(6)


def test_pair_cosine_distinguishes_duplicate_and_distinct_slots():
    duplicates = torch.ones(2, 8, 3)
    assert torch.allclose(mean_pair_cosine(duplicates), torch.ones(2, dtype=torch.float64))
    distinct = torch.eye(3)[None]
    assert mean_pair_cosine(distinct).item() == 0
