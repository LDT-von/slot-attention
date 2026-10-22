"""Native SlotSPE pruning contracts, using invented tensors only.

These tests check compact active-slot execution and codebook lifecycle; they
do not read patient data or establish predictive or biological benefit.
"""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from survot_rank.research.slotspe_proof.model import build_model, make_args
from survot_rank.research.slotspe_proof.native_pruning import (
    NativePruning,
    PrototypeBank,
    PruningConfig,
    cosine_clusters,
    forward_compact,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_torch_runtime():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        yield
    torch.set_num_threads(previous_threads)


def make_fixture(training=True, batch_size=1):
    torch.manual_seed(9103)
    args = make_args([4, 4], encoding_dim=8)
    args.wsi_projection_dim = 16
    args.slot_num_wsi = args.slot_num_omics = 4
    args.slot_iters = 3
    model = build_model(ROOT / "third_party/SlotSPE", args, "original")
    model.train(training)
    batch = {
        "x_wsi": torch.randn(batch_size, 7, 8),
        "x_omic1": torch.randn(batch_size, 4),
        "x_omic2": torch.randn(batch_size, 4),
        "omic_missing": False,
        "y": torch.arange(batch_size) % 4,
        "c": torch.arange(batch_size) % 2,
    }
    return model, batch


def make_banks(config, dimension=16):
    generator = torch.Generator().manual_seed(181)
    return {
        modality: PrototypeBank(config).fit(
            torch.randn(32, dimension, generator=generator))
        for modality in ("wsi", "omic")
    }


def assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert_tree_equal(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_tree_equal(a, e)
    else:
        assert actual == expected


def test_cosine_clusters_merge_duplicates_and_preserve_orthogonal_contents():
    slots = torch.tensor([[1., 0., 0.], [3., 0., 0.],
                          [0., 1., 0.], [0., 4., 0.]])
    assert cosine_clusters(slots, .05) == [[0, 1], [2, 3]]
    assert cosine_clusters(slots[:1], .05) == [[0]]
    # Scaling a slot does not create an independent direction.
    assert cosine_clusters(slots * torch.tensor([[1.], [2.], [3.], [4.]]),
                           .05) == [[0, 1], [2, 3]]


def test_cosine_clusters_use_average_linkage_instead_of_chaining():
    angles = torch.deg2rad(torch.tensor([0., 12., 24.]))
    slots = torch.stack((angles.cos(), angles.sin()), dim=-1)
    # Adjacent distances are .02185, but the mean distance between
    # {0, 1} and {2} is .05415: a single-linkage chain is inappropriate.
    groups = cosine_clusters(slots, .03)
    assert sorted(len(group) for group in groups) == [1, 2]
    assert sorted(index for group in groups for index in group) == [0, 1, 2]
    assert all(not ({0, 2} <= set(group)) for group in groups)
    assert cosine_clusters(slots, .03) == groups


def test_prototype_lookup_quantizes_without_mutating_or_queuing():
    config = PruningConfig(prototype_count=2)
    vectors = torch.tensor([[1., 0.], [2., .01], [-1., 0.], [-2., -.01]])
    bank = PrototypeBank(config).fit(vectors)
    before = bank.prototypes.clone()
    slots = torch.tensor([[[1., 0.], [3., .01], [-1., 0.]]])
    indices, quantized = bank.lookup(slots)
    assert indices.shape == (1, 3)
    assert quantized.shape == slots.shape
    assert indices[0, 0] == indices[0, 1]
    assert indices[0, 0] != indices[0, 2]
    torch.testing.assert_close(quantized, bank.prototypes[indices])
    assert torch.equal(bank.prototypes, before)
    assert bank.updates == 0 and not bank.pending


def test_codebook_queue_is_detached_and_only_commit_changes_state():
    config = PruningConfig(prototype_count=2, ema_rate=.5, dead_timeout=2)
    bank = PrototypeBank(config).fit(torch.randn(32, 4))
    before = bank.prototypes.clone()
    queued = torch.tensor([[3., 2., 1., 0.], [0., 1., 2., 3.]],
                          requires_grad=True)
    bank.queue(queued)
    assert bank.pending and all(not item.requires_grad for item in bank.pending)
    assert torch.equal(bank.prototypes, before) and bank.updates == 0
    bank.commit()
    assert not bank.pending and bank.updates == 1
    assert torch.isfinite(bank.prototypes).all()
    assert not torch.equal(bank.prototypes, before)
    after = bank.prototypes.clone()
    bank.commit()
    assert bank.updates == 1 and torch.equal(bank.prototypes, after)


def test_codebook_roundtrip_keeps_lookup_and_update_history():
    config = PruningConfig(prototype_count=3, dead_timeout=1)
    bank = PrototypeBank(config).fit(torch.randn(32, 4))
    # Exercising stale-code refresh must remain finite even with a one-vector
    # queue; this case also catches empty-cluster divisions in EMA.
    bank.queue(torch.tensor([[1., 2., 3., 4.]]))
    bank.commit()
    clone = PrototypeBank(config)
    clone.load_state_dict(bank.state_dict())
    assert_tree_equal(clone.state_dict(), bank.state_dict())
    query = torch.randn(2, 4, 4)
    before = bank.lookup(query)
    after = clone.lookup(query)
    assert torch.equal(before[0], after[0])
    torch.testing.assert_close(before[1], after[1], atol=0, rtol=0)
    assert torch.isfinite(clone.prototypes).all()


@pytest.mark.parametrize("variant", ["S5", "S6"])
@pytest.mark.parametrize("training", [False, True])
def test_pruning_runs_full_native_head_and_reconstruction_with_compact_slots(
        variant, training):
    model, batch = make_fixture(training)
    config = PruningConfig(dias_distance=2., prototype_count=1)
    banks = make_banks(config) if variant == "S6" else None
    downstream_counts = {"wsi": [], "omic": []}
    hooks = []
    for modality in downstream_counts:
        decoder = getattr(model, "slot_decoder_" + modality)

        def record_count(_module, args, name=modality):
            slots = args[0]
            downstream_counts[name].append(slots.shape[1])
            assert slots.shape[1] == 1
            assert torch.isfinite(slots).all()
            assert not (slots == 0).all(-1).any()

        hooks.append(decoder.register_forward_pre_hook(record_count))
    try:
        with NativePruning(model, variant, config=config, banks=banks,
                           noise_seed=123, capture=True) as adapter:
            logits, auxiliary = model(**batch)
            main = model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None)
            assert logits.shape == (1, 4)
            assert torch.isfinite(logits).all() and torch.isfinite(main)
            assert torch.isfinite(torch.as_tensor(auxiliary))
            selector_parameters = tuple(parameter for name, parameter in model.named_parameters()
                                        if "pred_keep_slot" in name)
            selector_gradients = torch.autograd.grad(
                main, selector_parameters, allow_unused=True, retain_graph=True)
            assert selector_parameters and all(gradient is None for gradient in selector_gradients)
            (main + auxiliary).backward()
            assert all(torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters() if parameter.grad is not None)
            for modality in ("wsi", "omic"):
                value_gradient = getattr(model, "slot_attention_" + modality).to_v.weight.grad
                assert value_gradient is not None
                assert value_gradient.abs().sum() > 0
            assert downstream_counts == {"wsi": [1], "omic": [1]}
            expected_calls = [("wsi", 0), ("omic", 0)]
            if training:
                expected_calls += [("omic", 1)]
            assert [(r["modality"], r["call_index"]) for r in adapter.records] == expected_calls
            for record in adapter.records:
                assert record["mode"] == ("dias" if variant == "S5" else "metaslot")
                assert record["initial"].shape == record["pre_prune"].shape == (1, 4, 16)
                assert record["retained_before_refine"].shape == (1, 1, 16)
                assert record["final"].shape == (1, 1, 16)
                assert record["active_count"] == 1
                assert record["first_stage_iterations"] == (2 if variant == "S5" else 3)
                assert record["refine_iterations"] == (1 if variant == "S5" else 3)
                tokens = 7 if record["modality"] == "wsi" or record["call_index"] == 1 else 2
                assert record["source_input_count"] == tokens
                assert len(record["attentions"]) == record["refine_iterations"]
                for attention in record["attentions"]:
                    assert attention.shape == (1, 8, 1, tokens)
                    assert torch.isfinite(attention).all()
                assert not record["final"].requires_grad
                if variant == "S5":
                    assert record["groups"] == [[0, 1, 2, 3]]
                else:
                    assert torch.as_tensor(record["prototype_indices"]).numel() == 4
    finally:
        for hook in hooks:
            hook.remove()


def test_s6_requires_initialized_per_modality_codebooks():
    model, _ = make_fixture(False)
    with pytest.raises(ValueError):
        with NativePruning(model, "S6", banks=None):
            pass
    with pytest.raises(ValueError):
        with NativePruning(model, "S6", banks={"wsi": PrototypeBank(PruningConfig())}):
            pass


def test_s6_rejects_mismatched_codebook_configuration_and_pending_input():
    model, _ = make_fixture(False)
    config = PruningConfig(prototype_count=2)
    for changed in ({"prototype_count": 3}, {"ema_rate": .25},
                    {"dead_timeout": 101}, {"kmeans_steps": 11}):
        banks = make_banks(config)
        banks["wsi"] = PrototypeBank(replace(config, **changed)).fit(torch.randn(32, 16))
        with pytest.raises(ValueError, match="codebook configuration"):
            NativePruning(model, "S6", config=config, banks=banks)
    banks = make_banks(config)
    banks["omic"].queue(torch.randn(1, 16))
    with pytest.raises((ValueError, RuntimeError), match="pending"):
        NativePruning(model, "S6", config=config, banks=banks)


def test_s6_exception_discards_uncommitted_vectors_and_preserves_committed_state():
    model, batch = make_fixture(True)
    config = PruningConfig(prototype_count=2)
    banks = make_banks(config)
    committed = None
    with pytest.raises(RuntimeError, match="invented failed training batch"):
        with NativePruning(model, "S6", config=config, banks=banks) as adapter:
            model(**batch)
            adapter.commit_codebooks()
            committed = {name: bank.state_dict() for name, bank in banks.items()}
            assert all(bank.updates == 1 for bank in banks.values())
            model(**batch)
            assert all(bank.pending for bank in banks.values())
            raise RuntimeError("invented failed training batch")
    assert committed is not None
    for name, bank in banks.items():
        assert not bank.pending
        assert_tree_equal(bank.state_dict(), committed[name])


def test_compact_batch_failure_caught_inside_context_discards_queue_before_retry():
    model, batch = make_fixture(True, batch_size=2)
    config = PruningConfig(prototype_count=2)
    banks = make_banks(config)
    before = {name: bank.state_dict() for name, bank in banks.items()}
    calls = []

    def fail_second_example(_module, _args):
        calls.append(None)
        if len(calls) == 2:
            assert all(bank.pending for bank in banks.values())
            raise RuntimeError("invented second sample failure")

    with NativePruning(model, "S6", config=config, banks=banks,
                       capture=True) as adapter:
        handle = model.register_forward_pre_hook(fail_second_example)
        try:
            with pytest.raises(RuntimeError, match="invented second sample failure"):
                forward_compact(model, adapter, batch, native_seed=71, noise_seed=83)
        finally:
            handle.remove()
        for name, bank in banks.items():
            assert not bank.pending
            assert_tree_equal(bank.state_dict(), before[name])
        logits, auxiliary = forward_compact(
            model, adapter, batch, native_seed=71, noise_seed=83)
        assert torch.isfinite(logits).all() and torch.isfinite(auxiliary)
        expected_vectors = {
            name: sum(record["active_count"] for record in adapter.batch_records
                      if record["modality"] == name)
            for name in banks
        }
        adapter.commit_codebooks()
        for name, bank in banks.items():
            assert not bank.pending and bank.updates == 1
            assert bank.observations == before[name]["observations"] + expected_vectors[name]


@pytest.mark.parametrize("training", [False, True])
def test_s6_final_step_reaches_native_gaussian_initialization_without_full_unroll(training):
    model, batch = make_fixture(training)
    config = PruningConfig(prototype_count=2)
    banks = make_banks(config)
    grad_enabled = {"wsi": [], "omic": []}
    handles = []
    for modality in grad_enabled:
        gru = getattr(model, "slot_attention_" + modality).gru

        def observe_grad_mode(_module, _inputs, _output, name=modality):
            grad_enabled[name].append(torch.is_grad_enabled())

        handles.append(gru.register_forward_hook(observe_grad_mode))
    try:
        with NativePruning(model, "S6", config=config, banks=banks):
            logits, _ = model(**batch)
            main = model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None)
            parameters = []
            for modality in ("wsi", "omic"):
                attention = getattr(model, "slot_attention_" + modality)
                parameters.extend([attention.slots_mu, attention.slots_logsigma])
            gradients = torch.autograd.grad(main, parameters, allow_unused=True)
            for gradient in gradients:
                # The straight-through final query must train both learned
                # Gaussian parameters, not merely the Q/K/V projections.
                assert gradient is not None
                assert torch.isfinite(gradient).all()
                assert gradient.abs().sum() > 0
        # Three detached lookup iterations and two detached refinement steps
        # precede the only differentiable refinement. The reused omic module
        # must apply the same truncation during reconstruction.
        one_call = [False] * 5 + [True]
        assert grad_enabled["wsi"] == one_call
        assert grad_enabled["omic"] == one_call * (2 if training else 1)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("training", [False, True])
def test_s6_training_queues_then_explicitly_commits_but_eval_is_frozen(training):
    model, batch = make_fixture(training)
    config = PruningConfig(prototype_count=2)
    banks = make_banks(config)
    before = {name: bank.prototypes.clone() for name, bank in banks.items()}
    with NativePruning(model, "S6", config=config, banks=banks, capture=True) as adapter:
        model(**batch)
        for name, bank in banks.items():
            assert torch.equal(bank.prototypes, before[name]) and bank.updates == 0
            assert bool(bank.pending) is training
        adapter.commit_codebooks()
        for bank in banks.values():
            assert not bank.pending
            assert bank.updates == (1 if training else 0)


@pytest.mark.parametrize("variant", ["S5", "S6"])
def test_pruning_restores_original_forward_decoder_k_and_state_on_exception(variant):
    model, batch = make_fixture(False)
    # Native k can exceed the compact active count and must be temporarily
    # bounded at the decoder only, then restored after the intervention.
    model.slot_decoder_wsi.k = model.slot_decoder_omic.k = 3
    expected_methods = {name: module.forward for name, module in model.named_modules()}
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
    config = PruningConfig(dias_distance=2., prototype_count=1)
    banks = make_banks(config) if variant == "S6" else None
    with pytest.raises(RuntimeError, match="invented interruption"):
        with NativePruning(model, variant, config=config, banks=banks):
            model(**batch)
            assert model.slot_decoder_wsi.k <= 1
            assert model.slot_decoder_omic.k <= 1
            assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameter_ids
            raise RuntimeError("invented interruption")
    assert model.slot_decoder_wsi.k == model.slot_decoder_omic.k == 3
    assert all(module.forward == expected_methods[name]
               for name, module in model.named_modules())
    assert_tree_equal(model.state_dict(), state)


@pytest.mark.parametrize("variant", ["S5", "S6"])
def test_variable_slot_batch_is_explicitly_rejected_instead_of_zero_padded(variant):
    model, batch = make_fixture(False, batch_size=2)
    config = PruningConfig(prototype_count=2)
    banks = make_banks(config) if variant == "S6" else None
    with NativePruning(model, variant, config=config, banks=banks):
        with pytest.raises(ValueError):
            model(**batch)


@pytest.mark.parametrize("variant", ["S5", "S6"])
def test_compact_batch_helper_replays_per_sample_and_retains_training_graph(variant):
    model, batch = make_fixture(True, batch_size=2)
    config = PruningConfig(prototype_count=2)
    banks = make_banks(config) if variant == "S6" else None
    with NativePruning(model, variant, config=config, banks=banks,
                       capture=True) as adapter:
        first_logits, first_auxiliary = forward_compact(
            model, adapter, batch, native_seed=71, noise_seed=83)
        first_records = list(adapter.batch_records)
        second_logits, second_auxiliary = forward_compact(
            model, adapter, batch, native_seed=71, noise_seed=83)
        assert first_logits.shape == second_logits.shape == (2, 4)
        torch.testing.assert_close(first_logits, second_logits, atol=0, rtol=0)
        torch.testing.assert_close(first_auxiliary, second_auxiliary, atol=0, rtol=0)
        assert first_auxiliary.requires_grad
        direct_outputs, direct_losses = [], []
        for index in range(2):
            # Explicit pairing protocol documented by forward_compact.
            torch.manual_seed(71 + index * 10007)
            adapter.set_noise_seed(83 + index * 10007)
            item = {key: value[index:index + 1] if torch.is_tensor(value) else value
                    for key, value in batch.items()}
            direct_logits, direct_auxiliary = model(**item)
            direct_outputs.append(direct_logits)
            direct_losses.append(direct_auxiliary)
        torch.testing.assert_close(first_logits, torch.cat(direct_outputs), atol=0, rtol=0)
        torch.testing.assert_close(first_auxiliary, torch.stack(direct_losses).mean(), atol=0, rtol=0)
        assert len(first_records) == 6
        assert {record["sample_index"] for record in first_records} == {0, 1}
        for index in (0, 1):
            records = [record for record in first_records if record["sample_index"] == index]
            assert [(r["modality"], r["call_index"]) for r in records] == [
                ("wsi", 0), ("omic", 0), ("omic", 1)]
        main = model.loss_fn(first_logits, y=batch["y"], c=batch["c"], t=None) / 2
        (main + first_auxiliary).backward()
        assert all(torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters() if parameter.grad is not None)
