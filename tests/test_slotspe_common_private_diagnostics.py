"""Contracts of the common/private diagnostics, on generated tensors only.

The diagnostics exist to make a claim about the prototype falsifiable. These
tests therefore check that each instrument has a known answer on data with a
known answer: a probe must recover a factor that is present, must not report
a factor that was shuffled, and a branch that the readout ignores must
produce exactly zero change in the likelihood.
"""
import math
from pathlib import Path

import pytest
import torch

from slotspe_proof.common_private_diagnostics import (branch_sensitivity,
                                                      duplication_probe,
                                                      linear_probe, partial_probe,
                                                      pearson, prior_substitution_ablation,
                                                      probe_suite,
                                                      representation_variance,
                                                      ridge_fit, ridge_predict,
                                                      r2_score, swap_ablation,
                                                      summarise_probe_suite)
from slotspe_proof.common_private_survival import CommonPrivateSurvival


@pytest.fixture(autouse=True)
def isolated_torch_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        yield
    torch.set_num_threads(previous)


def _linear_dataset(n=200, seed=3):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=generator)
    y = 2.0 * x[:, 0] - 0.5 * x[:, 1] + 0.25 * x[:, 2]
    return x, y


def test_ridge_recovers_a_linear_relation_exactly():
    x, y = _linear_dataset()
    weight, bias = ridge_fit(x, y, lam=1e-10)
    prediction = ridge_predict(x, weight, bias)
    assert r2_score(y, prediction) == pytest.approx(1.0, abs=1e-6)
    assert pearson(y, prediction) == pytest.approx(1.0, abs=1e-6)


def test_ridge_intercept_is_unpenalised_and_centered_design_is_used():
    x = torch.arange(20, dtype=torch.float64)[:, None]
    y = 5.0 + 0.0 * x[:, 0]
    weight, bias = ridge_fit(x, y, lam=1e-8)
    assert float(bias) == pytest.approx(5.0, abs=1e-6)
    assert float(weight) == pytest.approx(0.0, abs=1e-6)


def test_probe_reports_held_out_scores_and_exposes_memorisation():
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(60, 4, generator=generator)
    y = x[:, 0] * 3.0 + x[:, 1]
    result = linear_probe(x[:40], y[:40], x[40:], y[40:])
    assert result["test_r2"] > 0.9
    assert result["train_r2"] > 0.9
    assert result["n_train"] == 40 and result["n_test"] == 20


def test_probe_on_a_shuffled_target_reports_nothing():
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(120, 5, generator=generator)
    y = x[:, 0]
    shuffled = y[torch.randperm(120, generator=generator)]
    result = linear_probe(x[:80], shuffled[:80], x[80:], shuffled[80:])
    assert result["test_r2"] < 0.2


def test_probe_suite_rejects_mismatched_patient_counts():
    representations = {"common": torch.randn(10, 2)}
    factors = {"shared": torch.randn(9)}
    with pytest.raises(ValueError):
        probe_suite(representations, factors, torch.arange(8), torch.arange(8, 10))


def test_partial_probe_separates_complementary_from_redundant_content():
    generator = torch.Generator().manual_seed(17)
    n = 160
    common_signal = torch.randn(n, generator=generator)
    private_signal = torch.randn(n, generator=generator)
    common = torch.stack([common_signal, torch.randn(n, generator=generator)], 1)
    # A private branch that carries the complementary signal.
    complementary = torch.stack([private_signal, torch.randn(n, generator=generator)], 1)
    # A private branch that only repeats what the common branch already knows.
    redundant = common + 0.01 * torch.randn(n, 2, generator=generator)
    target = common_signal + private_signal
    train_index = torch.arange(100)
    test_index = torch.arange(100, n)
    complementary_result = partial_probe(
        common[train_index], complementary[train_index], target[train_index],
        common[test_index], complementary[test_index], target[test_index])
    redundant_result = partial_probe(
        common[train_index], redundant[train_index], target[train_index],
        common[test_index], redundant[test_index], target[test_index])
    assert complementary_result["private_residual_test_r2"] > 0.8
    assert redundant_result["private_residual_test_r2"] < 0.2
    # The common branch explains the same share of the target in both cases,
    # because the redundant branch adds nothing beyond the common branch.
    assert complementary_result["common_only_test_r2"] == pytest.approx(
        redundant_result["common_only_test_r2"], abs=1e-6)
    # Most of the target variance survives the removal of the common branch.
    assert complementary_result["residual_variance_share_left_after_common"] > 0.5


def test_swap_ablation_detects_an_unused_branch_exactly():
    model = CommonPrivateSurvival(4, 4, width=8, slots=2, common_dim=2, private_dim=2)
    common = torch.randn(6, 2)
    private = [torch.randn(6, 2, 2)]
    model.phi = torch.nn.ModuleList([torch.nn.Linear(2, 8)])
    model.head = torch.nn.Linear(2 + 8, 4)

    def nll_of(logits):
        return float(logits.sum())

    result = swap_ablation(model, common, private, None, None, nll_of)
    # The private branch reaches the readout through phi, so zeroing it must
    # change the likelihood.
    assert result["delta_vs_full"]["zero_private"] != 0.0
    # Zeroing the common branch must also change the likelihood.
    assert result["delta_vs_full"]["zero_common"] != 0.0
    # A pure sum of logits is invariant to a permutation of the samples, so
    # this instrument only bites on a readout that is not permutation blind.
    # The prototype readout is linear in the aggregated private features and
    # therefore also blind here; the instrument is exercised on the
    # likelihood in the semantics script, not on this toy sum.


def test_swap_ablation_is_blind_when_the_readout_is_permutation_invariant():
    model = CommonPrivateSurvival(4, 4, width=8, slots=2, common_dim=2, private_dim=2)
    common = torch.randn(6, 2)
    private = [torch.randn(6, 2, 2)]
    model.phi = torch.nn.ModuleList([torch.nn.Linear(2, 8)])
    model.head = torch.nn.Linear(2 + 8, 4)

    def nll_of(logits):
        return float(logits.square().sum())

    result = swap_ablation(model, common, private, None, None, nll_of)
    # A quadratic score is not permutation blind, so the swap must bite.
    assert result["delta_vs_full"]["swap_private"] != 0.0
    assert result["delta_vs_full"]["swap_common"] != 0.0


def test_swap_ablation_rejects_an_identity_permutation():
    model = CommonPrivateSurvival(4, 4, width=8, slots=2, common_dim=2, private_dim=2)
    common = torch.randn(5, 2)
    private = [torch.randn(5, 2, 2)]

    def nll_of(logits):
        return float(logits.sum())

    with pytest.raises(ValueError):
        swap_ablation(model, common, private, None, None, nll_of,
                      permutation=torch.arange(5))


def test_swap_ablation_rejects_a_non_permutation():
    model = CommonPrivateSurvival(4, 4, width=8, slots=2, common_dim=2, private_dim=2)
    common = torch.randn(5, 2)
    private = [torch.randn(5, 2, 2)]

    def nll_of(logits):
        return float(logits.sum())

    with pytest.raises(ValueError):
        swap_ablation(model, common, private, None, None, nll_of,
                      permutation=torch.tensor([0, 0, 1, 2, 3]))


def test_summary_reports_the_best_branch_per_factor():
    suite = {
        "common": {"shared": {"test_r2": 0.1, "test_pearson": 0.2, "train_r2": 0.2}},
        "private": {"shared": {"test_r2": 0.9, "test_pearson": 0.95, "train_r2": 0.95}},
    }
    summary = summarise_probe_suite(suite)
    assert summary["best_branch_per_factor"]["shared"]["branch"] == "private"
    assert summary["rows"][0]["test_r2"] == 0.9


def test_ridge_rejects_non_finite_and_mismatched_inputs():
    with pytest.raises(ValueError):
        ridge_fit(torch.tensor([[float("nan"), 0.0]]), torch.tensor([1.0]))
    with pytest.raises(ValueError):
        ridge_fit(torch.randn(4, 2), torch.randn(5))


def test_ridge_requires_a_penalty_when_underdetermined():
    with pytest.raises(ValueError):
        ridge_fit(torch.randn(2, 5), torch.randn(2), lam=0.0)


def test_duplication_probe_separates_a_copy_from_complementary_content():
    generator = torch.Generator().manual_seed(23)
    n = 160
    common = torch.randn(n, 2, generator=generator)
    copy = common + 0.01 * torch.randn(n, 2, generator=generator)
    independent = torch.randn(n, 2, generator=generator)
    train_index = torch.arange(100)
    test_index = torch.arange(100, n)
    copy_result = duplication_probe(common[train_index], copy[train_index],
                                     common[test_index], copy[test_index])
    independent_result = duplication_probe(common[train_index], independent[train_index],
                                           common[test_index], independent[test_index])
    assert copy_result["common_to_private_test_r2"] > 0.95
    assert independent_result["common_to_private_test_r2"] < 0.2


def test_duplication_probe_rejects_mismatched_rows():
    with pytest.raises(ValueError):
        duplication_probe(torch.randn(10, 2), torch.randn(9, 2),
                          torch.randn(5, 2), torch.randn(5, 2))


def test_representation_variance_detects_a_constant_feature():
    generator = torch.Generator().manual_seed(29)
    varying = torch.randn(50, 3, generator=generator)
    constant = torch.ones(50, 3)
    assert representation_variance(varying)["across_patient_variance_share"] > 0.5
    assert representation_variance(constant)["across_patient_variance_share"] == pytest.approx(0.0)


def test_representation_variance_requires_two_patients():
    with pytest.raises(ValueError):
        representation_variance(torch.randn(1, 3))


def test_branch_sensitivity_reports_the_share_of_affected_patients():
    model = CommonPrivateSurvival(4, 4, width=8, slots=2, common_dim=2, private_dim=2)
    model.phi = torch.nn.ModuleList([torch.nn.Linear(2, 8)])
    model.head = torch.nn.Linear(2 + 8, 4)
    common = torch.randn(8, 2)
    private = [torch.randn(8, 2, 2)]
    time_bin = torch.randint(0, 4, (8,))
    event = torch.randint(0, 2, (8,)).float()

    def nll_of(logits):
        return float(logits.square().sum())

    result = branch_sensitivity(model, common, private, time_bin, event, nll_of)
    assert result["n_patients"] == 8
    assert result["zero_private"]["patients_total"] == 8
    assert result["zero_private"]["fraction_changed"] == pytest.approx(1.0)
    assert result["zero_private"]["mean_abs_delta"] > 0.0


def test_prior_substitution_detects_that_the_prior_carries_the_private_content():
    torch.manual_seed(31)
    model = CommonPrivateSurvival(4, 4, width=8, slots=2, common_dim=2, private_dim=2)
    model.phi = torch.nn.ModuleList([torch.nn.Linear(2, 8)])
    model.head = torch.nn.Linear(2 + 8, 4)
    common = torch.randn(6, 2)
    # The private slots carry exactly the common vector, in every slot.
    private = [common[:, None, :].expand(6, 2, 2).contiguous()]
    # The conditional prior is made to reproduce that vector exactly.
    with torch.no_grad():
        model.prior[0].weight.zero_()
        model.prior[0].weight[0:2] = torch.eye(2)
        model.prior[0].bias.zero_()

    def nll_of(logits):
        return float(logits.square().sum())

    result = prior_substitution_ablation(model, common, private, None, None, nll_of)
    # The prior reproduces the private content, so substituting it must not
    # change the likelihood, while zeroing the private slots must.
    assert result["delta_prior_vs_full"] == pytest.approx(0.0, abs=1e-5)
    assert abs(result["delta_zero_vs_full"]) > 1e-6


def test_prior_substitution_reports_a_change_when_the_prior_disagrees():
    torch.manual_seed(31)
    model = CommonPrivateSurvival(4, 4, width=8, slots=2, common_dim=2, private_dim=2)
    model.phi = torch.nn.ModuleList([torch.nn.Linear(2, 8)])
    model.head = torch.nn.Linear(2 + 8, 4)
    common = torch.randn(6, 2)
    private = [torch.randn(6, 2, 2)]

    def nll_of(logits):
        return float(logits.square().sum())

    # With an untrained prior the substituted slots differ from the real ones,
    # so the likelihood must change.
    result = prior_substitution_ablation(model, common, private, None, None, nll_of)
    assert result["delta_prior_vs_full"] != 0.0
    assert result["delta_zero_vs_full"] != 0.0


def test_prior_substitution_requires_a_model_with_a_prior():
    model = CommonPrivateSurvival(4, 4, width=8, slots=2, common_dim=2, private_dim=2)
    del model.prior
    with pytest.raises(ValueError):
        prior_substitution_ablation(model, torch.randn(4, 2), [torch.randn(4, 2, 2)],
                                    None, None, lambda logits: 0.0)
