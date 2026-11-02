"""Diagnostics for the common/private slot bottleneck prototype.

These helpers accept tensors and do not load patient files or checkpoints.
The research receipts in this project use generated inputs only; the helpers
can call supplied readouts for controlled forward perturbations.

The prototype in :mod:`slotspe_proof.common_private_survival` claims that the
private slots carry prognostic information that the common representation does
not carry. Naming a branch is not evidence for that claim, and the module
cannot observe it directly, so this module measures it with four independent
instruments:

``linear_probe``
    Closed-form ridge regression from a branch representation to a known
    synthetic factor. It reports held-out R^2 and Pearson correlation.

``partial_probe``
    Linear residual prediction after regressing out the common representation.
    A nonlinear common copy can produce a false incremental signal. This is
    not conditional MI or a conditional-independence test; use the later
    fit/validation/test nonlinear incremental probe for bounded evidence.

``swap_ablation``
    Fixed-network readout perturbations: exchange common representations or
    zero private slots and recompute likelihood. These measure forward
    sensitivity, not patient-level causality. Off-manifold perturbations and
    finite measurement resolution limit the interpretation of null effects.

``control ladder``
    Every measurement ships with instruments of known answer: an input upper
    bound that shows the probe can recover a factor when it is present, a
    label-shuffled control that must show nothing, and a constant baseline
    that must show nothing.

None of these measurements says anything about patient performance. They
measure the internal behaviour of one prototype on invented data.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence

import torch

__all__ = [
    "ridge_fit",
    "ridge_predict",
    "r2_score",
    "pearson",
    "linear_probe",
    "partial_probe",
    "swap_ablation",
    "branch_sensitivity",
    "representation_variance",
    "prior_substitution_ablation",
    "duplication_probe",
    "probe_suite",
    "summarise_probe_suite",
]


def _as_float_matrix(x, name):
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"{name} must be a torch tensor")
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(f"{name} must be a nonempty [n, d] matrix")
    if not x.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    x = x.double()
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} must be finite")
    return x


def ridge_fit(x, y, lam=1e-3):
    """Closed-form ridge regression with an unpenalised intercept.

    ``x`` is ``[n, d]``, ``y`` is ``[n]`` or ``[n, m]``. The returned pair is
    ``(weight, bias)`` with ``weight`` of shape ``[d, m]``. The ridge penalty
    is applied to the centred design only, so the intercept is exact.
    """
    x = _as_float_matrix(x, "x")
    y = _as_float_matrix(y, "y")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if not isinstance(lam, float) or not math.isfinite(lam) or lam < 0:
        raise ValueError("lam must be a finite non-negative float")
    n, d = x.shape
    if n <= d + 1 and lam <= 0:
        raise ValueError("an underdetermined fit needs a positive ridge penalty")
    x_mean = x.mean(0, keepdim=True)
    y_mean = y.mean(0, keepdim=True)
    xc = x - x_mean
    yc = y - y_mean
    gram = xc.T @ xc + lam * torch.eye(d, dtype=x.dtype)
    weight = torch.linalg.solve(gram, xc.T @ yc)
    bias = (y_mean - x_mean @ weight).squeeze(0)
    return weight, bias


def ridge_predict(x, weight, bias):
    """Apply a fitted ridge model."""
    x = _as_float_matrix(x, "x")
    if weight.ndim != 2 or weight.shape[0] != x.shape[1]:
        raise ValueError("weight does not match the feature dimension")
    return x @ weight + bias


def r2_score(y_true, y_pred):
    """Held-out R^2 of a prediction against a target."""
    y_true = _as_float_matrix(y_true, "y_true")
    y_pred = _as_float_matrix(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("prediction and target shapes differ")
    residual = (y_true - y_pred).square().sum()
    total = (y_true - y_true.mean(0, keepdim=True)).square().sum()
    if float(total) <= 0:
        return float("nan")
    return float(1.0 - residual / total)


def pearson(a, b):
    """Pearson correlation between two vectors."""
    a = _as_float_matrix(a, "a")
    b = _as_float_matrix(b, "b")
    if a.shape != b.shape:
        raise ValueError("vectors must have the same shape")
    ac = a - a.mean()
    bc = b - b.mean()
    denominator = float(ac.norm() * bc.norm())
    if denominator <= 0:
        return float("nan")
    return float((ac * bc).sum() / denominator)


def linear_probe(x_train, y_train, x_test, y_test, lam=1e-3):
    """Ridge probe from a representation to a target, with held-out scores.

    The probe is fitted on ``x_train``/``y_train`` only and scored on
    ``x_test``/``y_test``. The returned dictionary carries the held-out R^2,
    the held-out Pearson correlation and the fraction of target variance the
    training fit explains, so that a probe which merely memorises the
    training set is visible.
    """
    x_train = _as_float_matrix(x_train, "x_train")
    x_test = _as_float_matrix(x_test, "x_test")
    y_train = _as_float_matrix(y_train, "y_train")
    y_test = _as_float_matrix(y_test, "y_test")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("train and test feature dimensions differ")
    if y_train.shape[0] != x_train.shape[0] or y_test.shape[0] != x_test.shape[0]:
        raise ValueError("feature and target row counts differ")
    weight, bias = ridge_fit(x_train, y_train, lam=lam)
    train_prediction = ridge_predict(x_train, weight, bias)
    test_prediction = ridge_predict(x_test, weight, bias)
    return {
        "test_r2": r2_score(y_test, test_prediction),
        "test_pearson": pearson(y_test, test_prediction),
        "train_r2": r2_score(y_train, train_prediction),
        "weight_l2": float(weight.norm()),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
    }


def partial_probe(common_train, private_train, target_train,
                  common_test, private_test, target_test, lam=1e-3):
    """How much of the target survives after the common branch is regressed out.

    Step one regresses the target on the common representation and keeps the
    residual. Step two regresses that residual on the private representation
    and reports the held-out R^2 of the residual prediction. A private branch
    that merely copies what the common branch already knows yields a value
    near zero; a private branch with genuinely complementary content yields
    a positive value.
    """
    common_train = _as_float_matrix(common_train, "common_train")
    private_train = _as_float_matrix(private_train, "private_train")
    target_train = _as_float_matrix(target_train, "target_train")
    common_test = _as_float_matrix(common_test, "common_test")
    private_test = _as_float_matrix(private_test, "private_test")
    target_test = _as_float_matrix(target_test, "target_test")
    if common_train.shape[0] != target_train.shape[0] or private_train.shape[0] != target_train.shape[0]:
        raise ValueError("training rows do not line up")
    if common_test.shape[0] != target_test.shape[0] or private_test.shape[0] != target_test.shape[0]:
        raise ValueError("test rows do not line up")
    weight, bias = ridge_fit(common_train, target_train, lam=lam)
    residual_train = target_train - ridge_predict(common_train, weight, bias)
    residual_test = target_test - ridge_predict(common_test, weight, bias)
    total_variance = float((target_test - target_test.mean(0, keepdim=True)).square().sum())
    common_explained = float((residual_test).square().sum())
    private_weight, private_bias = ridge_fit(private_train, residual_train, lam=lam)
    residual_prediction = ridge_predict(private_test, private_weight, private_bias)
    result = {
        "common_only_test_r2": r2_score(target_test, ridge_predict(common_test, weight, bias)),
        "private_residual_test_r2": r2_score(residual_test, residual_prediction),
        "private_residual_test_pearson": pearson(residual_test, residual_prediction),
        "residual_variance_share_left_after_common": (
            common_explained / total_variance if total_variance > 0 else float("nan")),
        "residual_train_variance_share_left_after_common": (
            float((residual_train).square().sum()) /
            float((target_train - target_train.mean(0, keepdim=True)).square().sum())
            if float((target_train - target_train.mean(0, keepdim=True)).square().sum()) > 0 else float("nan")),
    }
    return result


def swap_ablation(model, common, private, time_bin, event, nll_function,
                  permutation=None):
    """Causal intervention on the readout of the prototype.

    ``model`` must expose a ``readout(common, private)`` method, which the
    prototype does. Four conditions are evaluated on the same batch:

    ``full``
        the model's own common representation and its own private slots.
    ``swap_common``
        the common representation of another patient, with the own private
        slots. A branch the readout ignores cannot change the likelihood.
    ``swap_private``
        the private slots of another patient, with the own common
        representation.
    ``zero_private``
        the private slots replaced by zeros.
    ``zero_common``
        the common representation replaced by zeros.

    The returned deltas are relative to ``full``. A branch that is actually
    used produces a non-zero delta; a branch that is bypassed produces a
    delta of exactly zero.
    """
    if permutation is None:
        permutation = torch.roll(torch.arange(common.shape[0]), 1)
    permutation = torch.as_tensor(permutation, dtype=torch.long)
    if permutation.shape[0] != common.shape[0]:
        raise ValueError("permutation length does not match the batch")
    if not torch.equal(permutation.sort().values, torch.arange(common.shape[0])):
        raise ValueError("permutation must be a permutation of the batch")
    if torch.equal(permutation, torch.arange(common.shape[0])):
        raise ValueError("permutation must not be the identity")
    zeros_private = [torch.zeros_like(u) for u in private]
    zeros_common = torch.zeros_like(common)
    conditions = {
        "full": model.readout(common, private),
        "swap_common": model.readout(common[permutation], private),
        "swap_private": model.readout(common, [u[permutation] for u in private]),
        "zero_private": model.readout(common, zeros_private),
        "zero_common": model.readout(zeros_common, private),
    }
    scores = {name: float(nll_function(logits)) for name, logits in conditions.items()}
    reference = scores["full"]
    return {
        "nll": scores,
        "delta_vs_full": {name: value - reference for name, value in scores.items()},
        "n_patients": int(common.shape[0]),
    }


def branch_sensitivity(model, common, private, time_bin, event, nll_function,
                       permutation=None):
    """Per-patient likelihood sensitivity of each branch of the readout.

    ``swap_ablation`` reports the change of the *cohort mean* likelihood. A
    branch that is used by only a few patients can move that mean very
    little, and a branch that is used but whose effect cancels across
    patients is invisible in the mean. This instrument instead computes, for
    each patient separately, the likelihood change caused by replacing that
    patient's common representation, or that patient's private slots, with the
    values of another patient.

    The returned dictionary reports the per-patient absolute change
    distribution and the fraction of patients whose likelihood changed at
    all. A branch that the readout genuinely consumes must change the
    likelihood for a large share of patients.
    """
    if permutation is None:
        permutation = torch.roll(torch.arange(common.shape[0]), 1)
    permutation = torch.as_tensor(permutation, dtype=torch.long)
    if permutation.shape[0] != common.shape[0]:
        raise ValueError("permutation length does not match the batch")
    if not torch.equal(permutation.sort().values, torch.arange(common.shape[0])):
        raise ValueError("permutation must be a permutation of the batch")
    if torch.equal(permutation, torch.arange(common.shape[0])):
        raise ValueError("permutation must not be the identity")

    def per_patient_nll(logits):
        """Negative log-likelihood of every patient, one value each."""
        return -survival_nll_per_patient(logits, time_bin, event)

    full = per_patient_nll(model.readout(common, private))
    swap_common = per_patient_nll(model.readout(common[permutation], private))
    swap_private = per_patient_nll(model.readout(common, [u[permutation] for u in private]))
    zero_common = per_patient_nll(model.readout(torch.zeros_like(common), private))
    zero_private = per_patient_nll(model.readout(common, [torch.zeros_like(u) for u in private]))

    def describe(delta):
        absolute = delta.abs()
        changed = int((absolute > 0).sum())
        return {
            "mean_abs_delta": float(absolute.mean()),
            "median_abs_delta": float(absolute.median()),
            "max_abs_delta": float(absolute.max()),
            "patients_changed": changed,
            "patients_total": int(delta.shape[0]),
            "fraction_changed": changed / int(delta.shape[0]),
        }

    return {
        "swap_common": describe(swap_common - full),
        "swap_private": describe(swap_private - full),
        "zero_common": describe(zero_common - full),
        "zero_private": describe(zero_private - full),
        "n_patients": int(common.shape[0]),
    }


def survival_nll_per_patient(logits, time_bin, event):
    """Per-patient negative log-likelihood under the prototype's censoring rule.

    This mirrors :func:`slotspe_proof.common_private_survival.survival_nll`
    exactly, but returns one value per patient instead of the batch mean, so
    that a branch can be tested for how many patients it actually affects.
    The censoring convention is the prototype's: ``event == 1`` means the
    event occurred inside the interval ``time_bin``, and ``event == 0`` means
    the patient is known to have survived through that interval.
    """
    from torch.nn import functional as F

    if time_bin.shape != logits.shape[:1] or event.shape != time_bin.shape:
        raise ValueError("one time and event indicator per example required")
    if not ((time_bin >= 0) & (time_bin < logits.shape[1])).all():
        raise ValueError("time bins out of range")
    if not ((event == 0) | (event == 1)).all():
        raise ValueError("event must be binary")
    axis = torch.arange(logits.shape[1], device=logits.device)[None]
    survived = axis < time_bin[:, None]
    survived = survived | ((axis == time_bin[:, None]) & ~event.bool()[:, None])
    loss = (F.softplus(logits) * survived).sum(1)
    loss = loss + F.softplus(-logits.gather(1, time_bin[:, None]).squeeze(1)) * event
    return loss


def representation_variance(features):
    """Spread of a representation across patients, and its shared fraction.

    A representation whose across-patient variance is a small fraction of its
    total variance carries almost no patient-specific information. The
    ``patient_share`` value is the ratio of the across-patient variance to
    the total variance of the feature matrix.
    """
    features = _as_float_matrix(features, "features")
    if features.shape[0] < 2:
        raise ValueError("at least two patients are required")
    centered = features - features.mean(0, keepdim=True)
    total = float(features.square().sum())
    across = float(centered.square().sum())
    return {
        "n_patients": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "mean_feature_norm": float(features.norm(dim=1).mean()),
        "across_patient_variance_share": (across / total) if total > 0 else float("nan"),
        "per_feature_std_mean": float(features.std(0, unbiased=False).mean()),
    }


def prior_substitution_ablation(model, common, private, time_bin, event, nll_function):
    """Replace the private slots by what the conditional prior predicts.

    The prototype conditions the prior on the common representation only. If
    the conditional prior carries the whole private content, then replacing
    the sampled private slots by the prior mean must leave the survival
    likelihood essentially unchanged, because the readout only ever sees
    information that is already inside ``common``.

    If the likelihood degrades, the private slots carry content that the
    common representation does not contain, which is the claim the
    architecture is supposed to make.

    The result is reported as the change in the cohort negative log
    likelihood, together with the same quantity for a control that replaces
    the private slots by their own per-patient mean (which removes only the
    sampling noise) and by zeros (which removes everything).
    """
    from torch.nn import functional as F

    if not hasattr(model, "prior"):
        raise ValueError("the model must expose the conditional prior")
    with torch.no_grad():
        prior_means = []
        for index, prior in enumerate(model.prior):
            mean, _ = prior(common).unsqueeze(1).chunk(2, -1)
            prior_means.append(mean)
        prior_slots = [prior_means[index].expand_as(private[index])
                       for index in range(len(private))]
        own_means = [u.mean(1, keepdim=True).expand_as(u) for u in private]
        zeros = [torch.zeros_like(u) for u in private]

        def score(slots):
            return float(nll_function(model.readout(common, slots)))

        full = score(private)
        prior_score = score(prior_slots)
        own_mean_score = score(own_means)
        zero_score = score(zeros)

    return {
        "nll_full": full,
        "nll_prior_substituted": prior_score,
        "nll_own_mean_substituted": own_mean_score,
        "nll_zero_substituted": zero_score,
        "delta_prior_vs_full": prior_score - full,
        "delta_own_mean_vs_full": own_mean_score - full,
        "delta_zero_vs_full": zero_score - full,
        "prior_retains_fraction_of_private_effect": (
            (zero_score - prior_score) / (zero_score - full)
            if abs(zero_score - full) > 1e-12 else float("nan")),
    }


def duplication_probe(common_train, private_train, common_test, private_test, lam=1e-3):
    """How much of the private branch can be predicted from the common branch.

    This is a model-free duplication measure. The conditional prior of the
    prototype is itself a learned network, so a low duplication measured
    through that prior could be an artefact of how the prior failed to
    train. This probe instead regresses the private representation on the
    common representation with a held-out ridge fit.

    A high held-out R^2 means the private branch is largely a deterministic
    function of the common branch, that is, a redundant copy rather than
    complementary content. A low value means the private branch carries
    information that the common branch does not determine.
    """
    common_train = _as_float_matrix(common_train, "common_train")
    private_train = _as_float_matrix(private_train, "private_train")
    common_test = _as_float_matrix(common_test, "common_test")
    private_test = _as_float_matrix(private_test, "private_test")
    if common_train.shape[0] != private_train.shape[0]:
        raise ValueError("training rows do not line up")
    if common_test.shape[0] != private_test.shape[0]:
        raise ValueError("test rows do not line up")
    weight, bias = ridge_fit(common_train, private_train, lam=lam)
    return {
        "common_to_private_test_r2": r2_score(private_test, ridge_predict(common_test, weight, bias)),
        "common_to_private_test_pearson": pearson(
            private_test, ridge_predict(common_test, weight, bias)),
        "private_variance_share_left": (
            float((private_test - private_test.mean(0, keepdim=True)).square().sum()) /
            float(private_test.square().sum())
            if float(private_test.square().sum()) > 0 else float("nan")),
    }


def probe_suite(representations: Mapping[str, torch.Tensor],
                 factors: Mapping[str, torch.Tensor],
                 train_index, test_index, lam=1e-3) -> Dict[str, Dict[str, dict]]:
    """Probe every representation against every factor, on held-out patients.

    ``representations`` maps a branch name to its per-patient feature matrix,
    ``factors`` maps a factor name to its per-patient value. The probe is
    fitted on the training patients and scored on the held-out patients.
    """
    if not representations or not factors:
        raise ValueError("representations and factors must not be empty")
    names = set(representations)
    for name, values in representations.items():
        if values.shape[0] != list(representations.values())[0].shape[0]:
            raise ValueError("representations must share the patient axis")
        if name in names and not isinstance(name, str):
            raise ValueError("representation names must be strings")
    suite: Dict[str, Dict[str, dict]] = {}
    for branch, features in representations.items():
        suite[branch] = {}
        for factor, target in factors.items():
            if target.shape[0] != features.shape[0]:
                raise ValueError("factor and representation patient counts differ")
            suite[branch][factor] = linear_probe(
                features[train_index], target[train_index],
                features[test_index], target[test_index], lam=lam)
    return suite


def summarise_probe_suite(suite: Mapping[str, Mapping[str, Mapping[str, float]]]) -> Dict[str, object]:
    """Condense a probe suite into the numbers a reader should check first."""
    rows = []
    for branch, factors in suite.items():
        for factor, scores in factors.items():
            rows.append({
                "branch": branch,
                "factor": factor,
                "test_r2": scores["test_r2"],
                "test_pearson": scores["test_pearson"],
                "train_r2": scores["train_r2"],
            })
    rows.sort(key=lambda row: (-(row["test_r2"] if row["test_r2"] == row["test_r2"] else -1e9), row["branch"], row["factor"]))
    best = {}
    for factor in sorted({row["factor"] for row in rows}):
        candidates = [row for row in rows if row["factor"] == factor]
        best[factor] = max(candidates, key=lambda row: row["test_r2"] if row["test_r2"] == row["test_r2"] else -1e9)
    return {"rows": rows, "best_branch_per_factor": best}
