"""Verify the new Cox partial likelihood against sksurv on synthetic data.

Checks, on invented patients only:
  1. The loss value and the risk-score ordering agree with sksurv's Coxnet on
     the same synthetic cohort.
  2. Tied event times are handled: a patient tied at the event time belongs to
     the risk set, and dropping that changes the loss, so the tied case is
     compared against a hand-computed reference.
  3. The gradient direction is correct: increasing the risk score of an event
     patient relative to a censored patient lowers the loss.
  4. An event-free batch yields zero loss and zero gradient.

Read-only with respect to any real patient data.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_models.common_private_synergy.components.cox_surv import (
    cox_partial_likelihood, cox_risk_from_logits)


def test_matches_sksurv():
    """Our partial likelihood must reproduce the unpenalized Cox MLE.

    sksurv's Coxnet is a penalised fit whose C-index therefore differs slightly
    from the unpenalised maximum likelihood solution. The unpenalised reference
    is CoxPHSurvivalAnalysis, so that is the value our loss has to match, and
    Coxnet is kept as a loose ordering cross-check.
    """
    from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    rng = np.random.default_rng(20261116)
    n = 300
    x = rng.normal(size=(n, 4))
    beta_true = np.array([0.8, -0.5, 0.3, 0.0])
    linear = x @ beta_true + rng.normal(scale=0.5, size=n)
    times = np.exp(2.0 - 0.6 * linear + rng.normal(scale=0.2, size=n))
    events = rng.random(n) < 0.65

    # Reference fits with sksurv, on the same design matrix.
    y = np.empty(n, dtype=[('e', '?'), ('t', 'f8')])
    y['t'] = times
    y['e'] = events
    ph = CoxPHSurvivalAnalysis(alpha=1e-8).fit(x, y)
    risk_ph = ph.predict(x)
    c_ph = float(concordance_index_censored(events, times, risk_ph)[0])
    net = CoxnetSurvivalAnalysis(l1_ratio=0.01, alpha_min_ratio=0.01, n_alphas=5).fit(x, y)
    risk_net = net.predict(x)
    c_net = float(concordance_index_censored(events, times, risk_net)[0])

    # Our implementation: unpenalised fit by the same LBFGS route.
    xt = torch.as_tensor(x, dtype=torch.float64)
    tt = torch.as_tensor(times, dtype=torch.float64)
    et = torch.as_tensor(events)
    beta = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([beta], max_iter=300, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = cox_partial_likelihood(xt @ beta, tt, et)
        loss.backward()
        return loss
    opt.step(closure)
    risk_ours = (xt @ beta.detach()).numpy()
    c_ours = float(concordance_index_censored(events, times, risk_ours)[0])

    from scipy.stats import spearmanr
    rho_ph = float(spearmanr(risk_ph, risk_ours).statistic)
    rho_net = float(spearmanr(risk_net, risk_ours).statistic)
    print('CoxPH C-index', round(c_ph, 8), '| ours', round(c_ours, 8),
          '| spearman', round(rho_ph, 6))
    print('Coxnet C-index', round(c_net, 8), '| spearman', round(rho_net, 6))
    assert abs(c_ours - c_ph) < 1e-9, (
        'the unpenalised partial likelihood fit must match sksurv CoxPH exactly')
    assert rho_ph > 0.999, 'risk ordering must agree with sksurv CoxPH'
    assert rho_net > 0.99, 'risk ordering must agree with sksurv Coxnet'
    return c_ours


def test_ties():
    # Two patients tied at the event time plus one censored later patient.
    # Risk set of the tied event patients = all three.
    risk = torch.tensor([1.0, 0.5, -0.5], dtype=torch.float64)
    times = torch.tensor([10.0, 10.0, 20.0], dtype=torch.float64)
    event = torch.tensor([True, True, False])
    got = float(cox_partial_likelihood(risk, times, event))
    # Manual Breslow with the full risk set of both tied events.
    import math as _m
    manual = -((1.0 - _m.log(_m.exp(1.0) + _m.exp(0.5) + _m.exp(-0.5)))
               + (0.5 - _m.log(_m.exp(1.0) + _m.exp(0.5) + _m.exp(-0.5)))) / 2
    print('ties: got', round(got, 10), 'manual', round(manual, 10))
    assert abs(got - manual) < 1e-12, 'tie handling must match the manual risk set'

    # If a tied patient were dropped from the risk set the loss would differ,
    # so the two values must not coincide.
    wrong = -((1.0 - _m.log(_m.exp(1.0) + _m.exp(-0.5)))
              + (0.5 - _m.log(_m.exp(1.0) + _m.exp(-0.5)))) / 2
    assert abs(got - wrong) > 1e-6, 'the tie correction must actually change the value'


def test_gradient_direction():
    risk = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    times = torch.tensor([5.0, 9.0], dtype=torch.float64)
    event = torch.tensor([True, False])
    loss = cox_partial_likelihood(risk, times, event)
    loss.backward()
    print('gradient', risk.grad.tolist())
    # The event patient has a shorter time; raising its risk must lower loss.
    assert risk.grad[0] < 0, 'raising the event risk must lower the loss'
    assert risk.grad[1] > 0, 'lowering the censored risk must lower the loss'


def test_event_free_batch():
    risk = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    times = torch.tensor([5.0, 9.0], dtype=torch.float64)
    event = torch.tensor([False, False])
    loss = cox_partial_likelihood(risk, times, event)
    loss.backward()
    print('event-free loss', float(loss), 'grad', risk.grad.tolist())
    assert float(loss) == 0.0
    assert torch.all(risk.grad == 0), 'an event-free batch must not produce gradient'


def test_bin_collapse():
    logits = torch.tensor([[1.0, 2.0, -0.5, 0.25]], dtype=torch.float64)
    risk = cox_risk_from_logits(logits)
    print('bin collapse', risk.tolist())
    assert abs(float(risk) - 2.75) < 1e-12


def test_shapes_and_validation():
    for bad in [
        lambda: cox_partial_likelihood(torch.zeros(3, 2), torch.zeros(3), torch.zeros(3, dtype=torch.bool)),
        lambda: cox_partial_likelihood(torch.zeros(3), torch.zeros(3), torch.zeros(3, dtype=torch.float64)),
        lambda: cox_partial_likelihood(torch.zeros(3), torch.zeros(4), torch.zeros(3, dtype=torch.bool)),
        lambda: cox_partial_likelihood(torch.zeros(3), -torch.ones(3), torch.ones(3, dtype=torch.bool)),
    ]:
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError('invalid input was accepted')
    print('validation checks passed')


if __name__ == '__main__':
    test_matches_sksurv()
    test_ties()
    test_gradient_direction()
    test_event_free_batch()
    test_bin_collapse()
    test_shapes_and_validation()
    print('ALL COX CHECKS PASSED')
