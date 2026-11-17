"""Cox partial likelihood for the empirical-role survival task.

Added as a NEW module. The archived matched35 sources are left byte for byte so
that the historical receipts keep their provenance and hashes.

Sign convention, taken from the original SlotSPE trainer at
third_party/SlotSPE/utils/loss_func.py (class SurvPLE): the original uses the
convention E == 1 means censored and E == 0 means the event was observed. This
module uses the explicit boolean ``event`` flag instead, where True means the
event was observed, which matches the research repository's existing
discrete-hazard loss. The two conventions are related by ``event = (censor < 1)``
in the prepared manifest.

The partial likelihood is the Breslow form with an explicit tie handling. The
risk set of a patient with an observed event at time t contains every patient
whose time is greater than or equal to t, including other patients whose time
equals t. A naive ``argsort`` implementation silently drops tied patients from
each other's risk sets, which biases the loss whenever the survival times are
discretised and ties appear. The prepared training fold contains tied event
times, so this is a real case and not a hypothetical one.

No fitting, no binning, no patient values: this module only maps a batch of
risk scores, times and event flags onto a scalar loss.
"""
from typing import Literal

import torch
from torch import Tensor


def _floating_matrix(name: str, x: Tensor) -> None:
    if not isinstance(x, Tensor) or x.ndim != 1 or x.numel() == 0:
        raise ValueError(f'{name} must be a nonempty [patients] tensor')
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f'{name} requires float32/float64')
    if not torch.isfinite(x).all():
        raise ValueError(f'{name} contains nonfinite values')


def cox_partial_likelihood(
    risk: Tensor,
    times: Tensor,
    event: Tensor,
    *,
    reduction: Literal['none', 'mean', 'sum'] = 'mean',
) -> Tensor:
    """Negative Breslow partial log-likelihood of a single risk score per patient.

    ``risk[b]`` is the linear risk score of patient b; a larger score must mean a
    larger risk of the event, matching the direction the concordance index
    expects. ``times[b]`` is the observed or censored time and ``event[b]`` is
    True when the event was observed for patient b.

    The loss is the mean over the patients with an observed event of

        risk_b - log( sum over the risk set of b of exp(risk) )

    where the risk set of b contains every patient whose time is greater than
    or equal to times[b]. Patients tied at the same time belong to each other's
    risk set. Patients whose time is smaller than times[b] are not in the risk
    set and do not contribute.

    A batch that contains no observed event has an undefined partial
    likelihood; the result is a zero loss with zero gradient rather than a
    NaN, so that a single event-free batch cannot poison an epoch.

    This loss is monotone in the ordering of the risk scores only. It does not
    model interval censoring, informative censoring or a competing risk, and
    it does not by itself certify that the risk score is calibrated.
    """
    _floating_matrix('risk', risk)
    if not isinstance(times, Tensor) or not isinstance(event, Tensor):
        raise ValueError('times and event must be tensors')
    if times.shape != risk.shape or event.shape != risk.shape:
        raise ValueError('one risk score, time and event flag per patient required')
    if times.dtype != risk.dtype or event.dtype != torch.bool:
        raise ValueError('times must share the risk dtype and event must be bool')
    if times.device != risk.device or event.device != risk.device:
        raise ValueError('risk, times and event must share device')
    if (times < 0).any():
        raise ValueError('times must be nonnegative')
    if reduction not in ('none', 'mean', 'sum'):
        raise ValueError('invalid reduction')

    # Descending time order. Stable ordering keeps tied patients contiguous and
    # in a deterministic order, which is what makes the tie correction below
    # reproducible across runs.
    order = torch.argsort(times, descending=True, stable=True)
    sorted_times = times[order]
    sorted_risk = risk[order]
    sorted_event = event[order]

    # Group boundaries of equal times. Each group is one distinct time value;
    # all of its members belong to each other's risk set.
    unique_times, counts = torch.unique_consecutive(sorted_times, return_counts=True)
    group_id = torch.repeat_interleave(
        torch.arange(len(unique_times), device=risk.device), counts)

    # Cumulative risk mass per group. The groups are ordered by DESCENDING
    # time, so a forward cumulative sum at group g covers exactly the groups
    # whose time is greater than or equal to this group's time. That is the
    # risk set. A reverse cumulative sum would collect the shorter times and
    # would invert the model.
    group_mass = torch.zeros(len(unique_times), dtype=risk.dtype, device=risk.device)
    group_mass = group_mass.index_add(0, group_id, torch.exp(sorted_risk))
    risk_set_mass = torch.cumsum(group_mass, dim=0)
    per_patient_risk_set = risk_set_mass[group_id]

    observed = sorted_event
    if not bool(observed.any()):
        zero = risk.sum() * 0.0
        if reduction == 'none':
            return zero.expand(risk.shape[0])
        return zero
    contribution = sorted_risk[observed] - torch.log(per_patient_risk_set[observed])
    loss = -contribution.sum()
    if reduction == 'none':
        # Per-patient loss, zero for censored rows so that a mean over the batch
        # is the mean over the events.
        full = torch.zeros_like(sorted_risk)
        full[observed] = -contribution
        return full[order]
    if reduction == 'sum':
        return loss
    return loss / float(int(observed.sum()))


def cox_risk_from_logits(logits: Tensor) -> Tensor:
    """Collapse per-bin hazard logits into one monotone risk score.

    The discrete hazard head emits one logit per time bin. A Cox objective
    needs a single score per patient, so the bins are combined by their sum.
    The sum is monotone in the total log-hazard, which is the quantity the
    discrete loss already trains, so the two objectives optimise compatible
    directions of the same readout.
    """
    if not isinstance(logits, Tensor) or logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError('expected nonempty [patients, bins] logits')
    if logits.dtype not in (torch.float32, torch.float64):
        raise ValueError('logits require float32/float64')
    if not torch.isfinite(logits).all():
        raise ValueError('logits contain nonfinite values')
    return logits.sum(dim=1)
