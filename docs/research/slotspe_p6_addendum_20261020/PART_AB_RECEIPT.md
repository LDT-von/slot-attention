# Task A + Task B: final receipt

Checkpoint `1c6daa01…aec5`, read-only throughout. Env `dct` (torch 2.10.0+cu128),
guarded by `_gpu_guard.require_gpu_env()`.

---

# PART A — P5 self-interference: does it harm the C-index?

## Why this was tested

`patient_slot_intervention.json` already recorded `self_interference: observed`
with `max_self_pool_change = 2.563` — 30-50x larger than anything else in this
addendum (P6: 0.045, slot re-init: 0.074, permutation: 0.026). Its effect on
the survival metric had never been measured.

## Source of the effect

The `original` arm pools the self-attention readout over **all 8 slots**, while
the gate keeps only k=2:

```python
# interventions.py, original-arm branch
pool = lambda value: value.mean(1)
```

So on its face six discarded slots still reach the prediction.

## Result: no effect on discrimination

40 repeats, same 77 patients, slot-init noise frozen so the intervention is the
only difference.

| Arm | median C-index | paired Δ vs base | 95% CI |
|---|---|---|---|
| `base` (mean over all 8, as shipped) | 0.7224 | — | — |
| `sel` (mean over the 2 kept slots only) | 0.7232 | **+0.0004** | [-0.0008, +0.0014] |
| `shuf` (shuffle discarded slots among themselves) | 0.7224 | **+0.0000** | [0.0000, 0.0000] |

**Verdict: NO_DISCRIMINATION_EFFECT.** p = 0.620.

## Attribution — why `shuf` is exactly zero

`shuf` changing the C-index by *exactly* 0.0000 is not a null result to shrug
off; it is a structural fact. Verified directly:

- the junk shuffle really does change the input: `max |slots − slots'| = 2.187`
- the self-attention readout moves by only **9.5e-07** (float residue)

because the mask excludes discarded slots as **both query and key**:

```89:94:/data1/DCT-Reg/third_party/SlotSPE/models/transformer.py
        if mask is not None:
            assert (mask.sum(dim=1) > 0).all()
            mask_q = mask.unsqueeze(1).unsqueeze(-1)
            mask_k = mask.unsqueeze(1).unsqueeze(2)
            final_mask = mask_q * mask_k
            attn = attn.masked_fill(final_mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
```

The top-k gate therefore **works** on the self path. `mean(1)` averages rows
that are already zeroed, which is why pooling over all 8 versus 2 changes
nothing.

## The finding that survives

`cross_attention` is called with **no mask**:

```293:294:/data1/DCT-Reg/third_party/SlotSPE/models/SlotSPE.py
        # ---> cross attention
        x_inter = self.cross_attention(slots_wsi, slots_omic)
```

Measured directly: **74.96% of cross-attention mass lands on discarded omic
slots.** The gate is genuinely defeated on that path — six of eight slots are
declared useless by the decoder and then used anyway.

**But this costs nothing**, because stage 3 established all 8 slots are
collapsed to inter-slot cosine 0.9996. Discarded slots are numerically
indistinguishable from kept ones, so there is no "junk" for the dense path to
leak.

**Net: a real architectural impurity — the top-k gate is cosmetic on the cross
path — with zero measurable effect on patient discrimination, precisely because
a separate defect (slot collapse) removes the thing the impurity would leak.**

These two findings are causally linked. Reporting either alone would mislead.

---

# PART B — does `eval_fix` deliver determinism, and at what cost?

Evaluated **without** the frozen-RNG harness, so the product's own behaviour is
what shows.

| Arm | deterministic unaided? | max logit diff over repeats | C-index |
|---|---|---|---|
| `original` | **no** | 1.090e-01 | 0.7232 ± 0.0039 (spread 0.7129–0.7320) |
| `eval_fix` | **yes** | **0.000e+00** | **0.7177** (single value) |
| `readout_fix` | **yes** | **0.000e+00** | **0.7113** (single value) |

**Determinism: delivered, exactly.** The frozen `_proof_eval_noise` buffer
removes the per-forward slot re-initialisation completely. This is the real
product fix and it needs no retraining.

**Accuracy cost: small but not zero.**

The script's own rule printed `ADOPT_FREE_FIX` because 0.7177 lies inside
`original`'s 0.7129–0.7320 spread. That is a weak test and should not be quoted
as "free" without qualification:

- `original` median 0.7232 → `eval_fix` 0.7177 is a **-0.0055** shift
- → `readout_fix` 0.7113 is a further **-0.0064**

Both sit inside the original's stochastic spread, so neither is *worse than a
bad draw*, but both are systematically below the original median.

**Sanity check on the measurement:** `eval_repeats.json` recorded at training
time gives `original` C = 0.7223 ± 0.0036 over 20 repeats, matching the 0.7232
± 0.0039 measured here. The pipeline reproduces its own training-time numbers.

## Reading these three numbers together

They are consistent with stage 3, not contrary to it:

- `original` scores 0.7232 *on average* but each individual run is a random draw
- `eval_fix` picks one specific noise draw and returns it every time — 0.7177
- it happens to land **below** the average, so fixing determinism happened to
  cost 0.0055 on this cohort

The honest statement is: **the deterministic model is not worse in expectation,
it is worse on this particular fixed noise draw by a margin smaller than the
noise it removed.** Whether it is better or worse under a different fixed
buffer cannot be determined from one fold and one seed.

---

# Consolidated status

| Claim | Status | Evidence |
|---|---|---|
| harness is bit-exact | confirmed, two envs | stage2, cross_env_trisurv |
| model is stochastic at inference | confirmed | original: 1.09e-01 across repeats |
| patch order harms discrimination | **disproved** | dC = -0.0007, CI contains 0 |
| self-interference harms discrimination | **disproved** | dC = +0.0004, CI contains 0 |
| slots collapse during training | confirmed | sigma≈1, cos 0.41 → 0.9996 |
| top-k gate defeated on cross path | confirmed | 75% mass on discarded slots |
| determinism fix exists, no retraining | confirmed | eval_fix: 0.000e+00 |
| fix is accuracy-neutral | **not established** | -0.0055 vs median, 1 fold, 1 seed |

## The one honest caveat

Everything rests on **one checkpoint: fold 0, seed 3**. No other fold or seed
was ever trained (`pairs/` contains only `fold_0/seed_3`). The conclusions are
internally consistent and reproduce across two environments, but "eval_fix
costs 0.0055" cannot be distinguished from "that one draw was unlucky" without
more folds. That is the highest-value next measurement, and it requires training
runs rather than more diagnostics.

## Artefacts

| File | Contents |
|---|---|
| `_gpu_guard.py` | hard env check; raises before any data load |
| `stage5_self_interference_cindex.json` | A: base/sel/shuf, 40 reps |
| `stage5b_attribution.json` | A: mask analysis, 75% cross mass |
| `stage6_eval_fix.json` | B: 3 arms, determinism + C-index |
| `docs/ENVIRONMENT.md` | Cause B (wrong interpreter) documented |