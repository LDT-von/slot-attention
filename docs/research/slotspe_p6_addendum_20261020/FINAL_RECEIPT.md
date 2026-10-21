# Stage 4A-2 + Stage 3 results: final receipt

All stages eval-only. Checkpoint `1c6daa01…aec5` read-only, unchanged throughout.
Environment note: the default `python` on PATH (torch 2.6.0+cu124) **cannot run on
this machine's RTX 5090 (sm_120)** — it fails with "no kernel image available".
All numbers below come from `/home/ubuntu/.conda/envs/dct/bin/python`
(torch 2.10.0+cu128). Any prior measurement taken with the broken interpreter is
suspect and was re-taken.

## 1. The measurement instrument is sound

| Test | Result |
|---|---|
| 5 repeats, same seed, same batch | bitwise identical (logit hash `452a9011b059cd1f`) |
| full 77-patient sweep replayed | max abs diff **exactly 0.0** |
| slot-cosine replay | drift **exactly 0.0** |

The residual drift previously seen (cos 0.9667 → 0.9777) **does not reproduce**.
It was not non-determinism. The earlier cuDNN-autotuning hypothesis was wrong.

## 2. What is actually stochastic in this model

The `original` arm re-draws all 8 slot initialisations from Gaussian noise on
**every forward**:

```80:80:/data1/DCT-Reg/third_party/SlotSPE/models/slot_attention.py
        slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)
```

Consequence: **the same patient scored twice can get different risks.** Seed
change alone moves logits by up to **7.39e-02** — larger than the patch-order
effect (4.55e-02). The model's own irreducible C-index spread, identical input,
60 draws: **median 0.7228, range [0.7137, 0.7304], std 0.0034**.

## 3. Slot collapse is real — this rescues P2

Stage 3 Part B measured the initialisation directly:

| | value |
|---|---|
| `slots_logsigma.exp()` mean | **1.0045** (wsi) / 1.0001 (omic) — order 1 |
| initial inter-slot cosine | max **0.5605**, mean 0.4053 |
| post-training inter-slot cosine | max **0.9996** |

Init starts the slots genuinely apart (cos ≈ 0.4–0.56); training drives them to
cos ≈ 0.9996. **`sigma` never shrank** — so this is not an initialisation
artefact. Training collapsed them. The earlier "maybe it's just the
parameterisation" hedge was unfounded; P2 stands.

## 4. Why slot permutation changes the output — mechanism isolated

| Stage | Permutation effect |
|---|---|
| cross-attention mean-pool | 9.5e-07 |
| self-attention mean-pool | 1.9e-06 |
| roundoff control (identical, twice) | **exactly 0.0** |
| **full readout** | **2.55e-02** |

The attention blocks are permutation-equivariant to float precision. The break
is the **top-k gate**: k=2 of 8, and the k-th vs (k+1)-th score gap has mean
9.9e-03, min **2.1e-04**. Among 8 near-identical slots the gate is
numerically arbitrating, so 29 mask elements per batch change under permutation.

So the earlier P4 reversal ("permutation is not invariant") is **confirmed, but
for a completely different reason than stated** — not because slots carry
model-visible identity, but because slot collapse makes the selector arbitrary.

## 5. Does any of this hurt patients? No.

Corrected test (Stage 4A-2). The first Stage 4A compared perturbations against a
single seed-0 draw and reported "MATERIAL_HARM" — **that was wrong**, because all
40 perturbations scored *higher* than the baseline, meaning the reference was an
unlucky low draw. Comparing against a distribution instead:

| Arm | C-index median | min | max | std |
|---|---|---|---|---|
| re-init only (control) | 0.7228 | 0.7137 | 0.7304 | 0.0034 |
| patch order permuted | 0.7224 | 0.7121 | 0.7296 | 0.0036 |

Paired difference (same seed, patch order changed): **-0.0007, CI95 [-0.0018, +0.0005]**
— contains zero. Patch order is **statistically indistinguishable** from simply
re-drawing the slot noise.

Clinically interpretable measure: discordant patient pairs flipped, against the
median-risk reference — **median 17, max 28 of 1261 pairs (1.34%)**. And that
flip rate is the *same* whether or not patches are permuted, so it is caused by
the slot re-initialisation, not by patch order.

**Verdict: NO_MEASURABLE_HARM.** Worst C-index drop against the median reference
is 0.0107, which is inside the model's own run-to-run spread.

## 6. What changed in the conclusions

| Claim | Before | After | Why |
|---|---|---|---|
| dropout caused drift | asserted | **RETRACTED** | no dropout in eval path |
| cuDNN non-determinism | suspected | **RETRACTED** | drift does not reproduce; bitwise exact |
| 8 slots near-duplicates | "maybe init artefact" | **CONFIRMED as training collapse** | sigma≈1, init cos 0.41→post 0.9996 |
| permutation changes output | "artifact" | **CONFIRMED, mechanism corrected** | top-k tie-break among collapsed slots |
| patch order matters to patients | unproven | **DISPROVED** | dC = -0.0007, CI contains 0 |
| C-index jitter is material | claimed 0.0198 | **RETRACTED** | baseline was an unlucky draw |

## 7. Project status

The decisive question is answered: **P6 order sensitivity does not harm
patient-level discrimination.** The output numbers move, but the ranking does
not degrade.

The real defect is different and simpler: **the model is stochastic at
inference** because slot attention re-initialises per forward. It does not hurt
this cohort's C-index, but a production deployment would return a different
risk score for the same patient on re-run — which is a reproducibility problem,
not a discrimination problem.

Stage B1 already completed (Section 3). The originally planned remediation
(B: retrain with an anti-collapse penalty) is **not justified by these results**
— the thing to fix is inference-time determinism, which is a one-line change
(the frozen `_proof_eval_noise` buffer already exists and is used by the
`eval_fix` arm), not a retraining problem. Recommend stopping here.

## 8. Output hashes

| File | SHA256 |
|---|---|
| `audit.md` | `65c43d1bf71e49286ecc8e42f7f2a47e30f487a090a3289970b7c00b2b2cd36f` |
| `stage2_findings.md` | `e11ae33c171c548ec34025a1b33407987d74c23aade04cbf7d7969a132f1eea7` |
| `bit_exact_probe.json` | `f43e92f48bd6e40d8966b31341a8e40854673936278a522c6e183ec433230ea4` |

Remaining artefacts: `stage3_corrected_s3_b1.json`,
`stage3b_permutation_decomposition.json`, `stage4a_cindex_impact.json` (superseded),
`stage4a2_corrected_cindex.json`.