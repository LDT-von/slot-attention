# Stage 2 results: bit-exactness probe — PASSED, plus two consequential findings

Script: `_p6_stage2_bitexact.py` (eval-only, no training, checkpoint untouched)
Output: `bit_exact_probe.json`, SHA256 recorded below.

## 1. Bit-exactness: confirmed, in both configurations

| Experiment | Setup | Result |
|---|---|---|
| E1 | 8 patients, 5 repeats, default determinism | **all pairwise diffs exactly 0.0**, identical logit hash `452a9011b059cd1f` |
| E2 | same, `use_deterministic_algorithms(True, warn_only=True)` | **bitwise identical**, same hash |
| E3 | full 77-patient sweep, replayed at same seed | **max abs diff exactly 0.0**, bitwise identical |
| E4 | slot-cosine replay, same seed | drift **exactly 0.0** |

The harness is bit-exact. The earlier drift between two reported runs
(residual cos 0.9667 -> 0.9777) **does not reproduce**; it was not
non-determinism in this path. Stage 1's hypothesis of cuDNN autotuning was
wrong. The drift was an artifact of the previous run, not of the model.

**Consequence:** Stage 3 (recompute S1/S3) requires no corrective rerun for
determinism reasons. S1 is confirmed valid on its own terms.

## 2. Finding A — the entire eval randomness is slot re-initialisation

E3 shows that changing *only* the seed moves logits by up to **7.39e-02**. That
is the same magnitude as the "slot permutation" effect previously reported
(7.74e-02). Both are dominated by the same mechanism:

```80:80:/data1/DCT-Reg/third_party/SlotSPE/models/slot_attention.py
        slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)
```

The `original` arm re-draws all 8 slot initialisations from Gaussian noise on
**every forward**. The frozen-RNG harness neutralises this, so S1's order-effect
numbers are clean. But this also means the shipped model's output for a given
patient is genuinely stochastic across repeated inference — which is the real,
demonstrable P6 finding, and it is *larger* than the previously reported patch
order effect.

## 3. Finding B — S3's comparison arm was invalid; its conclusion is retracted

The S3 routine compares a manually reconstructed permuted path against the
model's own full forward:

```505:524:/data1/DCT-Reg/_s2026_1020_addendum.py
            cross = model.cross_attention(slots_wsi, slots_omic)
            intra_wsi = model.self_attention_wsi(
                slots_wsi, mask=keep_wsi_base.detach() > 0.5
            )
            # ... permuted arm ...
            cross_p = model.cross_attention(slots_wsi_p, slots_omic)
            intra_wsi_p = model.self_attention_wsi(
                slots_wsi_p, mask=keep_wsi_p.detach() > 0.5
            )
            fused_p = torch.cat([cross_p.mean(1), pooled_wsi_p, pooled_omic_b], dim=1)
```

The defect: `slots_wsi`, `slots_omic` and the baseline `logits_base` are all
obtained **outside** any RNG freeze (lines 454-459 run with a live, advancing
CUDA RNG stream), so the baseline and the permuted arm are evaluated under
**different slot noise draws**. Per Finding A, that alone produces ~7e-2.

Therefore the previously reported "full-model slot permutation changes the
output by up to 0.077" is **not attributable to slot permutation**. It is
dominated by slot re-initialisation noise. The earlier reversal of the P4
conclusion ("permutation is not invariant") was itself an artifact.

Additionally the permuted arm rebuilds the fuse path by hand, so any drift
between the reconstruction and the true forward is also folded into the number.

**Retraction:** the S3 result (`S3_p4_full_perm`, max 0.0774) does not support
"slots carry model-visible identity" and must not be cited. The correct test is
re-run in Stage 3 with the baseline and permuted arms sharing one frozen RNG
draw.

## 4. What survives

| Prior claim | Status after Stage 2 |
|---|---|
| S1 patch-order effect 0.0455 | **stands** — correctly frozen, bit-exact |
| S1 zero-change control = 0.0 | **stands** — confirmed at full-sweep scale |
| S3 full permutation 0.0774 | **RETRACTED** — confounded by slot re-init noise |
| P4 "permutation not invariant" | **RETRACTED** — was an artifact of the above |
| eval randomness is real | **stands, and is now quantified**: 7.39e-02 per re-init |

Proceed to Stage 3: re-run S3 correctly, then Stage 4A, then B1.