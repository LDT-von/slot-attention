# Stage 1 audit: sources of non-determinism and scope correction

Recorded: 2026-10-20. Scope: eval-only audit, **no training, no new checkpoint**.
Checkpoint under audit: `1c6daa01969879c40612bc67dd17ab7b9d3ed9e5ea56671e7f8278dd4c30aec5`
(arm `original`, BLCA fold 0, seed 3, epoch 3, 77 val patients).

## 0. Retraction of an earlier claim

An earlier working note stated that residual drift across repeated evals was
caused by an un-frozen `Dropout` layer (p=0.1) in the slot attention block.
**That claim was wrong and is retracted.** It was asserted without reading the
module. `grep -rnE 'Dropout|dropout'` over `third_party/SlotSPE/models/` returns
no dropout in either `MultiHeadSlotAttention` or `MoESlotDecoder`. The drift has
a different cause, identified below.

## 1. Inventory of every randomness source in the eval path

| # | Site | File / line | Active in eval? | Status |
|---|---|---|---|---|
| R1 | `torch.randn(mu.shape)` slot re-init | `third_party/SlotSPE/models/slot_attention.py:80` | **YES** | real RNG draw on **every forward, every patient** |
| R2 | `slots.topk` gate selection | `survot_rank/.../model.py:244` | YES | deterministic (no gumbel outside `self.training`) |
| R3 | patch subset sampling | `survot_rank/.../patient.py:274-275` | YES | deterministic — `sha256(pid)`-derived seed, per-patient constant |
| R4 | fold shuffling | `survot_rank/.../patient.py:568` | NO | `torch.Generator` seeded per epoch; train loop only |
| R5 | `_rng(seed,...)` | `survot_rank/.../patient.py:366-371` | YES | sets global seeds but cannot prevent R1 |

**R1 is the entire story.** In the `original` arm the module class is left
unrepaired, so every forward re-draws the 8 slot initialisations from Gaussian
noise. `apply_repairs` returns the source model untouched for this arm:

```339:341:/data1/DCT-Reg/survot_rank/research/slotspe_proof/model.py
    if arm == "original":
        if type(net) is not source.SlotSPE:
            raise ValueError("The original arm must be an unrepaired source model")
        return net
```

The repaired arms instead use a frozen per-module buffer, constant across
patients, batches and repeats:

```353:356:/data1/DCT-Reg/survot_rank/research/slotspe_proof/model.py
    for index, name in enumerate(("slot_attention_wsi", "slot_attention_omic")):
        module = getattr(net, name)
        module.__class__ = classes.ReproducibleSlotAttention
        generator = torch.Generator(device="cpu").manual_seed(160018 + index)
        noise = torch.randn(1, module.num_slots, module.dim, generator=generator)
        module.register_buffer("_proof_eval_noise", noise, persistent=False)
```

## 2. Consequence: which prior numbers are affected

This audit *re-scopes* earlier findings rather than merely confirming them.

- **The zero-change control (S1) is valid and is in fact stronger than
  claimed.** A 0.0 diff across 200 pairs is achievable precisely because the
  harness suppresses R1. The measurement instrument is sound; no bit-exactness
  probe is required to justify S1/S3, though one will still be run as a guard.
- **S1/S3 are measured *with slot-init noise frozen*.** The `0.0455` mean patch
  order effect and the `0.0774` max full-permutation effect are therefore clean
  order effects, not contaminated by R1.
- **The original `evaluation_randomness` flag is genuine model behaviour, not a
  measurement artifact.** Re-running the identical input through the `original`
  arm *does* change the answer. This is a property of the shipped model.
- **The residual cos drift between two reported runs (0.9667 -> 0.9777) is not
  explained by R1**, which is frozen in that measurement. It remains an open
  item; the hypothesis is non-bit-exact float accumulation in the slot
  iteration (`einsum` + `GRUCell` + `softmax`) under cuDNN autotuning. Stage 2
  tests it directly.

## 3. The finding that changes the slot-collapse conclusion

Slot initialisation is

```77:80:/data1/DCT-Reg/third_party/SlotSPE/models/slot_attention.py
        mu = repeat(self.slots_mu, '1 1 d -> b s d', b=b, s=n_s)
        sigma = repeat(self.slots_logsigma.exp(), '1 1 d -> b s d', b=b, s=n_s)

        slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)
```

All 8 slots share a single learned `mu` and differ only by `sigma * noise`.
Therefore high inter-slot cosine similarity is **the expected outcome of this
parameterisation**, independently of any training pathology. It is not
evidence of collapse. Stage B1 measures `slots_logsigma` directly to separate:

- (a) `sigma` small post-training => the model *did* collapse the slots; versus
- (b) `sigma` order-1 => slots start close because the parameterisation forces
  them to, and the residual cosine structure is what the module can express.

Only (a) licenses the original "slot collapse" framing.

## 4. Non-determinism hazards that remain

| Hazard | Present? | Mitigation for Stage 2 |
|---|---|---|
| `nn.Dropout` | **NO** — none in eval path | n/a |
| cuDNN autotune non-bit-exact kernels | probable cause of the 0.9667/0.9777 drift | `torch.use_deterministic_algorithms(True, warn_only=True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False` |
| `einsum`/`GRUCell` float accumulation order | probable | fixed batch shape, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| RNG read during batch construction (R3) | NO — sha256-derived | none needed |
| RNG stream position at forward time | controlled by `_rng` | freeze per repeat |

## 5. Verdict

Stage 1 passes. The instrument is trustworthy; S1 and S3 stand as measured. The
audit yields one correction and one downgraded claim:

1. **Correction:** the dropout explanation is retracted (Section 0).
2. **Downgrade:** "8 slots are near-duplicates" is demoted from *model defect* to
   *unresolved between collapse (a) and parameterisation (b)*, pending B1.

Proceed to Stage 2.