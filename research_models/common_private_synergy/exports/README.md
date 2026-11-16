# matched35 82-epoch aggregate export

De-identified aggregate view of the completed common-start matched35 paired run.

## What this is

`matched35_82epoch_aggregates.json` is the read-only aggregate export of the
82-epoch matched35 protocol (one shared 12-epoch role warmup, then two
35-epoch task arms, `joint` and `continued_role_only`, forked from the same
common start).

It was produced by `scripts/export_matched35_aggregates.py`, which reads only
the receipts that the run already wrote. The exporter does not import torch,
does not load any checkpoint, does not touch the patient cache, does not
execute a model and does not use a GPU. It reports `model_executed: false` and
`patient_rows_exported: false`.

- exporter SHA256: `bac2938eab5289e275ef93aee1b5d75aaecd23ac41a8ba26dfcb85aa02369270`
- aggregate SHA256: `a8148305634592a4d29093270b5570c1c094a9f39403c9f667e25f691b9dcd74`

The aggregate document embeds the exporter SHA256 in its `exporter_SHA256`
field, so the two artefacts can be checked against each other.

## Reproducing it

The exporter takes the run root and a fresh output path:

```bash
python -B scripts/export_matched35_aggregates.py \
  --run-root local_evidence/full_slot_matched35_20261113_01 \
  --output /tmp/aggregates.json
sha256sum /tmp/aggregates.json
```

The output path must not already exist; the exporter refuses to overwrite.
Two independent exports of this run produced byte-identical files with the
SHA256 above.

The run root itself is patient-linked private evidence and is not published
here; only the aggregate view is.

## What is inside

Top-level keys: `schema`, `scope`, `exporter_SHA256`, `paired_receipt`,
`shared_role`, `arms`.

- `paired_receipt`: the paired-run status, epoch budget, the common-start,
  matched-input-draw and task-optimizer-reset verification flags, the
  optimizer policy, the shared checkpoint hash, the config and the source
  file hashes.
- `shared_role`: the 12 shared role warmup epochs with per-epoch training
  loss, role loss and role target MSE.
- `arms`: for `joint` and `continued_role_only`, the status, epoch count,
  best and last, the early-stop field, the gradient policy, the reload and
  reference checks, the common-start hashes, the optimizer state checks, the
  artifact hashes, and `epoch_rows`.

Each task `epoch_row` carries the epoch and phase, the training losses, the
validation C-index and NLL, the role target MSE for all four roles, the
per-epoch `input_draw_SHA256`, the read diagnostics for all four roles on
both the train and the validation side, the raw reading mass, the ablation
NLL and the signed ablation delta NLL, the decoded role and slot
diagnostics, and the raw slot gradient means for the `combined`, `survival`
and `role` terms.

Read diagnostics are summarised per modality, role, slot and iteration:
`query_norm`, `key_norm_real_mean`, `key_norm_real_max`, `raw_mass`,
`log_pre_epsilon_mass`, `epsilon_uniform_fraction`, `logit_gap_real_mean`
and `logit_gap_real_min`.

## De-identification

The export contains no patient identifiers, no per-patient arrays, no
predictions, no weights and no survival times or labels. It was audited for
this: no TCGA-style identifiers and no UUID-style strings are present, there
are no arrays whose length equals the 304 training or 76 validation patient
count, and there is no key naming an identifier, a survival time, a censoring
label or a per-patient index.

The only patient-shaped keys are counts and aggregate statistics computed
over the cohort, for example `patients: 76`, `across_patient_std_rms` and
`centered_patient_effective_rank`. `epoch_rows` and `role_epoch_rows` are
per-epoch rows, 35 and 12 of them respectively, not per-patient rows.

## Scope limits

These are development-fold measurements from one fold and one seed. The
paired comparison is a mechanism comparison on the development validation
split, not an independent test and not a patient-performance claim. The
trajectory standard deviation is not a generalisation error, and the best
values are development selection results. The signed per-slot deletion NLL
was recorded as the protocol required, but the deletion changes the pooling
and may leave the training representation distribution, so it is not a
measure of conditional counterfactual inference or of unique biological
information contribution.

CUDA-level bitwise reproducibility was not tested for this run and is not
claimed.
