# SlotSPE problem discovery

Read README.md and docs/PROGRESS.json first. Maintain only the SlotSPE problem line in this repository.
Original executable source: third_party/SlotSPE/. Preserve it byte for byte and keep provenance.
Separate source facts, synthetic counterexamples, historical local patient observations, server reports, audited server artifacts, predictive harm, and correction efficacy.
P8 does not apply to original SlotSPE. Do not add transport or new innovation modules as part of this audit.
Patient features, weights, predictions, identifiers, and raw per-patient arrays remain in ignored local_evidence/. Never stage them.
Do not delete archives or original data: ARCHIVED.txt was incorrectly used to call raw arrays/report folders identical; see P5 correction.
Do not start training, inference, or patient interventions without explicit execution authorization. Plan/help/metadata/hash checks are separate from execution.
Default queue config is the existing clean original-only fold0/seed3, 30 epochs, 4096 real patches, 20 repeats, splits/5fold_uni2h. Only --run executes.
Split identity is a measured fact, not a name: the original loader hardcodes the 5fold directory (third_party/SlotSPE/dataset/dataset_survival.py _load_split), so the splits_dir setting never reaches cohort selection. The completed 30/4096/20 best queue and its prepared manifest hashed splits/5fold/blca/fold_*.csv and its 15 reused S0/R4/R3 results were produced under 5fold. 5fold and 5fold_uni2h are materially different cohorts (only 12-17 of 76 validation patients shared per fold; 381 vs 380 patients). Any arm that reuses those baselines must inherit 5fold; the AGENTS.md 5fold_uni2h default above does not describe them.
Patch handling follows the original loader: every real patch is cached, a fresh subset is drawn without replacement on each access, and bags shorter than the budget are zero-padded rather than excluding the patient (third_party/SlotSPE/dataset/dataset_survival.py lines 285-292).
Retain A/B/C: A original self pool; B selected self pool with final selection-weight gradient detached; C same forward with this gradient connected. Their auxiliary losses/gradients, selection, numeric handling, eval and training-fold bins must match.
Do not edit historical reports to change their old absolute paths or hashes. Local copies have a path map in ignored LOCAL_MIGRATION_MANIFEST.json; relocation does not automatically make old measurement auditors portable.
Update docs/PROGRESS.json as the single independent progress entry. E:/DCT-Reg records this repository as the new continuation location, while preserving its historical evidence.
Every completed stage reports six items: route/task; completed work; paths/commit; checks and actual server execution/raw receipt; gaps/reasons; next action/user inputs.
Never claim synthetic or packaging checks prove patient performance. Problems not observed must be recorded honestly.
