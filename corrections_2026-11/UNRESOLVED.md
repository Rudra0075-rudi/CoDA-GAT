# Unresolved items

Each item is classified by what it would take to resolve:

- **[artifact]**: missing files would settle it;
- **[omit]**: the claim should be dropped or qualified now;
- **[retrain]**: only future retraining or re-preprocessing can resolve it;
- **[inference]**: possible with exact saved checkpoints, but not done here, for the reason given.

Each item is also marked as a **demonstrated error** (the archive shows it is wrong) or **missing evidence** (the archive cannot show whether it is right).

| # | Item | Kind | Status | What is needed |
|---|---|---|---|---|
| 1 | **H2 ordinal accuracy / MAE** (`h2_decision/per_fold.csv`) | Demonstrated error, uncorrectable | [omit] [artifact] [retrain] | The B4 test-set H2 logits or the B4 arm checkpoints; neither was saved. The file also covers only folds 0–4 of 8 and lacks `summary.csv` / `readout.txt`, so the B4 run was incomplete. Neither the B1 fold-0 model nor the B7 checkpoints are the B4 arms. Until the artifacts are found, omit the claim. A corrected value needs a new B4 run with the fixed decoder (`h2_decision_b4.py --out-dir <new>`), which is retraining. |
| 2 | **Global ComBat** (`shared/bulk_expression.parquet`; a no-ComBat matrix is also shipped) | Missing evidence of the fitting scope; a leakage risk if it was fitted across all cohorts including held-out ones | [retrain] | The preprocessing code that shows the ComBat design. If ComBat was fitted globally, every LODO result includes test-cohort information in its inputs. Fixing that requires per-fold correction (or none) and retraining. The historical numbers must not be presented as if from a leakage-free pipeline, and they must not be relabelled. Report it as a limitation. |
| 3 | **Group C (fold-fitted features 17–25) and validation labels** | Missing evidence (stated by rule, not verifiable) | [retrain] | `fold_info.json` says Group C is "fitted on THIS fold's train". The early-stopping validation set is a 15% split drawn from that same `train/` after the features exist (`train_test_split` in B1/B4/B7). If the rule held, the labels used for early stopping also shaped Group C, so model selection is optimistic. Fixing it needs Group C refitted on the fit-only split, and retraining. Report as a limitation. |
| 4 | **Preprocessing provenance** | Missing evidence | [artifact] | The archive has no code or notebook for: expression processing, ComBat, the 26 node features, the deviation targets, Group C fitting, STRING graph construction, or label harmonisation. `README.txt` cites "the notebook (Cell 4)", which is absent. Without it, items 2, 3 and 9 cannot be verified. |
| 5 | **Ablation "Q"** | Missing evidence | [omit] [artifact] | No script, output or text in the archive refers to it. |
| 6 | **Test B rate sweep and fig22 on exact-stage records** | Affected by grouped labels; not recomputable from saved files | [inference] | Test B regenerated the calibration MC-dropout pools instead of loading Test A's, and dropout is not seeded. Its p40 operating point gives 44 flags, versus Test A's saved 52 at p40, so Test B was not run on Test A's flags. Neither run saved its pools. Re-running inference with the exact B7 checkpoints is permitted, but it would create a third, non-identical pool, so it was not done. `test_b_rate.py` is patched so that a future run uses exact stages. Until then, report fig22 / Test B with the historical coding and say so, or omit them. |
| 7 | **flagging v4 task 1 (ensemble variants), stage metrics** | Affected by grouped labels; per-patient flags not saved | [inference] | Its odds ratio, stage counts and two-stage share used the stored stage codes. A rerun of `flagging_v4.py` (inference) now uses exact stages. Task 2 (planted errors: 470 exact, 0 grouped) and task 3 (direction tests, binary) are valid as they are. |
| 8 | **flagging v2/v3 per-cohort tables** (`per_cohort_values.csv`) | Affected by grouped labels; not regenerated | [inference] or a small analysis | The per-cohort odds ratios for GSE213621 and GSE31803 include grouped records. GSE31803 has no exact-stage records, so its per-cohort stage test is undefined. The pooled results were regenerated (`outputs/stage_enrichment/`); the per-cohort tables were not. |
| 9 | **Masking dependencies** | Missing evidence | [artifact] | Whether unmasked nodes' Group A features (deviation-rank z, pathway scores) depend on masked genes' expression cannot be checked without the preprocessing code. In either case, describe masking as feature perturbation. |
| 10 | **m=7 augmentation arm** | Not run | [omit] or [retrain] | `README.txt` asks for {0, 1, 3, 7}; only 0, 1 and 3 exist. |
| 11 | **`directional_ps` Head-3 comparison** | Not run | [omit] or [retrain] | It was configured but never trained or evaluated. |
| 12 | **README CatBoost "± 0.047"** | Not reproducible | [omit] | Neither tied CatBoost model's between-cohort SD (0.0443, 0.0492) equals it. Use a stated, recomputed SD. |
| 13 | **Single-seed baselines vs 5-seed GAT** | Design asymmetry, not an error | [omit claim of a like-for-like spread] | Baselines have no seed variance. Compare means under named aggregations and do not compare the spreads directly. |
| 14 | **Multiplicity in the re-analysed stage tests** | Analysis caveat | [omit strong wording] | Exact-stage p-values cross 0.05 in both directions across 18 flag sets and 3 consensus levels, with no adjustment. Treat them as descriptive. |
| 15 | **`graph_utility/summary.txt`** | Known overwrite (already documented in the README) | [artifact] | Only its TEST 3 section survives. Cite the per-test CSVs instead. |

Not affected by any correction (verified):

- All binary predictions and metrics of the 440 saved GAT runs (B7 m=0, m=1, m=3 and B9), recomputed from the saved arrays: AUROC, macro-F1 and balanced accuracy differ by at most 3.3e-16.
- The B3 baselines' saved decisions.
- The planted-error eligibility.
- The B4 binary columns, which do not use the decoder.

The conformal (LAC/APS) results were not re-verified here. They use only binary probabilities, which are unchanged, and no correction touches them.
