# Required manuscript changes

The manuscript text is not in the archive, so each item names the claim by its content and its archived source. Apply them when the manuscript is revised. Every number comes from `outputs/` (regenerate with `run_analysis_only.sh`).

Describe all of these as **retrospective corrections made after the study**. None of them may be presented as preregistered.

## A. Remove

1. **Remove the ordinal (H2) stage accuracy and the "1.4–1.8 stage" MAE** from B4 / `h2_decision/per_fold.csv`.
   - These values were decoded incorrectly, and the archive cannot correct them.
   - Do not substitute a value. If H2 is discussed, say that ordinal quality was not validly evaluated.
2. **Remove "the real (and shuffled) graph produces better flags than no graph"** (flagging v2 graph comparison).
   - On exact-stage records no comparison excludes zero: real @52 Δlog-OR +0.59 [−0.22, 1.47]; real @192 +0.02 [−0.33, 0.38].
   - If any graph statement is kept, state that the comparison is inconclusive.
3. **Remove any claim that an ablation "Q" was run or what it showed.** No file, script or output in the archive refers to it.
4. **Remove "m=1 / m=3 not run".** Both arms are complete. Either report them, or state that they were run and not reported:

   | Arm | Mean per-seed AUROC | 5-seed-ensemble AUROC |
   |---|---|---|
   | m=1 | 0.82498 | 0.82945 |
   | m=3 | 0.82614 | 0.83393 |

   - m=7 was never run. Do not imply the {0, 1, 3, 7} sweep is complete.
5. **Remove any statement that a `directional_ps` Head-3 comparison was completed.** Only the `directional` target was ever trained.

## B. Change numbers

| Claim | Historical | Corrected |
|---|---|---|
| Consensus failures (≥19/23 models wrong) | 138 | **139** |
| Unanimous failures (23/23) | 57 | 57 (unchanged) |
| Consensus ≥21/23 | 108 | 108 (unchanged) |
| Consensus ∩ Test A flags; Jaccard | 41; 0.275 | 41; **0.273** |
| Between-family error Jaccard, family pairs weighted equally: observed / analytical / permutation mean | 0.439 / "chance" 0.166 | **0.439 / 0.166 analytical / 0.168 permutation** (name both references) |
| Same, with model pairs weighted equally | (0.441 / 0.165) | 0.441 / 0.165 / 0.167 |
| Graph removal (empty graph), Δ macro-F1 | +0.010 | **+0.024** (p = 0.74, n = 8 folds) |
| No fold-fitted features, p ≈ 0.016 | if attributed to AUROC | **applies to macro-F1** (Δ −0.034); AUROC Δ −0.011, p = 0.38 |
| CoDA-GAT AUROC | 0.822 ± 0.030 | **0.822** mean per-seed AUROC, between-cohort SD **0.027** (0.030 is the SD over 40 runs; say which) |
| CoDA-GAT 5-seed ensemble | (if reported) | **0.830**, between-cohort SD 0.029, labelled as a 5-model ensemble |
| Leading baseline | CatBoost 0.842 ± 0.047 | **0.8416**, single seed; between-cohort SD 0.044 (`panel72_full`) / 0.049 (`panel72_combined`, tied at 0.8416). The 0.047 is not reproducible. |
| Empty / shuffled graph AUROC | 0.827 ± 0.044 / 0.829 ± 0.050 | Means unchanged; the ± values are SDs over 40 runs. Between-cohort SDs are 0.044 / 0.045. |

## C. Re-word stage-level (adjacency) claims

These replace every F1/F2 "boundary enrichment" result: Test A, consensus failures, flagging v2/v3, figs 19 and 21, and the flagging-v2 stage panel.

1. **State the analysis set** wherever stage enrichment is reported. For example: "Stage-level tests use the 800 exact-stage records; 137 grouped F0–F1 and 127 grouped F3–F4 records were excluded from these tests only."
   - Add that the subset contains no GSE31803 records, and that GSE213621 contributes only its 107 exact-F2 records.
   - Do not describe it as representative of all 1,064 records.
2. **Replace the numbers** (one-sided Fisher OR, p). Also report the exact-stage n of flags:

   | Result | Historical | Exact-stage |
   |---|---|---|
   | Test A | OR 2.73, p 0.0005 | OR 2.57, p 0.0049 (45 of the 52 flags) |
   | Consensus ≥19/23 | OR 2.03, p 8.4e-05 | OR 2.47, p 5.0e-05 (108 of 139) |
   | Consensus 23/23 | OR 1.54, p 0.073 | OR 1.87, p 0.046 (43 of 57) |
   | Flagging v2, V1 | OR 3.01 | OR 2.78, p 0.0034 |
   | Flagging v2, V2 logreg | p 0.005 | OR 2.00, p 0.051 (no longer < 0.05) |
   | Flagging v2, V3 stable | p 0.036 | OR 2.37, p 0.14 (no longer < 0.05) |
   | Flagging v3 CL, real graph | OR 2.28 | OR 2.20, p 4.3e-05 |
   | Flagging v3 CL, empty graph | OR 1.87 | OR 1.62, p 0.012 |

   The full table is `outputs/stage_enrichment/adjacency_enrichment_by_coding.csv`.
3. **Do not present the 23/23 result (p = 0.046) as newly significant evidence.** It is a retrospective re-analysis at one of three consensus levels, without multiplicity adjustment. The exact subset is compositionally different, and every exact GSE213621 record is F2. Report the without-GSE213621 sensitivity alongside it.
4. **Soften the inference.** F1/F2 enrichment is *consistent with* label noise at the boundary. It does not demonstrate it: shared inputs and genuinely boundary-ambiguous biology predict the same pattern. Do not write "kappa-free confirmation".
5. **Per-stage counts and the "two-stage share" must not include grouped records under F0 or F3.** This applies to the 2×5 tables, fig19, fig21 and the flagging-v2 stage panel. Use the corrected figures.

## D. Correct experiment descriptions

1. **Shuffled graph:** "node features permuted across nodes over the unchanged STRING edge set (equivalently, a random relabelling of STRING)". It is not a rewired graph.
2. **Random graph:** "642 directed edges drawn uniformly without symmetrisation (595–607 unordered pairs vs 321 for STRING; only 35–47 reciprocated pairs)". It matches the directed-edge count only.
3. **Learned graphs:** "321 unordered pairs mirrored into 642 directed edges", which matches STRING's budget.
4. **Masking:** "feature perturbation: features 0–7 of selected nodes replaced by training means; no other feature is recomputed". It is not gene removal or gene dropout.
5. **Augmentation:** m=0, 1 and 3 completed; m=7 not run.
6. **Head 3:** trained with the `directional` deviation target. `directional_ps` was configured but not evaluated.
7. **Planted errors:** recovery of synthetic label flips at exact F1/F2 records (470 eligible, 0 grouped).
   - It is not validation against real biopsy error.
   - It does not show that the graph is superior.
8. **Aggregation:** name it at every AUROC. Use "mean per-seed AUROC" or "AUROC of the 5-seed ensemble", and say that baselines are single-seed.
9. **Agreement null:** name the reference. Use "analytical independence expectation" or "within-fold permutation null"; they are not interchangeable.
10. **Group means:** state whether families or individual model pairs are weighted equally.

## E. Add to Limitations

See `UNRESOLVED.md`. At minimum add:

- global ComBat across cohorts;
- Group C features fitted with training labels that include the early-stopping validation split;
- missing preprocessing provenance;
- H2 ordinal quality not evaluable;
- single-seed baselines compared with 5-seed GAT summaries;
- 8 cohorts, so fold-level tests are descriptive.

Passing these numerical checks does not by itself make the manuscript ready for submission. The claims above still have to be revised.
