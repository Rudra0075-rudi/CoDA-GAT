# Change log: 2026-11 corrections

Each entry gives the issue, the evidence, the code change, the result change and the affected manuscript claim.

- **"Demonstrated error"** means the archive shows the error directly.
- **"Missing evidence"** means the archive cannot show whether a claim is right. Those items are listed in `UNRESOLVED.md`.

All numbers below are produced by `run_analysis_only.sh`; see `outputs/VALIDATION.md`.

---

## 1. CORN ordinal decoder (demonstrated error)

**Evidence.** `corn_loss` trains logit *k* on the conditional task P(y>k | y>k−1). This is identical to the official `coral_pytorch` 1.4.0 `corn_loss` (relative difference 2e-8, from float32 targets).

The decoder thresholded those conditional probabilities directly. The official `corn_label_from_logits` thresholds their cumulative product instead. On 200,000 random logit rows, the historical rule disagreed with the official decoder on 75.4% of rows, was never lower, and over-counted by 1.18 stages on average.

Worked examples:

| Conditional probabilities (or implied distribution) | Historical rule | Correct decoding |
|---|---|---|
| [0.3, 0.9, 0.9, 0.9] | F3 | F0 |
| Implied distribution [0.40, 0.11, 0.10, 0.10, 0.29] | F4 | F1 (the median of that distribution) |

**Code.**

- `scripts/train_fold0_b1.py`: `corn_predict_stage` now applies `torch.cumprod` before thresholding. `corn_loss` and the binary head are unchanged.
- The only caller is `scripts/h2_decision_b4.py`, which inherits the fix. Its docstring now records that the historical ordinal columns used the old decoder. It refuses to overwrite `h2_decision/per_fold.csv` (new `--out-dir` option). It was not run.
- No other script decodes H2 logits. B7, B9 and the baselines train an H2 head but report only binary metrics.

**Result.**

- `h2_decision/per_fold.csv` covers folds 0–4 only. It has no `summary.csv` or `readout.txt`, and no H2 logits or B4 checkpoints exist anywhere in the archive.
  - `folds/fold0_GSE130970/outputs/weights.pt` is the B1 proof-of-life model. B1 seeds before loading the data, whereas B4 re-seeds immediately before building the model. Nothing links that checkpoint to a B4 arm, so it is not one.
  - The principal-experiment checkpoints are a different experiment and were not substituted.
- The ordinal accuracy (0.047–0.199) and MAE (1.38–1.84) therefore **cannot be corrected**, and no corrected value is given.
- The binary AUROC, macro-F1 and balanced-accuracy columns of that file do not use the decoder and are unaffected.

**Manuscript.** Remove the 1.4–1.8-stage ordinal error and any ordinal-accuracy figure from B4. Any H2 keep/kill statement may rest only on the binary columns, described as 5 of 8 folds, single seed.

---

## 2. Shared-error analysis: decisions (demonstrated error)

**Evidence.** `baselines_b3.py` computed `pred_test = p_test >= threshold` in memory and saved it. `consensus_failures.py` re-derived the decision from the CSV text instead, using pandas' default float parser. The tuned thresholds come from `np.arange` (0.45000000000000007, 0.12000000000000001). Five `bulk_top1000__random_forest` probabilities equal to 0.45 / 0.12 parsed to the same double as the threshold, which flipped the decision.

With round-trip parsing, all 14 baselines × 8 folds reproduce `pred_test` exactly. Sample order matches `folds/*/test/sample_ids.csv` and `y_true` matches `labels.parquet` in every file.

| Record | Cohort, label | Error status: historical → corrected |
|---|---|---|
| GSM6590212 | GSE213621, grouped F0–F1, y=0 | error → correct |
| GSM6590214 | GSE213621, grouped F0–F1, y=0 | error → correct |
| GSM6590269 | GSE213621, exact F2, y=1 | correct → error |
| GSM789158 | GSE31803, grouped F3–F4, y=1 | correct → error |
| GSM789177 | GSE31803, grouped F3–F4, y=1 | correct → error |

**Code.** `scripts/consensus_failures.py`:

- reads with `float_precision="round_trip"`;
- uses the saved `pred_test`, and raises an error if it disagrees with `p_test >= threshold` or if `y_true` disagrees with the labels;
- keeps the GAT ensemble rule unchanged (5-seed mean P(F2+) ≥ 0.5);
- writes to `--out-dir` and refuses to overwrite `consensus_failures/`;
- no longer imports `test_a_adjacency` (which pulls in torch); it uses the new `scripts/stage_labels.py`.

**Result.** The historical error matrix was first rebuilt independently and matches the saved file exactly.

| Models wrong | Historical | Corrected | Change |
|---|---|---|---|
| ≥19/23 | 138 | 139 | GSM789177 enters (18 → 19 models wrong) |
| ≥21/23 | 108 | 108 | none |
| 23/23 | 57 | 57 | none |

- The 231 of 253 pairwise rows that do not involve the changed model are reproduced with 0 deviation. Observed Jaccard moved by at most 0.010 for the 22 pairs that do.
- Overlap with Test A flags: 41 in both and Jaccard 0.273 (historical: 0.275).

**Manuscript.** Change 138 to 139. 57 is unchanged. Regenerate the figures (17, 18, 19, 20) and any table built from them.

---

## 3. Grouped stage labels in stage-level tests (demonstrated error)

**Evidence.** The 1,064 test records are 800 exact-stage and 264 grouped. `labels.parquet` stores the grouped ones under a single stage code:

- the 137 grouped F0–F1 records are stored as F0;
- the 127 grouped F3–F4 records are stored as F3.

The consensus, Test A, Test B and flagging v2/v3 scripts counted them under those codes. As a result:

- a grouped F0–F1 record (possibly an F1, i.e. adjacent to the boundary) was scored as a distant F0;
- grouped F3–F4 records inflated F3 in every 2×5 table.

**Code.**

- New `scripts/stage_labels.py` (torch-free). `enrichment_test` uses exact-stage records by default and raises an error if the `resolution` column is missing. `grouped_table` reports the grouped records separately.
- `test_a_adjacency.py`, `test_b_rate.py`, `consensus_failures.py` and `flagging_v2.py` (whose metric functions are shared by v3 and v4) now compute stage metrics on exact records only. Binary quantities (flag counts and rates, stability, direction, consensus counts) still use all 1,064 records.
- `flagging_v3.py` and `flagging_v4.py` now carry `resolution` in their frames.
- The patched `describe_variant` reproduces the independent exact-stage results for 12 of 12 flagging-v2 variants.

**Result.** The table is from `outputs/stage_enrichment/`: one-sided Fisher OR (p), flagged n in the exact subset. The `as_stored` coding reproduces all 15 historical summaries exactly.

| Flag set | Historical (as stored) | Exact-stage | Exact, without GSE213621 |
|---|---|---|---|
| Test A detector (52 flags) | 2.73 (0.00048) | 2.57 (0.0049), n=45 | 2.16 (0.030), n=33 |
| Consensus ≥19/23 | 2.03 (8.4e-05) [138 set] | 2.47 (5.0e-05), n=108 | 2.99 (1.6e-06), n=100 |
| Consensus ≥21/23 | 2.06 (3.1e-04) | 2.63 (1.3e-04), n=84 | 3.11 (1.1e-05), n=77 |
| Consensus 23/23 | 1.54 (0.073) | 1.87 (0.046), n=43 | 2.30 (0.011), n=41 |
| v2 V1 baseline | 3.01 (0.00024) | 2.78 (0.0034), n=43 | 2.41 (0.018) |
| v2 V2 combined/full logreg | 2.51 (0.0052) | 2.00 (0.051), n=34 | 1.97 (0.068) |
| v2 V3 stable ≥12/15 | 3.21 (0.036) | 2.37 (0.14), n=13 | 2.14 (0.21) |
| v2 V4 empty graph @52 | 1.63 (0.057) | 2.10 (0.029), n=39 | 2.37 (0.015) |
| v3 CL pruned, real graph | 2.28 (6.2e-07) | 2.20 (4.3e-05), n=146 | 2.32 (3.1e-05) |
| v3 CL pruned, empty graph | 1.87 (0.00034) | 1.62 (0.012), n=123 | 1.75 (0.006) |

Graph-comparison verdicts in flagging v2 (Δlog-OR real/shuffled minus empty, paired bootstrap):

| Comparison | Historical verdict | Exact-stage verdict |
|---|---|---|
| real @52 | BETTER, +0.61 [0.09, 1.19] | no difference, +0.59 [−0.22, 1.47] |
| real @192 | BETTER, +0.35 [0.06, 0.64] | no difference, +0.02 [−0.33, 0.38] |
| shuffled @52 | BETTER | no difference |

Caveat for the exact-stage verdicts: the original matched-count flags are kept, so the two arms are not count-matched within the exact subset. For example, at @52 the real graph has 43 exact-stage flags and the empty graph has 39.

Two things to keep in mind when reading the table:

- The direction of the change differs between flag sets, so no single correction factor applies.
- Some p-values cross 0.05 in each direction. These are retrospective, multiplicity-unadjusted re-analyses and must be described that way.

Planted-error eligibility was checked independently and is valid:

| Records | Eligible F1 (y=0) | Eligible F2 (y=1) | Total eligible |
|---|---|---|---|
| Exact-stage | 179 | 291 | 470 |
| Grouped | 0 | 0 | 0 |

The pools are read from `folds/*/test/y.csv`, and the stage and resolution there match `labels.parquet`. The planted-error experiment is unchanged.

Subset composition (`exact_stage_subset_composition.csv`):

- GSE31803 contributes 0 exact-stage records.
- GSE213621 contributes 107 of 299 records, all of them exact F2 (F2+ prevalence 1.00 vs 0.68 for the whole cohort).
- GSE213621's share falls from 28.1% of all records to 13.4% of the exact subset.

**Manuscript.** See `MANUSCRIPT_CHANGES.md` §3. Replace the figures with the corrected fig19, fig21 and figS.

---

## 4. Aggregation, null references and ablation numbers (labelling and arithmetic errors)

**AUROC.** All values are re-derived from the saved predictions. All 440 runs reproduce `master_results.csv` and `master_ablation_results.csv` to ≤3.3e-16, and every saved prediction equals the argmax. The primary predictions and metrics are therefore unchanged.

| Model | Mean per-seed AUROC | 5-seed-ensemble AUROC |
|---|---|---|
| CoDA-GAT off, m=0 | 0.82246 (between-cohort SD 0.02669) | 0.83017 (SD 0.02920) |
| CoDA-GAT marginal | 0.82073 | 0.82794 |
| CoDA-GAT conditional | 0.82139 | 0.82880 |
| m=1 | 0.82498 | 0.82945 |
| m=3 | 0.82614 | 0.83393 |
| Empty graph | 0.82683 | 0.83038 |
| Shuffled graph | 0.82900 | 0.83724 |
| Random graph | 0.82656 | 0.83682 |

Baselines have a single seed, so they have one AUROC each:

| Baseline | AUROC (between-cohort SD) |
|---|---|
| CatBoost, `panel72_full` and `panel72_combined` (tied for highest) | 0.84159 (0.04431 and 0.04921 respectively) |

Sources of the README spreads:

| README value | Source |
|---|---|
| CoDA-GAT ± 0.030 | SD over the 40 individual runs (0.0298) |
| Empty graph ± 0.044 | SD over the 40 individual runs (0.0435) |
| Shuffled graph ± 0.050 | SD over the 40 individual runs (0.0495) |
| CatBoost ± 0.047 | Neither model's between-cohort SD; not reproducible (the mean of the two is 0.0468) |

The README therefore mixes run-level and cohort-level spreads.

**Null references.**

- The historical summary reported the analytical independence expectation, E[J] = p₁p₂/(p₁+p₂−p₁p₂), as "chance", while describing the null as a within-fold permutation. These are two different references.
- The corrected outputs give both: an `analytical` column and a `perm_null_mean` column.
- Between architecture families, each family pair weighted equally:

| Between-family Jaccard | Corrected |
|---|---|
| Observed | 0.439 |
| Analytical reference | 0.166 |
| Permutation mean | 0.168 |

- Between architecture families, each model pair weighted equally:

| Between-family Jaccard | Corrected |
|---|---|
| Observed | 0.441 |
| Analytical reference | 0.165 |
| Permutation mean | 0.167 |

- Permutation z is 18.4–18.6.

**Ablations.** Recomputed from the saved tables; these match `ablations_b9/ablation_report.txt`:

| Ablation | Metric | Change (ablation − full) | Wilcoxon p (two-sided, n=8) |
|---|---|---|---|
| Empty graph | macro-F1 | **+0.0240** (not +0.010) | 0.74 |
| Empty graph | AUROC | +0.0044 | 0.95 |
| No fold-fitted features | macro-F1 | −0.0343 | **0.0156** |
| No fold-fitted features | balanced accuracy | −0.0306 | 0.0078 |
| No fold-fitted features | AUROC | −0.0110 | **0.38** |

The +0.0105 that appears in the README is the masking "flatness advantage" for macro-F1, a different quantity. `make_figures.fig24` already aggregates correctly and needs no change.

**Code.** No pipeline code was changed for item 4 beyond the consensus summary text and columns: `weighting`, `mean_jaccard_perm_null_mean` and `mean_jaccard_excess_vs_perm`. The recomputation is in `analysis/aggregation.py`.

---

## 5. Experiment descriptions (description errors; design unchanged)

`outputs/descriptions/` holds the evidence for each item below.

- **Shuffled graph.** `ablations_b9` applies `X[:, perm, :]` over the unchanged STRING edge list: node features are permuted and the edges are kept. The graph is still symmetric (321 pairs / 642 directed). Only 28–53 of the 321 gene pairs that meet along an edge are true STRING pairs, and 0–3 nodes keep their own features.
- **Random graph.** It draws 642 unique *ordered* pairs and is not symmetrised. That gives 595–607 unordered pairs (STRING has 321), and only 70–94 directed entries have their reverse present. It matches STRING's directed-edge count but not its symmetry or its unordered-pair coverage. The reconstruction matches `learned_graph/edge_budget_audit.csv` for all 8 folds.
- **Learned graphs.** Each `learned_edges_*.csv` file stores 321 unordered pairs once. `undirected_pairs_to_edge_index` mirrors them into 642 directed entries. This was verified for both variants in all 8 folds.
- **Masking.** `graph_utility.apply_mask` replaces only features 0–7 (Group A) of the masked nodes with training means. It recomputes nothing else. In particular, other nodes' Group A features (expression and deviation z-scores, pathway scores) are not rebuilt, because no preprocessing code exists to do so. Describe it as *feature perturbation of selected nodes*, not gene removal.
- **Augmentation.** `results_b7_m1` and `results_b7_m3` each contain 40/40 completed runs with `master_results.csv`, so "m=1/m=3: NOT YET RUN" in `ablations_b9/ablation_report.txt` is stale. m=7 was never run, although `README.txt` asks for all four rows.
- **`directional_ps`.** It is configured in `shared/handoff.json` (Head 3 arms; decision "deferred_to_ablation_row_B"). However, every training script uses `deviation_score`, which equals the `directional` column exactly. No output in the archive trains or evaluates a `directional_ps` model.
- **Planted-error recovery.** Labels were flipped at exact F1/F2 records, so recovery measures whether flags find *synthetic boundary flips*. It does not validate the detection of real biopsy errors. The graph conditions differ in recall only within that synthetic setting, so it also does not establish that the graph is superior.

---

## 6. Limitations

These are documented, not fixed; see `UNRESOLVED.md`. Global ComBat and Group C's dependence on training and early-stopping labels were not changed. Changing them would require retraining, and keeping the historical numbers alongside a changed pipeline would misstate what was evaluated.

## 7. Script edits made with the repository cleanup

Three scripts were edited after the results were produced. None changes a computation:

- `scripts/baselines_b3.py`: reads `shared/healthy_reference.parquet`, the deposited file and the file the version that produced the saved baseline results read. A later commit had switched it to `geo_control_reference.parquet`, which is not in the deposit; that name is kept as a fallback.
- `scripts/ablations_b9.py`, `scripts/flagging_v3.py`: a printed hint no longer names a local interpreter path.
