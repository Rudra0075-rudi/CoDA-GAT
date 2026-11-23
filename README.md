# CoDA-GAT

**Co**hort-invariant **D**eviation-**A**ware **G**raph **AT**tention Network for detecting
significant liver fibrosis (F2+) from bulk liver transcriptomics.

BRAC University undergraduate thesis · Team T25301033 · Supervisor: Dr. Amitabha Chakrabarty

---

## What this is

A multi-task graph attention network evaluated under **leave-one-dataset-out (LODO)**
cross-validation across 8 public GEO cohorts, with conformal prediction for per-sample
uncertainty and a cohort adversary for batch invariance.

This repository contains the complete experimental record: every
script, every saved run and every analysis output, including the results that did not go
our way. The processed dataset is deposited separately (see Data). The negative findings are in here on purpose. A reader who can only find the
favourable half of a study has no reason to believe the favourable half.

## Headline results

AUROC averaged over the **8 held-out cohorts**; ± is the between-cohort SD (ddof=1). GAT
rows average 5 training seeds per cohort; the baselines were run once (single seed), so
compare the means, not the spreads. Source: `corrections_2026-11/outputs/aggregation/summary.txt`.

| Model | Cross-cohort AUROC |
|---|---|
| Best tabular baseline (`panel72_full__catboost`, single seed) | **0.842 ± 0.044** |
| CoDA-GAT, primary configuration (adversary off, m=0), mean per seed | **0.822 ± 0.027** |
| CoDA-GAT, same, five-seed ensemble | 0.830 ± 0.029 |
| Empty graph (same network, no edges), mean per seed | 0.827 ± 0.044 |
| Shuffled STRING graph, mean per seed | 0.829 ± 0.045 |

`panel72_combined__catboost` ties at 0.842 (SD 0.049). Earlier versions of this README gave
0.842 ± 0.047 and 0.822 ± 0.030: the first SD is not reproducible from the saved
predictions, and the second is the SD over 40 runs, not between cohorts.

Three things follow, and we state them plainly:

1. **CoDA-GAT does not beat a well-tuned gradient-boosted baseline.** These results do not establish an accuracy ceiling or prove that disputed labels are wrong.
2. **The STRING graph did not improve mean AUC in these experiments.** Empty, shuffled and random graphs
   tie or narrowly beat the real biological topology. This is reproduced independently in
   `ablations_b9/`, `graph_utility/` and `learned_graph/`.
3. **Graphs learned from the data do not rescue it either.** Co-expression and end-to-end
   learned topologies, held to STRING's exact edge budget, do not beat the empty graph under
   masking. The end-to-end variant's overlap with STRING (Jaccard 0.063) is at chance, and its
   cross-fold stability is 0.312. These observations alone do not establish the cause of instability.

The one place the graph does something measurable is **robustness to missing genes**, and even
there the result is metric-dependent and is reported as such:

| Flatness advantage vs no-graph, over 40 cohort × masking-rate cells | |
|---|---|
| AUC | **−0.0077** [−0.0100, −0.0055] — the graph degrades **faster** |
| macro-F1 | **+0.0105** [+0.0010, +0.0209] — the graph degrades **slower** |
| balanced accuracy | **+0.0094** [+0.0021, +0.0175] — the graph degrades **slower** |

Ranking quality falls faster with the graph; thresholded decision quality falls more slowly.
There is no single-number "robustness advantage", and any claim of one has to name its metric.

## Data

1,366 supervised samples across 12 cohorts; 8 of them (1,064 samples) form the LODO folds.
Node features are a (72 genes × 26 features) tensor per sample: 8 dynamic + 9 static +
9 fold-fitted. The graph is 72 nodes and 321 undirected STRING edges at confidence ≥ 700.

Each fold's calibration set is a **single separate cohort**, never a random split — coverage must still be evaluated empirically under cohort shift; using a separate cohort does not itself establish exchangeability or a coverage guarantee.

**The dataset is not stored in this repository.** It is deposited on Zenodo as
`Archive_RSOS.zip`, DOI [10.5281/zenodo.24470811](https://doi.org/10.5281/zenodo.24470811)
("Dataset for: Can gene networks help detect significant liver fibrosis? Graph attention and
tabular models tested on unseen patient cohorts"): `shared/` (expression matrices, the 148
controls, labels, graph, panel) and the 8 LODO folds (`train/`, `calibration/`, `test/` tensors
and `fold_info.json`), 94 files, each byte-identical to the archived 5 Aug 2026 handoff, with
SHA-256 checksums and a README describing the layout. To restore the inputs every script reads:

```bash
unzip Archive_RSOS.zip
cp -R CoDA-GAT_data/shared CoDA-GAT_data/folds .   # merges into folds/; no file overlaps git
```

The fold-0 model outputs in `folds/fold0_GSE130970/outputs*/` stay in git. The data files were
removed from git on 22 Nov 2026; each was SHA-256 identical to its archived copy.
`preprocessing/` rebuilds the dataset from public GEO data (see `preprocessing/README.md`).

## Layout

```
scripts/              training and analysis scripts (see below)
shared/               graph, gene panel, labels, expression matrices  [Zenodo deposit]
folds/foldN_GSE*/     the LODO protocol: train/ calibration/ test/  [Zenodo deposit]
preprocessing/        preprocessing from GEO + check against the archive
corrections_2026-11/  analysis-only corrections to the reported numbers (no training)
results_b7/           the main 120-run experiment (8 folds x 3 modes x 5 seeds)
results_b7_m{1,3}/    augmentation-ratio arms
baselines/            tabular, hybrid and graph-control baselines
ablations_b9/         component ablations + attention analysis
graph_utility/        does the graph help on any axis? (masking, edges, embeddings)
learned_graph/        learned topologies vs STRING, at a matched edge budget
decomposition_b8/     degradation decomposition (Shapley-symmetrised)
safety_checks_b5/     the gate that had to pass before the main experiment ran
verification_checks/  APS conformal bug diagnosis + attention kill test
test_a/b/c            adjacency, rate-consistency and label-cleaning experiments
consensus_failures/   do 23 models fail on the same samples?
figures/              publication figures + diagnosis.txt (superseded/ = not used in the paper)
preregistration_b6.*  hypotheses and analysis plan, locked before B7 ran
```

## Reproducing

Set up the environment once:

```bash
conda env create -f environment.yml
conda activate coda-gat
```

First restore the data from the Zenodo deposit (see Data); every step below reads `shared/`
and `folds/`.

**Check the reported numbers without training (minutes).** This recomputes every corrected
number in the paper from the saved predictions and cannot launch training:

```bash
corrections_2026-11/run_analysis_only.sh
```

**Rebuild the dataset from GEO (hours).** Run `preprocessing/preprocessing_clean.ipynb` from
`preprocessing/`, or check the downstream steps against the deposit with
`preprocessing/verify_against_archive.py` (see `preprocessing/README.md`).

**Re-run the experiments (days on one GPU).** Order matters; later blocks import from earlier ones.

```bash
python scripts/train_fold0_b1.py          # model + one fold, proof of life
python scripts/baselines_b3.py            # baselines
python scripts/safety_checks_b5.py        # GATE: must pass before B7
python scripts/run_full_experiment_b7.py  # the main 120 runs (long)
python scripts/decomposition_b8.py
python scripts/ablations_b9.py
python scripts/graph_utility.py
python scripts/learned_graph.py
python scripts/verification_checks.py
python scripts/make_figures.py            # figures
python scripts/fix_figures.py             # diagnosis + corrected figs 4,5,11-14
```

`run_full_experiment_b7.py` is resumable and saves incrementally: re-running skips any
`(fold, mode, seed)` that already has a `DONE.marker`.

## Things a reader should know

- **The conformal APS implementation had a bug.** The randomization term was omitted, which
  makes APS degenerate at K=2 classes: 19.4% of calibration scores are exactly 1.0, so the
  quantile pins at 1.0 and every prediction set contains both classes. `verification_checks/`
  diagnoses it and supplies corrected values; the figures use the corrected numbers and show
  the original alongside. The LAC results were never affected.
- **The baseline summaries in `baselines/summary_*.csv` are single-cohort (GSE130970) values,
  not cross-cohort means.** The 8-cohort means are recomputed from
  `baselines/per_sample_predictions/`; `figures/diagnosis.txt` has the full before/after
  table. Figures mark any model that was only ever evaluated on one cohort.
- **`graph_utility/summary.txt` was overwritten by a partial re-run** and retains only its
  TEST 3 section.
- **Cross-fold statements are underpowered at n=8** and are descriptive, not confirmatory.
  Sample-level analyses must also account for dependence within cohorts; a larger patient count alone does not establish adequate power.

## Not reported in the paper, and why

These outputs stay in the repository as part of the record, but the paper does not report them.

| Item | Where | Why |
|---|---|---|
| CORN ordinal head and its stage error | `h2_decision/` | The decoder had an error and the test logits and checkpoints were not saved (`corrections_2026-11/UNRESOLVED.md`, item 1). |
| "Ablation Q" | none | No script, output or file exists. |
| m=7 augmentation arm, `directional_ps` deviation target | none | Planned or configured, never run. Listed as deviations from the analysis plan. |
| Test B rate sweep and fig22 | `test_b_rate/`, `figures/superseded/` | Pools not saved, dropout not seeded (item 6). |
| Flagging v4 task-1 stage metrics | `flagging_v4/` | Used grouped stage codes; would need new inference (item 7). |
| "Label-noise ceiling" band in fig04 | `figures/superseded/` | An assumption-based sensitivity range, not a measurement. |
| Label-cleaning experiment (+0.016 macro-F1, p = 0.31) | `test_c_cleaning/` | Not significant; supplement as a null at most. |
| NanoString external data (GSE163211) | `nanostring` (not deposited) | Not a test of the F2+ endpoint. |

## Licence and data provenance

Code: MIT (`LICENSE`). Processed data deposit: CC BY 4.0. All expression data derive from
public GEO series: GSE130970, GSE135251, GSE162694, GSE193066, GSE207310, GSE213621, GSE240729,
GSE276114, GSE31803, GSE48452, GSE84044 and GSE89632. References: GTEx v8 (liver) and STRING
v12.0. Please cite the original studies for the data. Processed data: [10.5281/zenodo.24470811](https://doi.org/10.5281/zenodo.24470811).
Software DOI: **ZENODO_SOFTWARE_DOI_TBD** (to be added once a GitHub release is archived on Zenodo).

## Project context

This repository records undergraduate thesis experiments on graph learning for liver fibrosis staging. The accepted iCACCESS 2026 paper **Graph Attention Versus Tabular Models for Early Liver Fibrosis Staging: A Leakage-Controlled Multi-Cohort Benchmark** is a separate research output; do not assume its dataset, protocol or results are identical to this repository.

Research code only; not validated for clinical diagnosis. The empirical results above are retained from the existing experiment record and were not rerun during the documentation update.
