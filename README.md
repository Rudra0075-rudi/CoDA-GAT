# CoDA-GAT

**Co**hort-invariant **D**eviation-**A**ware **G**raph **AT**tention Network for detecting
significant liver fibrosis (F2+) from bulk liver transcriptomics.

BRAC University undergraduate thesis · Team T25301033 · Supervisor: Dr. Amitabha Chakrabarty

---

## What this is

A multi-task graph attention network evaluated under **leave-one-dataset-out (LODO)**
cross-validation across 8 public GEO cohorts, with conformal prediction for per-sample
uncertainty and a cohort adversary for batch invariance.

This repository contains the complete experimental record: the preprocessed dataset, every
script, every saved run, and every analysis output — including the results that did not go
our way. The negative findings are in here on purpose. A reader who can only find the
favourable half of a study has no reason to believe the favourable half.

## Headline results

All values are means over the **8 held-out cohorts** (5 training seeds each), read from the
files in this repository.

| Model | Cross-cohort AUC |
|---|---|
| Best tabular baseline (`panel72_*__catboost`) | **0.842 ± 0.047** |
| CoDA-GAT (primary configuration: adversary off, m=0) | **0.822 ± 0.030** |
| Empty graph (same network, no edges) | 0.827 ± 0.044 |
| Shuffled STRING graph | 0.829 ± 0.050 |
| *Estimated label-noise ceiling* | *0.87 – 0.90* |

Three things follow, and we state them plainly:

1. **CoDA-GAT does not beat a well-tuned gradient-boosted baseline.** Both sit a little below
   the label ceiling implied by inter-pathologist agreement on fibrosis staging. The task, as
   labelled, appears to be near its information limit rather than near a modelling limit.
2. **The STRING graph contributes nothing to accuracy.** Empty, shuffled and random graphs
   tie or narrowly beat the real biological topology. This is reproduced independently in
   `ablations_b9/`, `graph_utility/` and `learned_graph/`.
3. **Graphs learned from the data do not rescue it either.** Co-expression and end-to-end
   learned topologies, held to STRING's exact edge budget, do not beat the empty graph under
   masking. The end-to-end variant's overlap with STRING (Jaccard 0.063) is at chance, and its
   cross-fold stability (0.312) is what fitting noise looks like.

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

Each fold's calibration set is a **single separate cohort**, never a random split — this is
what makes the conformal guarantee meaningful under cohort shift.

See `README.txt` for the dataset layout and loading instructions, and
`FLAT_VIEW_WARNING.txt` for why the top-level `train/`, `calibration/` and `test/` folders
must not be used for the LODO protocol.

**Expression matrices are stored with [Git LFS](https://git-lfs.com).** Install it
(`brew install git-lfs && git lfs install`) before cloning, or the `*.parquet` files arrive as
text pointers. Everything else is plain git.

## Layout

```
scripts/              all 20 analysis scripts (see below)
shared/               graph, gene panel, labels, expression matrices  [LFS]
folds/foldN_GSE*/     the LODO protocol: train/ calibration/ test/
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
figures/              26 publication figures + diagnosis.txt
preregistration_b6.*  hypotheses and analysis plan, locked before B7 ran
```

## Reproducing

Order matters; later blocks import from earlier ones.

```bash
conda activate dagat

python scripts/train_fold0_b1.py          # model + one fold, proof of life
python scripts/baselines_b3.py            # baselines
python scripts/safety_checks_b5.py        # GATE: must pass before B7
python scripts/run_full_experiment_b7.py  # the main 120 runs (long)
python scripts/decomposition_b8.py
python scripts/ablations_b9.py
python scripts/graph_utility.py
python scripts/learned_graph.py
python scripts/verification_checks.py
python scripts/make_figures.py            # 26 figures
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
  Sample-level statements (n ≈ 1,064) are adequately powered. This distinction is enforced in
  the analysis scripts rather than left to the reader.

## Licence and data provenance

All expression data derives from public GEO accessions (GSE130970, GSE135251, GSE162694,
GSE193066, GSE213621, GSE240729, GSE31803, GSE84044, and others). Please cite the original
studies for the data. The code in `scripts/` is released for academic use.
