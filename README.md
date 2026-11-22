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
script, every saved run and every analysis output, with the preprocessed dataset on Google Drive (see Data) — including the results that did not go
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

See `README.txt` for the dataset layout and loading instructions, and
`FLAT_VIEW_WARNING.txt` for why the top-level `train/`, `calibration/` and `test/` folders
must not be used for the LODO protocol.

**The dataset is not stored in this repository.** Download `Archive.zip` (336,386,271 bytes,
MD5 `60e281bbb499152b2f9c756d56ad5a0e`,
SHA-256 `d869afaa4e40cf55b515a323a15588828c5a449488e6178e711dce06a68012a7`) from
[Google Drive](https://drive.google.com/file/d/1fhzXBUcLJjSVRjvHdaVMO9Az15mGOZu9/view?usp=drive_link)
and extract only the data folders into the repository root:

```bash
unzip -n Archive.zip 'shared/*' 'folds/*' 'train/*' 'calibration/*' 'test/*' \
  'nanostring/raw/*' 'nanostring/rcc/*' 'nanostring/interim/*' 'nanostring/processed/*' -x '*.DS_Store'
```

Do not unzip the whole archive over the repository: it also holds older copies of `README.md` and
eight scripts, which would overwrite the current ones. The command above restores `shared/`, the per-fold `train/`, `calibration/`
and `test/` splits and `fold_info.json` in `folds/`, the top-level `train/`, `calibration/` and
`test/` copy, and `nanostring/{raw,rcc,interim,processed}/`: the inputs every script reads.
The fold-0 model outputs in `folds/fold0_GSE130970/outputs*/` stay in git. These 444
files were removed from git on 22 Nov 2026; each was SHA-256 identical to its copy in the archive.
`preprocessing/` rebuilds and checks the downstream inputs (see `preprocessing/README.md`).

## Layout

```
scripts/              training and analysis scripts (see below)
shared/               graph, gene panel, labels, expression matrices  [in Archive.zip]
folds/foldN_GSE*/     the LODO protocol: train/ calibration/ test/  [in Archive.zip]
preprocessing/        cleaned preprocessing code + check against the archive
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
  Sample-level analyses must also account for dependence within cohorts; a larger patient count alone does not establish adequate power.

## Licence and data provenance

All expression data derives from public GEO accessions (GSE130970, GSE135251, GSE162694,
GSE193066, GSE213621, GSE240729, GSE31803, GSE84044, and others). Please cite the original
studies for the data. The code in `scripts/` is released for academic use.

## Project context

This repository records undergraduate thesis experiments on graph learning for liver fibrosis staging. The accepted iCACCESS 2026 paper **Graph Attention Versus Tabular Models for Early Liver Fibrosis Staging: A Leakage-Controlled Multi-Cohort Benchmark** is a separate research output; do not assume its dataset, protocol or results are identical to this repository.

Research code only; not validated for clinical diagnosis. The empirical results above are retained from the existing experiment record and were not rerun during the documentation update.
