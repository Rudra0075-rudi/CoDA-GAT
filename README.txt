NOTE (23 Nov 2026): this is the dataset note written with the 5 Aug 2026
archive, kept unchanged below for provenance (several scripts cite it).
It is out of date in three ways:
  - The data are not in this repository. Download the deposit described in
    README.md (Data); its own README.md replaces this note.
  - The flat train/ calibration/ test/ copy of fold 0 and external/
    (NanoString) are not part of the deposit.
  - "Conformal coverage guarantee" (binding rule 1): calibration and test are
    different cohorts, so coverage is nominal, not guaranteed.
------------------------------------------------------------------------

CoDA-GAT PREPROCESSED DATASET
generated 2026-08-05 14:58

LAYOUT
  shared/          assets identical across folds (graph, panel, expression, labels)
  folds/foldN_GSE*/ train/  calibration/  test/   <- the evaluation protocol
  train/ calibration/ test/  flat copy of fold 0 only (see warning file)
  external/        NanoString notes (not yet processed)

Each split folder contains:
  X.npy           (n_samples, 72, 26) float32 node-feature tensor
  y.csv           labels + metadata, indexed by sample_id
  sample_ids.csv  ordering, matches X.npy rows

NUMBERS
  supervised samples      1366   (all real; zero synthetic)
  cohorts with expression 12
  T1 LODO folds           8      (1064 samples)
  T2 training-only        4      (302 samples)
  T4 external             1      (GSE163211, not processed)
  healthy references      148    (held separately, excluded from N)
  genes                   10735
  graph                   72 nodes, 321 undirected STRING edges (score >= 700)
  node features           26 = 8 dynamic + 9 static + 9 fold-fitted

LOADING
  import numpy as np, pandas as pd
  X = np.load('folds/fold0_GSE130970/train/X.npy')     # (1043, 72, 26)
  y = pd.read_csv('folds/fold0_GSE130970/train/y.csv', index_col=0)
  E = pd.read_csv('shared/graph_edges.csv')            # src_idx, dst_idx, weight

TASKS
  y_T1_significant  PRIMARY   F2/F3/F4 = 1 vs F0/F1 = 0
  y_T2_ordinal      SECONDARY F0<F1<F2<F3<F4 (NaN where labels are grouped)
  y_T3_early_adv    SECONDARY F3/F4 = 1 vs F0/F1/F2 = 0

BINDING RULES
  1. Synthetic augmentation goes in train/ ONLY. A synthetic sample in
     calibration/ destroys the conformal coverage guarantee outright.
  2. test/ is one real cohort, whole and untouched.
  3. Synthesis is within-cohort and within-class, or the cohort/stage label
     of the synthetic sample is undefined.
  4. Features 17-25 are already fitted per fold. Do not refit globally.
  Guards are in the notebook (Cell 4). Do not add a bypass flag.

AUGMENTATION MULTIPLIER
  m is an ablation variable, not a setting. Sweep {0, 1, 3, 7} and report all
  four rows. At m=7 the train split reaches roughly 7,400-9,500 per fold.
  N_real stays 1366 - that is the dataset size for the paper.

KNOWN ISSUES FOR THE PAPER
  - GSE213621 lacks ALB/APOA1/APOB/FGA/FGB (annotation gap; retains 373/500
    of the top-expressed genes, so not abundance-filtered).
  - GSE31803 and part of GSE213621 carry grouped labels: usable for T1/T3,
    excluded from the ordinal task.
  - GSE193066 is PLS high-risk enriched; 58 follow-up biopsies dropped to
    prevent patient-level leakage.
  - Head 3 is a 3-arm ablation (off / directional / directional_ps).
  - Both ComBat and no-ComBat expression matrices are provided.
