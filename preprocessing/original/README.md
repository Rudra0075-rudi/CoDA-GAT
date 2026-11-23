# The October 2026 notebook (kept unchanged, for provenance)

`preprocessing_2026-10-08.ipynb` is the preprocessing notebook as supplied on 21 Nov 2026, with
the outputs of its Colab run of 8 Oct 2026 (working folder `CoDAGAT_preprocessing_fix_v1`).
It is a later revision of the 30 Aug 2026 notebook that `../preprocessing_clean.ipynb` is
built on. It shares the August acquisition and label code and adds later cells, including
Cell 18b (source-processing review) and Cell 24b (a different reload of GSE240729).

It does not reproduce the archived matrices (5 Aug 2026) that every reported result was
trained on:

| | Archive / `preprocessing_clean.ipynb` | This notebook (8 Oct run) |
|---|---|---|
| Genes after integration | 10,735 | 10,732 |
| GSE240729 no-ComBat median | 4.62 | 4.17 |
| Group C Cohen's d | per-class variance ddof=0 | ddof=1 |

Use `../preprocessing_clean.ipynb` to rebuild the deposited data.
