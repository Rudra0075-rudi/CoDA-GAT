# Result figures

Draws the result figures of the paper (`fig_p1` … `fig_p26`; `fig_p22` appears only in the
supplementary material, as a historical record) from the committed outputs and the deposited data.

```bash
# from the repository root, with the Zenodo data restored into shared/ and folds/
python paper_analyses/result_figures/results_figures.py . paper_analyses/additional_analyses OUTDIR [N ...]
```

`N` limits the run to the listed figure numbers. Each figure is written to `OUTDIR` as PNG and
PDF, and the numbers it plots to `OUTDIR/figure_data/fig_pN_data.csv`. The script prints
checks against the numbers stated in the text as it runs.

| File | Content |
|---|---|
| `results_figures.py` | Figures p1–p10 and the shared loaders; runs `results_figures_part2.py` for p11–p26 |
| `style.py` | Shared look: 135 mm wide, sans-serif type, one colour per model family |
| `figure_data/` | The numbers behind every figure, as produced by the script |
| `figure_data_in/` | Reference values for figures p4, p11 and p13; the script stops if its own recomputation differs |

Requires matplotlib, pandas, scipy, scikit-learn and pyarrow. With the Liberation Sans font
installed, the output matches the published figures pixel for pixel (checked with
matplotlib 3.11).
