# Superseded figures (not used in the paper)

| Figure | Why it is not used |
|---|---|
| `fig04_all_models_auc_ranked` | Draws a shaded "label-noise ceiling" band (0.84–0.94, `CEILING` in `scripts/make_figures.py`). That band is an assumption-based sensitivity range, not a measurement, so it must not appear as a ceiling on the results. The AUROC ranking itself is unaffected. |
| `fig22_testB_operating_curve` | Test B rate sweep. Its MC-dropout pools were not saved and dropout was not seeded, so it cannot be recomputed or matched to Test A (`corrections_2026-11/UNRESOLVED.md`, item 6). |

`figures/all_figures.pdf` still contains both pages as generated.
