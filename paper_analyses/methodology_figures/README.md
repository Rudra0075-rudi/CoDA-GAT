# Methodology figures

TikZ sources of the four methodology figures, each a standalone LaTeX file drawn at the
journal text width (135 mm).

| File | Figure |
|---|---|
| `fig_methodology_overview.tex` | Overview of the framework: three modules and the five tests |
| `fig_module1_cohorts_labels.tex` | Module 1: cohorts and label audit |
| `fig_module2_expression_reference.tex` | Module 2: expression harmonisation and healthy reference |
| `fig_module3_model_tests.tex` | Module 3: STRING graph, node features, model and framework tests |

```bash
pdflatex fig_methodology_overview.tex
```

Needs a TeX distribution with TikZ and the helvet font package. The edges of the STRING
graph in the Module 3 figure are the 321 edges of `shared/graph_edges.csv`, coloured by the
pathways in `shared/panel_genes.csv` and written into the source; the other pictograms are
schematic.
