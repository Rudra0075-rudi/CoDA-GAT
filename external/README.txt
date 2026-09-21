GSE163211 (NanoString, 318 samples) is tier T4 and is NOT processed by
this pipeline. It requires its own stage: RCC parsing, positive-control
and housekeeping normalisation, the 43/72 gene subset, and a 43-node
STRING subgraph.

It CANNOT validate the F2+ boundary: the NASH_F1_F4 group (82 samples)
straddles the decision boundary. Legitimate uses are a specificity probe
on the 236 confirmed non-fibrotic samples, coarse fibrotic detection, and
cross-platform conformal coverage.
