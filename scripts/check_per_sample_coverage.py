"""
Verification helper: which models does consensus_failures.py actually see?

Reads the SAME registry and the SAME loaders consensus_failures.py uses, so
what this prints is exactly what that analysis will find -- it is not a
re-implementation that could drift. Run it after a B3 persistence re-run to
confirm the linear / tree_ensemble / hybrid / mlp families are present before
spending time on the full consensus analysis.

Read-only: loads nothing but saved predictions, writes nothing.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from collections import defaultdict

import pandas as pd

from consensus_failures import (
    MODEL_REGISTRY, SEEDS, fold_test_index, load_model_probs,
)


def main():
    parser = argparse.ArgumentParser(description="which models have usable per-sample predictions?")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--quiet-loader", action="store_true",
                         help="suppress the per-model MISSING lines the loaders print")
    args = parser.parse_args()

    index_df = fold_test_index()
    print(f"held-out test samples: {len(index_df)} across {index_df['fold'].nunique()} folds\n")

    if args.quiet_loader:
        import consensus_failures
        consensus_failures.log = lambda *_a, **_k: None

    present, absent = defaultdict(list), defaultdict(list)
    for name, spec in MODEL_REGISTRY.items():
        loaded = load_model_probs(name, spec, index_df, args.seeds)
        (present if loaded is not None else absent)[spec["architecture_family"]].append(name)

    n_present = sum(len(v) for v in present.values())
    n_absent = sum(len(v) for v in absent.values())

    print(f"{'=' * 70}\nAVAILABLE ({n_present} models)\n{'=' * 70}")
    for fam in sorted(present):
        print(f"  {fam:26s} ({len(present[fam])}): {', '.join(sorted(present[fam]))}")

    if absent:
        print(f"\n{'=' * 70}\nNOT AVAILABLE ({n_absent} models)\n{'=' * 70}")
        for fam in sorted(absent):
            print(f"  {fam:26s} ({len(absent[fam])}): {', '.join(sorted(absent[fam]))}")

    families = sorted(present.keys())
    print(f"\narchitecture families present ({len(families)}): {families}")

    if len(families) >= 2:
        print("\nDIVERSITY GATE: PASS -- consensus_failures.py will produce a real verdict")
        print("(between-family agreement is computable, so shared-input correlation can be")
        print("distinguished from convergent failure).")
    else:
        print("\nDIVERSITY GATE: FAIL -- only one architecture family is present, so")
        print("consensus_failures.py will report NOT ESTABLISHED regardless of the adjacency")
        print("result. Run the B3 persistence re-run first.")

    rows = [{"model": m, "architecture_family": f, "available": True}
            for f, ms in present.items() for m in ms]
    rows += [{"model": m, "architecture_family": f, "available": False}
             for f, ms in absent.items() for m in ms]
    print("\n" + pd.DataFrame(rows).sort_values(["available", "architecture_family", "model"],
                                                 ascending=[False, True, True]).to_string(index=False))


if __name__ == "__main__":
    main()
