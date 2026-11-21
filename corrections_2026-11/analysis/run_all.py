"""
Analysis-only reproduction of the 2026-11 corrections.

    corrections_2026-11/run_analysis_only.sh            (or)
    <python> -I corrections_2026-11/analysis/run_all.py

Cannot train: guard.install() makes torch, torch_geometric, catboost, xgboost
and every training/inference script unimportable before any analysis code
runs. The CORN check runs in a subprocess that allows torch but still blocks
every training module. Reads saved files only; writes only to
corrections_2026-11/outputs/ (deleted and rebuilt on each run). Every tracked
file outside scripts/ and corrections_2026-11/ is hashed before and after the
run, and the run fails if any changed.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard  # noqa: E402

guard.install()

import pandas as pd  # noqa: E402

import aggregation  # noqa: E402
import consensus_run  # noqa: E402
import descriptions  # noqa: E402
import figures  # noqa: E402
import stage_enrichment  # noqa: E402


def protected_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=guard.REPO, capture_output=True, check=True).stdout
    files = [f for f in out.decode().split("\0") if f]
    return [f for f in files if not f.startswith(("scripts/", "corrections_2026-11/"))]


def snapshot(files):
    h = {}
    for f in files:
        p = guard.REPO / f
        h[f] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    return h


def main():
    t0 = time.time()
    assert guard.OUT.parent == guard.CORR and guard.OUT.name == "outputs"
    if guard.OUT.exists():
        shutil.rmtree(guard.OUT)
    guard.OUT.mkdir(parents=True)

    files = protected_files()
    before = snapshot(files)
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[run_all] {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    # 1. CORN decoder (subprocess; torch allowed, training modules blocked)
    r = subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("corn_check.py"))],
                       cwd=guard.REPO, env=dict(os.environ))
    corn = json.loads((guard.OUT / "corn_check" / "corn_check.json").read_text())
    check("CORN decoder matches official coral_pytorch decoding", r.returncode == 0 and corn["pass"],
          f"{corn['reference']}; {corn['new_vs_reference_mismatches']} mismatches on {corn['random_n']} random "
          f"inputs; historical rule disagreed on {corn['old_vs_reference_mismatch_rate']:.1%}")

    # 2. consensus analysis with saved baseline decisions
    c = consensus_run.main()
    check("historical consensus error matrix reproduced before correcting", c["reproduced_historical"],
          "default-parse rebuild == consensus_failures/error_matrix.csv")
    counts = {r["level"]: r for r in c["counts"]}
    check("consensus counts recomputed", True,
          f">=19/23: {counts[0.8]['historical']} -> {counts[0.8]['corrected']}; "
          f"23/23: {counts[1.0]['historical']} -> {counts[1.0]['corrected']}; "
          f"{c['n_changed_decisions']} model decisions changed")
    check("pairwise rows not involving the changed model are unchanged", c["untouched_pairs_max_dev"] == 0,
          f"max deviation {c['untouched_pairs_max_dev']}")

    # 3. exact-stage enrichment
    s = stage_enrichment.main()
    check("as-stored coding reproduces every historical enrichment summary", s["checks_pass"],
          "flagging_v2 (12 variants + graph_comparison), flagging_v3 CL pruned (2), Test A 2x2")
    el = s["eligibility"].set_index("resolution")
    check("planted-error pools contain exact-stage records only",
          int(el.loc[["grouped_F0_F1", "grouped_F3_F4"], "eligible_total"].sum()) == 0,
          f"eligible exact = {int(el.loc['exact', 'eligible_total'])}, grouped = "
          f"{int(el.loc[['grouped_F0_F1', 'grouped_F3_F4'], 'eligible_total'].sum())}")

    # 4. aggregation, primary-metric invariance, ablations
    a = aggregation.main()
    check("saved primary metrics re-derived from saved predictions", a["invariance_ok"],
          "AUROC / macro-F1 / balanced accuracy for all 440 B7, m=1, m=3 and B9 runs; predictions == argmax")

    # 5. descriptions
    d = descriptions.main()
    check("graph reconstructions match learned_graph/edge_budget_audit.csv", d["audit_match"],
          "random_graph and learned graphs, all 8 folds")

    # 6. figures
    made = figures.main()
    check("corrected figures written", len(made) == 6, ", ".join(made))

    after = snapshot(files)
    changed = [f for f in files if before[f] != after[f]]
    check("no historical file modified by this run", not changed,
          f"{len(files)} tracked files outside scripts/ hashed before and after; changed: {changed[:5]}")

    pd.DataFrame(checks).to_csv(guard.OUT / "validation_checks.csv", index=False)
    lines = ["# Validation checks (generated by run_all.py)", "",
             f"python {sys.version.split()[0]}, pandas {pd.__version__}; runtime {time.time() - t0:.0f} s", "",
             "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {c['check']} | {'PASS' if c['pass'] else 'FAIL'} | {c['detail']} |" for c in checks]
    (guard.OUT / "VALIDATION.md").write_text("\n".join(lines) + "\n")
    ok = all(c["pass"] for c in checks)
    print(f"[run_all] {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} -> {guard.OUT}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
