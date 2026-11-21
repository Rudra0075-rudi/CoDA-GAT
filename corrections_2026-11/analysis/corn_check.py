"""
CORN decoder check (issue 1). Runs in its own subprocess: torch allowed,
training modules and torch_geometric blocked (guard.install(allow_torch=True)).

What it does NOT do: import train_fold0_b1.py, build a model, load a
checkpoint, or fit anything. corn_predict_stage and corn_loss are compiled
straight out of the patched source with ast, so the check exercises exactly
the code that h2_decision_b4.py imports.

Reference implementation: coral_pytorch 1.4.0 (MIT), functions
coral_pytorch.dataset.corn_label_from_logits and coral_pytorch.losses.corn_loss.
If the package is importable (set CORAL_PYTORCH_PATH to its install dir) it is
used directly; otherwise the verbatim copies below are used and the report
says so.
"""

import ast
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard  # noqa: E402

guard.install(allow_torch=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

OUT = guard.OUT / "corn_check"


# ---- verbatim from coral_pytorch 1.4.0 (MIT licence, Sebastian Raschka) ----
def _vendored_corn_label_from_logits(logits):
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels


def _vendored_corn_loss(logits, y_train, num_classes):
    sets = []
    for i in range(num_classes-1):
        label_mask = y_train > i-1
        label_tensor = (y_train[label_mask] > i).to(torch.int64)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]

        if len(train_labels) < 1:
            continue

        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]

        loss = -torch.sum(F.logsigmoid(pred)*train_labels
                          + (F.logsigmoid(pred) - pred)*(1-train_labels))
        losses += loss

    return losses/num_examples
# ---------------------------------------------------------------------------


def reference_impl():
    path = os.environ.get("CORAL_PYTORCH_PATH")
    if path:
        sys.path.insert(0, path)
    try:
        from coral_pytorch.dataset import corn_label_from_logits
        from coral_pytorch.losses import corn_loss
        import coral_pytorch
        return corn_label_from_logits, corn_loss, f"coral_pytorch {coral_pytorch.__version__} (installed package)"
    except ImportError:
        return _vendored_corn_label_from_logits, _vendored_corn_loss, \
            "coral_pytorch 1.4.0 (verbatim vendored copy; package not importable)"


def load_project_functions():
    src_path = guard.SCRIPTS / "train_fold0_b1.py"
    tree = ast.parse(src_path.read_text())
    wanted = {"corn_loss", "corn_predict_stage"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted, "corn functions not found in train_fold0_b1.py"
    ns = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(src_path), "exec"), ns)
    return ns["corn_predict_stage"], ns["corn_loss"]


def historical_decoder(logits):
    """The pre-fix corn_predict_stage, kept only to quantify the error."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def conditional_logits_from_distribution(p):
    """CORN logits whose conditional probabilities are exactly those implied by
    the class distribution p: task k = P(y > k | y > k-1)."""
    p = np.asarray(p, dtype=float)
    surv = 1.0 - np.cumsum(p)[:-1]                 # P(y > k), k = 0..K-2
    prev = np.concatenate([[1.0], surv[:-1]])      # P(y > k-1)
    cond = surv / prev
    return torch.tensor(np.log(cond / (1 - cond)), dtype=torch.float64).unsqueeze(0), cond


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    new_decode, project_loss = load_project_functions()
    ref_decode, ref_loss, ref_name = reference_impl()
    report, lines = {"reference": ref_name}, []

    def say(x=""):
        lines.append(x)
        print(f"[corn] {x}", flush=True)

    say(f"reference implementation: {ref_name}")
    say("project functions compiled from scripts/train_fold0_b1.py (module not imported)")

    # 1. random agreement with the official decoder, and with the official loss
    g = torch.Generator().manual_seed(20261121)
    logits = torch.randn(200_000, 4, generator=g, dtype=torch.float64) * 3
    new, ref, old = new_decode(logits), ref_decode(logits), historical_decoder(logits)
    report["random_n"] = int(len(logits))
    report["new_vs_reference_mismatches"] = int((new != ref).sum())
    report["old_vs_reference_mismatch_rate"] = float((old != ref).double().mean())
    report["old_minus_new_mean"] = float((old - new).double().mean())
    report["old_ge_new_always"] = bool((old >= new).all())
    say(f"random logits (n={len(logits)}, K=5): fixed decoder vs reference mismatches = "
        f"{report['new_vs_reference_mismatches']}")
    say(f"  historical decoder disagrees with reference on {report['old_vs_reference_mismatch_rate']:.1%} "
        f"of inputs; it is never lower (old >= new: {report['old_ge_new_always']}), "
        f"mean over-count {report['old_minus_new_mean']:.3f} stages")

    y = torch.randint(0, 5, (512,), generator=g)
    lg = torch.randn(512, 4, generator=g, dtype=torch.float64)
    l_proj, l_ref = float(project_loss(lg, y, 5)), float(ref_loss(lg, y, 5))
    report["loss_project"], report["loss_reference"] = l_proj, l_ref
    say(f"corn_loss: project {l_proj:.12f} vs reference {l_ref:.12f} -> the training loss is the "
        f"official CONDITIONAL loss (unchanged), so the cumulative-product decoder is the matching one")

    # 2. worked examples
    examples = []
    doc = torch.tensor([[14.152, -6.1942, 0.47710, 0.96850], [65.667, 0.303, 11.500, -4.524]])
    for row in doc:
        r = row.unsqueeze(0)
        examples.append({"case": "coral_pytorch docstring example", "logits": [round(float(v), 4) for v in row],
                         "conditional_p": [round(float(v), 4) for v in torch.sigmoid(row)],
                         "cumulative_p": [round(float(v), 4) for v in torch.cumprod(torch.sigmoid(row), 0)],
                         "historical": int(historical_decoder(r)), "fixed": int(new_decode(r)),
                         "reference": int(ref_decode(r))})
    cond = torch.tensor([[0.3, 0.9, 0.9, 0.9]], dtype=torch.float64)
    r = torch.log(cond / (1 - cond))
    examples.append({"case": "conditional [0.3, 0.9, 0.9, 0.9]", "logits": [round(float(v), 4) for v in r[0]],
                     "conditional_p": [0.3, 0.9, 0.9, 0.9],
                     "cumulative_p": [round(float(v), 4) for v in torch.cumprod(cond[0], 0)],
                     "historical": int(historical_decoder(r)), "fixed": int(new_decode(r)),
                     "reference": int(ref_decode(r))})
    # the decoder must return the median of the class distribution the conditionals imply
    for p in ([0.10, 0.15, 0.50, 0.15, 0.10], [0.40, 0.11, 0.10, 0.10, 0.29], [0.05, 0.60, 0.10, 0.10, 0.15]):
        r, c = conditional_logits_from_distribution(p)
        # number of k with P(y > k) > 0.5, i.e. with P(y <= k) < 0.5
        median = int(np.searchsorted(np.cumsum(p), 0.5, side="left"))
        examples.append({"case": f"implied class distribution {p}", "logits": [round(float(v), 4) for v in r[0]],
                         "conditional_p": [round(float(v), 4) for v in c],
                         "cumulative_p": [round(float(v), 4) for v in 1 - np.cumsum(p)[:-1]],
                         "median_class": median,
                         "historical": int(historical_decoder(r)), "fixed": int(new_decode(r)),
                         "reference": int(ref_decode(r))})
    report["examples"] = examples
    say("")
    say("worked examples (rank 0..4 = F0..F4):")
    for e in examples:
        say(f"  {e['case']}")
        say(f"    conditional P(y>k | y>k-1) = {e['conditional_p']}")
        say(f"    cumulative  P(y>k)         = {e['cumulative_p']}")
        extra = f", median of implied distribution = {e['median_class']}" if "median_class" in e else ""
        say(f"    historical = {e['historical']}, fixed = {e['fixed']}, reference = {e['reference']}{extra}")
    report["examples_fixed_equal_reference"] = all(e["fixed"] == e["reference"] for e in examples)
    report["examples_fixed_equal_median"] = all(e["fixed"] == e["median_class"] for e in examples if "median_class" in e)

    # 3. what this means for the historical H2 numbers
    h2 = pd.read_csv(guard.REPO / "h2_decision" / "per_fold.csv")
    report["h2_per_fold_folds"] = h2["fold"].tolist()
    report["h2_historical_mae_range"] = [float(h2["h2_ordinal_mae"].min()), float(h2["h2_ordinal_mae"].max())]
    report["h2_historical_accuracy_range"] = [float(h2["h2_ordinal_accuracy"].min()),
                                              float(h2["h2_ordinal_accuracy"].max())]
    say("")
    say(f"h2_decision/per_fold.csv: folds {report['h2_per_fold_folds']} only; ordinal accuracy "
        f"{report['h2_historical_accuracy_range'][0]:.3f}-{report['h2_historical_accuracy_range'][1]:.3f}, "
        f"MAE {report['h2_historical_mae_range'][0]:.2f}-{report['h2_historical_mae_range'][1]:.2f} stages.")
    say("  These were decoded with the historical (direct-threshold) rule. The file holds per-fold")
    say("  aggregates only; no ordinal logits or B4 checkpoints exist in the archive, so the values")
    say("  cannot be re-decoded and NO corrected value is reported. Mark them for exclusion.")

    # the project casts CORN targets to float32, so agreement is to float32 precision
    report["loss_rel_diff"] = abs(l_proj - l_ref) / abs(l_ref)
    ok = (report["new_vs_reference_mismatches"] == 0 and report["loss_rel_diff"] < 1e-6
          and report["examples_fixed_equal_reference"] and report["examples_fixed_equal_median"])
    report["pass"] = bool(ok)
    say("")
    say(f"CORN CHECK: {'PASS' if ok else 'FAIL'}")
    (OUT / "corn_check.json").write_text(json.dumps(report, indent=2))
    (OUT / "corn_check.txt").write_text("\n".join(lines) + "\n")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
