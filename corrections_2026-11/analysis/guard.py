"""
Analysis-only guard for the 2026-11 correction run.

Installed before anything else is imported. It makes training impossible
rather than merely unlikely:

  * torch, torch_geometric, catboost, xgboost and every project script that
    defines or launches training cannot be imported at all (ImportError);
  * the only project modules that may be imported are the torch-free analysis
    modules listed in ALLOWED_SCRIPT_MODULES.

The CORN decoder check is the one step that needs torch. It runs in a
separate subprocess (corn_check.py) that blocks the training modules and
torch_geometric but allows torch itself, and it never imports
train_fold0_b1.py -- it compiles only the corn_predict_stage / corn_loss
function bodies out of that file's source.
"""

import importlib.abc
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
CORR = REPO / "corrections_2026-11"
OUT = CORR / "outputs"

# every scripts/*.py that trains, runs inference, or imports something that does
TRAINING_MODULES = {
    "train_fold0_b1", "train_fold0_b2_onthefly", "baselines_b3", "h2_decision_b4",
    "safety_checks_b5", "lock_prereg_b6", "run_full_experiment_b7", "decomposition_b8",
    "ablations_b9", "attention_analysis_b9", "graph_utility", "learned_graph",
    "verification_checks", "test_a_adjacency", "test_b_rate", "test_c_cleaning",
    "flagging_v2", "flagging_v3", "flagging_v4", "review_fixes", "check_graph_symmetry",
    "check_per_sample_coverage", "fix_figures", "make_figures",
}
ALLOWED_SCRIPT_MODULES = {"consensus_failures", "stage_labels"}
HEAVY_PACKAGES = {"torch", "torch_geometric", "catboost", "xgboost", "lightgbm"}


class _Blocker(importlib.abc.MetaPathFinder):
    def __init__(self, blocked):
        self.blocked = set(blocked)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.blocked:
            raise ImportError(f"analysis-only guard: import of '{fullname}' is blocked "
                              f"(it can train models or launch training)")
        return None


def install(allow_torch=False):
    blocked = set(TRAINING_MODULES)
    blocked |= (HEAVY_PACKAGES - {"torch"}) if allow_torch else HEAVY_PACKAGES
    already = sorted(m for m in sys.modules if m.split(".")[0] in blocked)
    if already:
        raise RuntimeError(f"analysis-only guard installed too late; already imported: {already}")
    sys.meta_path.insert(0, _Blocker(blocked))
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return blocked
