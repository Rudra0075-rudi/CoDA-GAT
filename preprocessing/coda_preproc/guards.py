"""Leakage and integrity guards (notebook Cell 4), unchanged in behaviour."""
import numpy as np
import pandas as pd

SYN_PREFIX = "SYN_"

TIER_RULES = {  # can_train, can_calibrate, can_test
    "T1": (True, True, True), "T2": (True, False, False), "T3": (True, False, False),
    "T4": (False, False, False), "AUDIT": (False, False, False),
    "REJECTED": (False, False, False),
}


class GuardViolation(AssertionError):
    pass


def check_expression(df, name):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"{name}: missing or empty matrix")
    if not df.index.is_unique or not df.columns.is_unique:
        raise ValueError(f"{name}: duplicate gene/sample IDs")
    if df.index.hasnans or df.columns.hasnans:
        raise ValueError(f"{name}: missing gene/sample IDs")
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: NaN or infinite expression values")
    return df


def assert_no_synthetic(ids, split, fold="NA"):
    bad = [s for s in map(str, ids) if s.startswith(SYN_PREFIX)]
    if bad:
        raise GuardViolation(f"fold {fold}: {len(bad)} synthetic samples in {split}")


def assert_cohort_disjoint(train, cal, test, fold="NA"):
    tr, ca, te = set(train.cohort), set(cal.cohort), set(test.cohort)
    if len(te) != 1:
        raise GuardViolation(f"fold {fold}: test must be one cohort, got {te}")
    if te & tr or te & ca:
        raise GuardViolation(f"fold {fold}: test cohort also in train/calibration")


def assert_tier_usage(df, split, fold="NA"):
    slot = {"proper_train": 0, "calibration": 1, "test": 2}[split]
    for coh, tier in df[["cohort", "tier"]].drop_duplicates().itertuples(index=False):
        if not TIER_RULES[tier][slot]:
            raise GuardViolation(f"fold {fold}: {coh} (tier {tier}) not allowed in {split}")


def assert_tensor_integrity(arr, name="tensor"):
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise GuardViolation(f"{name}: NaN or Inf values")
