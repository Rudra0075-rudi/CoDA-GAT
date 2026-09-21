# CoDA-GAT — Pre-Registration (Block B6)

**Drafted:** 2026-08-31
**Status of this document:** This markdown file is the human-readable content of the pre-registration. The *authoritative* timestamp is not the date typed above — it is the ISO-8601 UTC timestamp inside `preregistration_b6.LOCKED.json`, produced by hashing this exact file with `scripts/lock_prereg_b6.py`. If this file is edited after locking, the recorded hash will no longer match its contents, and that mismatch is the whole point: it is what makes the timestamp tamper-evident. **B7 must not be run until this file is locked**, and this file must not be edited after it is locked.

This document exists because the dataset has only **8 evaluation sites** (8 LODO folds). At that sample size there is exactly one statistical comparison that can be run without the result being a foregone conclusion of how many things were tried. This document commits to that comparison, in writing, before any B7 result exists.

---

## 1. Primary contrast — the ONE test

**Contrast:** CoDA-GAT with the cohort-adversary head (H4) in **marginal mode ON**, versus the **same architecture with H4 OFF** (`LOSS_WEIGHTS["h4"] = 0`, matching the same zero-weighting convention already used in Block B4 for H2 — the adversary head exists in both arms with identical initialization, only its loss contribution differs).

This is deliberately **not** "adversary ON in whichever of marginal/conditional mode looks best" — that framing would itself be an undeclared second comparison and a source of post-hoc selection. Conditional-mode adversarial training does not currently exist in `train_fold0_b1.py` (only the plain GRL/marginal cohort classifier is implemented); building it now, immediately before locking a primary contrast, would risk letting implementation choices for the new code be shaped — even unconsciously — by a desire to see a particular result. Marginal-mode is therefore the sole primary contrast. **Conditional mode, if built later, is a secondary/descriptive analysis (Section 3), never a second primary test.**

- **Task:** H1 (significant fibrosis, F2+ vs F0/F1) — the project's primary task.
- **Metric:** macro-F1 on each fold's held-out test cohort.
- **Unit of analysis:** one paired value per fold. Per fold, per arm (ON / OFF), the value is the **mean macro-F1 across the 5 seeds** specified for B7 (8 folds × 3 modes × 5 seeds). This gives exactly 8 paired observations (ON, OFF) — one pair per LODO fold — regardless of the within-fold seed averaging.
- **Test:** two-sided **Wilcoxon signed-rank test** on the 8 paired differences (ON − OFF).
- **Hypothesis and direction:** the pre-specified scientific hypothesis is *directional* — H4 exists specifically to improve cross-cohort generalization, so the predicted direction is macro-F1(ON) > macro-F1(OFF). The test itself is run two-sided (see below for why), but a **PASS additionally requires the observed effect to be in this predicted direction** — a two-sided-significant result in the *opposite* direction is a NULL, not a PASS, and is reported as such.
- **Alpha:** 0.05.
- **Why two-sided despite a directional hypothesis:** the stated p-value floor for an exact two-sided Wilcoxon signed-rank test at n=8 is 2/2⁸ ≈ 0.0078 ≈ **0.008** — this matches the project's own stated p-floor and is the more conservative, standard convention (report the two-sided p-value; require the pre-specified direction as an additional, separate condition for a PASS, rather than halving alpha via a one-sided test). This avoids the appearance of having chosen a one-sided test after the fact to make significance easier to reach.

## 2. Stated power limitation

- n = 8 paired folds. The exact Wilcoxon signed-rank test cannot produce a two-sided p-value below **0.008**, no matter how large the true effect, because 8 paired differences all agreeing in sign is the most extreme possible configuration and that configuration alone has probability 2/2⁸.
- The minimum detectable standardized effect size at n=8, alpha=0.05, power=0.80 is **≈1.15 σ_d** (Cohen's d for paired differences). Anything smaller than this is not reliably detectable at this sample size, regardless of whether it is scientifically real.
- **All results from this project — the primary contrast included — are underpowered relative to conventional standards and must be reported as suggestive, not confirmatory, even in the PASS case.** A PASS with p<0.05 is evidence consistent with a large effect; it is not proof of a moderate-or-smaller one, and a NULL is not proof of no effect.

## 3. Secondary analyses — descriptive only

Everything below is exploratory/descriptive. None of it is a hypothesis test, none of it has a pre-specified direction, and multiplicity across these is **not** corrected for significance (Section 6 explains why, and what the one correction that *is* applied covers).

- The **conditional-mode** adversary contrast (ON-conditional vs OFF, and ON-conditional vs ON-marginal), once/if implemented.
- All other pairwise mode contrasts not covered by the primary test.
- The **augmentation multiplier sweep** m ∈ {0, 1, 3, 7} (Section 5).
- Per-fold AUC, balanced accuracy, and coverage-baseline metrics (BBSE, ATC, DoC, OT-bound proxy, pseudo-calibration, embedding-distance metrics) from Block B3.
- The H2 keep/kill result from Block B4 (already decided prior to B6; not re-litigated here).
- The full degradation decomposition (Block B8) and its per-component values.
- Any ablation row from Block B9.

These are reported as observed values with descriptive uncertainty (means, spreads, CIs where computed), explicitly labeled as non-confirmatory. If a secondary contrast happens to look striking, it is reported as a candidate for *future, independently pre-registered* confirmation — never folded into this document's primary claim after the fact.

## 4. Decomposition method commitment (Block B8)

Block B5's Check 3 (order-permutation sanity) found that **Shapley symmetrization is required**: 45% of placebo draws showed an order-swing exceeding 20% of the total drop, meaning the two possible orderings of the prevalence/residual decomposition do not agree closely enough for either one alone to be trustworthy (see `safety_checks_b5/b5_summary.txt`).

Accordingly, Block B8 commits to:
- Computing the degradation decomposition via **Shapley-symmetrized averaging over both component orderings** (the same `decompose_drop()` machinery already validated in B5 — not a new, unvalidated decomposition function).
- Reporting each component as a **counterfactual contrast** (e.g., "reweighting the test cohort to the calibration cohort's prevalence changes macro-F1 by X"), **never** as a "percentage of total drop explained." Percent-of-total framing implies a level of additive precision the order-swing finding above shows this decomposition does not reliably have.
- Reporting platform/biology/annotation effects as an explicitly **inseparable residual** — never decomposed further than the data supports, per the project's own honesty rule.

## 5. Augmentation commitment (Block B2)

- Augmentation is **on-the-fly**, generated in memory during training, never written to disk, seeded deterministically as a function of `(run_seed, fold_id, epoch)` (`scripts/train_fold0_b2_onthefly.py`).
- The multiplier sweep m ∈ {0, 1, 3, 7} is a **descriptive ablation**, not a hypothesis test. All four rows are reported regardless of outcome, including the m=0 (no augmentation) row.
- **If augmentation shows no benefit at any m, that null result is reported honestly** — augmentation is not silently dropped from the write-up, and m is not retroactively re-swept or re-tuned after seeing results.

## 6. Multiple-comparisons plan

- The **primary contrast (Section 1) is a single pre-registered test and receives no multiplicity correction** — there is nothing to correct for when there is exactly one confirmatory test.
- Within any *declared family* of secondary/descriptive comparisons that is later presented with p-values for illustrative purposes (e.g., if per-fold AUC differences across modes are annotated with significance stars in a figure), that family uses **Benjamini–Hochberg FDR control at q=0.05**, and the family must be declared in advance of computing those p-values (not assembled after seeing which comparisons look interesting).
- Descriptive analyses reported only as point estimates, CIs, or effect sizes (no p-value attached) require no correction, since no family-wise error rate is being claimed.

## 7. What each outcome means

- **PASS** — the two-sided Wilcoxon signed-rank p-value on the 8 paired (ON, OFF) macro-F1 differences is < 0.05, **and** the direction is macro-F1(ON) > macro-F1(OFF). This is reported as: "under LODO evaluation, marginal cohort-adversarial training shows a statistically detectable improvement in significant-fibrosis macro-F1, at a sample size (n=8) that can only detect large effects (≈1.15σ_d or greater)." It is not reported as proof the mechanism works well, generalizes to conditional mode, or holds at other sample sizes.
- **NULL** — p ≥ 0.05, or p < 0.05 in the *wrong* direction. This is reported as: "no statistically detectable benefit of marginal cohort-adversarial training was found at n=8; given the ≈1.15σ_d minimum detectable effect, this null is consistent with either no true effect or a true effect too small to detect at this sample size — the data cannot distinguish these." **A NULL result is reported as the primary finding, in the primary results section, at the same prominence as a PASS would have received** — it is not demoted to a footnote, and the secondary/descriptive analyses (Section 3) are not then searched for a substitute "winner" to lead with instead.

---

## Appendix — the 8 LODO folds (for the record, from `shared/fold_assignment_lodo.csv`)

| Fold | Test cohort | Calibration source | n_train | n_cal | n_test | test prevalence | cal prevalence |
|---|---|---|---|---|---|---|---|
| 0 | GSE130970 | GSE135251 | 1043 | 206 | 78 | 0.321 | 0.587 |
| 1 | GSE135251 | GSE162694 | 1009 | 112 | 206 | 0.587 | 0.420 |
| 2 | GSE162694 | GSE193066 | 1109 | 106 | 112 | 0.420 | 0.594 |
| 3 | GSE193066 | GSE213621 | 922 | 299 | 106 | 0.594 | 0.676 |
| 4 | GSE213621 | GSE240729 | 961 | 67 | 299 | 0.676 | 0.612 |
| 5 | GSE240729 | GSE31803 | 1188 | 72 | 67 | 0.612 | 0.444 |
| 6 | GSE31803 | GSE84044 | 1131 | 124 | 72 | 0.444 | 0.492 |
| 7 | GSE84044 | GSE130970 | 1125 | 78 | 124 | 0.492 | 0.321 |

These 8 folds are the entire universe of the primary test's n=8. No fold is added, dropped, or reweighted after this document is locked.
