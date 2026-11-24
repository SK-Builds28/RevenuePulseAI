# Phase 2 Build Notes - RevenuePulse AI

**Status:** Phase 2 complete (transparent baseline, two ML models, validation
comparison, locked test evaluation, model artifact, model card and evaluation
report). No Streamlit UI, review queue, HITL logging, recommendation engine, CRM
integration or LLM work is included.

**Version:** phase2-1.0.0
**All results are synthetic-data results.** No real customer, company or personal
data was used.

## 1. Objective and scope

Implement model training and evaluation on top of the Phase 1 data foundation
without changing its contract:

- transparent rule-based baseline;
- train-fitted preprocessing;
- logistic regression and histogram gradient boosting;
- validation-only model selection, one locked test evaluation;
- Precision@K, recall@K, lift@K, PR-AUC, ROC-AUC, Brier, calibration, segment and
  error analysis;
- account-level bootstrap intervals;
- reason codes and a saved, reproducible model artifact;
- documentation and tests.

Out of scope and not built: UI, review queue, accept/edit/dismiss, SQLite HITL
logging, action recommendation, CRM integration, LLM text generation, pricing
automation, opportunity scoring.

## 2. What Phase 1 provides and what Phase 2 relies on

| Phase 1 artifact | How Phase 2 uses it |
|---|---|
| `config.yaml` | Single source for thresholds, K values, model parameters, bootstrap settings |
| `src/features/snapshots.py` | `FEATURE_COLUMNS` allow-list, target columns, time-ordering assertion |
| Quarter-end snapshots | Loaded as-is; no feature engineering in Phase 2 |
| Split dates | Train fits, validation selects, test evaluates once |
| Revenue-decline label | Unchanged; 3-month horizon, 20% threshold, revenue floor |
| Leakage tests | Retained and re-run in the full suite |

**No Phase 1 label, feature or split definition was changed.** A few technical
corrections and documented adjustments were made to the Phase 2 scaffolding that
already existed in the repository; all are listed in section 10.

## 3. Files added or changed

Added:

| File | Purpose |
|---|---|
| `src/models/train.py` | Fit candidates on train, validate, select, save artifact and metadata |
| `src/models/reasons.py` | Reason codes, logistic coefficients, permutation importance, alignment table |
| `src/evaluation/evaluate.py` | Locked evaluation CLI: metrics, calibration, bootstrap, segments, error tables |
| `tests/test_baseline.py` | Determinism, weights, thresholds, allow-list |
| `tests/test_preprocessing.py` | Train-only fitting, imputation, unknown categories, feature exclusion |
| `tests/test_metrics.py` | Precision@K, recall@K, lift, false alarms, calibration, account bootstrap |
| `tests/test_model_training.py` | Selection rule, reproducibility, artifact reload, metadata, no test scoring |
| `tests/test_phase2_documentation.py` | Model card / report / notes / README coverage |
| `docs/Model_Card.md`, `docs/Evaluation_Results.md`, `docs/Error_Analysis.md`, `docs/Phase_2_Build_Notes.md` | Documentation |

Changed:

| File | Change |
|---|---|
| `config.yaml` | Active `reasons` section; explicit `class_weight: null` with validation evidence; Phase 2 model and evaluation settings |
| `src/utils/config.py` | Reason settings accessor and validation; baseline threshold-key validation corrected |
| `src/models/baseline.py` | Threshold-key validation corrected (`*_pct` / `*_days`) |
| `README.md` | Phase 2 status, commands, results, artifact location |
| `AGENTS.md` | Current phase and Phase 2 boundaries |

Previously scaffolded and completed in this phase: `src/models/baseline.py`,
`src/models/preprocessing.py`, `src/evaluation/metrics.py`,
`src/evaluation/bootstrap.py`.

## 4. Models considered

| Model | Decision | Reason |
|---|---|---|
| Transparent weighted scorecard | Kept as reference baseline | Required control; explainable to a sales manager; no training |
| Logistic regression (standardised, one-hot) | Kept as candidate | Interpretable coefficients, stable with ~2.4k rows and 22 features |
| Histogram gradient boosting | Kept and selected | Handles non-linearity and missing values, fast, no extra dependency |
| Random forest / XGBoost / LightGBM | Rejected | Extra dependency or heavier tuning for no clear benefit at this size |
| Deep learning / LLM classifier | Rejected | Unjustifiable for tabular data of this size and violates the local-first rule |
| Calibrated classifier (Platt/isotonic) | Not applied | Raw probabilities already close; fitting a calibrator would add a validation-tuned layer without evidence of need |

Parameters were fixed before any validation scoring and were not searched:

```text
LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, class_weight=None)
HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                               min_samples_leaf=20, l2_regularization=1.0,
                               early_stopping=False, random_state=42)
```

## 5. Preprocessing decisions

- **Numeric:** median imputation fitted on train (`SimpleImputer(strategy="median",
  keep_empty_features=True)`). Standardisation (`StandardScaler`) only for logistic
  regression; trees do not need it.
- **Categorical:** `segment` with most-frequent imputation and
  `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`, so unseen segments
  encode to all zeros instead of crashing.
- **Remainder:** `drop`; `account_id`, target columns and metadata can never enter
  the matrix. `model_feature_frame()` selects exactly `FEATURE_COLUMNS` and raises
  if a column is missing.
- **No row dropping** for null short-history features
  (`revenue_trend_6m_pct`, `product_count_change_6m`, `support_trend_3m_pct`) or
  null engagement/activity months. All 2,395 train rows are used.
- Preprocessing is inside the `Pipeline`, so `predict_proba` on validation/test
  applies train statistics. Tests assert the imputer's learned median equals the
  train median and changes when rows are added to the fit set.

## 6. Class imbalance handling

Train prevalence is 13.7% (329/2,395). Two treatments were compared on the
**validation** split only, before the test split was read:

| Setting | LR P@50 | LR PR-AUC | LR Brier | HGB P@50 | HGB PR-AUC | HGB Brier |
|---|---|---|---|---|---|---|
| `class_weight=None` | 0.320 | 0.277 | 0.085 | 0.380 | 0.274 | 0.086 |
| `class_weight="balanced"` | 0.280 | 0.275 | 0.148 | 0.400 | 0.274 | 0.114 |

Decision: **no reweighting**. Reweighting gave no consistent ranking gain
(PR-AUC unchanged, LR worse at K=50, HGB better by one reviewed account), and it
materially degraded probability quality (Brier roughly doubled) while pushing the
0.50 threshold to flag 96-155 snapshots instead of 21-6. Since the product ranks
accounts and shows probabilities, the unweighted scale is preferable.

Imbalance is otherwise handled by the evaluation design: PR-AUC, Precision@K, lift
and false-alarm rate instead of accuracy, plus reporting prevalence alongside every
number.

## 7. Validation methodology

1. Fit both candidates on train only (2,395 rows).
2. Score validation (778 rows, 80 positives) with the baseline and both models.
3. Compare on `evaluation.model_selection_metric` = Precision@K at
   `evaluation.primary_k` = 50; tie-break on PR-AUC.
4. Apply the interpretability rule: if a non-LR model wins by <= 0.02 Precision@K,
   prefer logistic regression.
5. Save the artifact and metadata; only then score test once.

Selected: histogram gradient boosting. Validation P@50 0.380 vs LR 0.320 (gap
0.060 > margin 0.02), PR-AUC essentially tied (0.274 vs 0.277). The margin rule was
written into `config.yaml` before either split was scored, and was not adjusted
afterwards. Test was not used for any choice, including the binary threshold, which
stays at the configured 0.50.

## 8. Evaluation and reporting decisions

- **Primary metric:** Precision@K at K = 50 (configured capacity). Additional K
  values 10 and 25, plus top-10%/top-20% capacity fractions.
- **Lift** = Precision@K / prevalence.
- **False-alarm rate** = false positives / selected accounts (defined explicitly;
  not "1 - precision" borrowed from another context).
- **Acceptance rate** is deliberately not reported: no human review experiment
  exists, so any acceptance number would be fabricated.
- **ROC-AUC** is secondary only; with 12.6% prevalence it is optimistic-looking and
  less decision-relevant than PR-AUC.
- **Bootstrap:** account-level percentile bootstrap, 1,000 iterations, seed 42,
  resampling 380 accounts and keeping each account's snapshots together. Row-level
  resampling would treat 748 correlated rows as independent and understate
  uncertainty. Replicates missing one class are excluded from PR-AUC intervals and
  counted in `valid_replicates`.
- **Calibration:** Brier score plus a 10-bin equal-count table (a table, not a
  plot, to avoid a plotting dependency).
- **Segments:** top-20% ranking within each segment plus threshold-based metrics,
  with row/positive minimums and an explicit `insufficient_observations` status.
- **Explainability:** logistic coefficients and odds ratios on standardised inputs;
  permutation importance on validation for the booster; rule-based reason codes.

## 9. Assumptions

- One row per account per quarter-end; repeated accounts across snapshots are
  expected and correlated.
- 50 reviews per quarter is the planning capacity; other K values are sensitivity
  checks.
- A 20% three-month revenue drop is a material decline for this exercise.
- The four scorecard signals are the most defensible non-ML rule for a sales
  manager, so they form the baseline.
- Class weighting is unnecessary when ranking is the product and probabilities are
  displayed.
- The Phase 1 label is correct as specified; it was not changed to improve results.
- Synthetic currency units; no FX, pricing or cost model.
- Reason-code thresholds are descriptive business rules, not model internals.
- No intervention effect is assumed anywhere; a flagged account is not "saved".

## 10. Fixes and changes made during Phase 2 (with reasons)

1. **Baseline threshold-key validation mismatch.** `config.yaml` defines
   `baseline.thresholds` as `revenue_decline_pct`, `transaction_decline_pct`,
   `inactivity_days`, `engagement_decline_pct`, but `validate_config` and
   `BaselineScorecard` required keys named exactly like the four signals. Loading
   config therefore failed. Fixed by making both validators use the actual
   threshold keys. No threshold value was changed.
2. **Reason codes widened beyond the four baseline signals.** The initial version
   reused only the scorecard's decline rules and produced no code for 32 of the 50
   highest-scoring test snapshots, because the selected model's top drivers are
   six-month revenue trend and engagement level. Reason codes now cover seven
   configurable rules (`reasons` in `config.yaml`), all evaluated on the account's
   own features, including a growth code that is explicitly labelled as a property
   of the synthetic generator. Coverage after the change: 49 of 50 top snapshots.
   This is explainability only; it does not affect scores.
3. **Class weighting set to `null`** based on the validation comparison in
   section 6, replacing the scaffolded `"balanced"`.
4. **`revenue_trend_6m_pct`, `revenue_mean_3m` added to error-analysis columns**
   so the realised error tables can show the dominant driver.

## 11. Hardcoded items remaining

- The four scorecard signals and their feature mapping (`SIGNAL_FEATURES`); the
  weights and thresholds themselves are configurable.
- The seven reason-code rules, their features and directions
  (`REASON_RULES`); thresholds and the per-account limit are configurable.
- Rank tie-break order in the scorecard (revenue trend, transaction trend,
  engagement trend, `snapshot_id`).
- Permutation importance settings (`n_repeats=5`, AP scoring) in
  `src/models/reasons.py`.
- Bootstrap percentile levels (2.5/97.5), i.e. 95% intervals; iterations and seed
  are configurable.
- Segment analysis defaults (fraction 0.20, `min_rows=30`, `min_positives=5`).
- Error-table truncation (`limit=15` per table) and the top-50 reason export size.
- Binary threshold is config-driven (0.50) but not tuned; no cost model.
- Model hyperparameters live in config but were fixed, not searched.
- Single random seed for training; no rolling-origin folds.

## 12. Unexpected findings

1. **The dominant learned signal points the "wrong" way for a churn story.** In
   train data, accounts with a positive six-month revenue trend are more likely to
   be labelled (mean +0.051 vs -0.074), and the gap is larger in test (+0.109 vs
   -0.037). Higher current revenue level and slightly lower engagement mean are
   also associated with the label. Both fitted models reproduce this; the booster's
   top permutation importance is `revenue_trend_6m_pct` (0.128, next is 0.039).
   Plausible mechanism: the label compares the next three months to the trailing
   three months, so a recent run-up makes a 20% give-back mechanically easier, and
   the generator's persistent shocks land on high-revenue accounts. **This is a
   property of the synthetic generator, not a market finding**, and it is a serious
   reason not to extrapolate the results.
2. **The transparent baseline is near random on this data** (validation
   Precision@50 0.100, lift 0.97; test 0.140, lift 1.11). It was not degraded to
   make the model look good; it simply tests decline-style rules while the synthetic
   signal runs through six-month trend and engagement level. The model-versus-rule
   comparison therefore overstates what ML adds over a well-designed simple rule.
3. **Most high-scoring false positives did decline, just not by 20%** (8 of 12).
   The label threshold, not the ranking, drives much of the apparent error.
4. **Recall at 50 reviews is low (0.287).** The queue surfaces the top of the
   distribution, not the majority of decliners; this should be stated to any
   reviewer expecting coverage.
5. **Calibration was good without a calibrator** (Brier 0.094, mean bin gap 0.030),
   which is unusual for an unweighted booster with 13.7% prevalence and worth
   re-checking on real data.
6. **Logistic regression is close on test.** It trails the booster by 3 reviewed
   accounts at K=50 (24 vs 27 true positives) and 4 at K=25 (14 vs 18), and it lost
   validation selection by the same 3-account margin (0.320 vs 0.380 Precision@50).
   If governance favours coefficients, LR is a defensible fallback rather than a
   poor model.

## 13. Test evidence

Full suite (Phase 1 + Phase 2), from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Result: **105 passed** in 27.47s (13 benign scikit-learn/scipy `OptimizeWarning`
messages from the pinned solver combination). Phase 1 tests were not removed or
weakened; Phase 2 tests add preprocessing, leakage, determinism, artifact, metric,
reason-code and documentation coverage. Tests use in-memory snapshots where possible and
do not need network access.

## 14. Limitations

- Synthetic-only evidence; no real-world performance claim is possible.
- Label is a revenue-decline proxy, not churn.
- Two test quarter-ends, 94 positives; intervals are wide.
- Baseline is weak, so the ML advantage is not the right benchmark yet.
- No cost model, so K is a capacity assumption rather than a decision optimum.
- No human review, acceptance, or intervention-effect data.
- No calibration layer, no threshold tuning, no rolling-origin validation.
- Reason codes are descriptive, not exact per-account attributions.
- Fairness cannot be assessed on synthetic segments.
- The model learned a counter-intuitive direction that must be validated on real
  data before any use.

## 15. Recommendation

**Iterate**, not scale: proceed only to a controlled real-data pilot subject to
data, privacy and governance validation. Before that pilot, add a stronger simple
baseline (level/volatility/trend) to test whether ML is needed at all, and confirm
that the generator's growth-to-give-back relationship is not the only thing driving
performance. Full reasoning in `docs/Evaluation_Results.md`, section 16.

## 16. Phase 3 boundary

Not built and not started: Streamlit UI, review queue, account detail page,
accept/edit/dismiss, SQLite HITL logging, action recommendation engine, CRM
integration, LLM or email generation, pricing automation.
