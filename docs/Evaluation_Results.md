# Evaluation Results - RevenuePulse AI Phase 2

**Version:** phase2-1.0.0
**Evaluation command:** `python -m src.evaluation.evaluate`
**Raw results:** `artifacts/evaluation/` (`test_results.json`, CSV tables)

> **Synthetic-data warning.** Every number in this report is produced from
> generated account histories. These results demonstrate that the method works on
> this generator; they are **not** evidence of real-world churn or revenue
> prediction performance.

## 1. Objective

Test whether a learned model ranks accounts by 90-day revenue-decline risk better
than a transparent rule at a realistic review capacity, using only point-in-time
features, a time-ordered split and no test-set tuning. Secondary objectives:
quantify uncertainty, check calibration, explain drivers, and decide whether the
project should continue.

## 2. Dataset and temporal split

Synthetic dataset: 400 accounts x 36 months (2023-01 to 2025-12), seed 42,
14,400 account-month rows. Quarter-end snapshots with a complete 3-month forward
window and at least 6 months of history give 3,921 labelled snapshots.

| Split | Snapshot dates | Rows | Accounts | Positives | Prevalence |
|---|---|---|---|---|---|
| Train | 2023-06-30 to 2024-09-30 | 2,395 | 400 | 329 | 0.137 |
| Validation | 2024-12-31 to 2025-03-31 | 778 | 395 | 80 | 0.103 |
| Test | 2025-06-30 to 2025-09-30 | 748 | 380 | 94 | 0.126 |

Splits are assigned by snapshot date in `src/features/snapshots.py` and were not
changed. Test holds two quarter-ends (2025-06-30 and 2025-09-30) and 380 accounts;
368 accounts appear in both snapshots, which is why all intervals below resample
accounts rather than rows.

## 3. Revenue-decline label

`future_revenue_decline = 1` when the mean revenue of the three months after the
snapshot is at most 80% of the mean of the three months ending at the snapshot,
subject to the history and revenue-floor rules. Phase 1's definition is unchanged;
no threshold was relaxed to raise prevalence or improve results.

The label is a **revenue-decline proxy**. It does not identify cancellations, and
it treats a 20% drop caused by seasonality or billing timing the same as a
retention problem.

## 4. Transparent baseline

A four-signal additive scorecard (`src/models/baseline.py`) using only
point-in-time features:

| Signal | Feature | Rule | Weight |
|---|---|---|---|
| Recent revenue decline | `revenue_trend_3m_pct` | <= -5% | 2.0 |
| Reduced transaction frequency | `transaction_trend_3m_pct` | <= -5% | 2.0 |
| Increased inactivity | `days_since_activity_last` | >= 30 days | 1.5 |
| Weakening engagement | `engagement_trend_3m_pct` | <= -5% | 1.5 |

Scores run from 0 to 7; the binary risk flag uses `score >= 3.0`; the ranking
probability is `score / 7` and is explicitly **not** calibrated. Missing values
never add points. Ties are broken by worse revenue trend, then transaction trend,
then engagement trend, then `snapshot_id`. The scorecard is deterministic, needs no
training, and is documented in business language in `docs/Phase_2_Build_Notes.md`.

Its validation ranking is essentially random (Precision@50 0.100 against 10.3%
prevalence, lift 0.97). That is a real finding about this dataset: the dominant
synthetic signal runs through six-month trend and engagement level, which the
four decline rules do not read.

## 5. Models evaluated

| Model | Configuration | Preprocessing |
|---|---|---|
| Transparent scorecard | Fixed weights/thresholds from `config.yaml` | None (raw features) |
| Logistic regression | `C=1.0`, `lbfgs`, `max_iter=2000`, no class weighting | Median imputation, standardisation, one-hot segment |
| Histogram gradient boosting | `learning_rate=0.05`, `max_iter=200`, `max_leaf_nodes=15`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=false` | Median imputation, one-hot segment (no scaling) |

Preprocessing is a scikit-learn `Pipeline` fitted on train rows only. No
hyperparameter search was performed; parameters were chosen before validation was
scored, and no parameter was changed after seeing test results.

## 6. Validation results

Capacity k = 50 reviews (the configured primary K):

| Model | P@10 | P@25 | P@50 | R@50 | Lift@50 | PR-AUC | Brier |
|---|---|---|---|---|---|---|---|
| Transparent scorecard | 0.100 | 0.120 | 0.100 | 0.062 | 0.97 | 0.115 | 0.202 |
| Logistic regression | 0.500 | 0.360 | 0.320 | 0.200 | 3.11 | 0.277 | 0.085 |
| Histogram gradient boosting | 0.500 | 0.440 | 0.380 | 0.237 | 3.70 | 0.274 | 0.086 |

## 7. Model selection decision

Selection rule (declared in `config.yaml` before evaluation): rank ML candidates by
validation Precision@K, tie-break on PR-AUC, and prefer logistic regression when
the winner's lead is within the 0.02 interpretability margin.

Outcome: gradient boosting led logistic regression by 0.060 Precision@50
(0.380 vs 0.320), which is outside the margin, so **histogram gradient boosting**
was selected. Logistic regression was not chosen despite being more interpretable:
the gap is three additional true positives in the top 50 on validation, larger
than the pre-registered margin. PR-AUC was essentially tied (0.274 vs 0.277).

Model selection used train/validation only. The test split was read once, after the
artifact was saved; the pipelines were not refitted on validation or test data.

## 8. Locked test results

748 snapshot rows, 380 accounts, 94 positives, prevalence 0.126:

| Model | P@10 | P@25 | P@50 | R@50 | Lift@50 | False alarms@50 | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|---|---|---|---|---|---|
| Transparent scorecard | 0.200 | 0.200 | 0.140 | 0.074 | 1.11 | 0.860 | 0.136 | 0.540 | 0.243 |
| Logistic regression | 0.500 | 0.560 | 0.480 | 0.255 | 3.82 | 0.520 | 0.350 | 0.781 | 0.097 |
| Histogram gradient boosting (selected) | 0.600 | 0.720 | 0.540 | 0.287 | 4.30 | 0.460 | 0.382 | 0.756 | 0.094 |

Threshold-based classification at the configured 0.50 cut-off (secondary to the
ranking view):

| Model | Predicted positive | TP | FP | FN | Precision | Recall | F1 | False-alarm rate | Accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Transparent scorecard | 316 | 44 | 272 | 50 | 0.139 | 0.468 | 0.215 | 0.861 | 0.570 |
| Logistic regression | 8 | 5 | 3 | 89 | 0.625 | 0.053 | 0.098 | 0.375 | 0.877 |
| Histogram gradient boosting | 35 | 23 | 12 | 71 | 0.657 | 0.245 | 0.357 | 0.343 | 0.889 |

Accuracy is shown for completeness only. The scorecard's apparently respectable
recall comes from flagging 42% of all accounts, 86% of them incorrectly. A fixed
0.50 threshold is not the operational mechanism; capacity-based ranking is.

## 9. Precision@K comparison

**Why Precision@K matters here.** A manager cannot review 380 accounts per
quarter. If review capacity is 50, the only question that matters is: of the 50
accounts put in front of the manager, how many will actually decline materially?
Precision@K answers that directly, Recall@K says how much of the decliner
population the review reaches, Lift@K compares the reviewed group's hit rate to
random selection, and the false-alarm rate (`FP / selected`) states how much of
the reviewer's time is spent on accounts that do not decline. Overall accuracy is
uninformative at 12.6% prevalence: predicting "no decline" for everyone scores
0.874 and finds zero decliners.

| K | Model | TP | FP | Precision@K | Recall@K | Lift@K | False alarms / selected |
|---|---|---|---|---|---|---|---|
| 10 | Scorecard | 2 | 8 | 0.200 | 0.021 | 1.59 | 0.800 |
| 10 | Logistic regression | 5 | 5 | 0.500 | 0.053 | 3.98 | 0.500 |
| 10 | Gradient boosting | 6 | 4 | 0.600 | 0.064 | 4.77 | 0.400 |
| 25 | Scorecard | 5 | 20 | 0.200 | 0.053 | 1.59 | 0.800 |
| 25 | Logistic regression | 14 | 11 | 0.560 | 0.149 | 4.46 | 0.440 |
| 25 | Gradient boosting | 18 | 7 | 0.720 | 0.191 | 5.73 | 0.280 |
| 50 | Scorecard | 7 | 43 | 0.140 | 0.074 | 1.11 | 0.860 |
| 50 | Logistic regression | 24 | 26 | 0.480 | 0.255 | 3.82 | 0.520 |
| 50 | Gradient boosting | 27 | 23 | 0.540 | 0.287 | 4.30 | 0.460 |

At capacity 50 the learned model triples the scorecard's hit rate and uses 23 of
50 review slots on accounts that do not decline. Precision declines as K grows
(0.600 at 10, 0.720 at 25, 0.540 at 50 for the selected model), which is the
expected shape: the top of the queue is stronger than its tail.

The selected model's top-50 selection contains 46 distinct accounts; four accounts
appear twice because they are flagged at both 2025 quarter-ends.

## 10. Confidence intervals

Method: percentile bootstrap resampling **accounts** with replacement (not
snapshot rows), 1,000 iterations, seed 42, 380 accounts. Each sampled account
contributes all of its test snapshots, preserving within-account correlation.
Percentile 95% intervals; replicates in which one class was absent are excluded
from PR-AUC intervals and counted (`valid_replicates`).

| Metric | Point | 95% interval |
|---|---|---|
| Scorecard P@50 | 0.140 | 0.060 to 0.240 |
| Scorecard PR-AUC | 0.136 | 0.109 to 0.172 |
| Logistic regression P@50 | 0.480 | 0.320 to 0.620 |
| Logistic regression PR-AUC | 0.350 | 0.277 to 0.451 |
| HGB P@50 | 0.540 | 0.400 to 0.700 |
| HGB PR-AUC | 0.382 | 0.292 to 0.496 |
| HGB minus scorecard P@50 | +0.400 | +0.240 to +0.560 |
| HGB / scorecard P@50 | 3.86x | 2.15x to 9.33x |
| Logistic regression minus scorecard P@50 | +0.340 | +0.160 to +0.500 |

Limitations: the intervals cover sampling variability in the test window only.
They do not cover model-selection uncertainty, generator assumptions, or the
small number of independent quarters (two test quarter-ends). Widths of roughly
±0.14 Precision@50 reflect 94 positives; treat any ranking difference smaller than
about two reviewed accounts as noise.

## 11. Calibration

Brier score and equal-count calibration bins (10 bins) on the test split:

| Model | Brier | Mean absolute bin gap | Worst bin gap | Top decile predicted vs observed |
|---|---|---|---|---|
| Transparent scorecard | 0.243 | 0.276 | 0.738 | 0.831 vs 0.093 |
| Logistic regression | 0.097 | 0.030 | 0.084 | 0.348 vs 0.400 |
| Histogram gradient boosting | 0.094 | 0.030 | 0.050 | 0.490 vs 0.440 |

The selected model's probabilities are usable: bins track observed rates closely
and the top decile is slightly over-confident (0.49 predicted vs 0.44 observed).
The scorecard's `score / 7` is not a probability at all and should never be shown
as one.

No post-hoc calibration (Platt/isotonic) was applied: the raw probabilities are
already close on validation and test, and fitting a calibrator would add a
validation-tuned layer with no demonstrated need. Validation and test calibration
tables are in `artifacts/evaluation/calibration_*.csv`.

## 12. Error analysis

Full detail in `docs/Error_Analysis.md`; the four generated tables are in
`artifacts/evaluation/error_*.csv`. Summary:

- **12 high-scoring false positives** (score >= 0.50, label 0). Eight of them did
  decline, just not by 20% (median future change -6.4%). Eleven had a six-month
  revenue change of at least +15%. Plausible explanation: the model learned the
  generator's give-back pattern, and some of those accounts gave back less than
  the label threshold. This is partly a label-threshold effect, not a pure ranking
  miss.
- **15 low-scoring false negatives** (score <= 0.50, label 1). Their scores are
  very low (maximum 0.063) and their median future change is -24.6%. Six had a
  three-month revenue decline and only three had the growth signature the model
  keys on. Plausible explanation: declines driven by shock events that are not
  visible in trend, level or engagement features at the snapshot.
- **Baseline-only hits (9)**: label-1 accounts in the scorecard's top 50 but not
  the model's. Median three-month trend -18.3%, model score median 0.078. These
  are genuine decline cases that the decline rule catches and the booster ranks
  low.
- **Model-only hits (15)**: label-1 accounts in the model's top 50 but not the
  scorecard's. All 15 had six-month growth >= 15%, eight had positive three-month
  trend. This is the clearest evidence that the learned model and the rule use
  different, and partly opposing, signals.

## 13. Segment analysis

Selected model, top 20% of each segment's ranking (equal-count per segment, not a
global threshold):

| Segment | Rows | Accounts | Positives | Prevalence | P@20% | Recall@20% | Lift | PR-AUC | Threshold FA rate |
|---|---|---|---|---|---|---|---|---|---|
| Enterprise | 138 | 69 | 22 | 0.159 | 0.357 | 0.455 | 2.24 | 0.477 | 0.462 |
| Mid-Market | 258 | 129 | 27 | 0.105 | 0.288 | 0.556 | 2.76 | 0.501 | 0.111 |
| SMB | 352 | 182 | 45 | 0.128 | 0.296 | 0.467 | 2.31 | 0.326 | 0.385 |

Scorecard comparison at the same capacity: Enterprise P@20% 0.179 (lift 1.12),
Mid-Market 0.173 (1.65), SMB 0.085 (0.66).

Reading notes:

- All three segments have enough observations to report, none enough to conclude.
- Ranking quality is weakest for SMB (PR-AUC 0.326 against 12.8% prevalence) but
  still clearly above chance; the scorecard is at or below chance for SMB.
- The threshold false-alarm rate differs sharply from the ranking view because it
  is a 0.50-probability cut, not a capacity cut: only 9 Mid-Market snapshots
  exceed 0.50, while Enterprise flags 13. Do not read the FA column as fairness
  evidence.
- **No fairness conclusion can be drawn from synthetic segment results.** Segment
  labels are generated categories, sample sizes are small, and there is no
  real-world protected attribute or outcome data.

## 14. Limitations

1. **Synthetic-only evidence.** Performance is a property of the generator. The
   strongest driver, six-month revenue growth predicting a later give-back, may be
   absent or inverted in real accounts.
2. **Proxy label.** 20% three-month revenue change, not churn; sensitive to
   seasonality, billing timing and one-off events.
3. **One split, two test quarter-ends.** No rolling-origin validation across many
   quarters, so temporal stability is untested.
4. **Small test set.** 94 positives; Precision@50 intervals span roughly ±0.14.
5. **Weak baseline.** The scorecard tests only decline-style rules and is near
   random here, so the model-versus-rule gap overstates the value of ML against a
   well-designed level/volatility rule. Improving the baseline was deliberately
   left out to avoid tuning after seeing results; it is the first item for a next
   iteration.
6. **No calibration fit, no threshold tuning, no cost model.** No human-review
   cost or value of a caught decline is estimated, so the optimal K is a capacity
   assumption, not an optimum.
7. **No human-in-the-loop data.** Acceptance rate, reviewer agreement and
   intervention effect are unknown and were not estimated.
8. **Repeated accounts across snapshots.** 368 of 380 accounts appear in both
   test quarter-ends. Handled in the bootstrap, but the effective independent
   sample is 380 accounts, not 748 rows.
9. **Feature set is fixed by Phase 1.** No interaction features, no CRM/contact
   activity, no contract or pricing detail.
10. **Reason codes are descriptive**, not exact per-account attributions.

## 15. Pilot implication

If this were a real dataset, the shape of the result would justify a small,
controlled pilot: a ranked queue of 50 accounts per quarter with mandatory human
review, a frozen baseline, outcome capture, and no automated action. The pilot
would test three things this evaluation cannot: whether the ranking survives on
real accounts, whether the 20% proxy matches what finance and sales consider a
material decline, and whether reviewers act on the queue.

On this synthetic data the pilot question is narrower: confirm the data
generator is not doing the work. The give-back direction of the dominant feature
must be checked against domain expectation before any real-data claim.

## 16. Go / Iterate / Stop recommendation

**Recommendation: ITERATE - proceed only to a controlled real-data pilot, subject
to data, privacy and governance validation. Do not scale, and do not start UI or
automation work.**

Evidence for continuing:

- The selected model beats the transparent scorecard on test at every capacity
  (P@50 0.540 vs 0.140; lift 4.30 vs 1.11) and the account-level 95% interval for
  the difference excludes zero (+0.240 to +0.560).
- Calibration is acceptable without a fitted calibrator (Brier 0.094, mean bin gap
  0.030).
- Logistic regression performs nearly as well (P@50 0.480), so an interpretable
  fallback exists if the booster proves hard to govern.

Evidence against scaling:

- All data is synthetic; the dominant learned relationship (growth precedes
  give-back) contradicts normal churn intuition and may be an artefact.
- The baseline is uninformative on this data, so the comparison does not yet show
  value over a *well-designed* simple rule.
- Recall at 50 reviews is 0.287: most future decliners are not reached.
- Test evidence covers two quarter-ends and 94 positives.
- No human review, cost or intervention evidence exists.

Conditions that would change this recommendation to Stop or to Go:

- **Stop** if a real-data pilot shows Precision@K at or below a well-designed rule
  baseline, or if the label cannot be tied to a business-accepted definition of
  material decline.
- **Go (to a limited product build)** only after the real-data pilot reproduces the
  ranking advantage with acceptable false-alarm cost and reviewers report the
  queue is usable. That is outside Phase 2.
