# Error Analysis - RevenuePulse AI Phase 2

All figures come from the locked test split (748 snapshot rows, 380 accounts, 94
positives, prevalence 0.126) produced by `python -m src.evaluation.evaluate`.
Tables are written to `artifacts/evaluation/error_*.csv`.

**Synthetic data only.** These explanations describe generated accounts. They are
not evidence about real customers, and none of them is a causal claim.

## Method

1. The selected model (histogram gradient boosting) scores and ranks every test
   snapshot; the scorecard is ranked the same way.
2. Review capacity is the configured K = 50.
3. Four tables are produced:
   - `error_high_scoring_false_positives.csv`: label 0, score >= 0.50, in the top
     50, sorted by score descending (first 15).
   - `error_low_scoring_false_negatives.csv`: label 1, score <= 0.50, outside the
     top 50, sorted by score ascending (first 15).
   - `error_baseline_only_hits.csv`: label 1, in the scorecard's top 50 but not the
     model's (first 15 of 9 total).
   - `error_model_only_hits.csv`: label 1, in the model's top 50 but not the
     scorecard's (first 15 of 15 total).
4. Every row retains its point-in-time features and the realised future revenue
   change, so a reason can be checked against data rather than narrated.

The tables are truncated for readability; counts in the headings are the full
counts.

## 1. High-scoring false positives (12 rows)

The model placed 23 non-decliners in the top 50; 12 of them scored at or above the
0.50 probability threshold. Of those 12:

| Observation | Count |
|---|---|
| Actually declined, but by less than the 20% label threshold | 8 |
| Actually grew (future change >= 0%) | 4 |
| Six-month revenue change >= +15% (the model's dominant driver) | 11 |
| Three-month revenue trend at or below -5% | 3 |
| Median realised future revenue change | -6.4% |

Most of these are not random misses: they are accounts whose revenue did fall, just
not far enough to cross the label line. Example: `SYN-0317` (2025-09-30) scored
0.87 with a -8.4% three-month trend and a +25.9% six-month trend, and realised
-7.8%. `SYN-0357` scored 0.61 on a +29.4% six-month trend and realised -16.6%.

Plausible reason supported by the features: in this generator, strong recent
growth frequently precedes a partial give-back. When the give-back is smaller than
20%, the ranking is penalised even though the direction was right. This is a
**label-threshold sensitivity** issue as much as a ranking error. The remaining
four false positives grew outright and look like genuine false alarms driven by the
same growth signal.

What cannot be concluded: that these accounts would have been "saved" or that a
reviewer would consider the flag useless. A -6% revenue move may still be
commercially interesting even though the label calls it negative.

## 2. Low-scoring false negatives (15 rows)

All miss rows scored very low (maximum 0.063), and their median realised change
was -24.6%, i.e. real, material declines the queue would not surface.

| Observation | Count |
|---|---|
| Three-month revenue trend at or below -5% | 6 |
| Six-month revenue change >= +15% (model's main signal) | 3 |
| Transaction trend negative | 5 |
| Engagement trend negative | 13 |
| Engagement level above 40 (not "low engagement") | 9 |
| Median realised future revenue change | -24.6% |

Feature reading: these accounts do not look risky on the dimensions the booster
weights most. `SYN-0103` had a -25.5% three-month trend and -30.1% transaction
trend, but the model scored it 0.017; its six-month trend and engagement level did
not match the learned risk pattern. Most show a mildly weakening engagement trend
while their engagement level stays in the normal range, and five have positive
transaction trends, so none of the model's strongest cues is present.

Plausible reason: the generator's decline events are persistent shocks applied to
revenue, and a shock that begins after the snapshot is not observable in any
point-in-time feature. Those cases are irreducible with this feature set, which is
why recall at capacity 50 is 0.287.

## 3. Where the scorecard succeeds and the model fails (9 rows)

These are label-1 accounts the scorecard placed in its top 50 but the model did not.
They are the direct evidence that the decline rules still carry information:

- Median three-month revenue trend: **-18.3%** (all nine were declining).
- Median model score: **0.078**; maximum model score: 0.207.
- Only 2 of 9 had the +15% six-month growth signature the model keys on.

Interpretation: for accounts already in visible three-month decline, the simple
rule is right and the booster under-ranks them because its dominant driver is the
six-month trend, which can stay positive while the last quarter deteriorates.
`SYN-0272` is the clearest case: scorecard score 1.00 (all four signals), model
score 0.207, realised -20.3%.

This suggests a concrete next-iteration hypothesis (not implemented, and not tuned
on test): a combined score, or a booster that includes an explicit short-term
decline interaction, may recover these cases without losing the growth-driven ones.
Adding such a feature and re-running validation would be a legitimate Phase 3
experiment; changing it now, after seeing test results, would invalidate the locked
evaluation.

## 4. Where the model succeeds and the scorecard fails (15 rows)

- All 15 had a six-month revenue change of at least +15%.
- Eight had a positive three-month trend, i.e. the scorecard saw no decline at all.
- Median model score 0.662; these are confidently ranked.

Interpretation: the booster captures accounts that give back revenue after a strong
run, which a decline-only rule cannot see by construction. `SYN-0082` scored 0.81
against a scorecard score of 0.00 and realised -36.7%.

## 5. What the pattern implies

1. The two approaches encode different and partly opposing hypotheses: the rule
   says "declining accounts keep declining", the model learned "accounts that grew
   sharply give some back". On this synthetic data the second hypothesis carries
   more ranking signal.
2. Because the generator drives that pattern, the model's advantage is not
   automatically transferable. Real B2B revenue processes may show the opposite
   relationship, or none.
3. The baseline is not a fair strong-rule control. A rule that used six-month trend
   and engagement level would likely score far better than 0.140 Precision@50.
   Building that rule is the cheapest next experiment and would test whether ML is
   needed at all.
4. The label threshold (20%) creates a band of "declined but not labelled" accounts
   that the model ranks high. Any real pilot should agree the materiality threshold
   with the business before judging precision, otherwise reviewers will see
   sensible flags scored as errors.

## 6. Reproduction

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.evaluate
```

This rewrites `artifacts/evaluation/error_*.csv` and `test_results.json`. The
evaluation is deterministic given the committed seed and data, so the rows above
are reproducible without re-running training.
