# RevenuePulse AI - Executive Summary

**Sales Opportunity & Customer Revenue-Decline Prediction Copilot**
MBA ZG583 capstone · Suhani Kanwar · Phase 3 MVP submission

## Problem

B2B sales and account teams sit on fragmented account history: monthly revenue,
order frequency, product adoption, engagement, support friction and activity
recency, spread across billing, CRM and product tools. When a quarter-end review
arrives, a manager with 150-400 accounts cannot inspect all of them. Prioritisation
becomes recency bias, gut feel or a spreadsheet filter on "revenue down".

The cost of missing a deteriorating account is usually invisible until renewal: the
first signal is a revenue drop that has already happened, and the intervention
window has closed.

RevenuePulse AI addresses one narrow decision: **which accounts should a human
review first, because they are most likely to show a material revenue decline in
the next 90 days, and why?**

## Target user

- **Sales managers** who own a book of accounts and a fixed weekly review capacity.
- **Key account managers** who need evidence before starting a retention
  conversation.
- **Revenue strategy / RevOps analysts** who maintain account data and govern how
  risk is defined and reviewed.

## Product concept

A decision-support copilot that produces a quarterly review queue:

1. **Rank** accounts by a calibrated risk probability (histogram gradient boosting).
2. **Explain** each score with rule-based reason codes drawn from the account's own
   features ("factors associated with the risk score include ...").
3. **Suggest** a next-best action with the evidence that triggered it, using
   transparent rules and no LLM.
4. **Record** the human decision - accept, edit or dismiss - in a local audit log.

The system never contacts a customer, changes a price, executes an offer, sends an
email or writes to a CRM. Human approval is mandatory for every action.

## Why AI instead of a dashboard or rules alone

A dashboard shows history; it does not rank. A static rule ("flag accounts whose
revenue fell 20%") only sees decline that has already occurred, and on this dataset
a transparent four-signal scorecard reached Precision@50 of **0.140** - effectively
random against a 12.6% base rate.

A learned model can combine many weak, correlated signals into a single ordering
and is evaluated at the capacity a manager actually has. On the same locked test
split the gradient boosting model reached **Precision@50 = 0.540** and **Lift@50 =
4.30**, i.e. roughly 27 genuine future decliners in a 50-account review queue
instead of 7.

## MVP

A four-page Streamlit prototype using the frozen model artifact:

1. **Executive Dashboard** - review counts by risk band, model in use, synthetic-data
   disclaimer.
2. **Review Queue** - accounts ranked by model probability with segment, risk band,
   recent revenue, trend, engagement, activity recency, primary reason and review
   status; filters by segment, band and top N; CSV export.
3. **Account Detail** - probability and band, historical monthly revenue chart,
   commercial signals, reason codes, and a rule-based next-best-action with
   account evidence.
4. **Human Review** - accept / edit / dismiss with optional note, stored in
   `runtime/review_feedback.db` (SQLite) with CSV export.

## Actual synthetic evaluation evidence

All data is generated: 400 accounts x 36 months, quarter-end snapshots, 3,921
labelled snapshots at 12.8% prevalence. Time-ordered split: train to 2024-09-30,
validation to 2025-03-31, test to 2025-09-30. Test: 748 snapshot rows, 380 accounts,
94 positives (12.6%). Model selection used validation only; test was scored once.

| Model | P@10 | P@25 | P@50 | Recall@50 | Lift@50 | False alarm@50 | PR-AUC | Brier |
|---|---|---|---|---|---|---|---|---|
| Transparent scorecard | 0.200 | 0.200 | 0.140 | 0.074 | 1.11 | 0.860 | 0.136 | 0.243 |
| Logistic regression | 0.500 | 0.560 | 0.480 | 0.255 | 3.82 | 0.520 | 0.350 | 0.097 |
| Histogram gradient boosting (selected) | 0.600 | 0.720 | 0.540 | 0.287 | 4.30 | 0.460 | 0.382 | 0.094 |

Account-level bootstrap (1,000 iterations, seed 42, 380 accounts): HGB Precision@50
**0.540 [0.400, 0.700]**; improvement over the scorecard **+0.400 [+0.240, +0.560]**.
Calibration is acceptable without a fitted calibrator (Brier 0.094, mean absolute
bin gap 0.030).

**This is synthetic-data evidence. It does not establish real-world model
performance, and it must never be presented as churn accuracy.**

Notable finding that constrains interpretation: in this generator, a *positive*
six-month revenue trend precedes a later give-back (label mean +0.051 vs -0.074 in
train). The models learned that direction. It is a property of the synthetic data
and may not hold in real accounts - the strongest reason the recommendation is not
to scale.

## Business value hypothesis

If the ranking survives on real data, value comes from using a limited review
budget better:

```text
Potential value =
  hours saved by not manually scanning the book
  + at-risk revenue identified x intervention effectiveness x save rate
```

Every term is editable and hypothetical. No real hours, revenue, intervention
effect or save rate has been observed. A pilot would measure triage time, review
acceptance and Precision@K against the client's existing rule before any value
claim is made.

## Main risks

1. **Synthetic-to-real gap** - the learned signal may be an artefact of the
   generator, including the counter-intuitive growth driver.
2. **Weak baseline** - the current scorecard is a weak control; a well-designed
   level/volatility rule might close much of the gap.
3. **Label materiality** - the 20% three-month threshold is a proxy; 8 of 12
   high-scoring false positives had declined, just not by 20%.
4. **Recall at capacity** - at 50 reviews the model surfaces 28.7% of decliners;
   reviewers will miss accounts outside the queue.
5. **Automation bias** - a confident score can suppress reviewer judgement; the
   design forces an explicit decision and keeps reason codes visible.
6. **Data access and privacy** - the pilot needs account-level aggregates only, a
   privacy review and a deletion policy. If access is uneconomic, the model stops.
7. **Governance** - without role-based access, audit logs and a rollback path, the
   product cannot leave a demo environment.

## Recommendation

**ITERATE: proceed toward a controlled real-data paid pilot only if privacy, data
quality and governance conditions are satisfied. Do not scale and do not start
production build-out.**

Proposed path: a 12-16 hour diagnostic sprint (₹50k-₹75k), then a 6-8 week paid
pilot (₹1.5L-₹2.5L) on one segment with agreed success criteria, fixed fee, 50%
upfront, no success fee, two-week termination for convenience, monthly aggregates
only and a 30-day data-deletion policy.

Stop conditions are pre-committed: no meaningful lift over the client's existing
rule, unacceptable false alarms, poor reviewer acceptance, uneconomic data access,
uncontrollable governance, no measurable workflow benefit, or materially failing
real-data performance.

**Synthetic evaluation cannot substitute for that evidence.**
