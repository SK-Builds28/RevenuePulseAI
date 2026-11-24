# RevenuePulse AI - Final Presentation

**11 slides · MBA ZG583 capstone · Suhani Kanwar**
All metrics are from synthetic data. No real-world performance is claimed.

---

## SLIDE 1 - RevenuePulse AI

# RevenuePulse AI
### AI-assisted account prioritization for revenue-risk review

**Recommendation: ITERATE toward a controlled real-data paid pilot. Do not scale.**

Decision support only · human review mandatory · synthetic data

---

## SLIDE 2 - Problem and Business Stakes

# The review window is smaller than the book

- A manager owns **150-400 accounts** and can review perhaps **50 per quarter**.
- Account history is fragmented across billing, CRM and product tools.
- Dashboards show the past; they do not rank the future.
- Rules that flag declining revenue fire **after** the decline is visible.

**Stakes**
- Review effort follows recency and familiarity, not risk.
- Declines surface at renewal instead of at the point of intervention.
- Two managers with the same book would prioritise differently, and could not explain why.

*No market-size or loss figures are claimed; none were measured.*

---

## SLIDE 3 - Target User and Current Workflow

# Who this is for, and what they do today

**Primary users:** sales managers · key account managers · RevOps analysts

**Current workflow**
1. Export or open account data from several systems.
2. Manually scan revenue, engagement, activity and support history.
3. Form a judgement about which accounts look risky.
4. Decide outreach or no action.
5. Record a follow-up in notes or a CRM task.

**Failure modes:** inconsistent ordering, no shared definition of material decline, and accounts that are simply never looked at.

*(Workflow diagram: `diagrams/initial_workflow.puml`)*

---

## SLIDE 4 - Why AI / Why Not Rules Alone

# Ranking at review capacity, not another dashboard

| Approach | What it gives | Limit |
|---|---|---|
| Dashboard | History and filters | No ordering against a future outcome |
| Hand-written rule | Transparent and cheap | Encodes existing beliefs; misses interactions |
| Learned model | Combined signal, calibrated ordering | Needs data discipline, monitoring, explanation |

**Locked synthetic test at 50 reviews (12.6% prevalence)**

| Model | Precision@50 | Lift@50 | PR-AUC |
|---|---|---|---|
| Transparent scorecard | 0.140 | 1.11 | 0.136 |
| Logistic regression | 0.480 | 3.82 | 0.350 |
| Histogram gradient boosting | **0.540** | **4.30** | **0.382** |

27 of 94 future decliners in 50 review slots, versus 7 for the rule.
**Synthetic data only.** The rule baseline is weak, so this is not proof that ML is required.

---

## SLIDE 5 - Product Concept + MVP

# A quarterly queue, evidence, and a human decision

**Four-page Streamlit prototype**
1. **Executive Dashboard** - review counts by risk band, model in use, synthetic warning.
2. **Review Queue** - ranked accounts with probability, band, recent revenue, trend, engagement, activity recency, primary reason, review status; segment/band/top-N filters; CSV export.
3. **Account Detail** - monthly revenue history, commercial signals, "Factors associated with this risk score", rule-based next-best-action with evidence.
4. **Human Review** - accept / edit / dismiss with note; SQLite audit log; CSV export.

**Boundary:** no customer contact, no pricing, no offers, no emails, no CRM writes. Human approval is mandatory.

*(Workflow: `diagrams/final_workflow.puml`)*

---

## SLIDE 6 - Data, Label and Model Design

# Synthetic data, a transparent label, leakage-safe features

- **Data:** 400 synthetic accounts x 36 months; quarter-end snapshots; 3,921 labelled snapshots; no personal identifiers.
- **Label:** next 3-month mean revenue <= 80% of trailing 3-month mean (20% decline); >= 6 months history; reference >= 5,000 units. A **revenue-decline proxy, not churn**.
- **Features:** 22 allow-listed point-in-time features; 3/6/12-month revenue means and trends, transactions, engagement, activity recency, products, support, contract age.
- **Leakage controls:** as-of filtering, separate outcome window, explicit allow-list, time-ordered splits, duplicate rejection, mutation tests.
- **Preprocessing:** median imputation and one-hot encoding fitted on train only; no rows dropped for short-history nulls.
- **Models:** logistic regression and histogram gradient boosting; selection on validation only (Precision@50, PR-AUC tie-break, 0.02 interpretability margin).
- **No test-set tuning. No class weighting** - validation showed no ranking gain and worse calibration.

---

## SLIDE 7 - Evaluation Results

# Locked test results

**748 snapshot rows · 380 accounts · 94 positives · 12.6% prevalence**

| Model | P@10 | P@25 | P@50 | Recall@50 | Lift@50 | FA@50 | PR-AUC | Brier |
|---|---|---|---|---|---|---|---|---|
| Scorecard | 0.200 | 0.200 | 0.140 | 0.074 | 1.11 | 0.860 | 0.136 | 0.243 |
| Logistic regression | 0.500 | 0.560 | 0.480 | 0.255 | 3.82 | 0.520 | 0.350 | 0.097 |
| HGB (selected) | 0.600 | 0.720 | 0.540 | 0.287 | 4.30 | 0.460 | 0.382 | 0.094 |

- HGB Precision@50 **0.540 [0.400, 0.700]** (account-level bootstrap, 1,000 iterations)
- HGB minus scorecard Precision@50 **+0.400 [+0.240, +0.560]**
- Calibration acceptable without a calibrator: Brier 0.094, mean bin gap 0.030
- Recall@50 0.287: most decliners are outside the queue

**SYNTHETIC DATA - NOT REAL-WORLD PERFORMANCE.** The dominant learned driver (positive six-month trend) is a generator property.

---

## SLIDE 8 - Prototype Walkthrough

# From queue to audited decision

1. **Dashboard** - 370 accounts, 6 high / 29 medium / 335 low for 2025-09-30.
2. **Queue** - filtered to Enterprise and high/medium, top 25; high-risk rows highlighted; medium shown as "Review recommended".
3. **Account** - probability, band, revenue history, signals, reason codes (e.g. "recent revenue decline; revenue well above its prior six-month average").
4. **Suggestion** - rule-based action with the exact values that triggered it.
5. **Decision** - accept / edit / dismiss, optional note, stored with timestamp, score and AI text in `runtime/review_feedback.db`; log exportable to CSV.

**Failure behaviour:** missing data or model stops the page with a recovery command; accounts with no core fields are excluded, never scored.

Run: `streamlit run prototype/app.py`

---

## SLIDE 9 - Commercial Roadmap + Paid Pilot

# Sell the diagnostic, then the pilot, then the product

| Phase | Scope | Duration | Price |
|---|---|---|---|
| **0 - Diagnostic sprint** | Data audit, workflow discovery, baseline feasibility | 12-16 hours | ₹50k-₹75k |
| **1 - Paid pilot** | One segment, controlled prioritisation, human decisions logged | 6-8 weeks | ₹1.5L-₹2.5L |
| **2 - Productised SaaS** | Auth, roles, monitoring, standard onboarding | Post-pilot | Subscription / enterprise licence |
| **3 - Scale** | CRM integration, multi-user, multi-tenancy | Later | Enterprise tiers |

**Pilot terms:** fixed fee, 50% upfront, 50% at readout, two-week termination for convenience, no auto-renewal, no success fee.
**Data:** account IDs and monthly aggregates only, no names/emails, encrypted transfer, deleted after 30 days, no LLM by default.
**Success criteria agreed first:** triage time, reviewer acceptance, Precision@K vs client's rule, false-alarm tolerance, one documented intervention.
No product-market fit is claimed.

---

## SLIDE 10 - Risk, Governance and Human Oversight

# What could go wrong, and what controls it

| Risk | Control |
|---|---|
| Synthetic-to-real gap | Real-data pilot with pre-agreed stop conditions; no scale claim |
| Counter-intuitive driver | Disclosed in model card and queue context; validation on real data |
| False alarms / missed declines | Precision@K, lift and false-alarm rate reported; Recall@50 0.287 disclosed |
| Automation bias | Mandatory accept / edit / dismiss; override logged; notes encouraged |
| Privacy | Aggregate-only data, no personal identifiers, deletion policy |
| Unauthorised automation | No outbound integration in code; guardrail test; prohibited-use list |
| Drift and decay | Monitoring plan: drift, calibration, Precision@K per window, rollback |
| Missing auth in MVP | Documented limitation; role-based access is a pilot blocker |

**STOP if:** no lift over the client's rule, unacceptable false alarms, poor acceptance, uneconomic data access, uncontrollable governance, no workflow benefit, or failing real-data performance.

---

## SLIDE 11 - Recommendation and Ask

# Iterate, with a controlled real-data pilot

**What the evidence supports**
- The method is sound: leakage-safe features, validation-only selection, capacity-aware evaluation, account-level uncertainty.
- On synthetic data the model materially outranks a transparent rule (Precision@50 0.540 vs 0.140).

**What it does not support**
- Any claim about real accounts, churn, retention or revenue.

**Ask: approve controlled real-data pilot exploration, not scale.**
- Phase 0 diagnostic sprint (₹50k-₹75k, 12-16 hours) to validate data access, label materiality and the current workflow.
- Then a 6-8 week fixed-fee pilot with pre-agreed success criteria and a real stop option.

**Thank you.** Questions welcome on evaluation design, governance and the prototype.
