# RevenuePulse AI - PM Artifact Pack

All product claims below are design intent. Model numbers are synthetic-data numbers
from the locked test evaluation and are labelled as such.

## Problem Statement

B2B sales managers responsible for 150-400 recurring-revenue accounts cannot manually
inspect the whole book each quarter. Account history is fragmented across billing,
CRM and product tools, and existing dashboards show the past without ranking the
future. Review time therefore follows recency and familiarity, and material revenue
declines are often recognised only after they appear in revenue reporting - when the
intervention window has narrowed.

## Product Vision

Every account review starts from a defensible priority order: the accounts most
likely to decline materially in the next 90 days, each with checkable evidence and a
suggested next step, and every human decision recorded so the ranking can be audited
and improved. The product assists the judgement of a sales professional; it never
acts on a customer.

## Jobs to Be Done

| Job | When | Want | So that |
|---|---|---|---|
| Prioritise | Quarter-end review starts | Accounts ordered by likely revenue decline | Review time goes to accounts that need it |
| Explain | Reviewing one account | Evidence behind the score | I can justify action or non-action |
| Decide | After reading the suggestion | Accept / edit / dismiss | AI assists without replacing judgement |
| Govern | Reviewing the quarter | An audit trail of decisions | The process is defensible and improvable |

## Personas / Users

**Priya - Sales Manager (primary).** Owns 260 mid-market accounts. Has 90 minutes a
week for account review. Distrusts black-box scores. Success = a short queue she can
defend in a leadership meeting.

**Arjun - Key Account Manager (primary).** Runs 30 relationships. Needs a reason to
start a conversation that is not "your usage is down". Success = evidence he can
quote and a concrete first question.

**Meera - RevOps Analyst (secondary).** Owns definitions, data quality and
reporting. Success = a reproducible definition of risk, no manual spreadsheet
maintenance, and the ability to explain how the number was produced.

**Governance reviewer (secondary).** Approves data access. Success = aggregate-only
data, no customer contact, visible human control, documented failure modes.

## User Stories

- As a sales manager, I want accounts ranked by likely revenue decline, so that I can
  focus limited review capacity.
- As an account manager, I want to understand why an account received a high-risk
  score, so that I can decide whether action is justified.
- As a reviewer, I want to accept, edit or dismiss recommendations, so that AI
  assists rather than replaces my judgment.
- As a sales manager, I want to filter the queue by segment and risk band, so that I
  can run a segment-specific review meeting.
- As a reviewer, I want to see recent revenue, trend, engagement and days since
  activity next to the score, so that I can sanity-check the model.
- As a RevOps analyst, I want the review decisions exported as a CSV, so that I can
  report on what was reviewed and decided.
- As a governance reviewer, I want the system to refuse to score accounts with
  missing required fields, so that no decision rests on an invented number.

## PRD

### Functional requirements

| ID | Requirement | Acceptance summary |
|---|---|---|
| FR-1 | Select a review date from available quarter-ends | Dates come from the snapshot file, newest first |
| FR-2 | Score accounts using the frozen model artifact | Probabilities in [0, 1]; no hardcoded scores |
| FR-3 | Rank descending by probability, deterministic ties | Rank 1..N stable across reloads |
| FR-4 | Assign risk bands from configuration | high >= 0.65, medium >= 0.35, else low |
| FR-5 | Queue shows account, segment, probability, band, revenue, trend, engagement, recency, primary reason, status | All columns render |
| FR-6 | Filter by segment, band and top N | Filters compose; top N applies last |
| FR-7 | Highlight high risk; medium labelled "Review recommended" | Visual differentiation |
| FR-8 | Account detail with monthly revenue chart and signals | Chart uses months <= snapshot date |
| FR-9 | Reason codes from actual features | No code without a matching feature threshold |
| FR-10 | Deterministic next-best-action with evidence | Same input -> same action and evidence |
| FR-11 | Accept / edit / dismiss with optional note | Edit requires amended text |
| FR-12 | Persist decisions to SQLite | Timestamp, account, date, score, AI text, decision, edit, note |
| FR-13 | Export queue and log to CSV | Downloads contain the displayed rows |
| FR-14 | Controlled failure on missing data/model | Error message with recovery command; no fake score |
| FR-15 | Authentication and role-based access | Out of MVP scope; pilot prerequisite |
| FR-16 | Automated customer contact or CRM write | Explicitly excluded; guarded by test |

### Non-functional requirements

| Area | Requirement |
|---|---|
| Determinism | Fixed seeds; repeated scoring returns identical probabilities |
| Performance | Queue for ~400 accounts renders in under 5 seconds locally |
| Explainability | Every queue row exposes a reason or states that no rule fired |
| Auditability | Every decision stored with context; log exportable |
| Privacy | No personal identifiers; aggregate account data only |
| Fail-safe | No score without required fields; no silent fallback to fabricated values |
| Portability | Python 3.12, one virtual environment, no network at runtime |
| Honesty | Synthetic-data and non-churn disclaimers visible on every page |
| Extensibility | Thresholds, K values, model parameters and DB path in `config.yaml` |

## Acceptance Criteria

1. **Queue ranking** - Given a selected snapshot date and the frozen artifact, when
   the queue renders, then rows are ordered by descending risk probability, ranks are
   unique and sequential, and every probability is between 0 and 1.
2. **No invented score** - Given a snapshot row whose core revenue, transaction and
   engagement fields are all missing, when scoring runs, then the row is excluded and
   the excluded count is shown, never a placeholder probability.
3. **Reason integrity** - Given an account with no reason-code threshold triggered,
   when the detail page renders, then it states that no configured rule fired instead
   of inventing a reason.
4. **Human control** - Given a suggestion, when the reviewer chooses Accept, Edit or
   Dismiss and saves, then a record is written with timestamp, account, snapshot date,
   risk score, AI recommendation, decision, edited text and note, and no customer
   contact occurs.
5. **Fallback** - Given a missing model artifact, when the app starts, then a
   controlled error names the missing path and the retrain command.
6. **Persistence** - Given at least one saved decision, when the log is exported, then
   the CSV contains the same rows with all audit columns.
7. **Guardrail** - Given the product source tree, when guardrail tests run, then no
   outbound email, CRM, webhook or LLM integration exists.

## Feature Prioritization

| Feature | User value | Effort | Confidence | Priority |
|---|---|---|---|---|
| Ranked review queue | Core prioritisation | M | High | P0 |
| Risk bands and highlighting | Fast scanning | S | High | P0 |
| Reason codes | Trust and defensibility | M (reuses Phase 2) | High | P0 |
| Account detail with history | Evidence for action | M | High | P0 |
| Accept / edit / dismiss + audit log | Human-in-the-loop proof | M | High | P0 |
| CSV export | Workflow integration | S | High | P0 |
| Failure handling and disclaimers | Trust | S | High | P0 |
| Filters (segment/band/top N) | Capacity management | S | High | P1 |
| Rule-based next-best-action | Reduces "so what" | S | Medium | P1 |
| Auth + roles | Pilot prerequisite | L | Medium | P1 |
| Monitoring and drift | Governance | M | Medium | P1 |
| CRM write-back | Convenience | L | Low | P2 |
| LLM-generated narratives | Marginal value, added risk | M | Low | P3 (rejected) |

## MVP Definition

**In:** one reviewer, one snapshot date at a time, ranked queue with filters, account
detail with reasons and rule-based suggestion, accept/edit/dismiss with SQLite audit
log, CSV exports, defensive failures, visible disclaimers.

**Out:** authentication, multi-user roles, multi-tenancy, CRM integration, LLM text,
automated outreach, pricing actions, deployment.

**MVP success test:** a reviewer can go from opening the tool to a recorded, defensible
decision on a high-risk account in under two minutes, and the decision is auditable.

## Success Metrics

**Product**
- Time to prioritise accounts (baseline vs queue): *not yet measured*
- Recommendation acceptance rate: *not yet measured*

**Model (current synthetic values)**
- Precision@K: 0.540 at K=50 (0.600 at K=10, 0.720 at K=25)
- Recall@K: 0.287 at K=50
- Lift@K: 4.30 at K=50
- PR-AUC: 0.382

**Trust**
- False-alarm rate at K=50: 0.460
- Reviewer override rate: *not measured*
- Reason coverage: 49 of the top 50 test snapshots

**Business**
- Pilot completion: *not started*
- Account review time saved: *hypothesis only*
- Documented intervention signal: *not yet*

Unobserved metrics stay unobserved. The pilot exists to measure them.

## Roadmap Including Commercial Path

| Phase | Scope | Duration | Price |
|---|---|---|---|
| Phase 0 - Diagnostic sprint | Data audit, workflow discovery, baseline feasibility | 12-16 hours | ₹50k-₹75k |
| Phase 1 - Paid pilot | One segment, controlled account-risk prioritisation | 6-8 weeks | ₹1.5L-₹2.5L |
| Phase 2 - Productised SaaS | Auth, roles, monitoring, standard onboarding, subscription | Post-pilot | Recurring subscription / enterprise licence |
| Phase 3 - Scale | CRM integration, multi-user controls, multi-tenancy | Later | Enterprise tiers |

Pilot terms: fixed fee, 50% upfront, 50% at readout, two-week termination for
convenience, no automatic renewal, no success fee in the first pilot. Data: account
IDs and monthly aggregates only, encrypted transfer, deletion after 30 days, no LLM by
default.

Pilot success criteria agreed before start: triage time, reviewer acceptance,
Precision@K versus the client's existing rule, false-alarm tolerance, at least one
documented intervention. No product-market fit claim is made.
