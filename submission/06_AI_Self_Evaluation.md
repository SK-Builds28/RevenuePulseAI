# AI Self-Evaluation - Independent Review

**Role:** independent AI product reviewer assessing the finished RevenuePulse AI
repository against the nine assignment dimensions.

**Basis of review:** repository contents only (`README.md`, `docs/`, `submission/`,
`src/`, `prototype/`, `tests/`, generated artifacts). No real customer data exists, so
no real-world performance claim can be assessed favourably or unfavourably.

**Overall:** a technically coherent, honestly bounded capstone with a working
decision-support prototype. Its strongest dimension is evaluation discipline; its
weakest are commercial evidence and production readiness, both correctly acknowledged
rather than papered over.

## Scores

| # | Dimension | Score /5 |
|---|---|---|
| 1 | Product framing | 4 |
| 2 | Strategy | 3 |
| 3 | PM artifacts | 4 |
| 4 | UX / workflow | 3 |
| 5 | Data / model / evaluation | 4 |
| 6 | Business / scaling | 3 |
| 7 | Ethics / governance | 4 |
| 8 | Prototype | 4 |
| 9 | AI collaboration | 4 |

### 1. Product framing - 4/5

**Evidence:** Clear single decision ("which accounts to review first, and why"),
explicit non-goals (no contact, pricing, offers, CRM action), named users with
distinct jobs, and a label that is honestly framed as a revenue-decline proxy rather
than churn.

**Weakness:** The wedge is defined by account count and cadence rather than by an
observed painful workflow from interviews; no primary user research exists, so the
problem statement rests on plausible reasoning.

**Improvement:** Five structured discovery interviews with sales managers, documented
with themes (not quotes), to test whether quarter-end triage is the real bottleneck
versus data quality or meeting cadence.

### 2. Strategy - 3/5

**Evidence:** Build-vs-buy-vs-API reasoning, a staged commercial path (diagnostic
sprint, paid pilot, SaaS, scale), and a disciplined recommendation not to scale on
synthetic evidence.

**Weakness:** No competitive scan, no validated willingness to pay, no segment-level
market evidence, and the "why now" is asserted rather than argued with data.

**Improvement:** A documented scan of 5-8 adjacent tools (CRM scoring, CS platforms,
BI templates) on price, data needs and label transparency, plus 3-5 pricing
conversations before the pilot.

### 3. PM artifacts - 4/5

**Evidence:** PRD with functional and non-functional requirements, acceptance
criteria, personas, jobs-to-be-done, user stories, prioritisation table, MVP scope,
metrics framework and roadmap; requirements traceable to prototype screens.

**Weakness:** Acceptance criteria are written but not all are automated; requirements
are not versioned with owners, and there is no explicit release checklist or
definition-of-done per feature.

**Improvement:** Add a traceability matrix (requirement -> screen -> test), an owners
column, and a release checklist used at the pilot boundary.

### 4. UX / workflow - 3/5

**Evidence:** The four-page flow mirrors the review ritual: dashboard, queue, detail,
decision. Filters map to real capacity. Reason codes and evidence sit next to the
score. Failure states are handled.

**Weakness:** No usability testing, no keyboard-first review flow, no side-by-side
comparison of accounts, no bulk actions; visual design is functional rather than
considered, and the 0.50-threshold view can feel arbitrary to a reviewer.

**Improvement:** Time-to-decision study with three reviewers on ten accounts, then add
a compact review mode (single account, large type, next/previous) and bulk dismiss
with a required reason.

### 5. Data / model / evaluation - 4/5

**Evidence:** Point-in-time snapshots, allow-listed 22 features, leakage tests that
mutate future revenue, train-only preprocessing, validation-only selection, single
locked test pass, account-level bootstrap, calibration table, segment and error
analysis. Precision@50 0.540 with CI [0.400, 0.700] and a documented weak baseline.

**Weakness:** Synthetic only; the dominant driver is counter-intuitive; the baseline is
a weak control; two test quarter-ends; no rolling-origin validation; no drift
monitoring; no cost-weighted evaluation.

**Improvement:** Add a strong simple baseline (level/volatility/trend), rolling-origin
evaluation across at least four quarterly folds, and a cost model that turns K into an
explicit trade-off.

### 6. Business / scaling - 3/5

**Evidence:** Staged pricing hypotheses, pilot terms, data-handling terms, gross-margin
conditions and a value formula with editable components.

**Weakness:** CAC unknown, no LTV/payback calculation (correctly withheld), no
validated price point, no pilot run, no support-cost data. Scaling describes
architecture, not an operating plan.

**Improvement:** Run the diagnostic sprint with one design partner and publish actual
hours and outputs; use it to replace one hypothesis with measured data before the
pilot.

### 7. Ethics / governance - 4/5

**Evidence:** Synthetic/real boundary stated everywhere, prohibited uses, stop
conditions, human approval required, local audit log, guardrail test against outbound
integrations, drift and monitoring gaps listed.

**Weakness:** No authentication or role separation in the prototype; no retention
automation; fairness cannot be assessed; automation-bias mitigations are procedural
and untested.

**Improvement:** Add role-scoped access and an audit-log retention policy before any
real pilot, and define a segment-level monitoring rule that triggers review if a group
is systematically under-prioritised.

### 8. Prototype - 4/5

**Evidence:** Runs locally from a frozen artifact; four coherent pages; scores are
computed, not typed; reason codes and recommendations are evidence-linked; decisions
persist to SQLite and export to CSV; controlled errors for missing data/model; 130+
tests including product-layer guardrails.

**Weakness:** No auth or deployment; single-user; no screenshots committed; limited
visual polish; no UI automation beyond an import/AppTest smoke check.

**Improvement:** Deploy a private instance for the pilot with SSO, add Playwright
coverage for the accept/edit/dismiss path, and capture a short demo video.

### 9. AI collaboration - 4/5

**Evidence:** A specific, non-generic collaboration log: three summarised prompts, a
dated activity table with verification status, and four documented real failures
(config mismatch, narrow reason codes, unjustified class weighting, counter-intuitive
synthetic signal) plus the human corrections that followed.

**Weakness:** Some tool/model names are unverifiable from the repository; two failures
sometimes mentioned in conversation (folder/path and Word-to-PDF) are not documented,
so they are correctly not claimed; verification depth per row varies.

**Improvement:** Maintain a live log during the build with exact timestamps and tool
versions, and attach a verification artifact (test output, commit hash) to each row.

## Ambiguities Found

1. **Individual vs team submission.** The project notes record a conflict between the
   assignment packet (teams of 3-5) and the Topic Approval Form (individual). This
   submission treats it as individual; faculty confirmation is still referenced.
2. **"Churn" vs revenue decline.** The brief uses churn language, but no cancellation
   field exists. The project defines a revenue-decline proxy and refuses to call it
   churn; this is a deliberate interpretation, not a data finding.
3. **Materiality is undefined.** "Material decline" needs a business definition. The
   project fixed 20% over three months in Phase 1 and did not change it; a client may
   choose differently, which would change prevalence and every metric.
4. **Acceptance rate.** The assignment metric list includes recommendation acceptance,
   but no human review experiment exists. The project reports it as unmeasured rather
   than inventing it.
5. **How much UI is "MVP".** The brief allows a simple prototype; the four-page scope
   is an interpretation of "minimum functional".
6. **Pricing inputs.** The ₹50k-₹75k diagnostic and ₹1.5L-₹2.5L pilot ranges are given
   assumptions, not validated prices. The report labels them hypotheses.
7. **AI tool identity.** The brief asks for "the AI model used if known". Model
   versioning is not available in the repository, so it is marked not recorded.

## Hardcoded vs Configurable

| Element | Current state | Hardcoded / configurable | Improvement |
|---|---|---|---|
| Label rule (3-month horizon, 3-month reference, 20% threshold, 5,000 floor, 6-month history) | `config.yaml` -> `label` | Configurable | Add a client-facing materiality worksheet and re-evaluation when changed |
| Risk-band thresholds (0.65 / 0.35) | `config.yaml` -> `risk` | Configurable | Expose in the UI with a "why this threshold" note |
| Precision@K values and primary K (10/25/50) | `config.yaml` -> `evaluation` | Configurable | Derive K from measured reviewer capacity in the pilot |
| Model settings (LR `C`, iterations; HGB depth, leaf nodes, L2) | `config.yaml` -> `models` | Configurable, not searched | Document a small, pre-registered search only if data grows |
| Class weighting | `null` in config, with validation evidence in comments | Configurable | Re-check per client prevalence |
| Reason-code thresholds | `config.yaml` -> `reasons` | Configurable | Add per-segment calibration only with evidence |
| Reason-code rule set (7 rules, feature mapping, direction) | `src/models/reasons.py` `REASON_RULES` | Hardcoded | Move rule definitions to config if clients need custom codebooks |
| Recommendation action mapping and order | `src/product/recommendations.py` | Hardcoded (thresholds reused from config) | Add a config-driven action catalogue in Phase 3 |
| Scorecard weights and thresholds | `config.yaml` -> `baseline` | Configurable | Keep frozen for pilot comparability |
| Review database path | `config.yaml` -> `feedback.database_path` | Configurable | Point to an access-controlled store in production |
| Feedback decision vocabulary | `src/product/feedback.py` `DECISIONS` | Hardcoded | Add "deferred" / "escalated" if the pilot needs them |
| Pricing and cost assumptions | `submission/` documents | Hardcoded assumptions, labelled hypothetical | Replace with measured pilot data |
| Bootstrap iterations/seed, calibration bins | `config.yaml` -> `evaluation` | Configurable | Keep fixed for comparability across reports |
| Segment analysis minimums (30 rows, 5 positives) | `src/evaluation/metrics.py` defaults | Hardcoded | Configurable if cohort sizes differ |

## Requirement Coverage

| Requirement | Status | Evidence | Gap |
|---|---|---|---|
| Executive Summary | Done | `submission/01_Executive_Summary.md` | Not exported to PDF |
| Final Report (34 sections) | Done | `submission/02_Final_Report.md` | No PDF export yet |
| PM Artifacts | Done | `submission/03_PM_Artifact_Pack.md` | No wireframes folder |
| Functional Streamlit prototype | Done | `prototype/app.py`, AppTest smoke run | No auth, no deployment |
| Account prioritisation + explanations + recommendation + review log | Done | `src/product/`, SQLite log | Recommendation catalogue is fixed code |
| Fallback and trust design | Done | Controlled errors, disclaimers, band wording | No runtime monitoring |
| Minimal product tests | Done | `tests/test_product_queue.py`, `test_feedback.py`, `test_prototype_app.py` | No UI click automation |
| Ethics / Risk | Done | `submission/04_Ethics_Governance_Risk.md` | No fairness testing possible |
| AI Collaboration Log | Done | `submission/05_AI_Collaboration_Log.md` | Some tool names unverifiable |
| AI Self-Evaluation | Done | This document | Self-assessed, not external |
| Individual Contribution Statement | Done | `submission/07_Individual_Contribution_Statement.md` | Signatures/IDs placeholders |
| PlantUML workflows | Done | `diagrams/initial_workflow.puml`, `final_workflow.puml` | PNG only if tooling exists |
| Presentation deck | Done | `presentation/Final_Presentation.md` (+ PPTX if generated) | Visual polish |
| README landing page | Done | `README.md` | — |
| GitHub readiness | Done | Repository commits or `GITHUB_UPLOAD_INSTRUCTIONS.md` | Remote not configured |
| Real-world validation | Not done (out of scope) | Stated in every artifact | Requires a pilot |

## What would I improve with two more weeks?

1. **A strong simple baseline** (level, volatility, six-month trend). Without it, the
   model-versus-rule comparison overstates the value of ML; this is the single most
   important missing experiment.
2. **Rolling-origin evaluation** across four quarterly folds instead of two test
   quarters, so the ranking claim has temporal stability evidence rather than a single
   window.
3. **Label validation with a finance stakeholder**: what counts as material decline,
   and how billing timing and seasonality should be excluded. This changes prevalence
   and precision, and it should be settled before a pilot.
4. **Real-data ingestion adapter** with PII rejection, schema mapping and a data-quality
   report, tested on a synthetic "messy client export".
5. **Pilot-grade operations**: authentication, role-based access, monitoring (drift,
   calibration, Precision@K per window), audit-log retention and a rollback runbook.
6. **UX research**: time-to-decision with three reviewers, then a compact single-account
   review mode and a required-reason bulk dismiss.
7. **Automated UI tests** for the accept/edit/dismiss path, plus a short demo video for
   the appendix.
8. **Cost model** linking review capacity, false-alarm cost and value of caught decline,
   so K becomes an explicit business trade-off instead of a capacity assumption.

None of these is claimed as done. They are the honest next steps.
