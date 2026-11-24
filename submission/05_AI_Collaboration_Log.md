# AI Collaboration Log

**Project:** RevenuePulse AI - MBA ZG583 capstone
**Student:** Suhani Kanwar
**Build dates recorded in the repository:** 24 November 2026 (Phase 1, Phase 2 and the
final submission build were completed in the same working day).

> **Accuracy note on tools.** Tool names below reflect the toolchain used for this
> project and this build session. Where the underlying model name was not recorded, it
> is stated as not recorded rather than guessed. Confirm the exact names before
> submission if your tooling reports them differently.

## Tools Used

| Tool | Role in this project | Notes |
|---|---|---|
| OpenCode (CLI coding agent) | Repository scaffolding, Phase 1/2 implementation, test runs | Underlying model/version not recorded in the repository; confirm from your tool history |
| Freebuff / Codebuff coding agent | Final submission build: Streamlit prototype, tests, PM artifacts, diagrams, presentation | Agent name is recorded in the build session; model version not exposed to the repository |
| ChatGPT | Project planning and prompt drafting, if used | **Not verifiable from the repository.** Add the date/use here only if it was actually used; do not claim it otherwise |
| Python 3.12, scikit-learn 1.6.1, pandas 2.2.3, Streamlit 1.67.0 | Runtime stack | Versions recorded in `requirements.txt` and model metadata |
| PlantUML | Workflow diagrams | Source files in `diagrams/`; rendering optional |
| pytest | Automated verification | 130+ tests, including the Phase 1 leakage suite |

No AI tool was given real customer data, and no AI tool was used to generate
explanations shown to end users.

## Where AI Helped

- **Assignment interpretation** - converting a broad brief into phase boundaries
  (data foundation, model evaluation, product MVP) with explicit exclusions.
- **Architecture options** - comparing build vs buy vs API, and choosing a local,
  inspectable scikit-learn pipeline over hosted services.
- **PRD drafting** - functional/non-functional requirements, acceptance criteria,
  personas and user stories.
- **Synthetic data design** - account-month generator with archetypes, seasonality,
  shocks and realistic missingness, plus a committed demo subset.
- **Code scaffolding** - data schema and validation, snapshot builder, feature
  allow-list, label, split assignment, configuration loader.
- **Leakage testing** - tests that mutate future revenue and prove earlier features do
  not change; forbidden-column checks; time-ordering assertions.
- **Model evaluation** - baseline scorecard, logistic regression, histogram gradient
  boosting, validation-only selection, Precision@K/lift/PR-AUC, calibration,
  account-level bootstrap and segment analysis.
- **Streamlit prototype** - four pages, product service layer, SQLite review log,
  failure handling and disclaimers.
- **Documentation** - model card, evaluation report, error analysis, build notes,
  submission artifacts.
- **Risk review** - surfacing false positives/negatives, automation bias, drift and
  governance gaps, and drafting stop conditions.

## Example Prompts or Workflows

The prompts below are **faithful summaries**, not exact transcripts. They capture the
intent and constraints given to the coding agent.

1. **Phase 1 build prompt (summarised).** "Build the repository and data foundation for
   RevenuePulse AI: configurable assumptions, a reproducible synthetic B2B
   account-month generator, a data dictionary, strict quality checks, point-in-time
   quarter-end snapshots, a transparent revenue-decline proxy label with a 3-month
   horizon and 20% threshold, time-ordered train/validation/test splits, and leakage
   tests. No model training, no UI, no LLM. Synthetic data only; no personal data; no
   fabricated metrics."

2. **Phase 2 model/evaluation prompt (summarised).** "Implement only Phase 2: a
   transparent rule baseline, train-fitted preprocessing, logistic regression and
   histogram gradient boosting, validation-only model selection, one locked test
   evaluation with Precision@K, recall@K, lift@K, PR-AUC, Brier and account-level
   bootstrap intervals, calibration, reason codes, a saved artifact and a model card.
   Do not change the label to improve results, do not tune on test, do not add a UI or
   LLM, and label all results as synthetic."

3. **Final submission build prompt (summarised).** "In a short submission window,
   turn the existing repository into the most complete honest submission: a four-page
   Streamlit prototype using the frozen artifact with accept/edit/dismiss and a SQLite
   audit log; minimal product tests; executive summary, final report, PM artifacts,
   ethics note, AI collaboration log, self-evaluation and contribution statement;
   PlantUML before/after workflows; an 11-slide deck; a rewritten README; GitHub
   readiness. Do not retrain or tune models, do not fabricate evidence, do not
   recommend scale."

4. **Verification workflow used repeatedly.** "Run the full test suite, then regenerate
   data, rebuild snapshots, retrain, re-evaluate and confirm the metrics are identical
   to the documented values." This caught a configuration failure before it reached
   documentation.

## Where AI Was Weak or Wrong

These are real, documented issues from this repository. No failures are invented.

1. **Phase 2 configuration mismatch (real bug).** The configuration validator and the
   baseline expected different threshold key names: `config.yaml` used
   `revenue_decline_pct` / `transaction_decline_pct` / `inactivity_days` /
   `engagement_decline_pct`, while validation required keys named exactly like the four
   signals. Loading the config raised `ConfigError` and blocked all model work until the
   validators were aligned with the actual keys. No threshold value was changed.

2. **Reason codes too narrow on first attempt (real design miss).** The first reason
   implementation reused only the scorecard's four decline rules. After scoring the
   test split, 32 of the top 50 highest-risk snapshots had no reason code, because the
   selected booster's top drivers are six-month revenue trend and engagement level. The
   rules were widened to seven configurable, feature-grounded codes, reaching coverage
   of 49 of 50. This was an explainability fix only; scores were unchanged.

3. **Class weighting default was not evidence-based (real scaffold issue).** The Phase 2
   scaffold set `class_weight="balanced"`. A validation-only comparison showed no
   consistent ranking gain and a materially worse Brier score (HGB 0.086 to 0.114), so
   the configuration was changed to `null` with the evidence recorded in
   `config.yaml` and the build notes.

4. **Synthetic generator produced a counter-intuitive signal (real finding).** In the
   generated data, a *positive* six-month revenue trend is associated with the future
   decline label (train mean +0.051 vs -0.074), and the fitted models learn that
   direction. AI-generated data design created a pattern that contradicts normal churn
   intuition; the response was to disclose it in the model card and recommend a
   real-data pilot rather than to redesign the generator late in the project.

5. **Human judgement required for model selection (not an error, a limit).** AI can
   apply a pre-declared rule (Precision@K with a PR-AUC tie-break and an interpretability
   margin) but cannot decide whether a 0.06 Precision@50 gap is commercially meaningful.
   The student chose to keep the pre-registered rule and record the interpretability
   trade-off, rather than retrofit a preference after seeing results.

6. **Environment and document-format issues.** The project notes record that the machine
   initially had only the Microsoft Store Python stub, that Python 3.12 had to be
   installed into a local `.venv`, and that Git was not available at Phase 1 time. A
   Word-based PDF conversion failure and an early folder/path problem are **not**
   documented in this repository, so they are not claimed here. Add them only if your
   own records support them.

## Human Corrections

- **Fixed the configuration mismatch** by aligning both validators to the threshold
  keys actually used in `config.yaml`; no model or label value was changed.
- **Retained the transparent baseline** as a required control, even though it performed
  near randomly (Precision@50 0.140), instead of removing an unflattering comparison.
- **Rejected class weighting** because validation evidence did not justify it; the
  probability scale was kept interpretable for reviewers.
- **Preserved the human-in-the-loop boundary** - the prototype cannot contact a
  customer, change a price or write to a CRM, and a test scans the product layer for
  outbound integrations.
- **Chose not to claim real-world performance** anywhere, and kept every metric
  labelled synthetic, including in the executive summary and the deck.
- **Changed model selection only through the pre-declared validation rule**, not by
  observing test results, and kept the test evaluation single-pass.
- **Widened reason-code coverage** after seeing that the explanation layer failed on
  most high-risk accounts, without touching the model.

## Final Reflection

**AI accelerated:** drafting (documents, PRD, model card), coding (data pipeline,
snapshots, models, evaluation, Streamlit prototype), testing (130+ automated checks
including leakage and guardrails), and documentation consistency across phases.

**AI could not replace:** choosing the problem and the wedge; deciding that synthetic
evidence is insufficient to scale; accepting the commercial and ethical risk of a
pilot; judging whether a synthetic pattern is plausible; and interpreting what the
numbers do and do not mean. Those remained human decisions, and the most important
output of this project - the decision to recommend a controlled pilot rather than a
launch - was a human judgement that the AI's own results supported but could not make.

## Collaboration Record

Dates are the recorded build dates; earlier planning conversations are not timestamped
in the repository.

| Date | Tool | Task | Prompt/workflow | Output used | Verified? | Human correction | Artifact |
|---|---|---|---|---|---|---|---|
| 2026-11-24 | OpenCode | Phase 1 data foundation | Summarised Phase 1 build prompt | Schema, generator, quality checks, snapshots, tests | Yes - `pytest` (47 Phase 1 tests) | Fixed environment (Python stub, venv); reviewed label rule | `src/`, `tests/`, `docs/Data_Dictionary.md` |
| 2026-11-24 | OpenCode | Phase 2 baseline and models | Summarised Phase 2 prompt | Scorecard, LR, HGB pipelines, selection rule | Yes - full suite and regenerated metrics | Corrected baseline threshold-key mismatch | `src/models/`, `config.yaml` |
| 2026-11-24 | OpenCode | Phase 2 evaluation | Evaluation prompt with metrics and bootstrap | `test_results.json`, calibration, segments, bootstrap | Yes - reran from clean data; identical metrics | Rejected class weighting on validation evidence | `docs/Evaluation_Results.md` |
| 2026-11-24 | OpenCode | Reason codes | Explainability requirement in Phase 2 prompt | Seven rule-based codes and driver-alignment table | Yes - coverage check (49/50) | Widened rules after 32/50 had no code | `src/models/reasons.py` |
| 2026-11-24 | Freebuff / Codebuff | Product layer | Final submission prompt | Queue service, recommendations, SQLite feedback store | Yes - 130 tests including ranking and feedback | Kept decision-support boundary; excluded CRM/email | `src/product/` |
| 2026-11-24 | Freebuff / Codebuff | Streamlit prototype | Four-page MVP requirement | `prototype/app.py` with four pages | Yes - Streamlit AppTest run; dashboard rendered 370/6/29/335 | Simplified visual design; added explicit failure states | `prototype/app.py` |
| 2026-11-24 | Freebuff / Codebuff | Submission artifacts | Parts 4-15 of final prompt | Executive summary, report, PM pack, ethics, AI logs, deck | Partially - documents reviewed for factual consistency with artifacts | Enforced synthetic-only claims; removed unsupported metrics | `submission/`, `presentation/` |
| 2026-11-24 | Freebuff / Codebuff | README, diagrams, GitHub readiness | Final prompt | README, PlantUML workflows, upload instructions | Yes - commands rerun; diagrams syntax-checked where tooling exists | Final review before submission | `README.md`, `diagrams/`, `GITHUB_UPLOAD_INSTRUCTIONS.md` |
